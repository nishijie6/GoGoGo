"""Tests for the durable KataGo board-size curriculum."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from weiqi.rl_curriculum import (
    BenchmarkMeasurement,
    CurriculumConfigError,
    CurriculumController,
    CurriculumLockError,
    CurriculumMigrationError,
    CurriculumStateStore,
    CurriculumStateError,
    GlobalCurriculumLock,
    HealthSummary,
    MatchSummary,
    apply_retention_plan,
    build_benchmark_input,
    build_match_config,
    build_retention_plan,
    choose_autotune_batch,
    compute_health_window,
    create_clean_swa_checkpoint,
    evaluate_quality_gate,
    load_curriculum_config,
    next_evaluation_boundary,
    prepare_stage_migration,
    parse_benchmark_output,
    render_curriculum_dashboard,
    select_replay_window,
    select_retained_directories,
    summarize_match_sgfs,
    validate_training_profiles,
    wilson_interval,
    write_curriculum_dashboard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def curriculum_dict() -> dict[str, object]:
    return {
        "schema_version": 1,
        "state_directory": "training_runs/curriculum",
        "evaluation_interval_samples": 500_000,
        "evaluation_games": 200,
        "promotion_win_rate": 0.60,
        "wilson_lower_bound": 0.50,
        "required_consecutive_passes": 2,
        "health_window_cycles": 10,
        "replay_window_rows": 500_000,
        "keep_models": 20,
        "keep_shuffles": 10,
        "disk_cleanup_gib": 50,
        "disk_pause_gib": 30,
        "max_gpu_memory_gib": 13,
        "stages": [
            {
                "board_size": 9,
                "training_config": "config/rl_training.rtx5070ti.json",
                "min_stage_samples": 10_000_000,
                "autotune_batches": [],
                "indefinite": False,
            },
            {
                "board_size": 13,
                "training_config": "config/rl_training.rtx5070ti.13x13.json",
                "min_stage_samples": 25_000_000,
                "autotune_batches": [512, 1024, 1536, 2048],
                "indefinite": False,
            },
            {
                "board_size": 19,
                "training_config": "config/rl_training.rtx5070ti.curriculum.19x19.json",
                "min_stage_samples": None,
                "autotune_batches": [256, 512, 768, 1024, 1280],
                "indefinite": True,
            },
        ],
    }


def load_test_config(root: Path):
    path = root / "curriculum.json"
    path.write_text(json.dumps(curriculum_dict()), encoding="utf-8")
    return load_curriculum_config(path)


def passing_health() -> HealthSummary:
    return compute_health_window(
        [
            {
                "games": 128,
                "black_wins": 64,
                "white_wins": 64,
                "draws": 0,
                "immediate_double_pass_games": 0,
                "extreme_games": 0,
                "invalid_games": 0,
            }
            for _ in range(10)
        ]
    )


def passing_match() -> MatchSummary:
    lower, upper = wilson_interval(125, 200)
    return MatchSummary(
        requested_games=200,
        games=200,
        candidate_wins=125,
        baseline_wins=75,
        draws=0,
        candidate_black_games=100,
        candidate_white_games=100,
        black_wins=100,
        white_wins=100,
        no_result_games=0,
        damaged_games=0,
        win_rate=0.625,
        wilson_lower=lower,
        wilson_upper=upper,
        elo=0.0,
    )


class CurriculumConfigurationTests(unittest.TestCase):
    def test_pool_names_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = curriculum_dict()
            value["stages"][0]["fixed_baseline_models"] = ["../outside"]
            path = root / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(CurriculumConfigError):
                load_curriculum_config(path)

    def test_strict_schema_and_real_profiles(self) -> None:
        config = load_curriculum_config(
            PROJECT_ROOT / "config" / "rl_curriculum.rtx5070ti.json"
        )
        self.assertEqual([stage.board_size for stage in config.stages], [9, 13, 19])
        self.assertEqual(config.stages[1].autotune_batches, (512, 1024, 1536, 2048))
        self.assertTrue(config.stages[2].indefinite)
        validate_training_profiles(config, PROJECT_ROOT)

    def test_unknown_key_and_wrong_stage_order_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = curriculum_dict()
            value["typo"] = True
            path = root / "bad.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CurriculumConfigError, "未知字段"):
                load_curriculum_config(path)

            value = curriculum_dict()
            value["stages"][0]["board_size"] = 13  # type: ignore[index]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CurriculumConfigError, "顺序"):
                load_curriculum_config(path)

    def test_threshold_and_indefinite_invariants_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            value = curriculum_dict()
            value["disk_cleanup_gib"] = 20
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CurriculumConfigError, "必须大于"):
                load_curriculum_config(path)

            value = curriculum_dict()
            value["stages"][2]["min_stage_samples"] = 1  # type: ignore[index]
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CurriculumConfigError, "19x19"):
                load_curriculum_config(path)

    def test_boolean_cannot_impersonate_schema_version_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.json"
            value = curriculum_dict()
            value["schema_version"] = True
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(CurriculumConfigError, "schema_version"):
                load_curriculum_config(path)


class StatisticsAndSgfTests(unittest.TestCase):
    def test_wilson_interval_distinguishes_uncertain_and_clear_win(self) -> None:
        low_small, _ = wilson_interval(6, 10)
        low_large, _ = wilson_interval(120, 200)
        self.assertLess(low_small, 0.5)
        self.assertGreater(low_large, 0.5)

    def test_match_sgf_color_swaps_draw_and_damage(self) -> None:
        text = "\n".join(
            (
                "(;GM[1]PB[candidate]PW[baseline]RE[B+3.5];B[aa];W[bb])",
                "(;GM[1]PB[baseline]PW[candidate]RE[W+R];B[cc];W[dd])",
                "(;GM[1]PB[candidate]PW[baseline]RE[0])",
                "(;GM[1]PB[baseline]PW[candidate]RE[W+1.5])",
                "(;GM[1]PB[candidate]PW[baseline])",
                "(;GM[1]PB[candidate]PW[baseline]RE[B+2.5]C[escaped \\] ) text])",
                "(;GM[1]PW[candidate]RE[W+1.5])",
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "match.sgfs"
            path.write_text(text, encoding="utf-8")
            result = summarize_match_sgfs(
                [path], candidate_name="candidate", requested_games=7
            )
        self.assertEqual(result.games, 5)
        self.assertEqual(result.candidate_wins, 4)
        self.assertEqual(result.draws, 1)
        self.assertEqual(result.candidate_black_games, 3)
        self.assertEqual(result.candidate_white_games, 2)
        self.assertEqual(result.no_result_games, 1)
        self.assertEqual(result.damaged_games, 2)
        self.assertGreater(result.win_rate, 0.8)

    def test_match_config_is_deterministic_and_mirrored(self) -> None:
        text = build_match_config(
            candidate_model=Path("candidate.bin.gz"),
            baseline_model=Path("baseline.bin.gz"),
            board_size=13,
            visits=300,
            opening_directory=Path("opening-001"),
        )
        self.assertIn("numBots = 2", text)
        self.assertIn("numGamesTotal = 200", text)
        self.assertIn("bSizes = 13", text)
        self.assertIn("komiMean = 6.5", text)
        self.assertIn("allowResignation = false", text)
        self.assertIn("resignThreshold = -1.0", text)
        self.assertIn("resignConsecTurns = 3", text)
        self.assertIn("rootNoiseEnabled = false", text)
        self.assertIn("chosenMoveTemperature = 0.0", text)
        self.assertIn("hintPosesDir = opening-001", text)
        self.assertIn("hintPosesProb = 1.0", text)
        self.assertIn("nnRandSeed = gogogo-curriculum-evaluation-v1", text)

    def test_health_window_requires_1280_games_and_all_limits(self) -> None:
        health = passing_health()
        self.assertTrue(health.complete)
        self.assertTrue(health.passed)
        incomplete = compute_health_window([{"games": 128, "black_wins": 64, "white_wins": 64}])
        self.assertFalse(incomplete.passed)
        unhealthy_records = [
            {
                "games": 128,
                "black_wins": 80,
                "white_wins": 48,
                "immediate_double_pass_games": 2,
                "extreme_games": 8,
                "invalid_games": 1,
            }
            for _ in range(10)
        ]
        unhealthy = compute_health_window(unhealthy_records)
        self.assertFalse(unhealthy.passed)
        self.assertGreaterEqual(len(unhealthy.reasons), 4)

    def test_health_accepts_real_metric_names_and_draws_score_half(self) -> None:
        records = [
            {
                "sgf_games": 128,
                "black_wins": 54,
                "white_wins": 54,
                "draws": 20,
                "opening_double_pass_games": 1,
                "extreme_result_games": 2,
                "no_result_games": 0,
                "damaged_games": 0,
            }
            for _ in range(10)
        ]
        health = compute_health_window(records)
        self.assertTrue(health.passed)
        self.assertAlmostEqual(health.black_win_rate, 0.5)
        self.assertEqual(health.immediate_double_pass_games, 10)
        self.assertEqual(health.extreme_games, 20)


class ControllerTests(unittest.TestCase):
    def test_adoption_boundary_and_two_consecutive_promotions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            store = CurriculumStateStore(root / "state")
            controller = CurriculumController(config, store)
            state = controller.adopt_existing_9x9(
                trained_samples=8_820_000,
                latest_model="gogogo-s8820000-d1",
                entry_model="gogogo-s100-d1",
            )
            self.assertEqual(state.active.next_evaluation_sample, 9_000_000)
            self.assertIn("gogogo-s100-d1", state.protected_models)
            self.assertTrue(
                controller.mark_cycle(
                    state,
                    trained_samples=10_000_000,
                    latest_model="gogogo-s10000000-d2",
                )
            )
            first = controller.record_evaluation(
                state,
                candidate_model=state.active.latest_model or "",
                match=passing_match(),
                health=passing_health(),
            )
            self.assertTrue(first.passed)
            self.assertEqual(state.active.consecutive_passes, 1)
            self.assertEqual(state.phase, "training")
            self.assertEqual(state.active.next_evaluation_sample, 10_500_000)
            controller.mark_cycle(
                state,
                trained_samples=10_500_000,
                latest_model="gogogo-s10500000-d3",
            )
            controller.record_evaluation(
                state,
                candidate_model=state.active.latest_model or "",
                match=passing_match(),
                health=passing_health(),
            )
            self.assertEqual(state.phase, "transition_ready")
            self.assertEqual(store.load().phase, "transition_ready")  # type: ignore[union-attr]

    def test_failed_gate_and_evaluation_error_continue_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            store = CurriculumStateStore(root / "state")
            controller = CurriculumController(config, store)
            state = controller.adopt_existing_9x9(
                trained_samples=10_000_000, latest_model="model"
            )
            bad = passing_match().__dict__.copy()
            bad.update(candidate_wins=100, baseline_wins=100, win_rate=0.5)
            match = MatchSummary(**bad)
            gate = controller.record_evaluation(
                state,
                candidate_model="model",
                match=match,
                health=passing_health(),
            )
            self.assertFalse(gate.passed)
            self.assertEqual(state.phase, "training")
            controller.record_evaluation_error(state, "match process exited 2")
            self.assertEqual(state.phase, "training")
            self.assertEqual(state.active.consecutive_passes, 0)
            self.assertEqual(state.warnings[-1]["kind"], "evaluation")

    def test_evaluation_save_failure_does_not_mutate_live_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            store = CurriculumStateStore(root / "state")
            controller = CurriculumController(config, store)
            state = controller.adopt_existing_9x9(
                trained_samples=10_000_000, latest_model="candidate"
            )
            original_save = store.save
            store.save = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("state disk unavailable")
            )
            try:
                with self.assertRaises(OSError):
                    controller.record_evaluation(
                        state,
                        candidate_model="candidate",
                        match=passing_match(),
                        health=passing_health(),
                    )
            finally:
                store.save = original_save
            self.assertEqual(state.phase, "training")
            self.assertEqual(state.active.consecutive_passes, 0)
            self.assertEqual(state.active.evaluations, [])

    def test_migration_switch_occurs_only_after_explicit_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            controller = CurriculumController(config, CurriculumStateStore(root / "state"))
            state = controller.adopt_existing_9x9(
                trained_samples=10_000_000, latest_model="seed9"
            )
            state.active.consecutive_passes = 2
            state.phase = "transition_ready"
            controller.begin_migration(state)
            self.assertEqual(state.active.board_size, 9)
            controller.migration_failed(state, "smoke failed")
            self.assertEqual(state.active.board_size, 9)
            self.assertEqual(state.phase, "training")
            state.phase = "transition_ready"
            controller.begin_migration(state)
            controller.complete_migration(state, seed_model="seed13", autotune_batch=1536)
            self.assertEqual(state.active.board_size, 13)
            self.assertEqual(state.active.samples, 0)
            self.assertEqual(state.active.autotune_batch, 1536)
            self.assertEqual(state.active.baseline_model, "seed13")

    def test_disk_pause_recovers_without_state_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            controller = CurriculumController(config, CurriculumStateStore(root / "state"))
            state = controller.adopt_existing_9x9(
                trained_samples=1, latest_model="model"
            )
            self.assertEqual(controller.update_disk(state, 29.9), "paused")
            self.assertEqual(state.phase, "paused_disk")
            self.assertEqual(controller.update_disk(state, 45), "cleanup")
            self.assertEqual(state.phase, "training")

            state.phase = "migrating"
            self.assertEqual(controller.update_disk(state, 29.9), "paused")
            reloaded = controller.store.load()
            assert reloaded is not None
            self.assertEqual(reloaded.phase, "paused_disk")
            self.assertEqual(reloaded.resume_phase, "migrating")
            self.assertEqual(controller.update_disk(reloaded, 60), "healthy")
            self.assertEqual(reloaded.phase, "migrating")
            self.assertIsNone(reloaded.resume_phase)

    def test_global_dashboard_is_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            controller = CurriculumController(config, CurriculumStateStore(root / "state"))
            state = controller.adopt_existing_9x9(
                trained_samples=8_820_000, latest_model="gogogo-s8820000-d1"
            )
            state.disk = {"free_gib": 125.5, "status": "healthy"}
            output = root / "dashboard.html"
            write_curriculum_dashboard(state, config, output)
            page = output.read_text(encoding="utf-8")
            self.assertIn("KataGo 9×9 → 13×13 → 19×19", page)
            self.assertIn("8,820,000", page)
            self.assertIn("125.5 GiB", page)
            self.assertIn("固定评测门槛", render_curriculum_dashboard(state, config))

    def test_state_json_is_restartable_and_duplicate_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_test_config(root)
            store = CurriculumStateStore(root / "state")
            state = CurriculumController(config, store).adopt_existing_9x9(
                trained_samples=123, latest_model="model"
            )
            self.assertEqual(store.load().active.samples, 123)  # type: ignore[union-attr]
            with GlobalCurriculumLock(root / "state"):
                with self.assertRaises(CurriculumLockError):
                    with GlobalCurriculumLock(root / "state"):
                        pass
            self.assertEqual(store.load().to_dict(), state.to_dict())  # type: ignore[union-attr]


class RetentionAndAutotuneTests(unittest.TestCase):
    def test_model_retention_keeps_latest_twenty_and_protected_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(25):
                path = root / f"gogogo-s{index}-d{index}"
                path.mkdir()
                paths.append(path)
            kept, deleted = select_retained_directories(
                paths, keep_latest=20, protected=[paths[0].name]
            )
            self.assertIn(paths[0], kept)
            self.assertEqual(len(kept), 21)
            self.assertEqual(len(deleted), 4)

    def test_replay_selection_keeps_newest_whole_files_over_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(5):
                path = root / f"{index}.npz"
                path.touch()
                os.utime(path, (100 + index, 100 + index))
                paths.append(path)
            kept, deleted = select_replay_window(
                paths, target_rows=250, row_counter=lambda _: 100
            )
            self.assertEqual(kept, tuple(paths[2:]))
            self.assertEqual(deleted, tuple(paths[:2]))

    def test_retention_plan_never_deletes_sgf_csv_or_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for parent in ("models", "torchmodels_toexport", "shuffleddata"):
                for index in range(3):
                    path = root / parent / f"gogogo-s{index}-d{index}"
                    path.mkdir(parents=True)
            tdata = root / "selfplay" / "model" / "tdata"
            tdata.mkdir(parents=True)
            npz = []
            for index in range(3):
                path = tdata / f"{index}.npz"
                path.touch()
                os.utime(path, (100 + index, 100 + index))
                npz.append(path)
            sgf = root / "selfplay" / "model" / "sgfs" / "games.sgfs"
            sgf.parent.mkdir(parents=True)
            sgf.write_text("(;RE[B+1.5])", encoding="utf-8")
            csv = root / "metrics" / "history.csv"
            csv.parent.mkdir(parents=True)
            csv.write_text("x", encoding="utf-8")
            plan = build_retention_plan(
                root,
                protected_models=["gogogo-s0-d0"],
                keep_models=1,
                keep_shuffles=1,
                replay_window_rows=150,
                row_counter=lambda _: 100,
            )
            apply_retention_plan(plan, root)
            self.assertTrue(sgf.is_file())
            self.assertTrue(csv.is_file())
            self.assertFalse(npz[0].exists())
            self.assertTrue(npz[1].exists())

    def test_autotune_excludes_oom_and_over_13_gib_then_persists(self) -> None:
        measurements = {
            512: BenchmarkMeasurement(512, True, 1000, 8),
            1024: BenchmarkMeasurement(1024, True, 1800, 12.5),
            1536: BenchmarkMeasurement(1536, True, 2200, 13.2),
            2048: BenchmarkMeasurement(2048, False, oom=True, error="CUDA OOM"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "autotune.json"
            result = choose_autotune_batch(
                [512, 1024, 1536, 2048],
                measurements.__getitem__,
                max_gpu_memory_gib=13,
                persist_path=path,
            )
            self.assertEqual(result.selected_batch_size, 1024)
            self.assertEqual(json.loads(path.read_text())["selected_batch_size"], 1024)

    def test_benchmark_input_tiles_rows_and_output_parser_reads_official_format(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "smoke.npz"
            target = root / "benchmark" / "benchmark_input.npz"
            np.savez_compressed(
                source,
                binaryInputNCHW=np.arange(2 * 3).reshape(2, 3),
                globalInputNC=np.arange(2 * 2).reshape(2, 2),
            )
            build_benchmark_input(source, target, minimum_rows=5)
            with np.load(target) as data:
                self.assertEqual(data["binaryInputNCHW"].shape[0], 5)
                self.assertEqual(data["globalInputNC"].shape[0], 5)
            measurement = parse_benchmark_output(
                "Throughput: 1,234.5 samples/s (x)\n"
                "Peak GPU memory (rank 0): 12.75 GiB\n",
                batch_size=1024,
            )
            self.assertTrue(measurement.success)
            self.assertEqual(measurement.throughput, 1234.5)
            self.assertEqual(measurement.peak_gpu_gib, 12.75)


class CheckpointMigrationTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("torch") is not None, "PyTorch required"
    )
    def test_clean_checkpoint_uses_swa_and_resets_training_state(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ckpt"
            target = root / "target.ckpt"
            torch.save(
                {
                    "model": {"weight": torch.tensor([1.0])},
                    "swa_model": {
                        "n_averaged": torch.tensor(99),
                        "module.weight": torch.tensor([7.0]),
                    },
                    "optimizer": {"bad": True},
                    "train_state": {"global_step_samples": 99_000},
                    "running_metrics": {"loss": 9},
                    "config": {"version": 15},
                },
                source,
            )
            create_clean_swa_checkpoint(source, target)
            clean = torch.load(target, map_location="cpu", weights_only=False)
            self.assertEqual(clean["model"]["weight"].item(), 7.0)
            self.assertEqual(clean["swa_model"]["n_averaged"].item(), 1)
            for key in ("optimizer", "train_state", "running_metrics", "metrics"):
                self.assertNotIn(key, clean)

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("torch") is not None, "PyTorch required"
    )
    def test_stage_tree_switches_atomically_only_after_smoke(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.ckpt"
            torch.save(
                {
                    "model": {"weight": torch.tensor([1.0])},
                    "swa_model": {
                        "n_averaged": torch.tensor(2),
                        "module.weight": torch.tensor([2.0]),
                    },
                    "config": {"version": 15},
                },
                source,
            )
            target = root / "13x13"
            with self.assertRaises(CurriculumMigrationError):
                prepare_stage_migration(
                    source_checkpoint=source,
                    target_stage_root=target,
                    target_board_size=13,
                    load_verifier=lambda _checkpoint, _board: None,
                    smoke_verifier=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
                )
            self.assertFalse(target.exists())
            self.assertFalse(any(root.glob(".13x13.migration-*.tmp")))

            observed = []
            result = prepare_stage_migration(
                source_checkpoint=source,
                target_stage_root=target,
                target_board_size=13,
                load_verifier=lambda checkpoint, board: observed.append(
                    ("load", checkpoint.is_file(), board)
                ),
                smoke_verifier=lambda run_root, checkpoint, board: observed.append(
                    ("smoke", run_root.is_dir() and checkpoint.is_file(), board)
                ),
            )
            self.assertTrue(result.checkpoint.is_file())
            self.assertEqual(observed, [("load", True, 13), ("smoke", True, 13)])


class OpponentPoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_test_config(self.root)
        self.controller = CurriculumController(self.config, CurriculumStateStore(self.root / "state"))
        self.state = self.controller.adopt_existing_9x9(
            trained_samples=10_000_000, latest_model="new", entry_model="anchor")
        self.state.active.champion_model = "champion"
        self.state.active.fixed_baseline_models = ["anchor", "history"]
        self.state.protected_models = ["anchor", "champion", "history"]

    def test_strength_promotion_independent_of_course_health(self):
        health = replace(passing_health(), passed=False, reasons=("极端结果比例超标",))
        results = {name: passing_match() for name in ("champion", "anchor", "history")}
        gate = self.controller.record_evaluation(
            self.state, candidate_model="new", match=results["anchor"],
            health=health, opponent_results=results)
        self.assertFalse(gate.passed)
        self.assertEqual(self.state.active.champion_model, "new")
        self.assertTrue(self.state.active.champion_established)
        self.assertEqual(self.state.phase, "training")
        self.assertIn("new", self.state.protected_models)
        self.assertIn("history", self.state.protected_models)
        self.assertNotIn("champion", self.state.protected_models)
        self.assertEqual(len(self.state.active.evaluations[-1]["opponents"]), 3)

    def test_fixed_baseline_win_does_not_promote_losing_challenger(self):
        loss = replace(passing_match(), candidate_wins=80, baseline_wins=120, win_rate=.4, wilson_lower=.33)
        results = {"champion": loss, "anchor": passing_match(), "history": passing_match()}
        gate = self.controller.record_evaluation(
            self.state, candidate_model="new", match=results["anchor"],
            health=passing_health(), opponent_results=results)
        self.assertTrue(gate.passed)
        self.assertEqual(self.state.active.champion_model, "champion")
        self.assertEqual(self.state.active.consecutive_passes, 1)

    def test_deduplicates_shared_roles_and_skips_self(self):
        self.state.active.champion_model = "anchor"
        self.state.active.fixed_baseline_models = ["anchor", "new"]
        self.assertEqual(self.controller.evaluation_opponents(self.state),
                         {"anchor": ["champion", "fixed"]})

    def test_incomplete_suite_does_not_mutate_state(self):
        before = self.state.to_dict()
        with self.assertRaises(CurriculumStateError):
            self.controller.record_evaluation(
                self.state, candidate_model="new", match=passing_match(),
                health=passing_health(), opponent_results={"anchor": passing_match()})
        self.assertEqual(self.state.to_dict(), before)

    def test_no_promotion_for_low_confidence_or_unbalanced_games(self):
        results = {name: passing_match() for name in ("champion", "anchor", "history")}
        results["champion"] = replace(passing_match(), wilson_lower=.49)
        self.controller.record_evaluation(
            self.state, candidate_model="new", match=results["anchor"],
            health=passing_health(), opponent_results=results)
        self.assertEqual(self.state.active.champion_model, "champion")
        results["champion"] = replace(passing_match(), candidate_black_games=200, candidate_white_games=0)
        with self.assertRaises(CurriculumStateError):
            self.controller.record_evaluation(
                self.state, candidate_model="new", match=results["anchor"],
                health=passing_health(), opponent_results=results)

    def test_page_preserves_legacy_opponent_identity(self):
        from weiqi.rl_evaluation_dashboard import render_opponent_pools
        self.state.active.evaluations = [{
            "baseline_model": "old-weak-seed", "candidate_model": "new", "stage_samples": 9000000,
            "match": replace(passing_match(), win_rate=1., elo=3600.).to_dict(),
        }]
        page = render_opponent_pools(self.state.active)
        self.assertIn("old-weak-seed", page)
        self.assertIn("暂定冠军", page)
        self.assertIn("待评测", page)
        self.assertIn("饱和，无法估计", page)
        self.assertNotIn("+3600", page)


if __name__ == "__main__":
    unittest.main()
