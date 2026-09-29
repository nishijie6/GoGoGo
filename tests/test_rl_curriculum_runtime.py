"""Tests for the side-effecting KataGo curriculum runtime."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from weiqi.rl_curriculum import (
    BenchmarkMeasurement,
    CurriculumMigrationError,
    CurriculumState,
    CurriculumStateError,
    DiskStatus,
    StageProgress,
)
from weiqi.katago_rl import KataGoRLRunnerError
from weiqi.replay_accounting import ReplayRowLedger
from weiqi.rl_curriculum_runtime import (
    CommandExecution,
    CurriculumRuntime,
    ensure_evaluation_opening_suite,
    parse_benchmark_output,
    prepare_benchmark_npz,
)
from weiqi.rl_config import load_rl_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_profiles(root: Path) -> tuple[Path, dict[int, Path]]:
    source_names = {
        9: "rl_training.rtx5070ti.json",
        13: "rl_training.rtx5070ti.13x13.json",
        19: "rl_training.rtx5070ti.curriculum.19x19.json",
    }
    profiles: dict[int, Path] = {}
    for board, name in source_names.items():
        value = json.loads((PROJECT_ROOT / "config" / name).read_text(encoding="utf-8"))
        value["overrides"]["runtime"]["output_directory"] = str(root / f"{board}x{board}")
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        profiles[board] = path
    curriculum = {
        "schema_version": 1,
        "state_directory": str(root / "curriculum"),
        "evaluation_interval_samples": 500_000,
        "evaluation_games": 200,
        "promotion_win_rate": 0.6,
        "wilson_lower_bound": 0.5,
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
                "training_config": str(profiles[9]),
                "min_stage_samples": 10_000_000,
                "autotune_batches": [],
                "indefinite": False,
            },
            {
                "board_size": 13,
                "training_config": str(profiles[13]),
                "min_stage_samples": 25_000_000,
                "autotune_batches": [512, 1024, 1536, 2048],
                "indefinite": False,
            },
            {
                "board_size": 19,
                "training_config": str(profiles[19]),
                "min_stage_samples": None,
                "autotune_batches": [256, 512, 768, 1024, 1280],
                "indefinite": True,
            },
        ],
    }
    curriculum_path = root / "curriculum.json"
    curriculum_path.write_text(json.dumps(curriculum), encoding="utf-8")
    return curriculum_path, profiles


class _FakeMetricStore:
    def __init__(self, root: Path):
        self.jsonl_path = root / "metrics" / "history.jsonl"


class _FakeRunner:
    def __init__(
        self,
        *,
        config_path: Path,
        project_root: Path,
        run_root_override: Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.project_root = project_root
        self.config = load_rl_training_config(config_path)
        self.run_root = run_root_override or Path(self.config.runtime.output_directory)
        self.models_dir = self.run_root / "models"
        self.metric_store = _FakeMetricStore(self.run_root)
        self.katago_executable = project_root / "katago" / "katago.exe"
        self.python_source = project_root / "katago" / "source" / "python"
        self.model_kind = "b10c128"

    @contextlib.contextmanager
    def lock(self):
        yield

    def _environment(self) -> dict[str, str]:
        return dict(os.environ)


class _LockedFakeRunner(_FakeRunner):
    @contextlib.contextmanager
    def lock(self):
        raise KataGoRLRunnerError("busy")
        yield


def _make_model(root: Path, name: str) -> None:
    model = root / "models" / name / "model.bin.gz"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    checkpoint = root / "torchmodels_toexport" / name / "model.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")


def _write_npz(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        binaryInputNCHWPacked=np.zeros((rows, 2, 4), dtype=np.uint8),
        globalInputNC=np.zeros((rows, 3), dtype=np.float32),
        scalar=np.array(7),
    )


class BenchmarkRuntimeTests(unittest.TestCase):
    def test_parse_benchmark_output_and_oom(self) -> None:
        measurement = parse_benchmark_output(
            "Throughput: 12,345.6 samples/s (batch 1024)\n"
            "Peak GPU memory (rank 0): 12.75 GiB\n",
            batch_size=1024,
        )
        self.assertTrue(measurement.success)
        self.assertEqual(measurement.throughput, 12345.6)
        self.assertEqual(measurement.peak_gpu_gib, 12.75)
        failed = parse_benchmark_output(
            "CUDA out of memory", batch_size=2048, returncode=1
        )
        self.assertFalse(failed.success)
        self.assertTrue(failed.oom)

    def test_prepare_benchmark_npz_tiles_only_row_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.npz", root / "target.npz"
            _write_npz(source, 3)
            prepare_benchmark_npz(source, target, 8)
            with np.load(target, allow_pickle=False) as data:
                self.assertEqual(data["globalInputNC"].shape[0], 8)
                self.assertEqual(data["binaryInputNCHWPacked"].shape[0], 8)
                self.assertEqual(data["scalar"].item(), 7)

    def test_fixed_opening_suite_is_unique_stable_and_singleton(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ensure_evaluation_opening_suite(
                root, board_size=9, opening_count=100
            )
            second = ensure_evaluation_opening_suite(
                root, board_size=9, opening_count=100
            )
            self.assertEqual(first, second)
            samples = [
                (path / "opening.startposes.txt").read_text(encoding="utf-8")
                for path in first
            ]
            self.assertEqual(len(set(samples)), 100)
            for sample in samples:
                parsed = json.loads(sample)
                self.assertEqual(parsed["nextPla"], "B")
                self.assertEqual(len(parsed["moveLocs"]), len(parsed["movePlas"]))
                self.assertEqual(parsed["movePlas"][::2], ["B"] * (len(parsed["movePlas"]) // 2))


class CurriculumRuntimeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config_path, self.profiles = _write_profiles(self.root)
        self.runtime = CurriculumRuntime(
            self.config_path,
            project_root=PROJECT_ROOT,
            runner_factory=_FakeRunner,
            disk_probe=lambda *_args, **_kwargs: DiskStatus(100.0, 500.0, "healthy"),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_adoption_protects_oldest_baseline_before_latest(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s2048-d100")
        _make_model(run_root, "gogogo-s9000000-d900")
        state = self.runtime._adopt_existing_9x9()
        self.assertEqual(state.active.baseline_model, "gogogo-s2048-d100")
        self.assertEqual(state.active.latest_model, "gogogo-s9000000-d900")
        self.assertEqual(state.active.samples, 9_000_000)
        persisted = json.loads(self.runtime.store.path.read_text(encoding="utf-8"))
        self.assertIn("gogogo-s2048-d100", persisted["protected_models"])
        self.assertIn("gogogo-s9000000-d900", persisted["protected_models"])

    def test_health_ignores_export_without_real_sgf_games(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s2048-d100")
        metrics = run_root / "metrics" / "history.jsonl"
        metrics.parent.mkdir(parents=True)
        real = {
            "sgf_entries": 128,
            "black_wins": 64,
            "white_wins": 64,
            "draws": 0,
            "opening_double_pass_games": 2,
            "extreme_result_games": 0,
            "no_result_games": 0,
            "damaged_games": 0,
        }
        records = [dict(real) for _ in range(10)]
        records.append({"selfplay_games": 128, "black_wins": 0, "white_wins": 0})
        metrics.write_text(
            "".join(json.dumps(item) + "\n" for item in records),
            encoding="utf-8",
        )
        health = self.runtime._health(self.runtime._make_runner(self.runtime.config.stages[0]))
        self.assertEqual(health.games, 1280)
        self.assertAlmostEqual(health.immediate_double_pass_rate, 20 / 1280)

    def test_evaluating_phase_failure_is_recorded_and_training_resumes(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s2048-d100")
        _make_model(run_root, "gogogo-s500000-d500")
        state = self.runtime._adopt_existing_9x9()
        state.phase = "evaluating"
        state.active.latest_model = "gogogo-s500000-d500"
        state.active.samples = 500_000
        state.active.next_evaluation_sample = 500_000
        self.runtime.store.save(state)
        self.runtime.command_runner = lambda *_args, **_kwargs: CommandExecution(7, "boom")
        self.runtime._evaluate(state)
        recovered = self.runtime.store.load()
        assert recovered is not None
        self.assertEqual(recovered.phase, "training")
        self.assertEqual(recovered.active.consecutive_passes, 0)
        self.assertGreater(recovered.active.next_evaluation_sample, 500_000)
        self.assertEqual(recovered.warnings[-1]["kind"], "evaluation")

    def test_artifact_failure_after_evaluation_commit_does_not_undo_pass(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s1-d1")
        _make_model(run_root, "gogogo-s10000000-d100")
        state = self.runtime._adopt_existing_9x9()
        state.phase = "evaluating"
        state.active.latest_model = "gogogo-s10000000-d100"
        state.active.samples = 10_000_000
        state.active.next_evaluation_sample = 10_000_000
        metrics = run_root / "metrics" / "history.jsonl"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        health_record = {
            "sgf_entries": 128,
            "black_wins": 64,
            "white_wins": 64,
            "draws": 0,
            "opening_double_pass_games": 0,
            "extreme_result_games": 0,
            "no_result_games": 0,
            "damaged_games": 0,
        }
        metrics.write_text(
            "".join(json.dumps(health_record) + "\n" for _ in range(10)),
            encoding="utf-8",
        )
        self.runtime.store.save(state)

        def successful_pair(command, **_kwargs):
            sgf_dir = Path(command[command.index("-sgf-output-dir") + 1])
            sgf_dir.mkdir(parents=True, exist_ok=True)
            (sgf_dir / "pair.sgfs").write_text(
                "(;PB[candidate]PW[baseline]RE[B+1.5])\n"
                "(;PB[baseline]PW[candidate]RE[W+1.5])\n",
                encoding="utf-8",
            )
            return CommandExecution(0, "ok")

        self.runtime.command_runner = successful_pair
        import weiqi.rl_curriculum_runtime as runtime_module

        original = runtime_module._atomic_write_json

        def fail_summary(path, value):
            if path.name == "summary.json":
                raise OSError("summary disk error")
            return original(path, value)

        runtime_module._atomic_write_json = fail_summary
        try:
            self.runtime._evaluate(state)
        finally:
            runtime_module._atomic_write_json = original
        recovered = self.runtime.store.load()
        assert recovered is not None
        self.assertEqual(len(recovered.active.evaluations), 1)
        self.assertEqual(recovered.active.consecutive_passes, 1)
        self.assertEqual(recovered.active.next_evaluation_sample, 10_500_000)
        self.assertEqual(recovered.warnings[-1]["kind"], "evaluation_artifact")

    def test_retention_keeps_protected_model_and_twenty_longterm_checkpoints(self) -> None:
        run_root = self.root / "9x9"
        for index in range(24):
            name = f"gogogo-s{index * 1000 + 1}-d{index}"
            _make_model(run_root, name)
        for index in range(12):
            folder = run_root / "shuffleddata" / f"shuffle-{index:02d}" / "train"
            folder.mkdir(parents=True)
        longterm = run_root / "train" / "gogogo" / "longterm_checkpoints"
        longterm.mkdir(parents=True)
        for index in range(24):
            checkpoint = longterm / f"checkpoint-{index:02d}.ckpt"
            checkpoint.write_bytes(bytes([index]))
            os.utime(checkpoint, (1000 + index, 1000 + index))
        stale = run_root / "models" / "aborted.tmp-123"
        stale.mkdir(parents=True)
        os.utime(stale, (time.time() - 50_000, time.time() - 50_000))
        self.runtime.config = replace(
            self.runtime.config, keep_models=20, keep_shuffles=10
        )
        protected = "gogogo-s1-d0"
        report = self.runtime._retention_for_root(run_root, [protected])
        self.assertTrue((run_root / "models" / protected).is_dir())
        self.assertEqual(len(list((run_root / "models").glob("gogogo-*"))), 21)
        self.assertEqual(len(list((run_root / "torchmodels_toexport").glob("gogogo-*"))), 21)
        self.assertEqual(len(list((run_root / "shuffleddata").glob("shuffle-*"))), 10)
        self.assertEqual(len(list(longterm.glob("*.ckpt"))), 20)
        self.assertFalse(stale.exists())
        self.assertGreater(report["directories"], 0)
        self.assertGreater(report["bytes"], 0)

    def test_retention_preserves_logical_rows_in_replay_ledger(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s1-d0")
        paths = []
        for index in range(3):
            path = run_root / "selfplay" / "seed" / "tdata" / f"{index}.npz"
            _write_npz(path, 100)
            os.utime(path, (100 + index, 100 + index))
            paths.append(path)
        self.runtime.config = replace(self.runtime.config, replay_window_rows=150)

        self.runtime._retention_for_root(run_root, [])
        snapshot = ReplayRowLedger(run_root).sync()

        self.assertFalse(paths[0].exists())
        self.assertTrue(paths[1].exists())
        self.assertEqual(snapshot.deleted_rows_offset, 100)
        self.assertEqual(snapshot.physical_rows, 200)
        self.assertEqual(snapshot.logical_rows, 300)

    def test_only_stale_exact_migration_temps_are_removed(self) -> None:
        target = self.root / "13x13"
        stale = self.root / ".13x13.migration-deadbeef.tmp"
        fresh = self.root / ".13x13.migration-live.tmp"
        unrelated = self.root / ".19x19.migration-deadbeef.tmp"
        for path in (stale, fresh, unrelated):
            path.mkdir()
            (path / "artifact").write_bytes(b"x")
        old = time.time() - 50_000
        os.utime(stale, (old, old))
        os.utime(unrelated, (old, old))
        report = self.runtime._cleanup_stale_migrations(target)
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual(report["directories"], 1)
        self.assertEqual(report["bytes"], 1)

    def test_migrating_phase_with_atomic_target_can_complete_without_reprepare(self) -> None:
        state = CurriculumState(
            schema_version=1,
            active_stage_index=0,
            phase="migrating",
            stages={
                "9x9": StageProgress(
                    board_size=9,
                    samples=11_000_000,
                    entry_model="gogogo-s1-d1",
                    baseline_model="gogogo-s1-d1",
                    latest_model="gogogo-s11000000-d9",
                    next_evaluation_sample=11_500_000,
                    consecutive_passes=2,
                )
            },
            protected_models=["gogogo-s1-d1", "gogogo-s11000000-d9"],
        )
        source_root = self.root / "9x9"
        _make_model(source_root, "gogogo-s11000000-d9")
        target = self.root / "13x13"
        (target / "train" / "gogogo").mkdir(parents=True)
        (target / "train" / "gogogo" / "checkpoint.ckpt").write_bytes(b"seed")
        _make_model(target, "gogogo-s0-d0")
        (target / "autotune.json").write_text(
            json.dumps({"selected_batch_size": 1024}), encoding="utf-8"
        )
        source_checkpoint = (
            source_root
            / "torchmodels_toexport"
            / "gogogo-s11000000-d9"
            / "model.ckpt"
        )
        state.migration = {
            "id": "recover-me",
            "source_model": "gogogo-s11000000-d9",
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
            "target_board_size": 13,
            "seed_model": "gogogo-s0-d0",
            "started_at": "2026-01-01T00:00:00+00:00",
        }
        target_checkpoint = target / "train" / "gogogo" / "checkpoint.ckpt"
        (target / "migration_manifest.json").write_text(
            json.dumps(
                {
                    **state.migration,
                    "selected_batch_size": 1024,
                    "seed_checkpoint_sha256": hashlib.sha256(
                        target_checkpoint.read_bytes()
                    ).hexdigest(),
                    "completed_at": "2026-01-01T00:01:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        self.runtime.store.save(state)
        import weiqi.rl_curriculum_runtime as runtime_module

        original = runtime_module.verify_checkpoint_loads
        original_read_text = Path.read_text
        attempts = 0
        manifest_denials = 0

        def transient_manifest_read(path, *args, **kwargs):
            nonlocal manifest_denials
            if (
                path == target / "migration_manifest.json"
                and manifest_denials == 0
            ):
                manifest_denials += 1
                raise PermissionError("transient manifest read")
            return original_read_text(path, *args, **kwargs)

        def transient_then_success(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient checkpoint read")

        runtime_module.verify_checkpoint_loads = transient_then_success
        Path.read_text = transient_manifest_read
        try:
            self.runtime._migrate(state)
            retrying = self.runtime.store.load()
            assert retrying is not None
            self.assertEqual(retrying.phase, "migrating")
            self.assertEqual(retrying.migration["id"], "recover-me")
            self.assertTrue(target.is_dir())
            self.assertEqual(manifest_denials, 1)
            Path.read_text = original_read_text
            self.runtime._migrate(retrying)
            retrying_again = self.runtime.store.load()
            assert retrying_again is not None
            self.assertEqual(retrying_again.phase, "migrating")
            self.assertEqual(retrying_again.migration["id"], "recover-me")
            self.runtime._migrate(retrying_again)
        finally:
            Path.read_text = original_read_text
            runtime_module.verify_checkpoint_loads = original
        recovered = self.runtime.store.load()
        assert recovered is not None
        self.assertEqual(recovered.active_stage_index, 1)
        self.assertEqual(recovered.phase, "training")
        self.assertEqual(recovered.active.samples, 0)
        self.assertEqual(recovered.active.autotune_batch, 1024)

    def test_published_target_survives_state_commit_failure(self) -> None:
        state = CurriculumState(
            schema_version=1,
            active_stage_index=0,
            phase="transition_ready",
            stages={
                "9x9": StageProgress(
                    board_size=9,
                    samples=11_000_000,
                    entry_model="gogogo-s1-d1",
                    baseline_model="gogogo-s1-d1",
                    latest_model="gogogo-s11000000-d9",
                    next_evaluation_sample=11_500_000,
                    consecutive_passes=2,
                )
            },
            protected_models=["gogogo-s1-d1", "gogogo-s11000000-d9"],
        )
        source_root = self.root / "9x9"
        _make_model(source_root, "gogogo-s11000000-d9")
        target = self.root / "13x13"
        self.runtime.store.save(state)

        import weiqi.rl_curriculum_runtime as runtime_module

        original_prepare = runtime_module.prepare_stage_migration
        original_verify = runtime_module.verify_checkpoint_loads
        original_checks = self.runtime._migration_checks
        original_save = self.runtime.store.save
        original_load = self.runtime.store.load
        failed_commit = False
        failed_reload = False

        def publish_target(**kwargs):
            checkpoint = target / "train" / "gogogo" / "checkpoint.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"published-seed")
            model = target / "models" / "gogogo-s0-d0" / "model.bin.gz"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            (target / "autotune.json").write_text(
                json.dumps({"selected_batch_size": 1024}), encoding="utf-8"
            )
            kwargs["load_verifier"](checkpoint, 13)
            kwargs["smoke_verifier"](target, checkpoint, 13)
            return SimpleNamespace(stage_root=target, checkpoint=checkpoint)

        def fail_first_completed_save(value):
            nonlocal failed_commit
            if value.active_stage_index == 1 and not failed_commit:
                failed_commit = True
                raise OSError("transient curriculum state commit")
            return original_save(value)

        def fail_first_recovery_load():
            nonlocal failed_reload
            if failed_commit and not failed_reload:
                failed_reload = True
                raise PermissionError("transient curriculum state reload")
            return original_load()

        runtime_module.prepare_stage_migration = publish_target
        runtime_module.verify_checkpoint_loads = lambda *_args, **_kwargs: None
        self.runtime._migration_checks = (
            lambda *_args, **_kwargs: (1024, None)
        )
        self.runtime.store.save = fail_first_completed_save
        self.runtime.store.load = fail_first_recovery_load
        try:
            with self.assertRaises(CurriculumStateError):
                self.runtime._migrate(state)
            retrying = original_load()
            assert retrying is not None
            self.assertTrue(failed_commit)
            self.assertTrue(failed_reload)
            self.assertTrue(target.is_dir())
            self.assertEqual(retrying.phase, "migrating")
            migration_id = retrying.migration["id"]
            self.assertEqual(state.active_stage_index, 0)
            self.assertEqual(state.migration["id"], migration_id)

            self.runtime.store.load = original_load
            self.runtime._migrate(retrying)
            recovered = original_load()
            assert recovered is not None
            self.assertEqual(recovered.active_stage_index, 1)
            self.assertEqual(recovered.phase, "training")
            self.assertEqual(recovered.active.autotune_batch, 1024)
            manifest = json.loads(
                (target / "migration_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["id"], migration_id)
        finally:
            self.runtime.store.load = original_load
            self.runtime.store.save = original_save
            self.runtime._migration_checks = original_checks
            runtime_module.verify_checkpoint_loads = original_verify
            runtime_module.prepare_stage_migration = original_prepare

    def test_source_hash_permission_error_keeps_transition_ready(self) -> None:
        state = CurriculumState(
            schema_version=1,
            active_stage_index=0,
            phase="transition_ready",
            stages={
                "9x9": StageProgress(
                    board_size=9,
                    samples=11_000_000,
                    entry_model="gogogo-s1-d1",
                    baseline_model="gogogo-s1-d1",
                    latest_model="gogogo-s11000000-d9",
                    next_evaluation_sample=11_500_000,
                    consecutive_passes=2,
                )
            },
            protected_models=["gogogo-s1-d1", "gogogo-s11000000-d9"],
        )
        _make_model(self.root / "9x9", "gogogo-s11000000-d9")
        self.runtime.store.save(state)

        import weiqi.rl_curriculum_runtime as runtime_module

        original = runtime_module._sha256_file
        runtime_module._sha256_file = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("checkpoint temporarily shared")
        )
        try:
            self.runtime._migrate(state)
        finally:
            runtime_module._sha256_file = original
        retrying = self.runtime.store.load()
        assert retrying is not None
        self.assertEqual(retrying.phase, "transition_ready")
        self.assertEqual(retrying.active_stage_index, 0)
        self.assertEqual(retrying.active.consecutive_passes, 2)
        self.assertEqual(retrying.migration, {})
        self.assertEqual(retrying.warnings[-1]["kind"], "migration_retry")

    def test_migration_refuses_busy_preexisting_target_without_touching_it(self) -> None:
        source_root = self.root / "9x9"
        _make_model(source_root, "gogogo-s11000000-d9")
        target = self.root / "13x13"
        target.mkdir()
        marker = target / "standalone-owner.txt"
        marker.write_text("keep", encoding="utf-8")

        def selective_runner_factory(**kwargs):
            config = load_rl_training_config(kwargs["config_path"])
            runner_type = _LockedFakeRunner if config.game.board_size == 13 else _FakeRunner
            return runner_type(**kwargs)

        runtime = CurriculumRuntime(
            self.config_path,
            project_root=PROJECT_ROOT,
            runner_factory=selective_runner_factory,
            disk_probe=lambda *_args, **_kwargs: DiskStatus(100.0, 500.0, "healthy"),
        )
        state = runtime._adopt_existing_9x9()
        state.phase = "transition_ready"
        runtime.store.save(state)
        runtime._migrate(state)
        recovered = runtime.store.load()
        assert recovered is not None
        self.assertEqual(recovered.active_stage_index, 0)
        self.assertEqual(recovered.phase, "migrating")
        self.assertTrue(marker.is_file())
        self.assertIn("busy", recovered.warnings[-1]["message"])
        self.assertEqual(recovered.warnings[-1]["kind"], "migration_retry")

    def test_paused_disk_restores_pending_phase(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s2048-d100")
        state = self.runtime._adopt_existing_9x9()
        state.phase = "transition_ready"
        self.runtime.store.save(state)
        self.runtime.disk_probe = lambda *_args, **_kwargs: DiskStatus(20.0, 500.0, "paused")
        self.assertEqual(self.runtime._update_disk(state), "paused")
        self.assertEqual(state.phase, "paused_disk")
        self.assertEqual(state.resume_phase, "transition_ready")
        self.runtime.disk_probe = lambda *_args, **_kwargs: DiskStatus(100.0, 500.0, "healthy")
        self.assertEqual(self.runtime._update_disk(state), "healthy")
        self.assertEqual(state.phase, "transition_ready")

    def test_autotune_with_no_eligible_batch_blocks_migration(self) -> None:
        stage = self.runtime.config.stages[1]
        runner = _FakeRunner(
            config_path=self.profiles[13],
            project_root=PROJECT_ROOT,
        )
        source = self.root / "smoke.npz"
        _write_npz(source, 2)
        self.runtime._benchmark_measurement = lambda _runner, _data, batch, _logs: BenchmarkMeasurement(
            batch_size=batch,
            success=False,
            oom=True,
            error="OOM",
        )
        with self.assertRaises(CurriculumMigrationError):
            self.runtime._autotune(stage, runner, source)
        persisted_text = (runner.run_root / "autotune.json").read_text(encoding="utf-8")
        persisted = json.loads(persisted_text)
        self.assertTrue(persisted["failed"])
        self.assertIsNone(persisted["selected_batch_size"])
        self.assertEqual(len(persisted["measurements"]), 4)
        self.assertNotIn("Infinity", persisted_text)

    def test_retention_skips_stage_while_training_lock_is_busy(self) -> None:
        run_root = self.root / "9x9"
        for index in range(24):
            _make_model(run_root, f"gogogo-s{index + 1}-d{index}")
        state = self.runtime._adopt_existing_9x9()
        locked = CurriculumRuntime(
            self.config_path,
            project_root=PROJECT_ROOT,
            runner_factory=_LockedFakeRunner,
            disk_probe=lambda *_args, **_kwargs: DiskStatus(100.0, 500.0, "healthy"),
        )
        locked._apply_retention(state)
        self.assertEqual(len(list((run_root / "models").glob("gogogo-*"))), 24)
        self.assertEqual(state.warnings[-1]["kind"], "retention_lock")

    def test_evaluation_directories_do_not_collide_within_one_second(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s2048-d100")
        state = self.runtime._adopt_existing_9x9()
        self.assertNotEqual(
            self.runtime._evaluation_directory(state),
            self.runtime._evaluation_directory(state),
        )

    def test_suite_uses_both_pools_and_commits_once(self):
        from test_rl_curriculum import passing_match, passing_health
        from unittest.mock import patch
        run_root = self.root / "9x9"
        for name in ("gogogo-s1-d1", "gogogo-s2-d2", "gogogo-s3-d3", "gogogo-s10000000-d4"):
            _make_model(run_root, name)
        state = self.runtime._adopt_existing_9x9()
        state.active.champion_model = "gogogo-s3-d3"
        state.active.fixed_baseline_models = ["gogogo-s1-d1", "gogogo-s2-d2"]
        calls = []
        def match(_state, _runner, candidate, opponent, roles, path):
            calls.append((opponent, roles))
            path.mkdir(parents=True)
            return passing_match()
        with patch.object(self.runtime, "_match_opponent", side_effect=match), patch.object(self.runtime, "_health", return_value=passing_health()):
            self.runtime._evaluate(state)
        self.assertEqual([roles for _, roles in calls], [["champion"], ["fixed"], ["fixed"]])
        self.assertEqual(len(state.active.evaluations), 1)
        self.assertEqual(state.active.champion_model, "gogogo-s10000000-d4")
        self.assertIn("gogogo-s2-d2", state.protected_models)

    def test_missing_pool_model_prevents_initialization_and_cleanup(self):
        from unittest.mock import patch
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s1-d1")
        state = self.runtime._adopt_existing_9x9()
        definition = replace(self.runtime.config.stages[0], fixed_baseline_models=("missing",))
        self.runtime.config = replace(self.runtime.config, stages=(definition, *self.runtime.config.stages[1:]))
        before = state.to_dict()
        with self.assertRaises(CurriculumStateError):
            self.runtime._ensure_opponent_pools(state)
        self.assertEqual(state.to_dict(), before)

    def test_dashboard_contains_course_health_and_storage(self) -> None:
        run_root = self.root / "9x9"
        _make_model(run_root, "gogogo-s2048-d100")
        state = self.runtime._adopt_existing_9x9()
        state.disk = {"free_gib": 100.5, "total_gib": 500, "status": "healthy"}
        self.runtime._render_dashboard(state)
        document = self.runtime.dashboard_path.read_text(encoding="utf-8")
        self.assertIn("9×9 → 13×13 → 19×19", document)
        self.assertIn("开局双停", document)
        self.assertIn("磁盘", document)


if __name__ == "__main__":
    unittest.main()
