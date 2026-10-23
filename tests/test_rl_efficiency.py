"""Behavioral checks for faster evaluation, recovery and adaptive scheduling."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_rl_curriculum import curriculum_dict, passing_health, passing_match
from test_rl_curriculum_runtime import _FakeRunner, _make_model, _write_profiles
from weiqi.katago_hardware import HardwareProfile
from weiqi.rl_curriculum import CurriculumController, CurriculumStateStore, load_curriculum_config
from weiqi.rl_curriculum_runtime import CurriculumRuntime
from weiqi.rl_match_jobs import run_opening_pairs
from weiqi.rl_metric_store import RLMetricStore
from weiqi.katago_rl import KataGoRLRunner, KataGoRLRunnerError
from scripts.benchmark_rl_efficiency import main as benchmark_main, validate_checkpoint_update


PAIR = "(;PB[candidate]PW[baseline]RE[B+1.5])\n(;PB[baseline]PW[candidate]RE[W+1.5])\n"


class PairedMatchExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.openings = [self.root / f"opening-{i:03d}" for i in range(8)]
        self.build = lambda opening: f"hintPosesDir = {opening}\nnumGamesTotal = 2\n"

    @staticmethod
    def success(_config, sgfs, _log):
        (sgfs / "pair.sgfs").write_text(PAIR, encoding="utf-8")
        return 0

    def test_parallel_jobs_are_bounded_and_keep_every_color_pair(self):
        lock = threading.Lock()
        barrier = threading.Barrier(4, timeout=5)
        active = peak = 0

        def execute(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait()
                return self.success(*args)
            finally:
                with lock:
                    active -= 1

        result = run_opening_pairs(
            self.openings, self.root, parallel_pairs=4,
            config_builder=self.build, execute=execute,
        )
        self.assertEqual(peak, 4)
        self.assertEqual(result.games, 16)
        self.assertEqual(result.candidate_black_games, 8)
        self.assertEqual(result.candidate_white_games, 8)
        self.assertEqual(result.damaged_games, 0)
        progress = json.loads((self.root / "progress.json").read_text())
        self.assertEqual(progress["completed_pairs"], 8)

    def test_recovery_reuses_complete_pairs_and_keeps_incomplete_attempt(self):
        run_opening_pairs(self.openings, self.root, parallel_pairs=1,
                          config_builder=self.build, execute=self.success)
        broken = self.root / "pairs/opening-003/sgfs/pair.sgfs"
        broken.write_text(PAIR.splitlines()[0], encoding="utf-8")
        called = []

        def execute(config, sgfs, log):
            called.append(config)
            return self.success(config, sgfs, log)

        result = run_opening_pairs(self.openings, self.root, parallel_pairs=4,
                                  config_builder=self.build, execute=execute)
        self.assertEqual(len(called), 1)
        self.assertIn("retry-", str(called[0]))
        self.assertEqual(broken.read_text(), PAIR.splitlines()[0])
        self.assertEqual(result.games, 16)  # The failed attempt is never counted twice.
        progress = json.loads((self.root / "progress.json").read_text())
        self.assertEqual(progress["reused_pairs"], 7)
        run_opening_pairs(self.openings, self.root, parallel_pairs=4,
                          config_builder=self.build,
                          execute=lambda *_: self.fail("complete pairs were run again"))

    def test_changed_config_forces_fresh_pairs(self):
        run_opening_pairs(self.openings[:1], self.root, parallel_pairs=1,
                          config_builder=self.build, execute=self.success)
        calls = []
        run_opening_pairs(self.openings[:1], self.root, parallel_pairs=1,
                          config_builder=lambda p: self.build(p) + "maxVisits = 400\n",
                          execute=lambda *args: calls.append(args) or self.success(*args))
        self.assertEqual(len(calls), 1)

    def test_unbalanced_or_failed_pairs_cannot_publish_results(self):
        def unbalanced(_config, sgfs, _log):
            (sgfs / "bad.sgfs").write_text(PAIR.splitlines()[0] * 2)
            return 0

        for execute in (unbalanced, lambda *_: 7):
            with self.subTest(execute=execute), self.assertRaises(RuntimeError):
                run_opening_pairs(self.openings[:1], self.root, parallel_pairs=1,
                                  config_builder=self.build, execute=execute)
        self.assertFalse((self.root / "match-result.json").exists())

    def test_truncated_extra_sgf_cannot_be_reused_as_a_complete_pair(self):
        run_opening_pairs(self.openings[:1], self.root, parallel_pairs=1,
                          config_builder=self.build, execute=self.success)
        path = self.root / "pairs/opening-000/sgfs/pair.sgfs"
        path.write_text(PAIR + "(;PB[candidate]")
        calls = []
        result = run_opening_pairs(self.openings[:1], self.root, parallel_pairs=1,
                                  config_builder=self.build,
                                  execute=lambda *args: calls.append(args) or self.success(*args))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.games, 2)


class AdaptiveEvaluationTests(unittest.TestCase):
    def test_invalid_parallelism_and_intervals_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for change in ({"evaluation_parallel_pairs": True},
                           {"evaluation_parallel_pairs": 0},
                           {"evaluation_parallel_pairs": 17},
                           {"evaluation_midstage_interval_samples": 750_000},
                           {"evaluation_midstage_interval_samples": 100_000},
                           {"evaluation_near_gate_samples": 0}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    path.write_text(json.dumps({**curriculum_dict(), **change}))
                    load_curriculum_config(path)

    def test_midstage_interval_returns_to_dense_checks_before_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = curriculum_dict()
            value.update(evaluation_parallel_pairs=4,
                         evaluation_midstage_interval_samples=2_000_000,
                         evaluation_near_gate_samples=1_000_000)
            path = root / "config.json"
            path.write_text(json.dumps(value))
            config = load_curriculum_config(path)
            controller = CurriculumController(config, CurriculumStateStore(root))
            self.assertEqual(controller.next_evaluation_sample(1, 20_004_864), 22_000_000)
            self.assertEqual(controller.next_evaluation_sample(1, 23_900_000), 24_000_000)
            self.assertEqual(controller.next_evaluation_sample(1, 24_000_000), 24_500_000)
            self.assertEqual(controller.next_evaluation_sample(1, 25_000_000), 25_500_000)
            self.assertEqual(controller.next_evaluation_sample(2, 500_000), 2_000_000)
            state = controller.adopt_existing_9x9(trained_samples=8_000_000, latest_model="net")
            controller.record_evaluation(state, candidate_model="net", match=passing_match(),
                                         health=passing_health())
            self.assertFalse(state.active.evaluations[-1]["passed"])
            self.assertEqual(state.active.consecutive_passes, 0)
            self.assertEqual(config.evaluation_games, 200)

    def test_resume_requires_same_models_openings_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = _write_profiles(root)
            runtime = CurriculumRuntime(path, project_root=root, runner_factory=_FakeRunner)
            runtime.config = replace(runtime.config, evaluation_games=4)
            for name in ("gogogo-s1-d1", "gogogo-s2-d2"):
                _make_model(root / "9x9", name)
            state = runtime._adopt_existing_9x9()
            runner = runtime._stage_runner(state)
            opponents = runtime.controller.evaluation_opponents(state)
            output = runtime._evaluation_directory(state)
            name, roles = next(iter(opponents.items()))
            _, manifest = runtime._match_manifest(state, runner, state.active.latest_model, name, roles)
            child = output / f"opponent-00-{name}"
            child.mkdir(parents=True)
            (child / "evaluation_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(runtime._pending_evaluation_directory(
                state, runner, state.active.latest_model, opponents), output)
            manifest["candidate_sha256"] = "wrong"
            (child / "evaluation_manifest.json").write_text(json.dumps(manifest))
            self.assertIsNone(runtime._pending_evaluation_directory(
                state, runner, state.active.latest_model, opponents))

    def test_small_devices_do_not_inherit_four_engine_concurrency(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = _write_profiles(Path(temporary))
            runtime = CurriculumRuntime(path, project_root=Path(temporary), runner_factory=_FakeRunner)
            runtime.config = replace(runtime.config, evaluation_parallel_pairs=4)
            runtime.hardware_profile = HardwareProfile("cpu", "cpu", "float32", None, 0, 4, False)
            self.assertEqual(runtime._parallel_pairs(), 1)
            runtime.hardware_profile = HardwareProfile("low", "cuda:0", "amp_float16", "GPU", 4, 8, True)
            self.assertEqual(runtime._parallel_pairs(), 1)
            runtime.hardware_profile = HardwareProfile("high", "cuda:0", "amp_bfloat16", "GPU", 16, 16, True)
            self.assertEqual(runtime._parallel_pairs(), 4)


class EfficiencyMetricTests(unittest.TestCase):
    def test_waiting_and_sample_reuse_are_distinct_from_training_speed(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RLMetricStore(Path(temporary))
            store.metrics_dir.mkdir(parents=True)
            records = [
                {"model": "gogogo-s1000-d500", "trained_samples": 1000, "data_rows": 500,
                 "timestamp": "2026-10-15T10:00:00+08:00", "cycle_seconds": 100},
                {"model": "gogogo-s5000-d1000", "trained_samples": 5000, "data_rows": 1000,
                 "timestamp": "2026-10-15T11:00:00+08:00", "cycle_seconds": 100},
            ]
            store.jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in records))
            result = store.sync()[-1]
            self.assertEqual(result["new_data_rows_per_hour"], 18_000)
            self.assertEqual(result["wall_new_data_rows_per_hour"], 500)
            self.assertEqual(result["sample_reuse_ratio"], 8)
            self.assertIn("含评测等待", store.dashboard_path.read_text(encoding="utf-8"))
            self.assertIn("wall_new_data_rows_per_hour", store.csv_path.read_text(encoding="utf-8-sig"))


class LargerCycleTrainingTests(unittest.TestCase):
    def test_expanded_19x19_budget_preserves_evaluation_and_fits_fresh_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(__file__).resolve().parents[1] / "config/rl_training.rtx5070ti.curriculum.19x19.json"
            runner = KataGoRLRunner(profile, run_root_override=Path(temporary))
            self.assertEqual(runner.config.self_play.games_per_iteration, 2048)
            self.assertEqual(runner.config.evaluation.games, 200)
            self.assertEqual(runner.config.evaluation.promotion_win_rate, 0.6)
            data = runner.shuffled_dir / "latest/train/data.npz"
            data.parent.mkdir(parents=True)
            data.touch()
            with patch.object(runner, "_require_runtime"), patch.object(runner, "_run") as run:
                runner.train(batch_size=1280, new_data_rows=158_448)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-samples-per-epoch") + 1], 327_680)
                self.assertEqual(command[command.index("-max-train-bucket-per-new-data") + 1], 4)
                runner.train(batch_size=1280, new_data_rows=10_000)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("-samples-per-epoch") + 1], 39_680)

    def test_large_epoch_fits_the_fresh_data_bucket(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(__file__).resolve().parents[1] / "config/rl_training.rtx5070ti.13x13.json"
            runner = KataGoRLRunner(profile, run_root_override=Path(temporary))
            data = runner.shuffled_dir / "latest/train/data.npz"
            data.parent.mkdir(parents=True)
            data.touch()
            with patch.object(runner, "_require_runtime"), patch.object(runner, "_run") as run:
                runner.train(batch_size=2048, new_data_rows=12_000)
                command = run.call_args.args[0]
                samples = command[command.index("-samples-per-epoch") + 1]
                self.assertGreater(samples, 0)
                self.assertLessEqual(samples, 4 * 12_000)
                self.assertEqual(samples % 2048, 0)
                self.assertEqual(command[command.index("-max-train-bucket-per-new-data") + 1], 4)
                with self.assertRaises(KataGoRLRunnerError):
                    runner.train(batch_size=2048, new_data_rows=100)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_validation_rejects_nonfinite_optimizer_or_unchanged_weights(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            before, after = (Path(temporary) / name for name in ("before.ckpt", "after.ckpt"))
            payload = {"model": {"weight": torch.tensor([1.0])}, "swa_model": {},
                       "optimizer": {"state": {1: {"exp_avg": torch.tensor([0.0])}}}}
            torch.save(payload, before)
            torch.save(payload, after)
            with self.assertRaisesRegex(RuntimeError, "did not change"):
                validate_checkpoint_update(before, after)
            payload["model"]["weight"] = torch.tensor([2.0])
            torch.save(payload, after)
            self.assertEqual(validate_checkpoint_update(before, after)["changed_model_tensors"], 1)
            payload["optimizer"]["state"][1]["exp_avg"] = torch.tensor([float("nan")])
            torch.save(payload, after)
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                validate_checkpoint_update(before, after)

    def test_invalid_resource_requests_fail_before_opening_the_live_curriculum(self):
        cases = (
            ["--threads", "0"], ["--threads", "31"], ["--threads", "129"],
            ["--threads", "272"], ["--threads", "128", "128"],
            ["--training-steps", "0"], ["--mode", "training"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), patch(
                "sys.argv", ["benchmark_rl_efficiency.py", *arguments]
            ), patch("scripts.benchmark_rl_efficiency.CurriculumRuntime") as runtime:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    benchmark_main()
                self.assertEqual(raised.exception.code, 2)
                runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
