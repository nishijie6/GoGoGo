"""Check hardware routing across the official training and curriculum stages."""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from weiqi.katago_hardware import HardwareProfile
from weiqi.katago_rl import DEFAULT_RTX_PROFILE, KataGoRLRunner, build_selfplay_overrides
from weiqi.katago_rl_cli import build_parser, main
from weiqi.rl_benchmark import build_benchmark_command
from weiqi.rl_curriculum import CurriculumStage
from weiqi.rl_curriculum import compute_health_window, HealthThresholds
from weiqi.rl_curriculum_runtime import CurriculumRuntime
from weiqi.rl_match import build_match_config


def cpu_profile():
    return HardwareProfile("cpu", "cpu", "float32", None, 0, 4, False)


def low_profile():
    return HardwareProfile("low", "cuda:1", "amp_float16", "Test GPU", 4, 4, True)


class HardwareRoutingTests(unittest.TestCase):
    def test_cpu_engine_environment_and_selfplay_use_the_same_backend(self):
        profile = cpu_profile()
        runner = KataGoRLRunner(hardware_profile=profile)
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.assertEqual(runner.katago_executable.parent.name, "cpu")
        self.assertEqual(runner._environment()["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original)
        overrides = build_selfplay_overrides(runner.config, hardware_profile=profile)
        self.assertFalse(any(key.startswith("cuda") for key in overrides))
        self.assertEqual(overrides["nnMaxBatchSize"], 1)
        self.assertEqual(overrides["numGameThreads"], 4)
        self.assertEqual(overrides["numEigenThreadsPerModel"], 4)
        self.assertEqual(overrides["maxVisits"], 200)

    def test_smaller_hardware_caps_persisted_batch_without_changing_model(self):
        cpu = KataGoRLRunner(hardware_profile=cpu_profile())
        gpu = KataGoRLRunner(hardware_profile=low_profile())
        self.assertEqual(cpu.training_batch_size(2048), 16)
        self.assertEqual(gpu.training_batch_size(2048), 64)
        self.assertEqual(cpu.model_kind, gpu.model_kind)
        self.assertEqual(cpu.run_root, gpu.run_root)

    def test_train_respects_cpu_and_selected_gpu(self):
        for profile in (cpu_profile(), low_profile()):
            with self.subTest(tier=profile.tier), tempfile.TemporaryDirectory() as tmp:
                runner = KataGoRLRunner(run_root_override=Path(tmp), hardware_profile=profile)
                data = runner.shuffled_dir / "latest" / "train" / "data.npz"
                data.parent.mkdir(parents=True)
                data.touch()
                with patch.object(runner, "_require_runtime"), patch.object(runner, "_run") as run:
                    runner.train(batch_size=2048)
                command = [str(value) for value in run.call_args.args[0]]
                batch = int(command[command.index("-batch-size") + 1])
                self.assertEqual(batch, runner.training_batch_size(2048))
                if profile.tier == "cpu":
                    self.assertNotIn("-multi-gpus", command)
                    self.assertNotIn("-use-fp16", command)
                    self.assertNotIn("-use-bf16", command)
                    self.assertNotIn("-use-tf32-matmul", command)
                else:
                    self.assertEqual(command[command.index("-multi-gpus") + 1], "1")
                    self.assertIn("-use-fp16", command)
                    self.assertNotIn("-use-bf16", command)

    def test_match_cpu_keeps_fixed_protocol_without_cuda_settings(self):
        text = build_match_config(
            candidate_model=Path("candidate.bin.gz"), baseline_model=Path("baseline.bin.gz"),
            board_size=13, visits=300, games=200, hardware_profile=cpu_profile(),
            inference_batch_size=1,
        )
        self.assertNotIn("cuda", text)
        self.assertIn("numEigenThreadsPerModel = 4", text)
        self.assertIn("maxVisits = 300", text)
        self.assertIn("numGamesTotal = 200", text)
        self.assertIn("rootNoiseEnabled = false", text)

    def test_match_gpu_uses_selected_device(self):
        text = build_match_config(
            candidate_model=Path("candidate.bin.gz"), baseline_model=Path("baseline.bin.gz"),
            board_size=9, visits=200, hardware_profile=low_profile(),
        )
        self.assertIn("cudaDeviceToUse = 1", text)

    def test_cpu_autotune_skips_cuda_benchmark_and_persists_safe_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CurriculumRuntime.__new__(CurriculumRuntime)
            runtime.hardware_profile = cpu_profile()
            runner = KataGoRLRunner(
                config_path=DEFAULT_RTX_PROFILE.with_name("rl_training.rtx5070ti.13x13.json"),
                run_root_override=Path(tmp), hardware_profile=cpu_profile(),
            )
            stage = CurriculumStage(13, "unused.json", 25_000_000, (512, 1024), False)
            runtime._benchmark_measurement = Mock(side_effect=AssertionError("CUDA-only benchmark"))
            batch, warning = runtime._autotune(stage, runner, Path("nonexistent.npz"))
            self.assertEqual(batch, 8)
            self.assertIsNone(warning)
            self.assertTrue((runner.run_root / "autotune.json").is_file())
            runtime._benchmark_measurement.assert_not_called()

    def test_cycle_caps_a_saved_large_batch_for_new_hardware(self):
        runtime = CurriculumRuntime.__new__(CurriculumRuntime)
        runtime.hardware_profile = cpu_profile()
        runtime.config = SimpleNamespace(champion_evaluation_every_cycle=False)
        runner = KataGoRLRunner(hardware_profile=cpu_profile())
        runner.lock = contextlib.nullcontext
        runner.cycle = Mock()
        runtime._stage_runner = Mock(return_value=runner)
        runtime._refresh_progress = Mock(return_value=False)
        state = SimpleNamespace(active=SimpleNamespace(autotune_batch=2048), phase="training")
        runtime._cycle(state, smoke=False)
        self.assertEqual(runner.cycle.call_args.kwargs["training_batch_size"], 16)
        self.assertEqual(state.active.autotune_batch, 2048)

    def test_benchmark_matches_training_precision_and_gpu(self):
        for precision, enabled, excluded in (
            ("amp_float16", "-use-amp", "-use-bf16"),
            ("amp_bfloat16", "-use-bf16", "-use-amp"),
        ):
            command = build_benchmark_command(
                python_executable=Path("python"), benchmark_script=Path("benchmark.py"),
                data_file=Path("data.npz"), board_size=13, batch_size=32, gpu_index=1,
                precision=precision, allow_tf32=False,
            )
            self.assertIn(enabled, command)
            self.assertNotIn(excluded, command)
            self.assertNotIn("-use-tf32-matmul", command)
            self.assertEqual(command[command.index("-gpu") + 1], "1")

    def test_cli_defaults_to_auto_and_hardware_inspection_does_not_train(self):
        self.assertEqual(build_parser().parse_args(["curriculum"]).hardware, "auto")
        with patch("weiqi.katago_rl_cli.detect_hardware", return_value=cpu_profile()) as detect, \
                patch("weiqi.katago_rl_cli.KataGoRLRunner") as runner:
            main(["hardware", "--hardware", "cpu"])
        detect.assert_called_once_with("cpu")
        runner.assert_not_called()

    def test_reading_curriculum_status_does_not_probe_gpu(self):
        with patch("weiqi.katago_rl_cli.detect_hardware") as detect, \
                patch("weiqi.katago_rl_cli.CurriculumRuntime") as runtime:
            main(["curriculum-status"])
        detect.assert_not_called()
        runtime.return_value.status.assert_called_once()

    def test_cpu_production_cycle_accumulates_replay_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = KataGoRLRunner(run_root_override=Path(tmp), hardware_profile=cpu_profile())
            data = runner.selfplay_dir / "seed" / "tdata" / "small.npz"
            data.parent.mkdir(parents=True)
            np.savez_compressed(data, globalInputNC=np.zeros((100, 1), dtype=np.float32))
            with patch.object(runner, "selfplay", return_value={}), \
                    patch.object(runner, "shuffle") as shuffle, \
                    patch.object(runner, "train") as train, \
                    patch.object(runner.metric_store, "sync"):
                runner.cycle()
            shuffle.assert_not_called()
            train.assert_not_called()
            self.assertTrue(data.is_file())

    def test_cpu_production_cycle_trains_when_accumulated_replay_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = KataGoRLRunner(run_root_override=Path(tmp), hardware_profile=cpu_profile())
            data = runner.selfplay_dir / "seed" / "tdata" / "ready.npz"
            data.parent.mkdir(parents=True)
            np.savez_compressed(data, globalInputNC=np.zeros((2048, 1), dtype=np.float32))
            with patch.object(runner, "selfplay", return_value={}), \
                    patch.object(runner, "shuffle", return_value=None), \
                    patch.object(runner, "train") as train, \
                    patch.object(runner, "export_new_models", return_value=[Path("new-model")]), \
                    patch.object(runner, "status", return_value={"accepted_models": 1}), \
                    patch.object(runner.metric_store, "sync", return_value=[]):
                runner.cycle()
            train.assert_called_once()

    def test_cpu_health_window_retains_original_game_evidence(self):
        records = [{"games": 4, "black_wins": 2, "white_wins": 2} for _ in range(320)]
        self.assertFalse(compute_health_window(records[:10], minimum_games=1280).complete)
        health = compute_health_window(records, minimum_games=1280)
        self.assertTrue(health.passed)
        self.assertEqual(health.games, 1280)
        self.assertEqual(health.cycles, 320)

    def test_health_window_handles_mixed_hardware_cycles(self):
        records = ([{"games": 128, "black_wins": 64, "white_wins": 64}] * 10
                   + [{"games": 4, "black_wins": 2, "white_wins": 2}] * 10)
        health = compute_health_window(records, minimum_games=1280)
        self.assertTrue(health.passed)
        self.assertEqual(health.cycles, 20)
        self.assertEqual(health.games, 1320)

    def test_dashboard_health_does_not_depend_on_hardware_detection(self):
        runtime = CurriculumRuntime.__new__(CurriculumRuntime)
        runtime.hardware_profile = None
        runtime.config = SimpleNamespace(health_window_cycles=10, health=HealthThresholds())
        runtime._read_metric_records = Mock(return_value=[
            {"sgf_entries": 4, "black_wins": 2, "white_wins": 2} for _ in range(320)
        ])
        self.assertTrue(runtime._health(None).passed)


if __name__ == "__main__":
    unittest.main()
