"""Fixed wall budgets must stop cleanly and must never hide crashed training."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from weiqi.katago_hardware import HardwareProfile
from weiqi.katago_mixed import MixedBoardRunner
from weiqi.katago_mixed_experiments import prepare_mixed_experiment
from weiqi.katago_mixed_trials import run_budgeted_process, run_prepared_experiment, summarize_budget


class BudgetedMixedTrialTests(unittest.TestCase):
    def test_deadline_interrupts_training_and_records_shutdown_separately(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["trainer"], 2), 130]

        @contextmanager
        def start(*args, **kwargs):
            yield process

        with tempfile.TemporaryDirectory() as tmp, \
                patch("weiqi.katago_mixed_trials.managed_popen", start), \
                patch("weiqi.katago_mixed_trials.time.monotonic", side_effect=[10., 12., 12.5]):
            report = run_budgeted_process(["trainer"], Path(tmp)/"out.log", wall_seconds=2)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertEqual(report["returncode"], 130)
        self.assertEqual(report["budget_seconds"], 2)
        self.assertEqual(report["training_seconds"], 2)
        self.assertEqual(report["shutdown_seconds"], .5)

    def test_premature_exit_even_zero_is_not_a_completed_budget(self):
        process = Mock(pid=123)
        process.wait.return_value = 0

        @contextmanager
        def start(*args, **kwargs):
            yield process

        with tempfile.TemporaryDirectory() as tmp, \
                patch("weiqi.katago_mixed_trials.managed_popen", start):
            with self.assertRaisesRegex(RuntimeError, "before|提前"):
                run_budgeted_process(["trainer"], Path(tmp)/"out.log", wall_seconds=2)
        process.send_signal.assert_not_called()

    def test_crash_during_shutdown_is_reported_as_failure(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(["trainer"], 2), 1]

        @contextmanager
        def start(*args, **kwargs):
            yield process

        with tempfile.TemporaryDirectory() as tmp, \
                patch("weiqi.katago_mixed_trials.managed_popen", start):
            with self.assertRaisesRegex(RuntimeError, "exit|退出"):
                run_budgeted_process(["trainer"], Path(tmp)/"out.log", wall_seconds=2)

    def test_actual_deltas_and_completed_stage_seconds_are_explicit(self):
        initial = {"global_step_samples": 100, "total_num_data_rows": 50}
        final = {"global_step_samples": 300, "total_num_data_rows": 90}
        report = summarize_budget(initial, final, [
            {"model": "old", "trained_samples": 100, "train_seconds": 99},
            {"model": "new", "trained_samples": 300, "train_seconds": 3,
             "shuffle_seconds": 2, "selfplay_stage_seconds": 5, "export_seconds": 1,
             "selfplay_data_rows": 42},
        ])
        self.assertEqual(report["trained_samples_delta"], 200)
        self.assertEqual(report["trainer_data_rows_delta"], 40)
        self.assertEqual(report["completed_stages_seconds"]["train_seconds"], 3)
        self.assertEqual(report["generated_rows_in_completed_cycles"], 42)
        self.assertEqual(report["completed_cycles"], 1)

    def test_nonpositive_budget_is_rejected_before_starting_process(self):
        with patch("weiqi.katago_mixed_trials.managed_popen") as start:
            for value in (0, -1, True, float("inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run_budgeted_process(["trainer"], Path("unused"), wall_seconds=value)
        start.assert_not_called()


class PreparedTrialIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        config = self.root / "mixed.json"
        config.write_text(json.dumps({
            "schema_version": 1, "board_sizes": [7,8,9,13,19], "board_weights": [2,2,3,2,1],
            "shuffle_rows_per_file": 256,
            "training": {"schema_version": 1, "preset": "balanced", "overrides": {
                "game": {"board_size": 19}, "network": {"channels": 128, "residual_blocks": 10},
                "hardware": {"device": "cpu", "precision": "float32"},
                "runtime": {"output_directory": str(source)}}}}))
        MixedBoardRunner(config).ensure_manifest()
        self.checkpoint = source / "train/gogogo/checkpoint.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        torch.save({"model": {"weight": torch.tensor([1.])}, "optimizer": {"valid": True},
                    "train_state": {"global_step_samples": 100, "total_num_data_rows": 1}}, self.checkpoint)
        replay = source / "selfplay/seed/tdata/one.npz"
        replay.parent.mkdir(parents=True)
        masks = np.zeros((1,22,19,19), dtype=np.uint8); masks[0,0,:7,:7] = 1
        np.savez_compressed(replay, globalInputNC=np.zeros((1,1)),
                            binaryInputNCHWPacked=np.packbits(masks.reshape(1,22,361),axis=2),
                            valueTargetsNCHW=np.zeros((1,5,19,19)))
        self.output = self.root / "experiment"
        self.manifest = prepare_mixed_experiment(source, config, self.output, wall_seconds=2)
        self.profile = HardwareProfile("cpu", "cpu", "float32", None, 0, 1, False)

    @staticmethod
    def export(runner):
        for checkpoint in (runner.run_root / "torchmodels_toexport").glob("*/model.ckpt"):
            model = runner.models_dir / checkpoint.parent.name / "model.bin.gz"
            model.parent.mkdir(parents=True, exist_ok=True)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.write_text(str(state["model"]["weight"].item()))
        return []

    def test_both_arms_have_initial_models_before_timer_and_source_stays_unchanged(self):
        before_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        calls = []

        def budget(command, log, *, wall_seconds):
            config = Path(command[command.index("--mixed-config") + 1])
            runner = MixedBoardRunner(config)
            self.assertEqual(wall_seconds, 2)
            self.assertTrue((runner.models_dir / "gogogo-s100-d1/model.bin.gz").exists())
            checkpoint = runner.run_root / "train/gogogo/checkpoint.ckpt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(state["model"]["weight"].item(), 1)
            calls.append(config.parent.name)
            state["model"]["weight"] += len(calls)
            state["train_state"]["global_step_samples"] += 32
            torch.save(state, checkpoint)
            return {"budget_seconds": 2, "training_seconds": 2, "shutdown_seconds": 0, "returncode": 130}

        with patch.object(MixedBoardRunner, "export_new_models", self.export), \
                patch("weiqi.katago_mixed_trials.run_budgeted_process", side_effect=budget), \
                patch("weiqi.katago_mixed_trials.evaluate_mixed", return_value={"status": "complete"}) as evaluate:
            result = run_prepared_experiment(self.output, hardware_profile=self.profile, games_per_board=2)
        self.assertEqual(calls, ["control", "sgf_starts"])
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual([v["accounting"]["trained_samples_delta"] for v in result["arms"].values()], [32,32])
        self.assertEqual(evaluate.call_args_list[0].kwargs["baseline"], evaluate.call_args_list[1].kwargs["baseline"])
        self.assertEqual(hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(), before_hash)
        with self.assertRaisesRegex(ValueError, "already started"):
            run_prepared_experiment(self.output, hardware_profile=self.profile)

    def test_extra_arm_replay_is_rejected_before_any_process_or_export(self):
        extra = Path(self.manifest["arms"]["sgf_starts"]["run_root"]) / "selfplay/extra/tdata/extra.npz"
        extra.parent.mkdir(parents=True)
        np.savez_compressed(extra, globalInputNC=np.zeros((1,1)))
        with patch("weiqi.katago_mixed_trials.run_budgeted_process") as budget, \
                patch.object(MixedBoardRunner, "export_new_models") as export:
            with self.assertRaisesRegex(ValueError, "replay|Replay|inventory"):
                run_prepared_experiment(self.output, hardware_profile=self.profile)
        budget.assert_not_called()
        export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
