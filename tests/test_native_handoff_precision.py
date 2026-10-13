"""Resume across precision settings and different CUDA device inventories."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.rl_config import resolve_rl_training_config


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
CUDA_AVAILABLE = False
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.runner import Trainer, _restore_cuda_rng
    from weiqi.rl.state import Position
    from weiqi.rl.storage import cpu_state, fingerprint, load_checkpoint
    CUDA_AVAILABLE = torch.cuda.is_available()


class RecordingScaler:
    """Enabled scaler substitute: restoring an FP32 empty state would fail."""
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state):
        if not state:
            raise AssertionError("An empty FP32 scaler must not be restored into newly enabled AMP")
        self.loaded = deepcopy(state)


@unittest.skipUnless(AVAILABLE, "Optional training dependencies PyTorch/NumPy are not installed")
class NativePrecisionResumeTests(unittest.TestCase):
    def config(self):
        return resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32", "data_loader_workers": 0},
            "network": {"channels": 8, "residual_blocks": 1},
            "self_play": {"milestone_fraction": 0.0},
            "optimizer": {"batch_size": 2, "minimum_replay_size": 1,
                          "training_steps_per_iteration": 8, "target_sample_reuse": None,
                          "use_board_symmetry_augmentation": False},
            "runtime": {"pause_while_game_is_active": False},
        })

    def checkpoint(self, output):
        trainer = Trainer(self.config(), output)
        position = Position.new(9, self.config().game.komi)
        policy = position.legal.astype(np.float32)
        policy /= policy.sum()
        trainer.replay.extend([(position.features(), policy.copy(), 1.0) for _ in range(4)])
        trainer.train_updates()
        trainer.iteration = 3
        trainer.best_iteration = 2
        trainer.save(archive=False)
        payload, config = load_checkpoint(output / "latest.pt")
        self.assertFalse(payload["scaler"])
        self.assertTrue(payload["optimizer"]["state"])
        return payload, config

    def test_fp32_empty_scaler_keeps_newly_enabled_scaler_defaults(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            payload, config = self.checkpoint(root / "source")
            scaler = RecordingScaler()
            with patch("weiqi.rl.runner.torch.amp.GradScaler", return_value=scaler):
                restored = Trainer(config, root / "destination", payload)
            self.assertIsNone(scaler.loaded)
            self.assertEqual(fingerprint(cpu_state(restored.model)), fingerprint(payload["model"]))
            self.assertEqual(restored.training_steps, payload["training_steps"])
            self.assertEqual(restored.iteration, 3)
            self.assertEqual(len(restored.replay), 4)
            self.assertTrue(restored.optimizer.state)

    def test_nonempty_scaler_state_is_restored(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            payload, config = self.checkpoint(root / "source")
            payload["scaler"] = {"scale": 32768.0, "growth_factor": 2.0,
                                 "backoff_factor": 0.5, "growth_interval": 2000,
                                 "_growth_tracker": 11}
            scaler = RecordingScaler()
            with patch("weiqi.rl.runner.torch.amp.GradScaler", return_value=scaler):
                Trainer(config, root / "destination", payload)
            self.assertEqual(scaler.loaded, payload["scaler"])

    def test_cpu_destination_never_restores_saved_cuda_rng(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            payload, config = self.checkpoint(root / "source")
            payload["cuda_rng"] = [torch.zeros(10, dtype=torch.uint8)]
            with patch("weiqi.rl.runner._restore_cuda_rng", side_effect=AssertionError("CPU destination")):
                restored = Trainer(config, root / "destination", payload)
            self.assertEqual(restored.runtime.device.type, "cpu")

    def test_two_saved_devices_restore_only_one_available_device(self):
        states = [torch.tensor([index], dtype=torch.uint8) for index in range(2)]
        with patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_rng_state") as setter:
            _restore_cuda_rng({"cuda_rng": states}, torch.device("cuda:0"))
        self.assertEqual(setter.call_count, 1)
        self.assertIs(setter.call_args.args[0], states[0])
        self.assertEqual(setter.call_args.kwargs["device"], 0)

    def test_source_active_device_is_mapped_to_destination_active_device(self):
        states = [torch.tensor([index], dtype=torch.uint8) for index in range(2)]
        resume = {"cuda_rng": states, "config": {"hardware": {"device": "cuda:1"}}}
        with patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_rng_state") as setter:
            _restore_cuda_rng(resume, torch.device("cuda:0"))
        self.assertEqual(setter.call_count, 2)
        self.assertIs(setter.call_args.args[0], states[1])
        self.assertEqual(setter.call_args.kwargs["device"], 0)

    def test_auto_source_rng_maps_to_current_target_and_new_devices_stay_seeded(self):
        state = torch.tensor([5], dtype=torch.uint8)
        with patch("torch.cuda.device_count", return_value=3), \
                patch("torch.cuda.current_device", return_value=2), patch("torch.cuda.set_rng_state") as setter:
            _restore_cuda_rng({"cuda_rng": [state]}, torch.device("cuda"))
        self.assertEqual([call.kwargs["device"] for call in setter.call_args_list], [0, 2])
        self.assertTrue(all(call.args[0] is state for call in setter.call_args_list))

    def test_empty_saved_cuda_rng_never_queries_or_initializes_devices(self):
        with patch("torch.cuda.device_count", side_effect=AssertionError("No saved CUDA RNG")), \
                patch("torch.cuda.set_rng_state", side_effect=AssertionError("No saved CUDA RNG")):
            _restore_cuda_rng({"cuda_rng": []}, torch.device("cuda:0"))

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
    def test_cpu_fp32_checkpoint_restores_cuda_amp16_and_updates_with_finite_loss(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            payload, config = self.checkpoint(root / "source")
            target = replace(config, hardware=replace(config.hardware, device="cuda:0", precision="amp_float16"))
            restored = Trainer(target, root / "destination", payload)
            self.assertEqual(restored.runtime.device.type, "cuda")
            self.assertTrue(restored.scaler.is_enabled())
            self.assertEqual(fingerprint(cpu_state(restored.model)), fingerprint(payload["model"]))
            result = restored.train_updates()
            self.assertTrue(math.isfinite(result["loss"]))
            self.assertTrue(result["weights_updated"])
            self.assertEqual(restored.training_steps,
                             payload["training_steps"] + config.optimizer.training_steps_per_iteration)


if __name__ == "__main__":
    unittest.main()
