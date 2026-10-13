"""Native checkpoint handoff validates restore state without mutating a run."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.fileio import acquire_file_lock, release_file_lock, sha256_file
from weiqi.rl.control import RunLock
from weiqi.rl_config import resolve_rl_training_config
from weiqi.rl_handoff import HandoffError
from weiqi.rl_handoff_guard import HANDOFF_MARKER, IMPORT_PENDING_MARKER
from weiqi.rl_native_handoff import (
    apply_native_hardware, check_native_checkpoint, native_source_plan,
)


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.network import PolicyValueNet
    from weiqi.rl.state import FEATURE_VERSION, Position
    from weiqi.rl.storage import CHECKPOINT_VERSION, ReplayBuffer, cpu_state


class NativeHandoffGuardTests(unittest.TestCase):
    def test_native_hold_blocks_and_releases_the_run_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / HANDOFF_MARKER
            marker.write_text(json.dumps({"trainer": "native", "bundle_id": "relay-1"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "py train.py handoff-cancel --run-dir"):
                with RunLock(root):
                    self.fail("A handed-off source cannot start training")
            stream = acquire_file_lock(root / "run.lock")
            release_file_lock(stream)
            marker.unlink()
            with RunLock(root):
                pass

    def test_pending_import_ancestor_blocks_and_releases_the_run_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            (root / IMPORT_PENDING_MARKER).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "导入尚未完成"):
                with RunLock(run):
                    self.fail("Partial imports cannot start training")
            stream = acquire_file_lock(run / "run.lock")
            release_file_lock(stream)

    def test_machine_profile_allows_only_documented_hardware_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "hardware.json"
            config = resolve_rl_training_config()
            for changes in ({"network": {"channels": 8}}, {"game": {"board_size": 13}},
                            {"optimizer": {"learning_rate": 0.1}}):
                with self.subTest(changes=changes):
                    profile.write_text(json.dumps(changes), encoding="utf-8")
                    with self.assertRaises(HandoffError):
                        apply_native_hardware(config, profile)
            profile.write_text(json.dumps({"hardware": {"device": "cpu", "precision": "float32"},
                                           "self_play": {"workers": 1},
                                           "optimizer": {"batch_size": 4}}), encoding="utf-8")
            changed = apply_native_hardware(config, profile)
            self.assertEqual(changed.hardware.device, "cpu")
            self.assertEqual(changed.optimizer.batch_size, 4)
            self.assertEqual(changed.optimizer.learning_rate, config.optimizer.learning_rate)
            self.assertEqual(changed.network, config.network)


@unittest.skipUnless(AVAILABLE, "Optional training dependencies PyTorch/NumPy are not installed")
class NativeCheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "existing_run"
        self.run.mkdir()
        self.config = resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32", "data_loader_workers": 0},
            "network": {"channels": 8, "residual_blocks": 1},
            "optimizer": {"batch_size": 4, "replay_buffer_capacity": 100, "minimum_replay_size": 1},
            "runtime": {"output_directory": str(self.run), "pause_while_game_is_active": False},
        })
        with torch.random.fork_rng(devices=[]):
            model = PolicyValueNet(self.config)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=self.config.optimizer.learning_rate,
                                          weight_decay=self.config.optimizer.weight_decay)
            # Populated moments make optimizer recovery validation meaningful.
            sum(parameter.sum() for parameter in model.parameters()).backward()
            optimizer.step()
            position = Position.new(9, self.config.game.komi)
            policy = position.legal.astype(np.float32)
            policy /= policy.sum()
            replay = ReplayBuffer(100, 9)
            replay.extend([(position.features(), policy, 1.0)])
            self.payload = {"checkpoint_version": CHECKPOINT_VERSION, "feature_version": FEATURE_VERSION,
                            "config": self.config.to_dict(), "kind": "training",
                            "model": cpu_state(model), "best_model": cpu_state(model),
                            "optimizer": optimizer.state_dict(), "scaler": {}, "replay": replay.state(),
                            "iteration": 7, "training_steps": 35, "best_iteration": 4,
                            "numpy_rng": np.random.default_rng(3).bit_generator.state,
                            "torch_rng": torch.get_rng_state(), "cuda_rng": []}
        self.checkpoint = self.run / "latest.pt"
        self.save()

    def save(self):
        torch.save(self.payload, self.checkpoint)

    def test_complete_checkpoint_validation_is_read_only_and_preserves_rng(self):
        before = {path.relative_to(self.run): sha256_file(path) for path in self.run.rglob("*") if path.is_file()}
        rng = torch.get_rng_state().clone()
        # A stale or malformed sidecar must not override latest.pt config.
        (self.run / "config.resolved.json").write_text("broken", encoding="utf-8")
        before[Path("config.resolved.json")] = sha256_file(self.run / "config.resolved.json")
        with patch("torch.cuda.init", side_effect=AssertionError("Must remain CPU only")), \
                patch("weiqi.rl.runner.Trainer", side_effect=AssertionError("Must not instantiate Trainer")):
            plan = native_source_plan(self.root, Path("existing_run"))
            report = check_native_checkpoint(self.checkpoint)
        self.assertEqual(plan["trainer"], "native")
        self.assertEqual(plan["sources"][0]["lock"], "run.lock")
        self.assertEqual(plan["configs"]["config/training.json"]["overrides"]["network"]["channels"], 8)
        self.assertEqual(report["iteration"], 7)
        self.assertEqual(report["training_steps"], 35)
        self.assertEqual(report["replay_samples"], 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(before, {path.relative_to(self.run): sha256_file(path)
                                 for path in self.run.rglob("*") if path.is_file()})

    def test_model_exports_and_missing_complete_state_are_rejected(self):
        for mutation in (lambda value: value.update(kind="model"), lambda value: value.pop("optimizer"),
                         lambda value: value.pop("torch_rng"), lambda value: value.pop("best_model")):
            original = deepcopy(self.payload)
            mutation(self.payload)
            self.save()
            with self.assertRaises(HandoffError):
                native_source_plan(self.root, self.run)
            self.payload = original

    def test_broken_model_replay_optimizer_and_rng_are_rejected(self):
        def break_model(value):
            value["best_model"].pop(next(iter(value["best_model"])))
        def break_replay(value):
            value["replay"]["policies"] = torch.zeros((1, 82))
        def break_optimizer(value):
            state = next(iter(value["optimizer"]["state"].values()))
            state["exp_avg"] = torch.zeros(2)
        def break_rng(value):
            value["torch_rng"] = torch.zeros(1)
        def break_moments(value):
            next(iter(value["optimizer"]["state"].values())).pop("exp_avg_sq")
        def break_scaler(value):
            value["scaler"] = {"scale": "invalid"}
        def break_empty_optimizer(value):
            value["optimizer"]["state"] = {}
        for mutation in (break_model, break_replay, break_optimizer, break_rng,
                         break_moments, break_scaler, break_empty_optimizer):
            original = deepcopy(self.payload)
            mutation(self.payload)
            self.save()
            with self.assertRaises(HandoffError):
                check_native_checkpoint(self.checkpoint)
            self.payload = original

    def test_imported_config_allows_hardware_and_output_only(self):
        document = native_source_plan(self.root, self.run)["configs"]["config/training.json"]
        document["overrides"]["runtime"]["output_directory"] = str(self.root / "new_run")
        document["overrides"]["optimizer"]["batch_size"] = 8
        document["overrides"]["hardware"]["device"] = "cuda:0"
        path = self.root / "imported_config.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        check_native_checkpoint(self.checkpoint, path)
        for section, name, value in (("game", "komi", 7.5), ("network", "channels", 16),
                                     ("optimizer", "learning_rate", 0.5), ("search", "c_puct", 2.0)):
            changed = deepcopy(document)
            changed["overrides"][section][name] = value
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(section=section), self.assertRaisesRegex(HandoffError, "训练算法不一致"):
                check_native_checkpoint(self.checkpoint, path)


if __name__ == "__main__":
    unittest.main()
