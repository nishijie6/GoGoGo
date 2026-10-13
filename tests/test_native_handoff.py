"""Ordinary trainer relay preserves full state without touching real runs."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from weiqi.fileio import acquire_file_lock, release_file_lock
from weiqi.rl_config import load_rl_training_config, resolve_rl_training_config
from weiqi.rl_handoff import (
    HandoffError, cancel_handoff, check_import, export_handoff, import_handoff, verify_handoff,
)
from weiqi.rl_handoff_guard import ensure_training_available, HandoffLockedError


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.control import RunLock
    from weiqi.rl.runner import Trainer
    from weiqi.rl.state import Position
    from weiqi.rl.storage import cpu_state, fingerprint, load_checkpoint


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_json(path: Path, value: object) -> None:
    write_bytes(path, json.dumps(value, indent=2).encode("utf-8"))


def archive_with_changes(original: Path, destination: Path,
                         changes: dict[str, bytes]) -> None:
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(destination, "w") as target:
        for info in source.infolist():
            target.writestr(info, changes.get(info.filename, source.read(info)))


def small_config():
    return resolve_rl_training_config(overrides={
        "hardware": {"device": "cpu", "precision": "float32", "data_loader_workers": 0,
                     "compile_model": False},
        "network": {"channels": 8, "residual_blocks": 1},
        "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0,
                   "full_search_probability": 1.0},
        "self_play": {"games_per_iteration": 2, "milestone_fraction": 0.0},
        "optimizer": {"batch_size": 4, "minimum_replay_size": 1,
                      "training_steps_per_iteration": 2,
                      "use_board_symmetry_augmentation": False, "target_sample_reuse": None},
        "runtime": {"pause_while_game_is_active": False},
    })


class NativePackagingTests(unittest.TestCase):
    """Exercise transport/ownership without requiring an optional GPU runtime.

    Only source checkpoint inspection is replaced with an already inspected
    source plan. The public implementation still packages, hashes, imports,
    protects the old run, and validates its restored configuration.
    """

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "computer-a"
        self.target = self.root / "computer-b"
        self.run = self.source / "training_runs" / "ordinary"
        self.bundle = self.root / "relay.gogogo-handoff.zip"
        self.destination = self.target / "training_runs" / "received"
        for project in (self.source, self.target):
            write_bytes(project / "weiqi" / "identity.py", b"VERSION = 1\n")
            write_bytes(project / "train.py", b"# ordinary training entry\n")
            write_bytes(project / "train_rl.py", b"# KataGo training entry\n")
        self.config = small_config()
        raw = self.config.to_dict()
        document = {"schema_version": raw["schema_version"], "preset": raw["preset"],
                    "overrides": {key: value for key, value in raw.items()
                                  if key not in ("schema_version", "preset")}}
        document["overrides"]["runtime"]["output_directory"] = str(self.run)
        self.plan = {
            "trainer": "native", "kind": "single",
            "sources": [{"root": self.run, "prefix": "run", "board_size": 9,
                         "lock": "run.lock"}],
            "configs": {"config/training.json": document}, "active_prefix": "run",
        }
        for name, content in {
            "latest.pt": b"synthetic full training state\x00\xff",
            "candidate.pt": b"candidate model", "best.pt": b"accepted model",
            "checkpoints/iteration_000003.pt": b"archived full training state",
            "anchors/accepted_000002.pt": b"historical opponent",
            "anchors/index.json": b'{"iteration":2}',
            "games/iteration_000003/game_0000.sgf": b"(;GM[1]SZ[9]RE[B+R])",
            "events.jsonl": b'{"event":"checkpoint"}\n',
            "data/replay.bin": b"other accumulated data\x01",
            "config.resolved.json": json.dumps(raw).encode(),
        }.items():
            write_bytes(self.run / name, content)

    def export(self, bundle: Path | None = None) -> dict:
        with patch("weiqi.rl_native_handoff.native_source_plan", return_value=deepcopy(self.plan)):
            return export_handoff(self.source, bundle or self.bundle, native_run=self.run)

    def export_katago(self, bundle: Path | None = None) -> dict:
        run = self.source / "training_runs/katago"
        write_bytes(run / "train/gogogo/checkpoint.ckpt", b"synthetic KataGo full state")
        profile = self.source / "config/katago.json"
        document = deepcopy(self.plan["configs"]["config/training.json"])
        document["overrides"]["runtime"]["output_directory"] = str(run)
        write_json(profile, document)
        return export_handoff(self.source, bundle or self.bundle, training_config=profile)

    def import_bundle(self, **kwargs: object) -> dict:
        return import_handoff(self.target, self.bundle, self.destination,
                              expected_trainer="native", **kwargs)

    def test_snapshot_preserves_all_existing_model_data_bytes_and_explicit_backend(self) -> None:
        original = {path.relative_to(self.run): path.read_bytes()
                    for path in self.run.rglob("*") if path.is_file()}
        report = self.export()
        self.assertEqual(report["trainer"], "native")
        manifest = verify_handoff(self.bundle, expected_trainer="native")
        self.assertEqual(manifest["kind"], "single")
        self.assertEqual(manifest["trainer"], "native")
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertIn("run/latest.pt", archive.namelist())
            self.assertNotIn("run/run.lock", archive.namelist())
            for entry in manifest["files"]:
                self.assertEqual(hashlib.sha256(archive.read(entry["path"])).hexdigest(),
                                 entry["sha256"])
        imported = self.import_bundle()
        self.assertEqual(Path(imported["resume_checkpoint"]), self.destination / "run/latest.pt")
        for relative, content in original.items():
            with self.subTest(file=str(relative)):
                self.assertEqual((self.run / relative).read_bytes(), content)
                if relative.as_posix() not in {"config.resolved.json", "monitor.json"}:
                    self.assertEqual((self.destination / "run" / relative).read_bytes(), content)
        resolved = load_rl_training_config(self.destination / "config/training.json")
        self.assertEqual(Path(resolved.runtime.output_directory), self.destination / "run")
        self.assertFalse((self.destination / "run/HANDOFF.json").exists())
        checked = check_import(self.destination, expected_trainer="native")
        self.assertTrue(checked["verified"])
        self.assertFalse(checked["checkpoint_loaded"])

    def test_stop_and_pause_controls_stay_on_source_and_do_not_block_target(self) -> None:
        for name in ("STOP", "PAUSE"):
            write_bytes(self.run / name, b"old computer control")
        self.export()
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertNotIn("run/STOP", archive.namelist())
            self.assertNotIn("run/PAUSE", archive.namelist())
        self.import_bundle()
        for name in ("STOP", "PAUSE"):
            self.assertTrue((self.run / name).is_file())
            self.assertFalse((self.destination / "run" / name).exists())
        ensure_training_available(self.destination / "run")

    def test_old_source_is_frozen_until_explicit_cancel(self) -> None:
        self.export()
        with self.assertRaises(HandoffLockedError):
            ensure_training_available(self.run)
        with patch("weiqi.rl_native_handoff.native_source_plan", return_value=deepcopy(self.plan)):
            report = cancel_handoff(self.source, native_run=self.run)
        self.assertTrue(report["source_resumed"])
        ensure_training_available(self.run)
        self.assertTrue(self.bundle.is_file())

    def test_active_native_os_lock_rejects_export_and_does_not_freeze(self) -> None:
        handle = acquire_file_lock(self.run / "run.lock")
        try:
            with self.assertRaises(HandoffError):
                self.export()
        finally:
            release_file_lock(handle)
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.run / "HANDOFF.json").exists())

    def test_existing_target_models_are_never_overwritten(self) -> None:
        self.export()
        existing = self.destination / "run/latest.pt"
        write_bytes(existing, b"pre-existing trained accomplishment")
        with self.assertRaises(HandoffError):
            self.import_bundle()
        self.assertEqual(existing.read_bytes(), b"pre-existing trained accomplishment")

    def test_corrupted_checkpoint_fails_before_import_publication(self) -> None:
        self.export()
        corrupted = self.root / "corrupted.zip"
        archive_with_changes(self.bundle, corrupted, {"run/latest.pt": b"damaged checkpoint"})
        with self.assertRaises(HandoffError):
            verify_handoff(corrupted, expected_trainer="native")
        with self.assertRaises(HandoffError):
            import_handoff(self.target, corrupted, self.destination, expected_trainer="native")
        self.assertFalse(self.destination.exists())

    def test_wrong_backend_is_rejected_by_verify_import_and_check(self) -> None:
        self.export()
        with self.assertRaises(HandoffError):
            verify_handoff(self.bundle, expected_trainer="katago")
        with self.assertRaises(HandoffError):
            import_handoff(self.target, self.bundle, self.destination, expected_trainer="katago")
        self.assertFalse(self.destination.exists())
        self.import_bundle()
        with self.assertRaises(HandoffError):
            check_import(self.destination, expected_trainer="katago")

    def test_backend_manifest_tampering_cannot_relabel_native_payload_as_katago(self) -> None:
        self.export()
        with zipfile.ZipFile(self.bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        manifest["trainer"] = "katago"
        altered = self.root / "mislabeled.zip"
        archive_with_changes(self.bundle, altered, {"manifest.json": json.dumps(manifest).encode()})
        with self.assertRaises(HandoffError):
            verify_handoff(altered)

    def test_legacy_katago_manifest_is_supported_and_rejected_by_native_entry(self) -> None:
        self.export_katago()
        with zipfile.ZipFile(self.bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        manifest.pop("trainer", None)
        legacy = self.root / "legacy.zip"
        archive_with_changes(self.bundle, legacy, {"manifest.json": json.dumps(manifest).encode()})
        checked = verify_handoff(legacy, expected_trainer="katago")
        self.assertTrue(checked["verified"])
        with self.assertRaises(HandoffError):
            verify_handoff(legacy, expected_trainer="native")
        with self.assertRaises(HandoffError):
            import_handoff(self.target, legacy, self.destination, expected_trainer="native")
        import_handoff(self.target, legacy, self.destination, expected_trainer="katago")
        check_import(self.destination, expected_trainer="katago")

    def test_new_hardware_profile_changes_runtime_config_without_rewriting_checkpoint(self) -> None:
        self.export()
        original = (self.run / "latest.pt").read_bytes()
        hardware = self.root / "computer-b-hardware.json"
        changes = {"hardware": {"device": "cpu", "precision": "float32"},
                   "self_play": {"workers": 1, "inference_batch_size": 16},
                   "optimizer": {"batch_size": 2}}
        write_json(hardware, changes)
        report = self.import_bundle(hardware_profile=hardware)
        self.assertEqual((self.destination / "run/latest.pt").read_bytes(), original)
        self.assertEqual((self.run / "latest.pt").read_bytes(), original)
        profile = Path(report["hardware_profile"])
        self.assertEqual(profile.parent, self.destination / "config")
        self.assertEqual(json.loads(profile.read_text(encoding="utf-8")), changes)
        config = load_rl_training_config(Path(report["training_config"]))
        self.assertEqual(config.optimizer.batch_size, 2)
        self.assertEqual(config.self_play.workers, 1)
        check_import(self.destination, expected_trainer="native")

    def test_hardware_profile_cannot_silently_change_training_algorithm(self) -> None:
        self.export()
        hardware = self.root / "invalid-hardware.json"
        write_json(hardware, {"optimizer": {"learning_rate": 0.5}})
        with self.assertRaises(HandoffError):
            self.import_bundle(hardware_profile=hardware)
        self.assertFalse(self.destination.exists())

    def test_export_on_filesystem_without_hardlinks_still_publishes_verified_bundle(self) -> None:
        for trainer, export in (("native", self.export), ("katago", self.export_katago)):
            with self.subTest(trainer=trainer):
                bundle = self.root / f"no-hardlink-{trainer}.zip"
                with patch("weiqi.rl_handoff.os.link", side_effect=OSError(errno.EOPNOTSUPP, "no hardlinks")):
                    report = export(bundle)
                self.assertTrue(report["source_suspended"])
                self.assertTrue(verify_handoff(bundle, expected_trainer=trainer)["verified"])
                run = self.run if trainer == "native" else self.source / "training_runs/katago"
                with self.assertRaises(HandoffLockedError):
                    ensure_training_available(run)
                self.assertFalse(list(bundle.parent.glob(f".{bundle.name}.*.tmp")))

    def test_export_copy_failure_removes_partial_bundle_and_releases_source_hold(self) -> None:
        def partial_copy(source, destination, _chunk):
            destination.write(source.read(19))
            raise OSError(errno.ENOSPC, "simulated full destination disk")

        for trainer, export in (("native", self.export), ("katago", self.export_katago)):
            with self.subTest(trainer=trainer):
                bundle = self.root / f"failed-copy-{trainer}.zip"
                run = self.run if trainer == "native" else self.source / "training_runs/katago"
                if trainer == "katago":
                    write_bytes(run / "train/gogogo/checkpoint.ckpt", b"synthetic KataGo full state")
                original = {path.relative_to(run): path.read_bytes()
                            for path in run.rglob("*") if path.is_file()}
                with patch("weiqi.rl_handoff.os.link", side_effect=OSError(errno.EOPNOTSUPP, "no hardlinks")):
                    with patch("weiqi.rl_handoff.shutil.copyfileobj", side_effect=partial_copy):
                        with self.assertRaises((HandoffError, OSError)):
                            export(bundle)
                self.assertFalse(bundle.exists())
                self.assertFalse((run / "HANDOFF.json").exists())
                self.assertFalse(list(bundle.parent.glob(f".{bundle.name}.*.tmp")))
                ensure_training_available(run)
                for relative, content in original.items():
                    self.assertEqual((run / relative).read_bytes(), content)


@unittest.skipUnless(AVAILABLE, "Optional training dependencies PyTorch/NumPy are not installed")
class NativeRealCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.run = self.project / "training_runs" / "ordinary"
        self.run.mkdir(parents=True)
        self.bundle = self.root / "relay.gogogo-handoff.zip"
        self.destination = self.root / "computer-b" / "received"
        self.config = replace(small_config(), runtime=replace(
            small_config().runtime, output_directory=str(self.run)))
        with redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config, self.run)
            state = Position.new(9, 6.5)
            policy = state.legal.astype(np.float32)
            policy /= policy.sum()
            trainer.replay.extend([(state.features(), policy.copy(), float(index % 2 * 2 - 1))
                                   for index in range(16)])
            trainer.iteration = 3
            trainer.untrained_samples = 16
            self.assertTrue(trainer.train_updates()["weights_updated"])
            trainer.save()
        self.original_checkpoint = (self.run / "latest.pt").read_bytes()

    def transfer(self) -> dict:
        original_rng = torch.get_rng_state().clone()
        export_handoff(self.project, self.bundle, native_run=self.run)
        torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        return import_handoff(self.project, self.bundle, self.destination,
                              expected_trainer="native")

    def assert_same_state(self, left: object, right: object) -> None:
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_same_state(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_same_state(a, b)
        else:
            self.assertEqual(left, right)

    def test_full_training_checkpoint_restores_optimizer_replay_rng_and_identical_next_update(self) -> None:
        self.transfer()
        imported_path = self.destination / "run/latest.pt"
        self.assertEqual(imported_path.read_bytes(), self.original_checkpoint)
        self.assertEqual((self.run / "latest.pt").read_bytes(), self.original_checkpoint)
        report = check_import(self.destination, load_checkpoint=True, expected_trainer="native")
        self.assertTrue(report["checkpoint_loaded"])
        payload, source_config = load_checkpoint(self.run / "latest.pt")
        restored_payload, restored_config = load_checkpoint(imported_path)
        self.assert_same_state(payload, restored_payload)
        self.assertEqual(source_config, restored_config)
        self.assertEqual(payload["training_steps"], 2)
        self.assertEqual(len(payload["replay"]["values"]), 16)
        with redirect_stdout(io.StringIO()):
            source_copy = self.root / "isolated-a"
            source_copy.mkdir()
            expected = Trainer(source_config, source_copy, payload)
            expected.train_updates()
            expected_weights = fingerprint(cpu_state(expected.model))
            expected_optimizer = deepcopy(expected.optimizer.state_dict())
            restored = Trainer(restored_config, self.destination / "run", restored_payload)
            restored.train_updates()
        self.assertEqual(restored.training_steps, 4)
        self.assertEqual(fingerprint(cpu_state(restored.model)), expected_weights)
        self.assert_same_state(restored.optimizer.state_dict(), expected_optimizer)
        self.assert_same_state(restored.rng.bit_generator.state, expected.rng.bit_generator.state)

    def test_actual_run_lock_refuses_a_frozen_source_and_cancel_restores_it(self) -> None:
        self.transfer()
        with self.assertRaises((HandoffLockedError, ValueError)):
            with RunLock(self.run):
                self.fail("A transferred ordinary run must not start another writer")
        cancel_handoff(self.project, native_run=self.run)
        with RunLock(self.run):
            pass
        self.assertEqual((self.run / "latest.pt").read_bytes(), self.original_checkpoint)

    def test_real_running_trainer_lock_prevents_export(self) -> None:
        with RunLock(self.run):
            with self.assertRaises(HandoffError):
                export_handoff(self.project, self.bundle, native_run=self.run)
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.run / "HANDOFF.json").exists())

    def test_changed_batch_uses_new_runtime_settings_and_preserves_saved_optimizer(self) -> None:
        from weiqi.rl_native_handoff import apply_native_hardware

        export_handoff(self.project, self.bundle, native_run=self.run)
        hardware = self.root / "smaller-machine.json"
        write_json(hardware, {"hardware": {"device": "cpu", "precision": "float32"},
                              "optimizer": {"batch_size": 2}, "self_play": {"workers": 1}})
        report = import_handoff(self.project, self.bundle, self.destination,
                                hardware_profile=hardware, expected_trainer="native")
        check_import(self.destination, load_checkpoint=True, expected_trainer="native")
        path = self.destination / "run/latest.pt"
        self.assertEqual(path.read_bytes(), self.original_checkpoint)
        payload, saved_config = load_checkpoint(path)
        self.assertEqual(saved_config.optimizer.batch_size, 4)
        runtime_config = apply_native_hardware(saved_config, Path(report["hardware_profile"]))
        self.assertEqual(runtime_config.optimizer.batch_size, 2)
        self.assertEqual(runtime_config.self_play.workers, 1)
        with redirect_stdout(io.StringIO()):
            resumed = Trainer(runtime_config, self.destination / "run", payload)
            self.assert_same_state(resumed.optimizer.state_dict(), payload["optimizer"])
            self.assertTrue(resumed.train_updates()["weights_updated"])
        self.assertEqual(resumed.training_steps, payload["training_steps"] + 2)

    def test_candidate_export_cannot_substitute_for_full_latest_checkpoint(self) -> None:
        (self.run / "latest.pt").write_bytes((self.run / "candidate.pt").read_bytes())
        with self.assertRaises(HandoffError):
            export_handoff(self.project, self.bundle, native_run=self.run)
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.run / "HANDOFF.json").exists())

    def test_opaque_latest_pt_fails_export_without_modifying_existing_artifacts(self) -> None:
        opaque = b"not a PyTorch checkpoint\x00\xff"
        (self.run / "latest.pt").write_bytes(opaque)
        with self.assertRaises(HandoffError):
            export_handoff(self.project, self.bundle, native_run=self.run)
        self.assertEqual((self.run / "latest.pt").read_bytes(), opaque)
        self.assertFalse(self.bundle.exists())
        self.assertFalse((self.run / "HANDOFF.json").exists())


if __name__ == "__main__":
    unittest.main()
