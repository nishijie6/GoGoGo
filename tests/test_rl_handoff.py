"""Synthetic KataGo handoff tests; never touch an installed training run."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from weiqi import rl_handoff
from weiqi.fileio import acquire_file_lock, release_file_lock
from weiqi.rl_config import load_rl_training_config
from weiqi.rl_curriculum_config import load_curriculum_config
from weiqi.rl_curriculum_state import CurriculumState, StageProgress
from weiqi.rl_handoff import (
    HandoffError,
    cancel_handoff,
    check_import,
    export_handoff,
    import_handoff,
    verify_handoff,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_PROFILE = PROJECT_ROOT / "config" / "rl_curriculum.rtx5070ti.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def rewrite_archive(source: Path, target: Path, *, replace: dict[str, bytes] | None = None,
                    append: list[tuple[str, bytes]] | None = None) -> None:
    """Produce malicious/corrupted fixtures without using extractall."""
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as edited:
        for info in original.infolist():
            edited.writestr(info, (replace or {}).get(info.filename, original.read(info)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content in append or []:
                edited.writestr(name, content)


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "computer-a"
        self.target = self.root / "computer-b"
        self.bundle = self.root / "handoff.zip"
        self.destination = self.target / "training_runs" / "received"
        for project in (self.source, self.target):
            write_bytes(project / "weiqi" / "trainer_identity.py", b"VERSION = 1\n")
            write_bytes(project / "train_rl.py", b"# synthetic training entry point\n")
        curriculum = json.loads(CURRICULUM_PROFILE.read_text(encoding="utf-8"))
        self.config_path = self.source / "config" / "curriculum.json"
        self.profiles: dict[int, Path] = {}
        for stage in curriculum["stages"]:
            board = stage["board_size"]
            profile = json.loads((PROJECT_ROOT / stage["training_config"]).read_text(encoding="utf-8"))
            profile["overrides"]["runtime"]["output_directory"] = f"training_runs/{board}x{board}"
            profile_path = self.source / "config" / f"{board}x{board}.json"
            write_json(profile_path, profile)
            stage["training_config"] = str(profile_path.relative_to(self.source))
            self.profiles[board] = profile_path
        write_json(self.config_path, curriculum)
        self.state_root = self.source / "training_runs" / "curriculum"
        self.stage_root = self.source / "training_runs" / "9x9"
        initial_stage = curriculum["stages"][0]
        self.champion = initial_stage["initial_champion_model"]
        self.baselines = initial_stage["fixed_baseline_models"]
        progress = StageProgress(
            board_size=9,
            samples=32_000_000,
            entry_model=self.baselines[0],
            baseline_model=self.baselines[0],
            latest_model=self.champion,
            champion_model=self.champion,
            champion_established=True,
            fixed_baseline_models=list(self.baselines),
            next_evaluation_sample=32_500_000,
            autotune_batch=1024,
        )
        state = CurriculumState(
            schema_version=1,
            active_stage_index=0,
            phase="training",
            stages={"9x9": progress},
            protected_models=[self.champion, *self.baselines],
        )
        write_json(self.state_root / "state.json", state.to_dict())
        self.populate_stage(self.stage_root, [self.champion, *self.baselines])
        write_bytes(self.state_root / "evaluations" / "accepted.json", b'{"games": 200}\n')

    def populate_stage(self, stage: Path, names: list[str]) -> None:
        write_bytes(stage / "train" / "gogogo" / "checkpoint.ckpt", b"full optimizer and train state\x00\xff")
        for name in names:
            write_bytes(stage / "models" / name / "model.bin.gz", f"model {name}".encode())
            write_bytes(stage / "torchmodels_toexport" / name / "model.ckpt", f"export {name}".encode())
        write_bytes(stage / "selfplay" / "games" / "tdata" / "part.npz", b"synthetic replay rows\x00\x01")
        write_bytes(stage / "selfplay" / "games" / "sgfs" / "part.sgfs", b"(;GM[1]SZ[9]RE[B+R])\n")
        write_bytes(stage / "shuffleddata" / "batch" / "train" / "part.npz", b"synthetic shuffled rows")
        write_json(stage / "replay_rows.json", {"schema_version": 1, "deleted_rows": 1234})
        os.utime(stage / "selfplay" / "games" / "tdata" / "part.npz",
                 ns=(1_710_000_000_123_456_700, 1_710_000_000_123_456_700))

    def export(self) -> dict:
        return export_handoff(self.source, self.bundle, curriculum_config=self.config_path)

    def import_bundle(self, **kwargs: object) -> dict:
        return import_handoff(self.target, self.bundle, self.destination, **kwargs)

    def assert_source_unmarked(self) -> None:
        self.assertFalse((self.state_root / "HANDOFF.json").exists())
        self.assertFalse((self.stage_root / "HANDOFF.json").exists())

    def altered_manifest(self, destination: Path, changes: dict[str, object]) -> Path:
        with zipfile.ZipFile(self.bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        manifest.update(changes)
        rewrite_archive(self.bundle, destination, replace={
            "manifest.json": json.dumps(manifest).encode("utf-8"),
        })
        return destination

    def test_curriculum_round_trip_preserves_resume_models_data_and_progress(self) -> None:
        source_files = {
            path.relative_to(self.stage_root): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.stage_root.rglob("*") if path.is_file()
        }
        original_state = (self.state_root / "state.json").read_bytes()
        report = self.export()
        self.assertEqual(report["kind"], "curriculum")
        self.assertTrue(self.bundle.is_file())
        verified = verify_handoff(self.bundle)
        self.assertIsInstance(verified, dict)
        with zipfile.ZipFile(self.bundle) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(manifest["kind"], "curriculum")
            self.assertIn("stages/9x9/train/gogogo/checkpoint.ckpt", archive.namelist())
            self.assertIn("curriculum/state.json", archive.namelist())
            for entry in manifest["files"]:
                content = archive.read(entry["path"])
                self.assertEqual(entry["size"], len(content))
                self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
        self.import_bundle()
        imported_stage = self.destination / "stages" / "9x9"
        for relative, (content, modified) in source_files.items():
            with self.subTest(path=str(relative)):
                self.assertEqual((imported_stage / relative).read_bytes(), content)
                self.assertEqual((imported_stage / relative).stat().st_mtime_ns, modified)
                self.assertEqual((self.stage_root / relative).read_bytes(), content)
        self.assertEqual((self.destination / "curriculum" / "state.json").read_bytes(), original_state)
        self.assertTrue((self.destination / "handoff_receipt.json").is_file())
        self.assertFalse((imported_stage / "HANDOFF.json").exists())
        curriculum = load_curriculum_config(self.destination / "config" / "curriculum.json")
        self.assertEqual(curriculum.state_root(self.target), self.destination / "curriculum")
        for stage in curriculum.stages:
            self.assertEqual(stage.config_path(self.target), self.destination / "config" / f"{stage.key}.json")
            profile = load_rl_training_config(stage.config_path(self.target))
            self.assertEqual(Path(profile.runtime.output_directory), self.destination / "stages" / stage.key)
        self.assertIsInstance(check_import(self.destination), dict)

    def test_single_run_round_trip_uses_full_resume_checkpoint(self) -> None:
        export_handoff(self.source, self.bundle, training_config=self.profiles[9])
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["kind"], "single")
            self.assertIn("run/train/gogogo/checkpoint.ckpt", archive.namelist())
        self.import_bundle()
        self.assertEqual(
            (self.destination / "run" / "train" / "gogogo" / "checkpoint.ckpt").read_bytes(),
            (self.stage_root / "train" / "gogogo" / "checkpoint.ckpt").read_bytes(),
        )
        config = load_rl_training_config(self.destination / "config" / "training.json")
        self.assertEqual(Path(config.runtime.output_directory), self.destination / "run")
        self.assertIsInstance(check_import(self.destination), dict)

    def test_existing_destination_is_never_overwritten(self) -> None:
        self.export()
        write_bytes(self.destination / "keep-model.ckpt", b"pre-existing accomplishment")
        with self.assertRaises(HandoffError):
            self.import_bundle()
        self.assertEqual((self.destination / "keep-model.ckpt").read_bytes(), b"pre-existing accomplishment")

    def test_concurrently_created_empty_destination_is_not_replaced(self) -> None:
        self.export()
        publish = rl_handoff._publish_directory
        created_identity = []

        def concurrent_publish(source: Path, destination: Path) -> None:
            destination.mkdir()
            created_identity.append(destination.stat().st_ino)
            publish(source, destination)

        with patch.object(rl_handoff, "_publish_directory", side_effect=concurrent_publish):
            with self.assertRaises(HandoffError):
                self.import_bundle()
        self.assertTrue(self.destination.is_dir())
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertEqual(self.destination.stat().st_ino, created_identity[0])
        self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.handoff-*.tmp")), [])

    def test_unsupported_native_rename_falls_back_to_complete_verified_import(self) -> None:
        self.export()
        with patch.object(rl_handoff, "_native_publish_directory", side_effect=OSError(errno.EINVAL, "unsupported filesystem")):
            self.import_bundle()
        self.assertFalse((self.destination / "HANDOFF_IMPORTING.json").exists())
        self.assertEqual(
            (self.destination / "stages" / "9x9" / "train" / "gogogo" / "checkpoint.ckpt").read_bytes(),
            (self.stage_root / "train" / "gogogo" / "checkpoint.ckpt").read_bytes(),
        )
        source_data = self.stage_root / "selfplay" / "games" / "tdata" / "part.npz"
        restored_data = self.destination / "stages" / "9x9" / "selfplay" / "games" / "tdata" / "part.npz"
        self.assertEqual(restored_data.read_bytes(), source_data.read_bytes())
        self.assertEqual(restored_data.stat().st_mtime_ns, source_data.stat().st_mtime_ns)
        self.assertIsInstance(check_import(self.destination), dict)
        self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.handoff-*.tmp")), [])

    def test_fallback_refuses_destination_created_after_native_rename_attempt(self) -> None:
        self.export()
        existing_file = self.destination / "keep.ckpt"

        def concurrent_unsupported_rename(_source: Path, destination: Path) -> None:
            destination.mkdir()
            existing_file.write_bytes(b"existing accomplishment")
            raise OSError(errno.EINVAL, "unsupported filesystem")

        with patch.object(rl_handoff, "_native_publish_directory", side_effect=concurrent_unsupported_rename):
            with self.assertRaises(HandoffError):
                self.import_bundle()
        self.assertEqual(existing_file.read_bytes(), b"existing accomplishment")
        self.assertEqual(list(self.destination.iterdir()), [existing_file])
        self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.handoff-*.tmp")), [])

    def test_fallback_failure_cleans_only_its_partial_publication(self) -> None:
        self.export()
        original = rl_handoff._publish_file
        published = []

        def failing_file_publish(source: Path, target: Path) -> None:
            if len(published) == 2:
                def partial_copy(input_stream, output_stream, _length) -> None:
                    output_stream.write(input_stream.read(4))
                    raise OSError(errno.EIO, "simulated storage failure after partial write")

                with patch.object(rl_handoff.os, "link", side_effect=OSError(errno.EXDEV, "no hard links")):
                    with patch.object(rl_handoff.shutil, "copyfileobj", side_effect=partial_copy):
                        original(source, target)
                return
            original(source, target)
            published.append(target)

        with patch.object(rl_handoff, "_native_publish_directory", side_effect=OSError(errno.EINVAL, "unsupported filesystem")):
            with patch.object(rl_handoff, "_publish_file", side_effect=failing_file_publish):
                with self.assertRaises(HandoffError):
                    self.import_bundle()
        self.assertEqual(len(published), 2)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.handoff-*.tmp")), [])
        self.assertTrue((self.stage_root / "train" / "gogogo" / "checkpoint.ckpt").is_file())

    def test_fallback_failure_preserves_external_file_and_pending_marker(self) -> None:
        self.export()
        original = rl_handoff._publish_file
        published = []
        external = self.destination / "external-accomplishment.ckpt"

        def failing_with_external_file(source: Path, target: Path) -> None:
            if len(published) == 2:
                external.write_bytes(b"created by another process")
                raise OSError(errno.EIO, "simulated storage failure")
            original(source, target)
            published.append(target)

        with patch.object(rl_handoff, "_native_publish_directory", side_effect=OSError(errno.EINVAL, "unsupported filesystem")):
            with patch.object(rl_handoff, "_publish_file", side_effect=failing_with_external_file):
                with self.assertRaises(HandoffError):
                    self.import_bundle()
        self.assertEqual(external.read_bytes(), b"created by another process")
        self.assertTrue((self.destination / "HANDOFF_IMPORTING.json").is_file())
        self.assertFalse((self.destination / "stages" / "9x9" / "train" / "gogogo" / "checkpoint.ckpt").exists())
        for path in published:
            self.assertFalse(path.exists())
        with self.assertRaises(HandoffError):
            check_import(self.destination)
        self.assertEqual(list(self.destination.parent.glob(f".{self.destination.name}.handoff-*.tmp")), [])

    def test_existing_export_file_is_never_overwritten_or_source_marked(self) -> None:
        self.bundle.write_bytes(b"previous verified handoff")
        with self.assertRaises(HandoffError):
            self.export()
        self.assertEqual(self.bundle.read_bytes(), b"previous verified handoff")
        self.assert_source_unmarked()

    def test_corrupted_payload_rejects_before_publishing(self) -> None:
        self.export()
        bad = self.root / "corrupt.zip"
        rewrite_archive(self.bundle, bad, replace={"stages/9x9/train/gogogo/checkpoint.ckpt": b"broken"})
        with self.assertRaises(HandoffError):
            verify_handoff(bad)
        existing = set(self.target.rglob("*"))
        with self.assertRaises(HandoffError):
            import_handoff(self.target, bad, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(set(self.target.rglob("*")), existing)

    def test_traversal_and_windows_absolute_paths_reject_before_extraction(self) -> None:
        self.export()
        for index, unsafe in enumerate(("../escape.ckpt", "/escape.ckpt", "C:/escape.ckpt", "..\\escape.ckpt")):
            with self.subTest(path=unsafe):
                bad = self.root / f"unsafe-{index}.zip"
                rewrite_archive(self.bundle, bad, append=[(unsafe, b"outside")])
                with self.assertRaises(HandoffError):
                    verify_handoff(bad)
                with self.assertRaises(HandoffError):
                    import_handoff(self.target, bad, self.destination)
                self.assertFalse(self.destination.exists())
        self.assertFalse((self.root / "escape.ckpt").exists())

    def test_duplicate_zip_member_rejects(self) -> None:
        self.export()
        bad = self.root / "duplicate.zip"
        replay = (self.stage_root / "replay_rows.json").read_bytes()
        rewrite_archive(self.bundle, bad, append=[("stages/9x9/replay_rows.json", replay)])
        with self.assertRaises(HandoffError):
            verify_handoff(bad)
        with self.assertRaises(HandoffError):
            import_handoff(self.target, bad, self.destination)
        self.assertFalse(self.destination.exists())

    def test_held_curriculum_or_stage_lock_refuses_export(self) -> None:
        for lock_path in (self.state_root / "curriculum.lock", self.stage_root / "training.lock"):
            with self.subTest(lock=str(lock_path)):
                handle = acquire_file_lock(lock_path)
                try:
                    with self.assertRaises(HandoffError):
                        self.export()
                finally:
                    release_file_lock(handle)
                self.assertFalse(self.bundle.exists())
                self.assert_source_unmarked()

    def test_export_checkpoint_is_not_a_substitute_for_resume_checkpoint(self) -> None:
        (self.stage_root / "train" / "gogogo" / "checkpoint.ckpt").unlink()
        self.assertTrue((self.stage_root / "torchmodels_toexport" / self.champion / "model.ckpt").is_file())
        with self.assertRaises(HandoffError):
            self.export()
        self.assertFalse(self.bundle.exists())
        self.assert_source_unmarked()

    def test_missing_protected_model_or_export_checkpoint_refuses_export(self) -> None:
        for parent, filename in (("models", "model.bin.gz"), ("torchmodels_toexport", "model.ckpt")):
            with self.subTest(parent=parent):
                path = self.stage_root / parent / self.champion / filename
                original = path.read_bytes()
                path.unlink()
                with self.assertRaises(HandoffError):
                    self.export()
                write_bytes(path, original)
                self.assertFalse(self.bundle.exists())
                self.assert_source_unmarked()

    def test_global_protected_model_not_in_stage_references_must_exist(self) -> None:
        path = self.state_root / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["protected_models"].append("gogogo-historical-protected")
        write_json(path, state)
        with self.assertRaises(HandoffError):
            self.export()
        self.assertFalse(self.bundle.exists())
        self.assert_source_unmarked()

    def test_in_progress_evaluation_and_migration_refuse_export(self) -> None:
        path = self.state_root / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        for phase in ("evaluating", "migrating", "transition_ready", "paused_disk"):
            with self.subTest(phase=phase):
                state["phase"] = phase
                state["resume_phase"] = "evaluating" if phase == "paused_disk" else None
                write_json(path, state)
                with self.assertRaises(HandoffError):
                    self.export()
                self.assertFalse(self.bundle.exists())
                self.assert_source_unmarked()

    def test_disk_pause_can_export_only_when_resume_phase_is_training(self) -> None:
        path = self.state_root / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(phase="paused_disk", resume_phase="training")
        write_json(path, state)
        self.export()
        self.import_bundle()
        self.assertEqual(json.loads((self.destination / "curriculum" / "state.json").read_text()), state)

    def test_uninitialized_curriculum_cannot_export(self) -> None:
        (self.state_root / "state.json").unlink()
        with self.assertRaises(HandoffError):
            self.export()
        self.assert_source_unmarked()

    def test_temporary_and_scratch_files_are_excluded_without_source_deletion(self) -> None:
        excluded = (
            self.stage_root / "train" / "gogogo" / ".checkpoint.ckpt.1.tmp",
            self.stage_root / ".migration-staging.tmp" / "checkpoint.ckpt",
            self.stage_root / "scratch" / "smoke" / "result.txt",
        )
        for path in excluded:
            write_bytes(path, b"incomplete or disposable")
        self.export()
        with zipfile.ZipFile(self.bundle) as archive:
            names = archive.namelist()
            for path in excluded:
                self.assertNotIn("stages/9x9/" + path.relative_to(self.stage_root).as_posix(), names)
                self.assertTrue(path.is_file())
            self.assertFalse(any(name.endswith(".lock") or name.endswith("HANDOFF.json") for name in names))

    def test_cancel_removes_only_handoff_markers_and_preserves_training_data(self) -> None:
        state_before = (self.state_root / "state.json").read_bytes()
        checkpoint = self.stage_root / "train" / "gogogo" / "checkpoint.ckpt"
        checkpoint_before = checkpoint.read_bytes()
        self.export()
        self.assertTrue((self.state_root / "HANDOFF.json").is_file())
        self.assertTrue((self.stage_root / "HANDOFF.json").is_file())
        cancel_handoff(self.source, curriculum_config=self.config_path)
        self.assert_source_unmarked()
        self.assertEqual((self.state_root / "state.json").read_bytes(), state_before)
        self.assertEqual(checkpoint.read_bytes(), checkpoint_before)
        self.assertTrue(self.bundle.is_file())

    def test_cancel_under_active_stage_lock_keeps_markers(self) -> None:
        self.export()
        handle = acquire_file_lock(self.stage_root / "training.lock")
        try:
            with self.assertRaises(HandoffError):
                cancel_handoff(self.source, curriculum_config=self.config_path)
        finally:
            release_file_lock(handle)
        self.assertTrue((self.state_root / "HANDOFF.json").is_file())
        self.assertTrue((self.stage_root / "HANDOFF.json").is_file())

    def test_second_export_requires_cancel_of_existing_handoff(self) -> None:
        self.export()
        marker = self.state_root / "HANDOFF.json"
        marker_before = marker.read_bytes()
        second = self.root / "duplicate-handoff.zip"
        with self.assertRaises(HandoffError):
            export_handoff(self.source, second, curriculum_config=self.config_path)
        self.assertFalse(second.exists())
        self.assertEqual(marker.read_bytes(), marker_before)

    def test_different_training_code_requires_explicit_import_override(self) -> None:
        self.export()
        write_bytes(self.target / "weiqi" / "trainer_identity.py", b"VERSION = 2\n")
        with self.assertRaises(HandoffError):
            self.import_bundle()
        self.assertFalse(self.destination.exists())
        self.import_bundle(allow_version_mismatch=True)
        self.assertTrue((self.destination / "curriculum" / "state.json").is_file())

    def test_hardware_override_changes_batch_and_resets_stale_autotune(self) -> None:
        self.export()
        override = self.root / "smaller-gpu.json"
        write_json(override, {
            "hardware": {"device": "cuda:0", "precision": "amp_float16", "gpu_memory_fraction": 0.7},
            "optimizer": {"batch_size": 256},
            "self_play": {"workers": 2, "inference_batch_size": 32},
        })
        self.import_bundle(hardware_profile=override)
        profile = load_rl_training_config(self.destination / "config" / "9x9.json")
        self.assertEqual(profile.optimizer.batch_size, 256)
        self.assertEqual(profile.hardware.precision, "amp_float16")
        self.assertEqual(profile.self_play.workers, 2)
        state = json.loads((self.destination / "curriculum" / "state.json").read_text())
        self.assertIsNone(state["stages"]["9x9"]["autotune_batch"])
        self.assertEqual(state["stages"]["9x9"]["samples"], 32_000_000)
        self.assertEqual(state["stages"]["9x9"]["champion_model"], self.champion)
        self.assertIsInstance(check_import(self.destination), dict)

    def test_hardware_profile_cannot_change_game_or_network(self) -> None:
        self.export()
        for section, values in (("game", {"board_size": 9}), ("network", {"channels": 128})):
            with self.subTest(section=section):
                override = self.root / f"bad-{section}.json"
                write_json(override, {section: values})
                with self.assertRaises(HandoffError):
                    self.import_bundle(hardware_profile=override)
                self.assertFalse(self.destination.exists())

    def test_worker_only_hardware_override_preserves_selected_training_batch(self) -> None:
        for saved_batch in (1024, 512):
            with self.subTest(batch=saved_batch):
                state_path = self.state_root / "state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["stages"]["9x9"]["autotune_batch"] = saved_batch
                write_json(state_path, state)
                bundle = self.root / f"worker-only-{saved_batch}.zip"
                destination = self.target / "training_runs" / f"worker-only-{saved_batch}"
                export_handoff(self.source, bundle, curriculum_config=self.config_path)
                override = self.root / "worker-only.json"
                write_json(override, {"self_play": {"workers": 2, "inference_batch_size": 32}})
                import_handoff(self.target, bundle, destination, hardware_profile=override)
                restored = json.loads((destination / "curriculum" / "state.json").read_text())
                self.assertEqual(restored["stages"]["9x9"]["autotune_batch"], saved_batch)
                profile = load_rl_training_config(destination / "config" / "9x9.json")
                self.assertEqual(profile.self_play.workers, 2)
                self.assertEqual(profile.optimizer.batch_size, 1024)
                cancel_handoff(self.source, curriculum_config=self.config_path)

    def test_state_active_board_must_match_manifest_active_prefix(self) -> None:
        next_stage = self.source / "training_runs" / "13x13"
        self.populate_stage(next_stage, ["gogogo-s0-d0"])
        state_path = self.state_root / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update(active_stage_index=1, active_board_size=13)
        progress = StageProgress(
            board_size=13,
            entry_model="gogogo-s0-d0", baseline_model="gogogo-s0-d0",
            latest_model="gogogo-s0-d0", champion_model="gogogo-s0-d0",
            champion_established=True, fixed_baseline_models=["gogogo-s0-d0"],
        )
        state["stages"]["13x13"] = progress.__dict__
        state["protected_models"].append("gogogo-s0-d0")
        write_json(state_path, state)
        self.export()
        bad = self.altered_manifest(self.root / "wrong-active.zip", {"active_prefix": "stages/9x9"})
        with zipfile.ZipFile(bad) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            state_record = next(entry for entry in manifest["files"] if entry["path"] == "curriculum/state.json")
            self.assertEqual(state_record["sha256"], hashlib.sha256(archive.read("curriculum/state.json")).hexdigest())
        with self.assertRaises(HandoffError):
            import_handoff(self.target, bad, self.destination)
        self.assertFalse(self.destination.exists())

    def test_invalid_manifest_types_raise_public_handoff_error(self) -> None:
        self.export()
        with zipfile.ZipFile(self.bundle) as archive:
            original = json.loads(archive.read("manifest.json"))
        bad_sources = [{**entry} for entry in original["sources"]]
        bad_sources[0]["prefix"] = []
        for index, changes in enumerate((
            {"configs": [{}]},
            {"active_prefix": []},
            {"sources": bad_sources},
        )):
            with self.subTest(changes=changes):
                bad = self.altered_manifest(self.root / f"invalid-type-{index}.zip", changes)
                with self.assertRaises(HandoffError):
                    verify_handoff(bad)
                with self.assertRaises(HandoffError):
                    import_handoff(self.target, bad, self.destination)
                self.assertFalse(self.destination.exists())

    def test_windows_reserved_and_control_character_names_are_rejected(self) -> None:
        self.export()
        for index, basename in enumerate(("CON.ckpt", "COM1.txt", "trailing.", "trailing ",
                                          "control\x01.ckpt", "line\nfeed.ckpt", "tab\t.ckpt", "delete\x7f.ckpt")):
            with self.subTest(name=repr(basename)):
                bad = self.root / f"invalid-name-{index}.zip"
                name = "stages/9x9/" + basename
                content = b"invalid path payload"
                with zipfile.ZipFile(self.bundle) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                manifest["files"].append({
                    "path": name, "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(), "mtime_ns": 0,
                })
                rewrite_archive(self.bundle, bad, replace={"manifest.json": json.dumps(manifest).encode()},
                                append=[(name, content)])
                with self.assertRaises(HandoffError):
                    verify_handoff(bad)
                with self.assertRaises(HandoffError):
                    import_handoff(self.target, bad, self.destination)
                self.assertFalse(self.destination.exists())

    @unittest.skipUnless(shutil.which("git"), "Git is required for archive staging safeguards")
    def test_model_archive_inside_git_repo_must_be_ignored(self) -> None:
        subprocess.run(["git", "init", str(self.source)], check=True, capture_output=True)
        write_bytes(self.source / ".gitignore", b"training_runs/\nhandoff_bundles/\n")
        visible = self.source / "visible.zip"
        with self.assertRaises(HandoffError):
            export_handoff(self.source, visible, curriculum_config=self.config_path)
        self.assertFalse(visible.exists())
        self.assert_source_unmarked()
        ignored = self.source / "handoff_bundles" / "snapshot.zip"
        export_handoff(self.source, ignored, curriculum_config=self.config_path)
        self.assertTrue(ignored.is_file())
        check = subprocess.run(["git", "-C", str(self.source), "check-ignore", "--", str(ignored)],
                               capture_output=True, check=False)
        self.assertEqual(check.returncode, 0)

    def test_completed_stage_migration_metadata_paths_are_relocated(self) -> None:
        next_stage = self.source / "training_runs" / "13x13"
        self.populate_stage(next_stage, ["gogogo-s0-d0"])
        path = self.state_root / "state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["active_stage_index"] = 1
        state["active_board_size"] = 13
        state["stages"]["13x13"] = {
            **StageProgress(board_size=13).__dict__,
            "entry_model": "gogogo-s0-d0", "baseline_model": "gogogo-s0-d0",
            "latest_model": "gogogo-s0-d0", "champion_model": "gogogo-s0-d0",
            "champion_established": True, "fixed_baseline_models": ["gogogo-s0-d0"],
        }
        state["protected_models"].append("gogogo-s0-d0")
        write_json(path, state)
        source_checkpoint = self.stage_root / "torchmodels_toexport" / self.champion / "model.ckpt"
        migration = {
            "source_model": self.champion,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
            "target_board_size": 13,
            "seed_model": "gogogo-s0-d0",
            "selected_batch_size": 512,
            "seed_checkpoint_sha256": hashlib.sha256((next_stage / "train" / "gogogo" / "checkpoint.ckpt").read_bytes()).hexdigest(),
            "completed_at": "2026-10-12T12:00:00+08:00",
        }
        write_json(next_stage / "migration_manifest.json", migration)
        self.export()
        self.import_bundle()
        imported = json.loads((self.destination / "stages" / "13x13" / "migration_manifest.json").read_text())
        expected_checkpoint = self.destination / "stages" / "9x9" / "torchmodels_toexport" / self.champion / "model.ckpt"
        self.assertEqual(Path(imported["source_checkpoint"]), expected_checkpoint)
        self.assertEqual(imported["source_sha256"], migration["source_sha256"])
        self.assertIsInstance(check_import(self.destination), dict)

    def test_import_check_detects_post_import_model_or_config_tampering(self) -> None:
        self.export()
        self.import_bundle()
        for path in (
            self.destination / "stages" / "9x9" / "models" / self.champion / "model.bin.gz",
            self.destination / "config" / "9x9.json",
        ):
            with self.subTest(path=str(path)):
                content = path.read_bytes()
                original_stat = path.stat()
                path.write_bytes(b"accidentally overwritten")
                with self.assertRaises(HandoffError):
                    check_import(self.destination)
                path.write_bytes(content)
                os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                self.assertIsInstance(check_import(self.destination), dict)


if __name__ == "__main__":
    unittest.main()
