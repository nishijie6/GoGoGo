"""Training resumes only after the handoff marker is explicitly removed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from weiqi.fileio import acquire_file_lock, release_file_lock
from weiqi.katago_rl import DEFAULT_RTX_PROFILE, KataGoRLRunner, KataGoRLRunnerError
from weiqi.rl_curriculum_state import CurriculumLockError, GlobalCurriculumLock
from weiqi.rl_handoff_guard import (
    HANDOFF_MARKER,
    IMPORT_PENDING_MARKER,
    HandoffLockedError,
    ensure_training_available,
)


class HandoffGuardTests(unittest.TestCase):
    def test_existing_directory_without_marker_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.ckpt"
            checkpoint.write_bytes(b"existing training checkpoint")
            ensure_training_available(root)
            self.assertEqual(checkpoint.read_bytes(), b"existing training checkpoint")
            self.assertFalse((root / HANDOFF_MARKER).exists())

    def test_valid_marker_reports_handoff_metadata_and_cancel_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / HANDOFF_MARKER).write_text(
                json.dumps({"bundle_id": "handoff-0001", "bundle": "relay.zip"}),
                encoding="utf-8",
            )
            with self.assertRaises(HandoffLockedError) as caught:
                ensure_training_available(root)
            message = str(caught.exception)
            self.assertIn("已导出接力包", message)
            self.assertIn("handoff-0001", message)
            self.assertIn("relay.zip", message)
            self.assertIn("handoff-cancel", message)
            self.assertIn("其他电脑已停止", message)

    def test_malformed_marker_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / HANDOFF_MARKER
            for content in (b"{", b"[]", b"null", b"\xff"):
                with self.subTest(content=content):
                    marker.write_bytes(content)
                    with self.assertRaisesRegex(HandoffLockedError, "handoff-cancel"):
                        ensure_training_available(root)
                    self.assertEqual(marker.read_bytes(), content)

    def test_marker_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / HANDOFF_MARKER).mkdir()
            with self.assertRaises(HandoffLockedError):
                ensure_training_available(root)

    def test_pending_import_blocks_nested_run_before_export_hold_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stages" / "9x9"
            stage.mkdir(parents=True)
            (root / IMPORT_PENDING_MARKER).write_bytes(b"malformed metadata")
            (stage / HANDOFF_MARKER).write_text("{}", encoding="utf-8")
            with self.assertRaises(HandoffLockedError) as caught:
                ensure_training_available(stage)
            message = str(caught.exception)
            self.assertIn("导入尚未完成", message)
            self.assertIn("重新导入到新的目录", message)
            self.assertNotIn("handoff-cancel", message)

    def test_pending_import_blocks_nested_stage_lock_until_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stages" / "9x9"
            stage.mkdir(parents=True)
            marker = root / IMPORT_PENDING_MARKER
            marker.write_text("{}", encoding="utf-8")
            runner = KataGoRLRunner(DEFAULT_RTX_PROFILE, run_root_override=stage)
            with self.assertRaisesRegex(KataGoRLRunnerError, "导入尚未完成"):
                with runner.lock():
                    self.fail("A partially imported stage must not execute training")
            handle = acquire_file_lock(stage / "training.lock")
            release_file_lock(handle)
            marker.unlink()
            with runner.lock():
                pass

    def test_pending_import_blocks_nested_curriculum_lock_until_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            curriculum = root / "curriculum"
            marker = root / IMPORT_PENDING_MARKER
            marker.write_text("{}", encoding="utf-8")
            lock = GlobalCurriculumLock(curriculum)
            with self.assertRaisesRegex(CurriculumLockError, "导入尚未完成"):
                with lock:
                    self.fail("A partially imported curriculum must not execute training")
            self.assertIsNone(lock._file)
            handle = acquire_file_lock(curriculum / "curriculum.lock")
            release_file_lock(handle)
            marker.unlink()
            with lock:
                pass

    def test_stage_lock_blocks_training_and_releases_os_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = KataGoRLRunner(DEFAULT_RTX_PROFILE, run_root_override=root)
            marker = root / HANDOFF_MARKER
            for content in ("{}", "broken JSON"):
                with self.subTest(content=content):
                    marker.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(KataGoRLRunnerError, "handoff-cancel"):
                        with runner.lock():
                            self.fail("A suspended stage must not execute training")
                    handle = acquire_file_lock(root / "training.lock")
                    release_file_lock(handle)
            marker.unlink()
            with runner.lock():
                pass

    def test_curriculum_lock_blocks_training_and_releases_os_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = GlobalCurriculumLock(root)
            marker = root / HANDOFF_MARKER
            for content in ("{}", "broken JSON"):
                with self.subTest(content=content):
                    marker.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(CurriculumLockError, "handoff-cancel"):
                        with lock:
                            self.fail("A suspended curriculum must not execute training")
                    self.assertIsNone(lock._file)
                    handle = acquire_file_lock(root / "curriculum.lock")
                    release_file_lock(handle)
            marker.unlink()
            with lock:
                pass


if __name__ == "__main__":
    unittest.main()
