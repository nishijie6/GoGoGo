"""Tests for crash-safe raw replay row accounting."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import numpy as np

from weiqi.replay_accounting import ReplayAccountingError, ReplayRowLedger


def _write_npz(path: Path, rows: int, *, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, globalInputNC=np.zeros((rows, 1), dtype=np.float32))
    os.utime(path, (mtime, mtime))


def _write_checkpoint(path: Path, rows: int, *, mtime: float) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_state": {
                "total_num_data_rows": rows,
                "train_bucket_level_at_row": rows,
            }
        },
        path,
    )
    os.utime(path, (mtime, mtime))


class ReplayRowLedgerTests(unittest.TestCase):
    def _checkpoint_case(self, root: Path) -> tuple[Path, Path]:
        (root / "models" / "gogogo-s100-d1000").mkdir(parents=True)
        old = root / "selfplay" / "gogogo-s90-d900" / "tdata" / "old.npz"
        new = root / "selfplay" / "gogogo-s100-d1000" / "tdata" / "new.npz"
        _write_npz(old, 800, mtime=100)
        _write_checkpoint(
            root / "train" / "gogogo" / "checkpoint.ckpt",
            1000,
            mtime=200,
        )
        _write_npz(new, 100, mtime=300)
        return old, new

    def test_first_sync_counts_post_checkpoint_rows_as_untrained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._checkpoint_case(root)

            snapshot = ReplayRowLedger(root).sync()
            persisted = json.loads(snapshot.path.read_text(encoding="utf-8"))

            self.assertEqual(snapshot.physical_rows, 900)
            self.assertEqual(snapshot.deleted_rows_offset, 200)
            self.assertEqual(snapshot.logical_rows, 1100)
            self.assertEqual(len(snapshot.files), 2)
            self.assertEqual(len(persisted["files"]), 2)
            self.assertTrue(
                all(key.startswith("selfplay/") for key in persisted["files"])
            )

    def test_missing_file_after_interruption_is_accounted_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, _new = self._checkpoint_case(root)
            first = ReplayRowLedger(root).sync()
            self.assertEqual(first.deleted_rows_offset, 200)

            # Simulate retention deleting the file after its pre-cleanup sync,
            # followed by an immediate process interruption.
            old.unlink()
            recovered = ReplayRowLedger(root).sync()
            repeated = ReplayRowLedger(root).sync()

            self.assertEqual(recovered.deleted_rows_offset, 1000)
            self.assertEqual(recovered.physical_rows, 100)
            self.assertEqual(recovered.logical_rows, 1100)
            self.assertEqual(repeated.deleted_rows_offset, 1000)
            self.assertNotIn(
                "selfplay/gogogo-s90-d900/tdata/old.npz", repeated.files
            )
            json.loads(repeated.path.read_text(encoding="utf-8"))

    def test_ledgers_are_isolated_per_board_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = {9: base / "9x9", 13: base / "13x13"}
            cases = {
                9: (1000, 800, 100, 200),
                13: (2000, 1700, 50, 300),
            }
            old_paths: dict[int, Path] = {}
            for board, (watermark, old_rows, new_rows, _offset) in cases.items():
                root = roots[board]
                model = f"gogogo-s{watermark}-d{watermark}"
                (root / "models" / model).mkdir(parents=True)
                old = root / "selfplay" / "older" / "tdata" / "old.npz"
                new = root / "selfplay" / model / "tdata" / "new.npz"
                _write_npz(old, old_rows, mtime=100)
                _write_npz(new, new_rows, mtime=300)
                old_paths[board] = old

            nine = ReplayRowLedger(roots[9]).sync()
            thirteen = ReplayRowLedger(roots[13]).sync()
            self.assertEqual(nine.deleted_rows_offset, cases[9][3])
            self.assertEqual(thirteen.deleted_rows_offset, cases[13][3])

            old_paths[9].unlink()
            nine_after = ReplayRowLedger(roots[9]).sync()
            thirteen_after = ReplayRowLedger(roots[13]).sync()
            self.assertEqual(nine_after.deleted_rows_offset, 1000)
            self.assertEqual(thirteen_after.deleted_rows_offset, 300)
            self.assertEqual(thirteen_after.logical_rows, 2050)
            self.assertEqual(nine_after.path, roots[9].resolve() / "replay_rows.json")
            self.assertEqual(
                thirteen_after.path, roots[13].resolve() / "replay_rows.json"
            )
            for key in (*nine_after.files, *thirteen_after.files):
                self.assertFalse(PurePosixPath(key).is_absolute())
                self.assertNotIn("..", PurePosixPath(key).parts)

    def test_corrupt_ledger_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "replay_rows.json"
            root.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '{"schema_version":1,"deleted_rows_offset":-1}', encoding="utf-8"
            )

            with self.assertRaises(ReplayAccountingError):
                ReplayRowLedger(root).sync()
            self.assertIn('"deleted_rows_offset":-1', path.read_text(encoding="utf-8"))

    def test_clean_curriculum_seed_starts_new_board_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models" / "gogogo-s0-d0").mkdir(parents=True)
            checkpoint = root / "train" / "gogogo" / "checkpoint.ckpt"
            checkpoint.parent.mkdir(parents=True)
            import torch

            torch.save({"model": {}, "swa_model": {}, "config": {}}, checkpoint)

            initial = ReplayRowLedger(root).sync()
            self.assertEqual(initial.deleted_rows_offset, 0)
            self.assertEqual(initial.logical_rows, 0)

            raw = root / "selfplay" / "gogogo-s0-d0" / "tdata" / "new.npz"
            _write_npz(raw, 50, mtime=checkpoint.stat().st_mtime + 1)
            with_data = ReplayRowLedger(root).sync()
            self.assertEqual(with_data.deleted_rows_offset, 0)
            self.assertEqual(with_data.physical_rows, 50)
            self.assertEqual(with_data.logical_rows, 50)
if __name__ == "__main__":
    unittest.main()
