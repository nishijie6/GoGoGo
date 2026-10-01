"""Curriculum disk monitoring and bounded cleanup of every stage's run directory."""

from __future__ import annotations

import contextlib
import shutil
import time
from pathlib import Path
from typing import Iterable

from .fileio import local_timestamp
from .katago_rl import KataGoRLRunnerError
from .replay_accounting import ReplayAccountingError, ReplayRowLedger
from .rl_curriculum import CurriculumState, CurriculumStateError
from .rl_retention import (
    apply_retention_plan,
    assert_descendant,
    build_retention_plan,
    directory_size,
)


# Temporary directories younger than this may belong to a live writer.
STALE_TEMP_SECONDS = 6 * 60 * 60


class CleanupSteps:
    """Disk half of :class:`~weiqi.rl_curriculum_runtime.CurriculumRuntime`.

    Mixed into the runtime; uses its ``config``, ``store``, ``controller``,
    ``state_root``, ``project_root``, ``disk_probe`` and ``_make_runner``.
    """

    @staticmethod
    def _stale_temp_paths(run_root: Path, now: float) -> list[Path]:
        candidates: list[Path] = []
        for parent in (
            run_root / "models",
            run_root / "torchmodels_toexport",
            run_root / "shuffleddata",
            run_root / "shufflescratch",
        ):
            if not parent.is_dir():
                continue
            for path in parent.iterdir():
                if not path.is_dir():
                    continue
                is_temp = path.name.endswith(".tmp") or ".tmp-" in path.name
                with contextlib.suppress(OSError):
                    if is_temp and now - path.stat().st_mtime >= STALE_TEMP_SECONDS:
                        candidates.append(path)
        return candidates

    def _retention_for_root(
        self, run_root: Path, protected: Iterable[str]
    ) -> dict[str, int]:
        if not run_root.is_dir():
            return {"files": 0, "directories": 0, "bytes": 0}
        ledger = ReplayRowLedger(run_root)
        try:
            # This durable inventory is the write-ahead record for any NPZ
            # deletion performed below.
            ledger.sync()
        except ReplayAccountingError as error:
            raise CurriculumStateError(
                f"无法安全同步 {run_root.name} 回放行账本：{error}"
            ) from error
        plan = build_retention_plan(
            run_root,
            protected_models=protected,
            keep_models=self.config.keep_models,
            keep_shuffles=self.config.keep_shuffles,
            replay_window_rows=self.config.replay_window_rows,
        )
        longterm = run_root / "train" / "gogogo" / "longterm_checkpoints"
        checkpoints = sorted(
            (path for path in longterm.glob("*.ckpt") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        delete_checkpoints = checkpoints[:-self.config.keep_models]
        stale = self._stale_temp_paths(run_root, time.time())
        directory_targets = list(plan.delete_directories) + stale
        file_targets = list(plan.delete_npz) + delete_checkpoints
        unique_dirs = list(dict.fromkeys(directory_targets))
        unique_files = list(dict.fromkeys(file_targets))
        for path in (*unique_dirs, *unique_files):
            assert_descendant(path, run_root)
        deleted_bytes = sum(directory_size(path) for path in (*unique_dirs, *unique_files))
        # Apply the core plan first (which repeats exact-target validation),
        # then the runtime-only longterm/tmp rules.
        apply_retention_plan(plan, run_root)
        try:
            # If interruption happens before this write, the next pre-cleanup
            # sync observes the same missing inventory entries exactly once.
            ledger.sync()
        except ReplayAccountingError as error:
            raise CurriculumStateError(
                f"无法提交 {run_root.name} 回放清理账本：{error}"
            ) from error
        for path in delete_checkpoints:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        for path in stale:
            if path.is_dir():
                shutil.rmtree(path)
        return {
            "files": len(unique_files),
            "directories": len(unique_dirs),
            "bytes": deleted_bytes,
        }

    @staticmethod
    def _cleanup_stale_migrations(target_root: Path) -> dict[str, int]:
        """Remove only old, exact sibling migration-temp directories."""

        parent = target_root.resolve().parent
        if not parent.is_dir():
            return {"files": 0, "directories": 0, "bytes": 0}
        now = time.time()
        targets: list[Path] = []
        pattern = f".{target_root.name}.migration-*.tmp"
        for path in parent.glob(pattern):
            if not path.is_dir():
                continue
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime >= STALE_TEMP_SECONDS:
                    assert_descendant(path, parent)
                    targets.append(path)
        deleted_bytes = sum(directory_size(path) for path in targets)
        for path in targets:
            shutil.rmtree(path)
        return {
            "files": 0,
            "directories": len(targets),
            "bytes": deleted_bytes,
        }

    def _apply_retention(self, state: CurriculumState) -> dict[str, int]:
        totals = {"files": 0, "directories": 0, "bytes": 0}
        for stage in self.config.stages:
            runner = self._make_runner(stage)
            reports: list[dict[str, int]] = []
            if runner.run_root.is_dir():
                try:
                    with runner.lock():
                        reports.append(
                            self._retention_for_root(
                                runner.run_root, state.protected_models
                            )
                        )
                except KataGoRLRunnerError as error:
                    state.warnings.append(
                        {
                            "timestamp": local_timestamp(),
                            "kind": "retention_lock",
                            "message": str(error),
                        }
                    )
            reports.append(self._cleanup_stale_migrations(runner.run_root))
            for report in reports:
                for key in totals:
                    totals[key] += report[key]
        state.disk["last_cleanup"] = {
            **totals,
            "completed_at": local_timestamp(),
        }
        self.store.save(state)
        return totals

    def _update_disk(self, state: CurriculumState) -> str:
        status = self.disk_probe(
            self.state_root if self.state_root.exists() else self.project_root,
            cleanup_below_gib=self.config.disk_cleanup_gib,
            pause_below_gib=self.config.disk_pause_gib,
        )
        status_name = self.controller.update_disk(state, float(status.free_gib))
        state.disk["total_gib"] = float(status.total_gib)
        self.store.save(state)
        return status_name
