"""Disk checks and bounded cleanup of KataGo model, shuffle, and NPZ outputs."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


def assert_descendant(path: Path, root: Path) -> None:
    """Refuse to delete ``root`` itself or anything outside it."""

    resolved = path.resolve()
    root = root.resolve()
    if resolved == root:
        raise ValueError("拒绝删除训练根目录")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"拒绝删除训练目录外路径：{resolved}") from error


def directory_size(path: Path) -> int:
    if path.is_file():
        with contextlib.suppress(OSError):
            return path.stat().st_size
        return 0
    total = 0
    with contextlib.suppress(OSError):
        for child in path.rglob("*"):
            if child.is_file():
                with contextlib.suppress(OSError):
                    total += child.stat().st_size
    return total


@dataclass(frozen=True)
class DiskStatus:
    free_gib: float
    total_gib: float
    status: str


def disk_space_status(
    path: Path, *, cleanup_below_gib: float, pause_below_gib: float
) -> DiskStatus:
    usage = shutil.disk_usage(path.resolve())
    gib = 1024**3
    free = usage.free / gib
    if free < pause_below_gib:
        status = "paused"
    elif free < cleanup_below_gib:
        status = "cleanup"
    else:
        status = "healthy"
    return DiskStatus(free_gib=free, total_gib=usage.total / gib, status=status)


def _model_sort_key(path: Path) -> tuple[int, int, float, str]:
    match = re.search(r"-s(\d+)-d(\d+)$", path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), path.stat().st_mtime, path.name
    return 0, 0, path.stat().st_mtime, path.name


def select_retained_directories(
    paths: Iterable[Path], *, keep_latest: int, protected: Iterable[str]
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return ``(keep, delete)`` for model/checkpoint directories."""

    ordered = sorted(paths, key=_model_sort_key)
    protected_keys = {os.path.normcase(str(item)) for item in protected}

    def is_protected(path: Path) -> bool:
        candidates = {
            os.path.normcase(path.name),
            os.path.normcase(str(path)),
            os.path.normcase(str(path.resolve())),
        }
        return bool(candidates & protected_keys)

    keep_set = set(ordered[-keep_latest:])
    keep_set.update(path for path in ordered if is_protected(path))
    kept = tuple(path for path in ordered if path in keep_set)
    deleted = tuple(path for path in ordered if path not in keep_set)
    return kept, deleted


def npz_row_count(path: Path) -> int:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        for key in ("binaryInputNCHW", "globalInputNC", "policyTargetsNCMove"):
            if key in data and data[key].ndim:
                return int(data[key].shape[0])
        for key in data.files:
            if data[key].ndim:
                return int(data[key].shape[0])
    raise ValueError(f"无法判断 NPZ 行数：{path}")


def select_replay_window(
    paths: Iterable[Path],
    *,
    target_rows: int,
    row_counter: Callable[[Path], int] = npz_row_count,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Keep newest complete NPZ files covering at least ``target_rows``."""

    ordered = sorted(paths, key=lambda path: (path.stat().st_mtime, str(path)))
    kept: list[Path] = []
    rows = 0
    for path in reversed(ordered):
        if rows >= target_rows and kept:
            break
        kept.append(path)
        rows += max(0, row_counter(path))
    keep_set = set(kept)
    return (
        tuple(path for path in ordered if path in keep_set),
        tuple(path for path in ordered if path not in keep_set),
    )


@dataclass(frozen=True)
class RetentionPlan:
    delete_directories: tuple[Path, ...]
    delete_npz: tuple[Path, ...]


def build_retention_plan(
    run_root: Path,
    *,
    protected_models: Iterable[str],
    keep_models: int,
    keep_shuffles: int,
    replay_window_rows: int,
    row_counter: Callable[[Path], int] = npz_row_count,
) -> RetentionPlan:
    """Plan bounded model/checkpoint/shuffle/NPZ cleanup without touching SGFs."""

    run_root = run_root.resolve()
    model_dirs = [
        path
        for path in (run_root / "models").glob("*")
        if path.is_dir() and ".tmp" not in path.name
    ]
    export_dirs = [
        path
        for path in (run_root / "torchmodels_toexport").glob("*")
        if path.is_dir() and ".tmp" not in path.name
    ]
    _, delete_models = select_retained_directories(
        model_dirs, keep_latest=keep_models, protected=protected_models
    )
    _, delete_exports = select_retained_directories(
        export_dirs, keep_latest=keep_models, protected=protected_models
    )
    shuffle_dirs = sorted(
        (
            path
            for path in (run_root / "shuffleddata").glob("*")
            if path.is_dir() and not path.name.endswith(".tmp")
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    delete_shuffles = tuple(shuffle_dirs[:-keep_shuffles])
    npz_files = list((run_root / "selfplay").glob("**/tdata/*.npz"))
    _, delete_npz = select_replay_window(
        npz_files, target_rows=replay_window_rows, row_counter=row_counter
    )
    return RetentionPlan(
        delete_directories=tuple(
            dict.fromkeys((*delete_models, *delete_exports, *delete_shuffles))
        ),
        delete_npz=delete_npz,
    )


def apply_retention_plan(plan: RetentionPlan, run_root: Path) -> None:
    """Apply a precomputed plan after validating every exact target."""

    root = run_root.resolve()
    for path in (*plan.delete_directories, *plan.delete_npz):
        assert_descendant(path, root)
    for path in plan.delete_npz:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    for path in plan.delete_directories:
        if path.is_dir():
            shutil.rmtree(path)


__all__ = [
    "DiskStatus",
    "RetentionPlan",
    "apply_retention_plan",
    "assert_descendant",
    "build_retention_plan",
    "directory_size",
    "disk_space_status",
    "npz_row_count",
    "select_replay_window",
    "select_retained_directories",
]
