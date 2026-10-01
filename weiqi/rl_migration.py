"""Seed a new curriculum board size from the previous stage's SWA weights."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .fileio import atomic_open, local_timestamp
from .rl_curriculum import EXPECTED_BOARDS, CurriculumMigrationError


def create_clean_swa_checkpoint(source: Path, target: Path) -> None:
    """Materialize SWA weights with all optimizer/counter/EMA state removed."""

    try:
        import torch
    except ImportError as error:
        raise CurriculumMigrationError("迁移检查点需要 PyTorch") from error
    try:
        source_state = torch.load(source, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise CurriculumMigrationError(f"无法加载源检查点：{source}") from error
    if not isinstance(source_state, dict) or "config" not in source_state:
        raise CurriculumMigrationError("源检查点缺少网络 config")
    swa_state = source_state.get("swa_model")
    if not isinstance(swa_state, dict):
        raise CurriculumMigrationError("源检查点不含 SWA 权重")
    raw_model: dict[str, object] = {}
    for original_key, value in swa_state.items():
        if original_key == "n_averaged":
            continue
        key = original_key
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            elif key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        raw_model[key] = value
    if not raw_model:
        raise CurriculumMigrationError("SWA 权重为空")
    # A fresh one-snapshot SWA wrapper lets the normal exporter keep using
    # -use-swa, but carries no optimizer, loss EMA, counters, or data history.
    first_tensor = next(iter(raw_model.values()))
    n_averaged = torch.tensor(1, device=getattr(first_tensor, "device", "cpu"))
    fresh_swa: dict[str, object] = {"n_averaged": n_averaged}
    fresh_swa.update({f"module.{key}": value for key, value in raw_model.items()})
    clean = {
        "model": raw_model,
        "swa_model": fresh_swa,
        "config": source_state["config"],
        "curriculum_seed": {
            "created_at": local_timestamp(),
            "source": str(source.resolve()),
            "reset_fields": [
                "optimizer",
                "train_state",
                "metrics",
                "running_metrics",
                "last_val_metrics",
            ],
        },
    }
    with atomic_open(target, "wb") as output:
        torch.save(clean, output)


def verify_checkpoint_loads(
    checkpoint: Path,
    *,
    boards: Sequence[int],
    python_source: Path,
) -> None:
    """Load the same seed at every requested positional board length on CPU."""

    source_text = str(python_source.resolve())
    inserted = source_text not in sys.path
    if inserted:
        sys.path.insert(0, source_text)
    try:
        from katago.train.load_model import load_model

        for board in boards:
            if board not in EXPECTED_BOARDS:
                raise CurriculumMigrationError(f"不支持验证棋盘：{board}")
            model, swa_model, _ = load_model(
                str(checkpoint), use_swa=True, device="cpu", pos_len=board, verbose=False
            )
            if model is None or swa_model is None:
                raise CurriculumMigrationError(f"{board}x{board} 模型加载失败")
    except Exception as error:
        if isinstance(error, CurriculumMigrationError):
            raise
        raise CurriculumMigrationError(f"检查点跨棋盘加载验证失败：{error}") from error
    finally:
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(source_text)


@dataclass(frozen=True)
class MigrationResult:
    stage_root: Path
    checkpoint: Path
    seed_name: str


def prepare_stage_migration(
    *,
    source_checkpoint: Path,
    target_stage_root: Path,
    target_board_size: int,
    load_verifier: Callable[[Path, int], None],
    smoke_verifier: Callable[[Path, Path, int], None],
    seed_name: str = "gogogo-s0-d0",
) -> MigrationResult:
    """Build and smoke-test an isolated stage tree, then rename it atomically."""

    target = target_stage_root.resolve()
    if target.exists():
        raise CurriculumMigrationError(f"目标阶段目录已存在，拒绝覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.migration-{uuid.uuid4().hex}.tmp"
    try:
        checkpoint = temporary / "train" / "gogogo" / "checkpoint.ckpt"
        create_clean_swa_checkpoint(source_checkpoint, checkpoint)
        export_checkpoint = temporary / "torchmodels_toexport" / seed_name / "model.ckpt"
        export_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, export_checkpoint)
        load_verifier(checkpoint, target_board_size)
        smoke_verifier(temporary, checkpoint, target_board_size)
        os.replace(temporary, target)
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(error, CurriculumMigrationError):
            raise
        raise CurriculumMigrationError(
            f"{target_board_size}x{target_board_size} 迁移验证失败：{error}"
        ) from error
    return MigrationResult(
        stage_root=target,
        checkpoint=target / "train" / "gogogo" / "checkpoint.ckpt",
        seed_name=seed_name,
    )


__all__ = [
    "MigrationResult",
    "create_clean_swa_checkpoint",
    "prepare_stage_migration",
    "verify_checkpoint_loads",
]
