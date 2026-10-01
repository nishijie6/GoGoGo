"""Durable curriculum state, its atomic store, and the global run lock."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .fileio import FileLockHeld, acquire_file_lock, atomic_write_json, local_timestamp, release_file_lock
from .rl_curriculum_config import EXPECTED_BOARDS


STATE_SCHEMA_VERSION = 1


class CurriculumStateError(RuntimeError):
    """Raised when persisted curriculum state is missing or corrupt."""


class CurriculumLockError(RuntimeError):
    """Raised when another curriculum process owns the global lock."""


def progress_key(board_size: int) -> str:
    return f"{board_size}x{board_size}"


@dataclass
class StageProgress:
    board_size: int
    samples: int = 0
    entry_model: Optional[str] = None
    baseline_model: Optional[str] = None
    latest_model: Optional[str] = None
    champion_model: Optional[str] = None
    champion_established: bool = False
    fixed_baseline_models: list[str] = field(default_factory=list)
    next_evaluation_sample: int = 0
    consecutive_passes: int = 0
    evaluations: list[dict[str, object]] = field(default_factory=list)
    autotune_batch: Optional[int] = None
    started_at: str = field(default_factory=local_timestamp)
    completed_at: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StageProgress":
        try:
            progress = cls(
                board_size=int(value["board_size"]),
                samples=int(value.get("samples", 0)),
                entry_model=_optional_string(value.get("entry_model")),
                baseline_model=_optional_string(value.get("baseline_model")),
                latest_model=_optional_string(value.get("latest_model")),
                champion_model=_optional_string(value.get("champion_model")),
                champion_established=bool(value.get("champion_established", value.get("champion_model") is not None)),
                fixed_baseline_models=list(value.get("fixed_baseline_models", [])),
                next_evaluation_sample=int(value.get("next_evaluation_sample", 0)),
                consecutive_passes=int(value.get("consecutive_passes", 0)),
                evaluations=list(value.get("evaluations", [])),
                autotune_batch=(
                    int(value["autotune_batch"])
                    if value.get("autotune_batch") is not None
                    else None
                ),
                started_at=str(value.get("started_at", local_timestamp())),
                completed_at=_optional_string(value.get("completed_at")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CurriculumStateError("阶段状态字段无效") from error
        if progress.board_size not in EXPECTED_BOARDS or progress.samples < 0:
            raise CurriculumStateError("阶段状态棋盘或样本数无效")
        if not all(isinstance(name, str) and name for name in progress.fixed_baseline_models):
            raise CurriculumStateError("固定历史基准模型无效")
        return progress


def _optional_string(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None


@dataclass
class CurriculumState:
    schema_version: int
    active_stage_index: int
    phase: str
    stages: dict[str, StageProgress]
    protected_models: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    disk: dict[str, object] = field(default_factory=dict)
    resume_phase: Optional[str] = None
    migration: dict[str, object] = field(default_factory=dict)
    updated_at: str = field(default_factory=local_timestamp)

    @property
    def active(self) -> StageProgress:
        board = EXPECTED_BOARDS[self.active_stage_index]
        return self.stages[f"{board}x{board}"]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "active_stage_index": self.active_stage_index,
            "active_board_size": self.active.board_size,
            "phase": self.phase,
            "stages": {name: asdict(stage) for name, stage in self.stages.items()},
            "protected_models": list(self.protected_models),
            "warnings": list(self.warnings[-100:]),
            "disk": dict(self.disk),
            "resume_phase": self.resume_phase,
            "migration": dict(self.migration),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CurriculumState":
        if value.get("schema_version") != STATE_SCHEMA_VERSION:
            raise CurriculumStateError("state.json 版本不受支持")
        try:
            index = int(value["active_stage_index"])
            phase = str(value["phase"])
            raw_stages = value["stages"]
        except (KeyError, TypeError, ValueError) as error:
            raise CurriculumStateError("state.json 缺少必要字段") from error
        if index not in range(len(EXPECTED_BOARDS)):
            raise CurriculumStateError("active_stage_index 无效")
        if phase not in {
            "training",
            "evaluating",
            "transition_ready",
            "migrating",
            "paused_disk",
        }:
            raise CurriculumStateError(f"未知课程 phase：{phase}")
        if not isinstance(raw_stages, dict):
            raise CurriculumStateError("stages 状态必须是对象")
        stages = {
            str(name): StageProgress.from_dict(stage)
            for name, stage in raw_stages.items()
            if isinstance(stage, Mapping)
        }
        for board in EXPECTED_BOARDS[: index + 1]:
            if f"{board}x{board}" not in stages:
                raise CurriculumStateError(f"缺少 {board}x{board} 阶段状态")
        protected = value.get("protected_models", [])
        warnings = value.get("warnings", [])
        disk = value.get("disk", {})
        resume_phase = _optional_string(value.get("resume_phase"))
        migration = value.get("migration", {})
        if not isinstance(protected, list) or not all(
            isinstance(item, str) for item in protected
        ):
            raise CurriculumStateError("protected_models 无效")
        if (
            not isinstance(warnings, list)
            or not isinstance(disk, dict)
            or not isinstance(migration, dict)
        ):
            raise CurriculumStateError("warnings、disk 或 migration 状态无效")
        if resume_phase is not None and resume_phase not in {
            "training",
            "evaluating",
            "transition_ready",
            "migrating",
        }:
            raise CurriculumStateError("resume_phase 无效")
        return cls(
            schema_version=STATE_SCHEMA_VERSION,
            active_stage_index=index,
            phase=phase,
            stages=stages,
            protected_models=list(dict.fromkeys(protected)),
            warnings=[dict(item) for item in warnings if isinstance(item, dict)],
            disk=dict(disk),
            resume_phase=resume_phase,
            migration=dict(migration),
            updated_at=str(value.get("updated_at", local_timestamp())),
        )


class CurriculumStateStore:
    """Atomic state persistence; a failed write leaves the previous JSON intact."""

    def __init__(self, state_root: Path):
        self.state_root = state_root.resolve()
        self.path = self.state_root / "state.json"

    def load(self) -> Optional[CurriculumState]:
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CurriculumStateError(f"无法读取有效状态：{self.path}") from error
        if not isinstance(value, dict):
            raise CurriculumStateError("state.json 根节点必须是对象")
        return CurriculumState.from_dict(value)

    def save(self, state: CurriculumState) -> None:
        state.updated_at = local_timestamp()
        atomic_write_json(self.path, state.to_dict())


class GlobalCurriculumLock:
    """A nonblocking process lock shared by every board-size stage."""

    def __init__(self, state_root: Path):
        self.path = state_root.resolve() / "curriculum.lock"
        self._file: Any = None

    def __enter__(self) -> "GlobalCurriculumLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = acquire_file_lock(self.path)
        except FileLockHeld as error:
            raise CurriculumLockError(f"课程训练已在运行：{self.path}") from error
        return self

    def __exit__(self, *_: object) -> None:
        handle, self._file = self._file, None
        if handle is not None:
            release_file_lock(handle)

__all__ = [
    "CurriculumLockError",
    "CurriculumState",
    "CurriculumStateError",
    "CurriculumStateStore",
    "GlobalCurriculumLock",
    "STATE_SCHEMA_VERSION",
    "StageProgress",
    "progress_key",
]
