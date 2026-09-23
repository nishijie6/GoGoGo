"""Resumable curriculum orchestration primitives for KataGo reinforcement learning.

The module deliberately keeps process execution behind small callbacks.  The CLI
runner can therefore use the real KataGo executable while the safety-critical
state machine, promotion gates, retention rules, and migrations remain fast to
unit test.
"""

from __future__ import annotations

import contextlib
import copy
import html
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence


CURRICULUM_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
EXPECTED_BOARDS = (9, 13, 19)
DEFAULT_HEALTH_GAMES_PER_CYCLE = 128


class CurriculumConfigError(ValueError):
    """Raised when the curriculum configuration is invalid."""


class CurriculumStateError(RuntimeError):
    """Raised when persisted curriculum state is missing or corrupt."""


class CurriculumLockError(RuntimeError):
    """Raised when another curriculum process owns the global lock."""


class CurriculumMigrationError(RuntimeError):
    """Raised when a stage seed cannot be prepared and verified safely."""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_exact_keys(
    value: Mapping[str, object],
    *,
    path: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise CurriculumConfigError(
            f"{path} 缺少字段：{', '.join(sorted(missing))}"
        )
    if unknown:
        raise CurriculumConfigError(
            f"{path} 包含未知字段：{', '.join(sorted(unknown))}"
        )


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CurriculumConfigError(f"{path} 必须是正整数")
    return value


def _positive_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurriculumConfigError(f"{path} 必须是正数")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CurriculumConfigError(f"{path} 必须是正数")
    return result


def _probability(value: object, path: str) -> float:
    result = _positive_number(value, path)
    if result > 1:
        raise CurriculumConfigError(f"{path} 必须在 (0, 1] 内")
    return result


@dataclass(frozen=True)
class CurriculumStage:
    """One isolated board-size stage."""

    board_size: int
    training_config: str
    min_stage_samples: Optional[int]
    autotune_batches: tuple[int, ...]
    indefinite: bool
    fixed_baseline_models: tuple[str, ...] = ()
    initial_champion_model: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.board_size}x{self.board_size}"

    def config_path(self, project_root: Path) -> Path:
        path = Path(self.training_config)
        return (path if path.is_absolute() else project_root / path).resolve()


@dataclass(frozen=True)
class HealthThresholds:
    """Fixed self-play health gates agreed for the curriculum."""

    immediate_double_pass_rate: float = 0.01
    extreme_result_rate: float = 0.05
    black_win_rate_min: float = 0.45
    black_win_rate_max: float = 0.55
    invalid_rate: float = 0.005


@dataclass(frozen=True)
class CurriculumConfig:
    schema_version: int
    state_directory: str
    evaluation_interval_samples: int
    evaluation_games: int
    promotion_win_rate: float
    wilson_lower_bound: float
    required_consecutive_passes: int
    health_window_cycles: int
    replay_window_rows: int
    keep_models: int
    keep_shuffles: int
    disk_cleanup_gib: float
    disk_pause_gib: float
    max_gpu_memory_gib: float
    stages: tuple[CurriculumStage, ...]
    health: HealthThresholds = HealthThresholds()

    def state_root(self, project_root: Path) -> Path:
        path = Path(self.state_directory)
        return (path if path.is_absolute() else project_root / path).resolve()

    def stage(self, board_size: int) -> CurriculumStage:
        for stage in self.stages:
            if stage.board_size == board_size:
                return stage
        raise KeyError(board_size)


_ROOT_FIELDS = {
    "schema_version",
    "state_directory",
    "evaluation_interval_samples",
    "evaluation_games",
    "promotion_win_rate",
    "wilson_lower_bound",
    "required_consecutive_passes",
    "health_window_cycles",
    "replay_window_rows",
    "keep_models",
    "keep_shuffles",
    "disk_cleanup_gib",
    "disk_pause_gib",
    "max_gpu_memory_gib",
    "stages",
}
_STAGE_FIELDS = {
    "board_size",
    "training_config",
    "min_stage_samples",
    "autotune_batches",
    "indefinite",
}


def load_curriculum_config(path: Path) -> CurriculumConfig:
    """Load the strict, user-visible curriculum JSON schema."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CurriculumConfigError(f"无法读取课程配置：{path}") from error
    except json.JSONDecodeError as error:
        raise CurriculumConfigError(f"课程配置不是有效 JSON：{error}") from error
    if not isinstance(raw, dict):
        raise CurriculumConfigError("课程配置根节点必须是对象")
    _require_exact_keys(raw, path="课程配置", required=_ROOT_FIELDS)
    if (
        isinstance(raw["schema_version"], bool)
        or not isinstance(raw["schema_version"], int)
        or raw["schema_version"] != CURRICULUM_SCHEMA_VERSION
    ):
        raise CurriculumConfigError(
            f"schema_version 必须是 {CURRICULUM_SCHEMA_VERSION}"
        )
    state_directory = raw["state_directory"]
    if not isinstance(state_directory, str) or not state_directory.strip():
        raise CurriculumConfigError("state_directory 必须是非空路径")
    if "\x00" in state_directory:
        raise CurriculumConfigError("state_directory 不能包含空字符")

    interval = _positive_int(
        raw["evaluation_interval_samples"], "evaluation_interval_samples"
    )
    games = _positive_int(raw["evaluation_games"], "evaluation_games")
    if games % 2:
        raise CurriculumConfigError("evaluation_games 必须是偶数")
    promotion = _probability(raw["promotion_win_rate"], "promotion_win_rate")
    if promotion <= 0.5:
        raise CurriculumConfigError("promotion_win_rate 必须大于 0.5")
    wilson = _probability(raw["wilson_lower_bound"], "wilson_lower_bound")
    if wilson < 0.5:
        raise CurriculumConfigError("wilson_lower_bound 不能小于 0.5")
    required_passes = _positive_int(
        raw["required_consecutive_passes"], "required_consecutive_passes"
    )
    health_cycles = _positive_int(raw["health_window_cycles"], "health_window_cycles")
    replay_rows = _positive_int(raw["replay_window_rows"], "replay_window_rows")
    keep_models = _positive_int(raw["keep_models"], "keep_models")
    keep_shuffles = _positive_int(raw["keep_shuffles"], "keep_shuffles")
    cleanup_gib = _positive_number(raw["disk_cleanup_gib"], "disk_cleanup_gib")
    pause_gib = _positive_number(raw["disk_pause_gib"], "disk_pause_gib")
    if cleanup_gib <= pause_gib:
        raise CurriculumConfigError("disk_cleanup_gib 必须大于 disk_pause_gib")
    max_gpu_gib = _positive_number(raw["max_gpu_memory_gib"], "max_gpu_memory_gib")

    raw_stages = raw["stages"]
    if not isinstance(raw_stages, list) or len(raw_stages) != len(EXPECTED_BOARDS):
        raise CurriculumConfigError("stages 必须依次包含 9、13、19 三个阶段")
    stages: list[CurriculumStage] = []
    for index, item in enumerate(raw_stages):
        path_name = f"stages[{index}]"
        if not isinstance(item, dict):
            raise CurriculumConfigError(f"{path_name} 必须是对象")
        _require_exact_keys(
            item, path=path_name, required=_STAGE_FIELDS,
            optional={"fixed_baseline_models", "initial_champion_model"},
        )
        fixed_models = item.get("fixed_baseline_models", [])
        champion = item.get("initial_champion_model")
        if not isinstance(fixed_models, list) or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name)
            for name in fixed_models
        ) or len(set(fixed_models)) != len(fixed_models):
            raise CurriculumConfigError(f"{path_name}.fixed_baseline_models 必须是不重复的模型目录名")
        if champion is not None and (
            not isinstance(champion, str)
            or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", champion)
        ):
            raise CurriculumConfigError(f"{path_name}.initial_champion_model 必须是模型目录名")
        board = _positive_int(item["board_size"], f"{path_name}.board_size")
        training_config = item["training_config"]
        if not isinstance(training_config, str) or not training_config.strip():
            raise CurriculumConfigError(f"{path_name}.training_config 必须是非空路径")
        minimum = item["min_stage_samples"]
        if minimum is not None:
            minimum = _positive_int(minimum, f"{path_name}.min_stage_samples")
        batches = item["autotune_batches"]
        if not isinstance(batches, list):
            raise CurriculumConfigError(f"{path_name}.autotune_batches 必须是数组")
        parsed_batches = tuple(
            _positive_int(value, f"{path_name}.autotune_batches") for value in batches
        )
        if tuple(sorted(set(parsed_batches))) != parsed_batches:
            raise CurriculumConfigError(
                f"{path_name}.autotune_batches 必须严格递增且不能重复"
            )
        indefinite = item["indefinite"]
        if not isinstance(indefinite, bool):
            raise CurriculumConfigError(f"{path_name}.indefinite 必须是布尔值")
        stages.append(
            CurriculumStage(
                board_size=board,
                training_config=training_config,
                min_stage_samples=minimum,
                autotune_batches=parsed_batches,
                indefinite=indefinite,
                fixed_baseline_models=tuple(fixed_models),
                initial_champion_model=champion,
            )
        )

    if tuple(stage.board_size for stage in stages) != EXPECTED_BOARDS:
        raise CurriculumConfigError("stages 棋盘顺序必须严格为 9、13、19")
    for stage in stages[:-1]:
        if stage.indefinite or stage.min_stage_samples is None:
            raise CurriculumConfigError(f"{stage.key} 必须设置有限的 min_stage_samples")
    final_stage = stages[-1]
    if not final_stage.indefinite or final_stage.min_stage_samples is not None:
        raise CurriculumConfigError("19x19 必须 indefinite=true 且 min_stage_samples=null")
    if stages[0].autotune_batches:
        raise CurriculumConfigError("9x9 沿用现有批次，不应配置 autotune_batches")
    if not stages[1].autotune_batches or not stages[2].autotune_batches:
        raise CurriculumConfigError("13x13 和 19x19 必须配置 autotune_batches")
    if len({stage.training_config.casefold() for stage in stages}) != 3:
        raise CurriculumConfigError("三个阶段必须使用不同的 training_config 文件")

    return CurriculumConfig(
        schema_version=CURRICULUM_SCHEMA_VERSION,
        state_directory=state_directory,
        evaluation_interval_samples=interval,
        evaluation_games=games,
        promotion_win_rate=promotion,
        wilson_lower_bound=wilson,
        required_consecutive_passes=required_passes,
        health_window_cycles=health_cycles,
        replay_window_rows=replay_rows,
        keep_models=keep_models,
        keep_shuffles=keep_shuffles,
        disk_cleanup_gib=cleanup_gib,
        disk_pause_gib=pause_gib,
        max_gpu_memory_gib=max_gpu_gib,
        stages=tuple(stages),
    )


def validate_training_profiles(config: CurriculumConfig, project_root: Path) -> None:
    """Verify board sizes, a shared network, and physically isolated run roots."""

    from .rl_config import RLConfigError, load_rl_training_config

    networks: set[tuple[int, int]] = set()
    run_roots: set[str] = set()
    for stage in config.stages:
        path = stage.config_path(project_root)
        if not path.is_file():
            raise CurriculumConfigError(f"训练配置不存在：{path}")
        try:
            profile = load_rl_training_config(path)
        except RLConfigError as error:
            raise CurriculumConfigError(f"训练配置无效 {path}：{error}") from error
        if profile.game.board_size != stage.board_size:
            raise CurriculumConfigError(
                f"{path} 的 board_size={profile.game.board_size}，应为 {stage.board_size}"
            )
        networks.add((profile.network.residual_blocks, profile.network.channels))
        output = Path(profile.runtime.output_directory)
        output = (output if output.is_absolute() else project_root / output).resolve()
        output_key = os.path.normcase(str(output))
        if output_key in run_roots:
            raise CurriculumConfigError("不同棋盘的 output_directory 必须严格分离")
        run_roots.add(output_key)
    if len(networks) != 1:
        raise CurriculumConfigError("三个阶段必须使用同一残差网络结构")
    if networks != {(10, 128)}:
        raise CurriculumConfigError("课程训练必须使用同一 b10c128 网络")


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
    started_at: str = field(default_factory=_now)
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
                started_at=str(value.get("started_at", _now())),
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
    updated_at: str = field(default_factory=_now)

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
            updated_at=str(value.get("updated_at", _now())),
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
        state.updated_at = _now()
        _atomic_write_json(self.path, state.to_dict())


class GlobalCurriculumLock:
    """A nonblocking process lock shared by every board-size stage."""

    def __init__(self, state_root: Path):
        self.path = state_root.resolve() / "curriculum.lock"
        self._file: Any = None

    def __enter__(self) -> "GlobalCurriculumLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise CurriculumLockError(f"课程训练已在运行：{self.path}") from error
        handle.seek(0)
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        self._file = handle
        return self

    def __exit__(self, *_: object) -> None:
        handle = self._file
        self._file = None
        if handle is None:
            return
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def next_evaluation_boundary(samples: int, interval: int) -> int:
    """Return the first evaluation boundary strictly after ``samples``."""

    if samples < 0 or interval < 1:
        raise ValueError("samples 不能为负且 interval 必须为正")
    return (samples // interval + 1) * interval


def wilson_interval(
    successes: float, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """95% Wilson score interval, supporting half a success for a draw."""

    if trials < 1 or successes < 0 or successes > trials:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def approximate_elo(win_rate: float) -> float:
    """Convert a match score to the conventional logistic Elo approximation."""

    clipped = min(1.0 - 1e-9, max(1e-9, win_rate))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class MatchGame:
    black: Optional[str]
    white: Optional[str]
    result: Optional[str]
    winner: Optional[str]
    margin: Optional[float]
    damaged: bool = False


@dataclass(frozen=True)
class MatchSummary:
    requested_games: int
    games: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    candidate_black_games: int
    candidate_white_games: int
    black_wins: int
    white_wins: int
    no_result_games: int
    damaged_games: int
    win_rate: float
    wilson_lower: float
    wilson_upper: float
    elo: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_sgf_trees(text: str) -> Iterator[str]:
    start: Optional[int] = None
    depth = 0
    bracket = False
    escaped = False
    for index, char in enumerate(text):
        if bracket:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "]":
                bracket = False
            continue
        if char == "[" and depth:
            bracket = True
        elif char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def _sgf_value(tree: str, key: str) -> Optional[str]:
    match = re.search(rf"(?<![A-Z]){re.escape(key)}\[((?:\\.|[^\\\]])*)\]", tree)
    if match is None:
        return None
    return re.sub(r"\\(.)", r"\1", match.group(1)).strip()


def parse_sgf_game(tree: str) -> MatchGame:
    black = _sgf_value(tree, "PB")
    white = _sgf_value(tree, "PW")
    result = _sgf_value(tree, "RE")
    damaged = black is None or white is None or result is None
    winner: Optional[str] = None
    margin: Optional[float] = None
    if result:
        result_text = result.strip()
        upper = result_text.upper()
        if upper.startswith("B+"):
            winner = "B"
        elif upper.startswith("W+"):
            winner = "W"
        elif upper in {"0", "DRAW", "JIGO"}:
            winner = "D"
        elif upper not in {"?", "VOID", "NO RESULT"}:
            damaged = True
        if winner in {"B", "W"}:
            score = result_text.split("+", 1)[1]
            try:
                margin = float(score)
            except ValueError:
                margin = None
    return MatchGame(black, white, result, winner, margin, damaged)


def summarize_match_sgfs(
    paths: Iterable[Path], *, candidate_name: str, requested_games: int
) -> MatchSummary:
    """Parse KataGo ``match`` SGFs with candidate color accounting."""

    games: list[MatchGame] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        games.extend(parse_sgf_game(tree) for tree in _iter_sgf_trees(text))
    candidate_key = candidate_name.casefold()
    candidate_wins = baseline_wins = draws = 0
    candidate_black = candidate_white = 0
    black_wins = white_wins = no_result = damaged = 0
    valid_candidate_games = 0
    for game in games:
        is_black = bool(game.black and game.black.casefold() == candidate_key)
        is_white = bool(game.white and game.white.casefold() == candidate_key)
        if game.damaged:
            damaged += 1
        if game.winner is None:
            no_result += 1
            continue
        if game.damaged:
            continue
        if not (is_black or is_white):
            damaged += 1
            continue
        if is_black:
            candidate_black += 1
        if is_white:
            candidate_white += 1
        valid_candidate_games += 1
        if game.winner == "D":
            draws += 1
        elif game.winner == "B":
            black_wins += 1
            if is_black:
                candidate_wins += 1
            else:
                baseline_wins += 1
        elif game.winner == "W":
            white_wins += 1
            if is_white:
                candidate_wins += 1
            else:
                baseline_wins += 1
    successes = candidate_wins + 0.5 * draws
    rate = successes / valid_candidate_games if valid_candidate_games else 0.0
    lower, upper = wilson_interval(successes, valid_candidate_games)
    return MatchSummary(
        requested_games=requested_games,
        games=valid_candidate_games,
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        draws=draws,
        candidate_black_games=candidate_black,
        candidate_white_games=candidate_white,
        black_wins=black_wins,
        white_wins=white_wins,
        no_result_games=no_result,
        damaged_games=damaged,
        win_rate=rate,
        wilson_lower=lower,
        wilson_upper=upper,
        elo=approximate_elo(rate),
    )


def build_match_config(
    *,
    candidate_model: Path,
    baseline_model: Path,
    board_size: int,
    visits: int,
    games: int = 200,
    game_threads: int = 128,
    inference_batch_size: int = 128,
    gpu_index: int = 0,
    opening_directory: Optional[Path] = None,
) -> str:
    """Build a fixed-protocol, mirrored-color KataGo match configuration."""

    checked_paths = [candidate_model, baseline_model]
    if opening_directory is not None:
        checked_paths.append(opening_directory)
    for path in checked_paths:
        if "\n" in str(path) or "\r" in str(path):
            raise ValueError("评测路径不能包含换行")
    if board_size not in EXPECTED_BOARDS or visits < 1 or games < 2 or games % 2:
        raise ValueError("棋盘、visits 或偶数局数无效")
    values: list[tuple[str, object]] = [
        ("logSearchInfo", "false"),
        ("logMoves", "false"),
        ("logGamesEvery", 1),
        ("logToStdout", "true"),
        ("numBots", 2),
        ("botName0", "candidate"),
        ("botName1", "baseline"),
        ("nnModelFile0", candidate_model),
        ("nnModelFile1", baseline_model),
        ("numGameThreads", game_threads),
        ("numGamesTotal", games),
        ("maxMovesPerGame", math.ceil(board_size * board_size * 2.5)),
        ("allowResignation", "false"),
        # KataGo's match loader requires these companion keys even when
        # resignation itself is disabled.
        ("resignThreshold", -1.0),
        ("resignConsecTurns", 3),
        ("koRules", "POSITIONAL"),
        ("scoringRules", "AREA"),
        ("taxRules", "NONE"),
        ("multiStoneSuicideLegals", "false"),
        ("hasButtons", "false"),
        ("bSizes", board_size),
        ("bSizeRelProbs", 1),
        ("komiAuto", "false"),
        ("komiMean", 6.5),
        ("komiStdev", 0.0),
        ("handicapProb", 0.0),
        ("maxVisits", visits),
        ("numSearchThreads", 1),
        ("rootNoiseEnabled", "false"),
        ("chosenMoveTemperatureEarly", 0.0),
        ("chosenMoveTemperature", 0.0),
        ("chosenMoveTemperatureOnlyBelowProb", 1.0),
        ("rootNumSymmetriesToSample", 1),
        ("rootSymmetryPruning", "false"),
        ("useGraphSearch", "false"),
        ("initGamesWithPolicy", "false"),
        ("nnMaxBatchSize", inference_batch_size),
        ("nnCacheSizePowerOfTwo", 20),
        ("numNNServerThreadsPerModel", 2),
        ("cudaDeviceToUse", gpu_index),
        ("cudaUseFP16", "true"),
        ("cudaUseNHWC", "true"),
        ("nnRandomize", "false"),
        ("nnRandSeed", "gogogo-curriculum-evaluation-v1"),
    ]
    if opening_directory is not None:
        values.extend(
            [
                ("hintPosesDir", opening_directory),
                ("hintPosesProb", 1.0),
            ]
        )
    return "\n".join(f"{key} = {value}" for key, value in values) + "\n"


@dataclass(frozen=True)
class HealthSummary:
    cycles: int
    games: int
    black_wins: int
    white_wins: int
    draws: int
    immediate_double_pass_games: int
    extreme_games: int
    invalid_games: int
    black_win_rate: float
    immediate_double_pass_rate: float
    extreme_result_rate: float
    invalid_rate: float
    complete: bool
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def _record_int(record: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0, int(value))
    return 0


def compute_health_window(
    records: Sequence[Mapping[str, object]],
    *,
    window_cycles: int = 10,
    games_per_cycle: int = DEFAULT_HEALTH_GAMES_PER_CYCLE,
    thresholds: HealthThresholds = HealthThresholds(),
) -> HealthSummary:
    """Aggregate the most recent self-play cycles and apply all health gates."""

    selected = list(records[-window_cycles:])
    games = sum(
        _record_int(record, "games", "sgf_games", "selfplay_games")
        for record in selected
    )
    black = sum(_record_int(record, "black_wins") for record in selected)
    white = sum(_record_int(record, "white_wins") for record in selected)
    draws = sum(_record_int(record, "draws") for record in selected)
    double_pass = sum(
        _record_int(
            record,
            "immediate_double_pass_games",
            "opening_double_pass_games",
            "double_pass_games",
        )
        for record in selected
    )
    extreme = sum(
        _record_int(record, "extreme_games", "extreme_result_games")
        for record in selected
    )
    invalid = sum(
        _record_int(record, "invalid_games", "no_result_games")
        + _record_int(record, "damaged_games", "damaged_sgfs")
        for record in selected
    )
    complete = len(selected) >= window_cycles and games >= window_cycles * games_per_cycle
    decided = black + white + draws
    black_rate = (black + 0.5 * draws) / decided if decided else 0.0
    double_rate = double_pass / games if games else 1.0
    extreme_rate = extreme / games if games else 1.0
    invalid_rate = min(games, invalid) / games if games else 1.0
    reasons: list[str] = []
    if not complete:
        reasons.append("健康窗口不足")
    if double_rate > thresholds.immediate_double_pass_rate:
        reasons.append("开局双方立即停一手比例超标")
    if extreme_rate > thresholds.extreme_result_rate:
        reasons.append("极端结果比例超标")
    if not thresholds.black_win_rate_min <= black_rate <= thresholds.black_win_rate_max:
        reasons.append("黑方胜率超出 45%–55%")
    if invalid_rate > thresholds.invalid_rate:
        reasons.append("无结果或损坏棋谱比例超标")
    return HealthSummary(
        cycles=len(selected),
        games=games,
        black_wins=black,
        white_wins=white,
        draws=draws,
        immediate_double_pass_games=double_pass,
        extreme_games=extreme,
        invalid_games=min(games, invalid),
        black_win_rate=black_rate,
        immediate_double_pass_rate=double_rate,
        extreme_result_rate=extreme_rate,
        invalid_rate=invalid_rate,
        complete=complete,
        passed=not reasons,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class QualityGate:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_quality_gate(
    match: MatchSummary,
    health: HealthSummary,
    config: CurriculumConfig,
    *,
    stage_samples: int,
    minimum_samples: Optional[int],
    rolling_champion: bool = False,
) -> QualityGate:
    reasons: list[str] = []
    if minimum_samples is not None and stage_samples < minimum_samples:
        reasons.append("尚未达到阶段最低样本数")
    if match.games != config.evaluation_games:
        reasons.append("固定评测局数不完整")
    if match.win_rate < config.promotion_win_rate:
        reasons.append("评测胜率低于晋级门槛")
    if not rolling_champion and match.wilson_lower <= config.wilson_lower_bound:
        reasons.append("Wilson 置信区间下界未高于门槛")
    if not health.passed:
        reasons.extend(health.reasons)
    return QualityGate(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))


class CurriculumController:
    """Persistent 9 -> 13 -> 19 state transitions and quality decisions."""

    def __init__(self, config: CurriculumConfig, store: CurriculumStateStore):
        self.config = config
        self.store = store

    def adopt_existing_9x9(
        self,
        *,
        trained_samples: int,
        latest_model: str,
        entry_model: Optional[str] = None,
    ) -> CurriculumState:
        existing = self.store.load()
        if existing is not None:
            return existing
        if trained_samples < 0 or not latest_model:
            raise CurriculumStateError("无法接管无效的 9x9 训练状态")
        baseline = entry_model or latest_model
        progress = StageProgress(
            board_size=9,
            samples=trained_samples,
            entry_model=baseline,
            baseline_model=baseline,
            latest_model=latest_model,
            next_evaluation_sample=next_evaluation_boundary(
                trained_samples, self.config.evaluation_interval_samples
            ),
        )
        state = CurriculumState(
            schema_version=STATE_SCHEMA_VERSION,
            active_stage_index=0,
            phase="training",
            stages={progress_key(9): progress},
            protected_models=list(dict.fromkeys([baseline, latest_model])),
        )
        self.store.save(state)
        return state

    def mark_cycle(
        self,
        state: CurriculumState,
        *,
        trained_samples: int,
        latest_model: str,
    ) -> bool:
        if trained_samples < state.active.samples:
            raise CurriculumStateError("阶段训练样本数不能倒退")
        state.active.samples = trained_samples
        state.active.latest_model = latest_model
        due = trained_samples >= state.active.next_evaluation_sample
        if state.phase != "paused_disk":
            state.phase = "evaluating" if due else "training"
        self.store.save(state)
        return due

    def evaluation_baseline(self, state: CurriculumState) -> Optional[str]:
        return state.active.champion_model or state.active.baseline_model

    def evaluation_opponents(self, state: CurriculumState) -> dict[str, list[str]]:
        """One match per distinct opponent, even when it has both pool roles."""
        active = state.active
        opponents: dict[str, list[str]] = {}
        champion = self.evaluation_baseline(state)
        if champion and champion != active.latest_model:
            opponents.setdefault(champion, []).append("champion")
        for name in active.fixed_baseline_models or [active.baseline_model]:
            if name and name != active.latest_model:
                opponents.setdefault(name, []).append("fixed")
        return opponents

    def record_evaluation(
        self,
        state: CurriculumState,
        *,
        candidate_model: str,
        match: MatchSummary,
        health: HealthSummary,
        opponent_results: Optional[Mapping[str, MatchSummary]] = None,
    ) -> QualityGate:
        stage_definition = self.config.stages[state.active_stage_index]
        # Stage advancement uses a stable historical anchor. Champion promotion
        # measures playing strength independently of self-play health and samples.
        baseline = state.active.baseline_model
        gate = evaluate_quality_gate(
            match,
            health,
            self.config,
            stage_samples=state.active.samples,
            minimum_samples=stage_definition.min_stage_samples,
        )
        candidate = copy.deepcopy(state)
        record: dict[str, object] = {
            "timestamp": _now(),
            "candidate_model": candidate_model,
            "baseline_model": baseline,
            "stage_samples": state.active.samples,
            "passed": gate.passed,
            "reasons": list(gate.reasons),
            "match": match.to_dict(),
            "health": health.to_dict(),
        }
        champion = state.active.champion_model
        record["champion_before"] = champion
        if opponent_results is not None:
            expected = self.evaluation_opponents(state)
            if set(opponent_results) != set(expected):
                raise CurriculumStateError("两类评测对手结果不完整")
            if baseline not in opponent_results or opponent_results[baseline] != match:
                raise CurriculumStateError("课程门槛必须使用首个固定历史基准的成绩")
            if any(
                result.games != self.config.evaluation_games
                or result.candidate_black_games != self.config.evaluation_games // 2
                or result.candidate_white_games != self.config.evaluation_games // 2
                or result.damaged_games or result.no_result_games
                for result in opponent_results.values()
            ):
                raise CurriculumStateError("评测局数不完整或没有严格交换黑白")
            record["opponents"] = [
                {"model": name, "roles": roles, "match": opponent_results[name].to_dict()}
                for name, roles in expected.items()
            ]
            champion_match = opponent_results.get(champion) if champion else None
            if champion_match is not None and (
                champion_match.games == self.config.evaluation_games
                and champion_match.candidate_black_games == self.config.evaluation_games // 2
                and champion_match.candidate_white_games == self.config.evaluation_games // 2
                and not champion_match.damaged_games
                and not champion_match.no_result_games
                and champion_match.win_rate >= self.config.promotion_win_rate
                and champion_match.wilson_lower > self.config.wilson_lower_bound
            ):
                candidate.active.champion_model = candidate_model
                candidate.active.champion_established = True
                record["champion_promoted"] = True
        record["champion_after"] = candidate.active.champion_model
        if stage_definition.indefinite:
            candidate.active.consecutive_passes = 0
            candidate.phase = "training"
        else:
            candidate.active.consecutive_passes = (
                candidate.active.consecutive_passes + 1 if gate.passed else 0
            )
            candidate.phase = (
                "transition_ready"
                if candidate.active.consecutive_passes
                >= self.config.required_consecutive_passes
                else "training"
            )
        candidate.active.evaluations.append(record)
        while candidate.active.next_evaluation_sample <= candidate.active.samples:
            candidate.active.next_evaluation_sample += self.config.evaluation_interval_samples
        candidate.protected_models = list(dict.fromkeys(
            candidate.protected_models + candidate.active.fixed_baseline_models
            + ([candidate.active.champion_model] if candidate.active.champion_model else [])
        ))
        if champion and champion != candidate.active.champion_model:
            still_needed = {
                name for progress in candidate.stages.values()
                for name in [progress.entry_model, progress.baseline_model, progress.latest_model,
                             progress.champion_model, *progress.fixed_baseline_models]
            }
            if champion not in still_needed:
                candidate.protected_models = [name for name in candidate.protected_models if name != champion]
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)
        return gate

    def record_evaluation_error(
        self, state: CurriculumState, message: str
    ) -> None:
        candidate = copy.deepcopy(state)
        candidate.warnings.append(
            {"timestamp": _now(), "kind": "evaluation", "message": message}
        )
        candidate.active.evaluations.append(
            {
                "timestamp": _now(),
                "stage_samples": candidate.active.samples,
                "passed": False,
                "error": message,
            }
        )
        candidate.active.consecutive_passes = 0
        while candidate.active.next_evaluation_sample <= candidate.active.samples:
            candidate.active.next_evaluation_sample += self.config.evaluation_interval_samples
        candidate.phase = "training"
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)

    def begin_migration(self, state: CurriculumState) -> None:
        if state.phase != "transition_ready":
            raise CurriculumStateError("只有通过连续质量门槛后才能迁移")
        if state.active_stage_index >= len(self.config.stages) - 1:
            raise CurriculumStateError("19x19 是最终持续阶段")
        state.phase = "migrating"
        self.store.save(state)

    def migration_failed(self, state: CurriculumState, message: str) -> None:
        state.warnings.append(
            {"timestamp": _now(), "kind": "migration", "message": message}
        )
        # Keep training the old board and allow the next boundary to retry.
        state.active.consecutive_passes = 0
        state.phase = "training"
        state.migration = {}
        self.store.save(state)

    def complete_migration(
        self,
        state: CurriculumState,
        *,
        seed_model: str,
        autotune_batch: Optional[int] = None,
    ) -> None:
        if state.phase != "migrating":
            raise CurriculumStateError("迁移尚未开始")
        # Mutate a private candidate and only publish it to the caller after
        # the atomic state write succeeds.  This prevents a failed save from
        # leaving the in-memory object one stage ahead of durable state.
        candidate = copy.deepcopy(state)
        old = candidate.active
        old.completed_at = _now()
        candidate.active_stage_index += 1
        board = EXPECTED_BOARDS[candidate.active_stage_index]
        progress = StageProgress(
            board_size=board,
            samples=0,
            entry_model=seed_model,
            baseline_model=seed_model,
            latest_model=seed_model,
            champion_model=seed_model,
            fixed_baseline_models=[seed_model],
            next_evaluation_sample=self.config.evaluation_interval_samples,
            autotune_batch=autotune_batch,
        )
        candidate.stages[progress_key(board)] = progress
        candidate.protected_models.append(seed_model)
        candidate.protected_models = list(dict.fromkeys(candidate.protected_models))
        candidate.phase = "training"
        candidate.migration = {}
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)

    def update_disk(self, state: CurriculumState, free_gib: float) -> str:
        if free_gib < 0:
            raise ValueError("free_gib 不能为负")
        if free_gib < self.config.disk_pause_gib:
            status = "paused"
            if state.phase != "paused_disk":
                state.resume_phase = state.phase
            state.phase = "paused_disk"
        elif free_gib < self.config.disk_cleanup_gib:
            status = "cleanup"
            if state.phase == "paused_disk":
                state.phase = state.resume_phase or "training"
                state.resume_phase = None
        else:
            status = "healthy"
            if state.phase == "paused_disk":
                state.phase = state.resume_phase or "training"
                state.resume_phase = None
        state.disk = {"free_gib": free_gib, "status": status, "checked_at": _now()}
        self.store.save(state)
        return status


def progress_key(board_size: int) -> str:
    return f"{board_size}x{board_size}"


def _finite_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percent(value: object) -> str:
    number = _finite_float(value)
    return f"{number * 100:.1f}%" if number is not None else "—"


def _count_pair(left: object, right: object) -> str:
    left_number = _finite_float(left)
    right_number = _finite_float(right)
    if left_number is None or right_number is None:
        return "—"
    return f"{int(left_number):,} / {int(right_number):,}"


def render_curriculum_dashboard(
    state: CurriculumState,
    config: CurriculumConfig,
    *,
    generated_at: Optional[str] = None,
    latest_health: Optional[Mapping[str, object]] = None,
    last_training_at: Optional[str] = None,
) -> str:
    """Render the global, self-contained 9/13/19 curriculum dashboard."""

    from .rl_evaluation_dashboard import render_opponent_pools

    active = state.active
    stage_definition = config.stages[state.active_stage_index]
    minimum = stage_definition.min_stage_samples
    remaining = max(0, minimum - active.samples) if minimum is not None else None
    evaluations = active.evaluations
    latest_evaluation = evaluations[-1] if evaluations else {}
    health = latest_health if latest_health is not None else latest_evaluation.get("health", {})
    if not isinstance(health, Mapping):
        health = {}
    disk = state.disk
    disk_free = _finite_float(disk.get("free_gib"))
    disk_status_key = str(disk.get("status", "waiting"))
    disk_status = {
        "healthy": "磁盘空间正常",
        "cleanup": "磁盘偏低，正在清理",
        "paused": "磁盘不足，已暂停",
        "waiting": "等待磁盘检查",
    }.get(disk_status_key, disk_status_key)
    phase_labels = {
        "training": "训练阶段（保存状态）",
        "evaluating": "固定评测",
        "transition_ready": "等待迁移",
        "migrating": "安全迁移",
        "paused_disk": "磁盘不足，已安全暂停",
    }
    generated = generated_at or _now()
    cards = (
        ("当前棋盘", f"{active.board_size}×{active.board_size}"),
        ("课程状态", phase_labels.get(state.phase, state.phase)),
        ("阶段样本", f"{active.samples:,}"),
        ("距离最低门槛", "无限持续" if remaining is None else f"{remaining:,}"),
        ("下次两类评测", f"{active.next_evaluation_sample:,}"),
        ("连续达标", f"{active.consecutive_passes}/{config.required_consecutive_passes}"),
        ("黑方得分率", _percent(health.get("black_win_rate"))),
        (
            "黑胜 / 白胜",
            _count_pair(health.get("black_wins"), health.get("white_wins")),
        ),
        ("开局双停率", _percent(health.get("immediate_double_pass_rate"))),
        ("极端棋局率", _percent(health.get("extreme_result_rate"))),
        ("无效棋谱率", _percent(health.get("invalid_rate"))),
        ("磁盘剩余", f"{disk_free:.1f} GiB" if disk_free is not None else "—"),
    )
    cards_html = "".join(
        f"<article class=\"stat\"><span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong></article>"
        for label, value in cards
    )
    stage_rows = []
    for index, stage in enumerate(config.stages):
        progress = state.stages.get(stage.key)
        if index < state.active_stage_index:
            status = "已完成"
        elif index == state.active_stage_index:
            status = phase_labels.get(state.phase, state.phase)
        else:
            status = "等待"
        stage_rows.append(
            "<tr>"
            f"<td>{stage.board_size}×{stage.board_size}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{progress.samples:,}</td>" if progress else
            "<tr>"
            f"<td>{stage.board_size}×{stage.board_size}</td>"
            f"<td>{html.escape(status)}</td>"
            "<td>0</td>"
        )
        stage_rows[-1] += (
            f"<td>{'无限' if stage.min_stage_samples is None else f'{stage.min_stage_samples:,}'}</td>"
            f"<td>{html.escape(progress.latest_model if progress and progress.latest_model else '—')}</td>"
            "</tr>"
        )
    warning = state.warnings[-1].get("message") if state.warnings else "无"
    disk_class = " alarm" if state.phase == "paused_disk" else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15"><title>KataGo 自动课程训练</title>
<style>
:root{{--bg:#f3f6fb;--panel:#fff;--text:#172033;--muted:#6a7589;--border:#dfe5ef;--blue:#4c7df0;--green:#17a589;--red:#df5367;--grid:#e5eaf3}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f141e;--panel:#171e2b;--text:#edf2fa;--muted:#9ba8bb;--border:#2a3548;--grid:#293447}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif}}
main{{width:min(1440px,calc(100% - 28px));margin:auto;padding:26px 0 42px}}header{{display:flex;justify-content:space-between;gap:18px;align-items:flex-end;margin-bottom:18px}}
h1{{margin:0 0 5px;font-size:clamp(25px,3vw,38px)}}p{{margin:0;color:var(--muted)}}.badge{{padding:8px 12px;border-radius:999px;background:var(--panel);border:1px solid var(--border)}}.badge.alarm{{color:var(--red);font-weight:700}}
.stats{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:11px}}.stat,.panel{{background:var(--panel);border:1px solid var(--border);border-radius:15px}}
.stat{{padding:15px}}.stat span{{display:block;color:var(--muted);font-size:12px;margin-bottom:7px}}.stat strong{{font-size:clamp(17px,2vw,25px)}}.grid2{{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;margin-top:14px}}
a{{color:var(--blue)}}.panel{{padding:18px;overflow:auto}}h2{{font-size:17px;margin:0 0 12px}}svg{{display:block;width:100%;height:auto}}.grid{{stroke:var(--grid);stroke-width:1}}.threshold{{stroke:var(--red);stroke-dasharray:5 4}}.axis-label{{fill:var(--muted);font-size:11px}}.series{{fill:none;stroke:var(--blue);stroke-width:2.8}}circle{{fill:var(--blue)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted)}}.empty{{height:220px;display:grid;place-items:center;color:var(--muted)}}.empty-row{{text-align:center!important;color:var(--muted)}}.foot{{margin-top:14px;font-size:13px;line-height:1.6}}
@media(max-width:1000px){{.stats{{grid-template-columns:repeat(3,1fr)}}.grid2{{grid-template-columns:1fr}}}}@media(max-width:600px){{header{{align-items:flex-start;flex-direction:column}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<header><div><h1>KataGo 9×9 → 13×13 → 19×19</h1><p>固定评测门槛 · 滚动冠军 + 固定历史基准 · SWA 权重迁移</p></div><div class="badge{disk_class}">{html.escape(disk_status)}</div></header>
<section class="stats">{cards_html}</section>
<p class="foot">最近训练记录：{html.escape(last_training_at or '尚未读取')}。页面刷新不代表训练正在运行。</p>
<nav class="foot"><a href="../9x9/dashboard.html">9×9 损失与吞吐</a> · <a href="../13x13/dashboard.html">13×13 训练曲线（阶段启动后）</a> · <a href="../19x19/dashboard.html">19×19 训练曲线（阶段启动后）</a></nav>
{render_opponent_pools(active)}
<article class="panel" style="margin-top:16px"><h2>课程阶段</h2><table><thead><tr><th>棋盘</th><th>状态</th><th>样本</th><th>最低门槛</th><th>最新模型</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></article>
<p class="foot">最近警告：{html.escape(str(warning))}<br>生成于 {html.escape(generated)}；每 15 秒自动刷新。评测使用固定 6.5 贴目、面积计分、位置全局同形、关闭根噪声与温度且不认输。</p>
</main></body></html>"""


def write_curriculum_dashboard(
    state: CurriculumState,
    config: CurriculumConfig,
    path: Path,
) -> Path:
    """Atomically refresh the global dashboard."""

    _atomic_write_text(path, render_curriculum_dashboard(state, config))
    return path


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


def _assert_descendant(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"拒绝清理训练目录之外的路径：{resolved}") from error
    if resolved == root:
        raise ValueError("拒绝清理训练根目录")


def apply_retention_plan(plan: RetentionPlan, run_root: Path) -> None:
    """Apply a precomputed plan after validating every exact target."""

    root = run_root.resolve()
    for path in (*plan.delete_directories, *plan.delete_npz):
        _assert_descendant(path, root)
    for path in plan.delete_npz:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    for path in plan.delete_directories:
        if path.is_dir():
            shutil.rmtree(path)


@dataclass(frozen=True)
class BenchmarkMeasurement:
    batch_size: int
    success: bool
    throughput: float = 0.0
    peak_gpu_gib: float = math.inf
    oom: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class AutotuneResult:
    selected_batch_size: int
    measurements: tuple[BenchmarkMeasurement, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_batch_size": self.selected_batch_size,
            "measurements": [asdict(item) for item in self.measurements],
            "completed_at": _now(),
        }


def choose_autotune_batch(
    candidates: Sequence[int],
    benchmark: Callable[[int], BenchmarkMeasurement],
    *,
    max_gpu_memory_gib: float,
    persist_path: Optional[Path] = None,
) -> AutotuneResult:
    """Benchmark candidates, exclude OOM/>limit runs, and persist the fastest."""

    measurements: list[BenchmarkMeasurement] = []
    for batch in candidates:
        try:
            measurement = benchmark(batch)
            if measurement.batch_size != batch:
                raise ValueError("benchmark 返回了不同的 batch_size")
        except Exception as error:  # a failed candidate must not abort tuning
            message = str(error)
            measurement = BenchmarkMeasurement(
                batch_size=batch,
                success=False,
                oom="out of memory" in message.casefold() or "oom" in message.casefold(),
                error=message,
            )
        measurements.append(measurement)
    eligible = [
        item
        for item in measurements
        if item.success
        and not item.oom
        and item.throughput > 0
        and item.peak_gpu_gib <= max_gpu_memory_gib
    ]
    if not eligible:
        raise RuntimeError("没有批次通过 OOM、吞吐和峰值显存筛选")
    winner = max(eligible, key=lambda item: (item.throughput, item.batch_size))
    result = AutotuneResult(winner.batch_size, tuple(measurements))
    if persist_path is not None:
        _atomic_write_json(persist_path, result.to_dict())
    return result


def build_benchmark_command(
    *,
    python_executable: Path,
    benchmark_script: Path,
    data_file: Path,
    board_size: int,
    batch_size: int,
    gpu_index: int = 0,
    model_kind: str = "b10c128",
) -> list[str]:
    return [
        str(python_executable),
        str(benchmark_script),
        "-model-kind",
        model_kind,
        "-optimizer",
        "adam",
        "-batch-size",
        str(batch_size),
        "-data",
        str(data_file),
        "-gpu",
        str(gpu_index),
        "-pos-len",
        str(board_size),
        "-num-iters",
        "3",
        "-warmup-iters",
        "1",
        "-mode",
        "trainloop",
        "-use-bf16",
        "-use-tf32-matmul",
        "-no-compile",
    ]


def build_benchmark_input(
    source_npz: Path,
    target_npz: Path,
    *,
    minimum_rows: int,
) -> Path:
    """Tile target-board smoke data into an isolated benchmark-only NPZ."""

    import numpy as np

    if minimum_rows < 1:
        raise ValueError("minimum_rows 必须是正整数")
    with np.load(source_npz, allow_pickle=False) as source:
        if not source.files:
            raise ValueError("源 benchmark NPZ 为空")
        first = source[source.files[0]]
        if first.ndim < 1 or first.shape[0] < 1:
            raise ValueError("源 benchmark NPZ 没有数据行")
        source_rows = int(first.shape[0])
        repeats = math.ceil(minimum_rows / source_rows)
        arrays: dict[str, object] = {}
        for key in source.files:
            value = source[key]
            if value.ndim < 1 or value.shape[0] != source_rows:
                arrays[key] = value
                continue
            tiled = np.concatenate([value] * repeats, axis=0)
            arrays[key] = tiled[:minimum_rows]
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_npz.with_name(
        f".{target_npz.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target_npz)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target_npz


def parse_benchmark_output(output: str, *, batch_size: int) -> BenchmarkMeasurement:
    """Parse official benchmark_fresh_model.py trainloop output."""

    lowered = output.casefold()
    oom = "out of memory" in lowered or "cuda oom" in lowered
    throughput_match = re.search(
        r"Throughput:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+samples/s",
        output,
        flags=re.IGNORECASE,
    )
    memory_match = re.search(
        r"Peak GPU memory(?:\s*\([^)]*\))?:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB",
        output,
        flags=re.IGNORECASE,
    )
    success = throughput_match is not None and memory_match is not None and not oom
    return BenchmarkMeasurement(
        batch_size=batch_size,
        success=success,
        throughput=(
            float(throughput_match.group(1).replace(",", ""))
            if throughput_match
            else 0.0
        ),
        peak_gpu_gib=float(memory_match.group(1)) if memory_match else math.inf,
        oom=oom,
        error=None if success else "benchmark 输出不完整或发生 OOM",
    )


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
            "created_at": _now(),
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
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(clean, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    "AutotuneResult",
    "BenchmarkMeasurement",
    "CURRICULUM_SCHEMA_VERSION",
    "CurriculumConfig",
    "CurriculumConfigError",
    "CurriculumController",
    "CurriculumLockError",
    "CurriculumMigrationError",
    "CurriculumStage",
    "CurriculumState",
    "CurriculumStateError",
    "CurriculumStateStore",
    "DiskStatus",
    "GlobalCurriculumLock",
    "HealthSummary",
    "HealthThresholds",
    "MatchGame",
    "MatchSummary",
    "MigrationResult",
    "QualityGate",
    "RetentionPlan",
    "StageProgress",
    "apply_retention_plan",
    "approximate_elo",
    "build_benchmark_command",
    "build_benchmark_input",
    "build_match_config",
    "build_retention_plan",
    "choose_autotune_batch",
    "compute_health_window",
    "create_clean_swa_checkpoint",
    "disk_space_status",
    "evaluate_quality_gate",
    "load_curriculum_config",
    "next_evaluation_boundary",
    "npz_row_count",
    "parse_benchmark_output",
    "parse_sgf_game",
    "prepare_stage_migration",
    "select_replay_window",
    "select_retained_directories",
    "summarize_match_sgfs",
    "validate_training_profiles",
    "verify_checkpoint_loads",
    "wilson_interval",
    "render_curriculum_dashboard",
    "write_curriculum_dashboard",
]
