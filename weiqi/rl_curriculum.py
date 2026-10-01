"""Resumable curriculum orchestration primitives for KataGo reinforcement learning.

The module deliberately keeps process execution behind small callbacks.  The CLI
runner can therefore use the real KataGo executable while the safety-critical
state machine, promotion gates, retention rules, and migrations remain fast to
unit test.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .fileio import atomic_write_json, local_timestamp
from .rl_match import MatchSummary


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
    """Self-play health gates used for curriculum promotion."""

    immediate_double_pass_rate: float = 0.01
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
            "timestamp": local_timestamp(),
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
            {"timestamp": local_timestamp(), "kind": "evaluation", "message": message}
        )
        candidate.active.evaluations.append(
            {
                "timestamp": local_timestamp(),
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
            {"timestamp": local_timestamp(), "kind": "migration", "message": message}
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
        old.completed_at = local_timestamp()
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
        state.disk = {"free_gib": free_gib, "status": status, "checked_at": local_timestamp()}
        self.store.save(state)
        return status


def progress_key(board_size: int) -> str:
    return f"{board_size}x{board_size}"


__all__ = [
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
    "GlobalCurriculumLock",
    "HealthSummary",
    "HealthThresholds",
    "QualityGate",
    "StageProgress",
    "compute_health_window",
    "evaluate_quality_gate",
    "load_curriculum_config",
    "next_evaluation_boundary",
    "validate_training_profiles",
]
