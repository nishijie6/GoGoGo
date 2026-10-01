"""Strictly validated curriculum configuration (stages, gates, budgets)."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


CURRICULUM_SCHEMA_VERSION = 1
EXPECTED_BOARDS = (9, 13, 19)


class CurriculumConfigError(ValueError):
    """Raised when the curriculum configuration is invalid."""


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


__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "CurriculumConfig",
    "CurriculumConfigError",
    "CurriculumStage",
    "EXPECTED_BOARDS",
    "HealthThresholds",
    "load_curriculum_config",
    "validate_training_profiles",
]
