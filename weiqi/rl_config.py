"""Validated configuration shared by the reinforcement-learning trainer.

This module intentionally has no PyTorch dependency.  The GUI and tests can
therefore read, validate, and save training choices even on a computer that is
only used to play Go.  The optional trainer consumes :class:`RLTrainingConfig`
without duplicating defaults or silently accepting misspelled options.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


RL_CONFIG_VERSION = 1
DEFAULT_RL_PRESET = "balanced"
DEFAULT_RL_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "rl_training.json"
)


class RLConfigError(ValueError):
    """Raised when a reinforcement-learning configuration is invalid."""


@dataclass(frozen=True)
class GameTrainingConfig:
    """Rules shared by self-play, evaluation, and the desktop game."""

    board_size: int
    komi: float
    scoring_rule: str
    ko_rule: str
    allow_suicide: bool


@dataclass(frozen=True)
class HardwareTrainingConfig:
    """Hardware preferences resolved by the optional PyTorch runtime."""

    device: str
    precision: str
    gpu_memory_fraction: float
    allow_tf32: bool
    compile_model: bool
    data_loader_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class NetworkTrainingConfig:
    """Size of the policy-value residual network."""

    channels: int
    residual_blocks: int
    policy_channels: int
    value_channels: int
    value_hidden_size: int


@dataclass(frozen=True)
class SearchTrainingConfig:
    """Monte Carlo tree search settings used to produce policy targets."""

    simulations_per_move: int
    c_puct: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    root_temperature: float
    temperature_moves: int


@dataclass(frozen=True)
class SelfPlayTrainingConfig:
    """Parallel self-play generation settings."""

    workers: int
    games_per_iteration: int
    inference_batch_size: int
    max_game_length_factor: float
    resign_threshold: Optional[float]
    resign_min_move: int


@dataclass(frozen=True)
class OptimizerTrainingConfig:
    """Replay-buffer and gradient-update settings."""

    batch_size: int
    replay_buffer_capacity: int
    minimum_replay_size: int
    training_steps_per_iteration: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    use_board_symmetry_augmentation: bool


@dataclass(frozen=True)
class EvaluationTrainingConfig:
    """Cheap diagnostics, frozen references, and occasional promotion matches."""

    games: int
    simulations_per_move: int
    promotion_win_rate: float
    screen_games: int
    screen_simulations_per_move: int
    screen_min_score_rate: float
    full_every_iterations: int
    pool_games: int
    pool_simulations_per_move: int
    pool_every_iterations: int
    milestone_every_iterations: int
    position_suite_path: str
    teacher_labels_path: str
    position_simulations_per_move: int
    confidence_level: float


@dataclass(frozen=True)
class RuntimeTrainingConfig:
    """Operational safeguards for running training beside the GUI."""

    pause_while_game_is_active: bool
    checkpoint_every_iterations: int
    keep_checkpoints: int
    log_every_training_steps: int
    output_directory: str
    random_seed: int


@dataclass(frozen=True)
class RLTrainingConfig:
    """A fully resolved and validated reinforcement-learning configuration."""

    schema_version: int
    preset: str
    game: GameTrainingConfig
    hardware: HardwareTrainingConfig
    network: NetworkTrainingConfig
    search: SearchTrainingConfig
    self_play: SelfPlayTrainingConfig
    optimizer: OptimizerTrainingConfig
    evaluation: EvaluationTrainingConfig
    runtime: RuntimeTrainingConfig

    @property
    def action_size(self) -> int:
        """Policy output size: every intersection plus one pass action."""

        return self.game.board_size * self.game.board_size + 1

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete resolved configuration as plain values."""

        return asdict(self)


_BALANCED_PRESET: Dict[str, Any] = {
    "game": {
        "board_size": 9,
        "komi": 6.5,
        "scoring_rule": "area",
        "ko_rule": "positional_superko",
        "allow_suicide": False,
    },
    "hardware": {
        "device": "auto",
        "precision": "auto",
        "gpu_memory_fraction": 0.60,
        "allow_tf32": True,
        "compile_model": False,
        "data_loader_workers": 2,
        "pin_memory": True,
    },
    "network": {
        "channels": 64,
        "residual_blocks": 4,
        "policy_channels": 2,
        "value_channels": 1,
        "value_hidden_size": 64,
    },
    "search": {
        "simulations_per_move": 64,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.30,
        "dirichlet_epsilon": 0.25,
        "root_temperature": 1.0,
        "temperature_moves": 20,
    },
    "self_play": {
        "workers": 2,
        "games_per_iteration": 16,
        "inference_batch_size": 8,
        "max_game_length_factor": 2.5,
        # Early models must not poison their own data with false resignations.
        "resign_threshold": None,
        "resign_min_move": 0,
    },
    "optimizer": {
        "batch_size": 128,
        "replay_buffer_capacity": 50_000,
        "minimum_replay_size": 2_048,
        "training_steps_per_iteration": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "use_board_symmetry_augmentation": True,
    },
    "evaluation": {
        "games": 20,
        "simulations_per_move": 96,
        "promotion_win_rate": 0.55,
        "screen_games": 4,
        "screen_simulations_per_move": 8,
        "screen_min_score_rate": 0.25,
        "full_every_iterations": 3,
        "pool_games": 4,
        "pool_simulations_per_move": 8,
        "pool_every_iterations": 3,
        "milestone_every_iterations": 5,
        "position_suite_path": "config/rl_eval_positions_9x9.json",
        "teacher_labels_path": "config/rl_eval_teacher_9x9.json",
        "position_simulations_per_move": 8,
        "confidence_level": 0.95,
    },
    "runtime": {
        "pause_while_game_is_active": True,
        "checkpoint_every_iterations": 1,
        "keep_checkpoints": 10,
        "log_every_training_steps": 10,
        "output_directory": "training_runs/balanced",
        "random_seed": 2026,
    },
}


_HIGH_PERFORMANCE_PRESET: Dict[str, Any] = {
    "game": {
        "board_size": 19,
        "komi": 6.5,
        "scoring_rule": "area",
        "ko_rule": "positional_superko",
        "allow_suicide": False,
    },
    "hardware": {
        # ``auto`` keeps the file loadable everywhere; a trainer should prefer
        # CUDA when present and otherwise report its chosen fallback clearly.
        "device": "auto",
        "precision": "auto",
        "gpu_memory_fraction": 0.90,
        "allow_tf32": True,
        "compile_model": True,
        "data_loader_workers": 8,
        "pin_memory": True,
    },
    "network": {
        "channels": 160,
        "residual_blocks": 12,
        "policy_channels": 4,
        "value_channels": 2,
        "value_hidden_size": 256,
    },
    "search": {
        "simulations_per_move": 400,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.03,
        "dirichlet_epsilon": 0.25,
        "root_temperature": 1.0,
        "temperature_moves": 30,
    },
    "self_play": {
        "workers": 8,
        "games_per_iteration": 64,
        "inference_batch_size": 64,
        "max_game_length_factor": 2.5,
        "resign_threshold": None,
        "resign_min_move": 0,
    },
    "optimizer": {
        "batch_size": 256,
        "replay_buffer_capacity": 500_000,
        "minimum_replay_size": 20_000,
        "training_steps_per_iteration": 500,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "use_board_symmetry_augmentation": True,
    },
    "evaluation": {
        "games": 40,
        "simulations_per_move": 800,
        "promotion_win_rate": 0.55,
        "screen_games": 4,
        "screen_simulations_per_move": 32,
        "screen_min_score_rate": 0.25,
        "full_every_iterations": 3,
        "pool_games": 4,
        "pool_simulations_per_move": 32,
        "pool_every_iterations": 3,
        "milestone_every_iterations": 5,
        "position_suite_path": "",
        "teacher_labels_path": "",
        "position_simulations_per_move": 16,
        "confidence_level": 0.95,
    },
    "runtime": {
        "pause_while_game_is_active": True,
        "checkpoint_every_iterations": 1,
        "keep_checkpoints": 20,
        "log_every_training_steps": 20,
        "output_directory": "training_runs/high_performance",
        "random_seed": 2026,
    },
}


_PRESET_DATA = {
    "balanced": _BALANCED_PRESET,
    "high_performance": _HIGH_PERFORMANCE_PRESET,
}
RL_PRESET_NAMES = tuple(_PRESET_DATA)


def _config_error(path: str, message: str) -> RLConfigError:
    return RLConfigError(f"强化学习配置 {path}：{message}")


def _require_bool(path: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise _config_error(path, "必须是 true 或 false")


def _require_int(path: str, value: Any, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _config_error(path, "必须是整数")
    if value < minimum:
        raise _config_error(path, f"不能小于 {minimum}")


def _require_number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _config_error(path, "必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise _config_error(path, "必须是有限数字")
    return number


def _require_positive_number(path: str, value: Any) -> None:
    if _require_number(path, value) <= 0:
        raise _config_error(path, "必须大于 0")


def _require_probability(path: str, value: Any, include_zero: bool = True) -> None:
    number = _require_number(path, value)
    lower_ok = number >= 0 if include_zero else number > 0
    if not lower_ok or number > 1:
        boundary = "0 到 1" if include_zero else "大于 0 且不超过 1"
        raise _config_error(path, f"必须在{boundary}之间")


def _merge_known(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
    path: str = "overrides",
) -> Dict[str, Any]:
    """Deep-merge overrides while rejecting misspelled or future-only keys."""

    result = deepcopy(dict(base))
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise _config_error(path, "字段名必须是字符串")
        key_path = f"{path}.{key}"
        if key not in base:
            raise _config_error(key_path, "未知字段")
        base_value = base[key]
        if isinstance(base_value, Mapping):
            if not isinstance(value, Mapping):
                raise _config_error(key_path, "必须是对象")
            result[key] = _merge_known(base_value, value, key_path)
        else:
            if isinstance(value, Mapping):
                raise _config_error(key_path, "不能是对象")
            result[key] = deepcopy(value)
    return result


def _build_config(preset: str, data: Mapping[str, Any]) -> RLTrainingConfig:
    config = RLTrainingConfig(
        schema_version=RL_CONFIG_VERSION,
        preset=preset,
        game=GameTrainingConfig(**data["game"]),
        hardware=HardwareTrainingConfig(**data["hardware"]),
        network=NetworkTrainingConfig(**data["network"]),
        search=SearchTrainingConfig(**data["search"]),
        self_play=SelfPlayTrainingConfig(**data["self_play"]),
        optimizer=OptimizerTrainingConfig(**data["optimizer"]),
        evaluation=EvaluationTrainingConfig(**data["evaluation"]),
        runtime=RuntimeTrainingConfig(**data["runtime"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: RLTrainingConfig) -> None:
    game = config.game
    _require_int("game.board_size", game.board_size, 1)
    if game.board_size not in (9, 13, 19):
        raise _config_error("game.board_size", "目前只支持 9、13 或 19")
    komi = _require_number("game.komi", game.komi)
    if not 0 <= komi <= 20:
        raise _config_error("game.komi", "必须在 0 到 20 之间")
    if game.scoring_rule != "area":
        raise _config_error("game.scoring_rule", "当前训练环境只支持 area")
    if game.ko_rule != "positional_superko":
        raise _config_error(
            "game.ko_rule",
            "当前训练环境只支持 positional_superko",
        )
    _require_bool("game.allow_suicide", game.allow_suicide)
    if game.allow_suicide:
        raise _config_error("game.allow_suicide", "当前规则引擎禁止自杀")

    hardware = config.hardware
    if not isinstance(hardware.device, str) or not re.fullmatch(
        r"(?:auto|cpu|mps|cuda(?::\d+)?)",
        hardware.device,
    ):
        raise _config_error(
            "hardware.device",
            "必须是 auto、cpu、mps、cuda 或 cuda:编号",
        )
    if hardware.precision not in (
        "auto",
        "float32",
        "amp_float16",
        "amp_bfloat16",
    ):
        raise _config_error(
            "hardware.precision",
            "必须是 auto、float32、amp_float16 或 amp_bfloat16",
        )
    _require_probability(
        "hardware.gpu_memory_fraction",
        hardware.gpu_memory_fraction,
        include_zero=False,
    )
    _require_bool("hardware.allow_tf32", hardware.allow_tf32)
    _require_bool("hardware.compile_model", hardware.compile_model)
    _require_int("hardware.data_loader_workers", hardware.data_loader_workers)
    _require_bool("hardware.pin_memory", hardware.pin_memory)

    network = config.network
    for name, value in (
        ("channels", network.channels),
        ("policy_channels", network.policy_channels),
        ("value_channels", network.value_channels),
        ("value_hidden_size", network.value_hidden_size),
    ):
        _require_int(f"network.{name}", value, 1)
    if network.channels % 8:
        raise _config_error("network.channels", "必须是 8 的倍数，以便高效使用 GPU")
    _require_int("network.residual_blocks", network.residual_blocks, 1)

    search = config.search
    _require_int("search.simulations_per_move", search.simulations_per_move, 1)
    _require_positive_number("search.c_puct", search.c_puct)
    _require_positive_number("search.dirichlet_alpha", search.dirichlet_alpha)
    _require_probability("search.dirichlet_epsilon", search.dirichlet_epsilon)
    if _require_number("search.root_temperature", search.root_temperature) < 0:
        raise _config_error("search.root_temperature", "不能小于 0")
    _require_int("search.temperature_moves", search.temperature_moves)

    self_play = config.self_play
    _require_int("self_play.workers", self_play.workers, 1)
    _require_int("self_play.games_per_iteration", self_play.games_per_iteration, 1)
    _require_int("self_play.inference_batch_size", self_play.inference_batch_size, 1)
    if _require_number(
        "self_play.max_game_length_factor",
        self_play.max_game_length_factor,
    ) < 1:
        raise _config_error("self_play.max_game_length_factor", "不能小于 1")
    if self_play.resign_threshold is not None:
        threshold = _require_number(
            "self_play.resign_threshold",
            self_play.resign_threshold,
        )
        if not -1 < threshold < 0:
            raise _config_error(
                "self_play.resign_threshold",
                "启用认输时必须在 -1 到 0 之间",
            )
    _require_int("self_play.resign_min_move", self_play.resign_min_move)

    optimizer = config.optimizer
    _require_int("optimizer.batch_size", optimizer.batch_size, 1)
    _require_int(
        "optimizer.replay_buffer_capacity",
        optimizer.replay_buffer_capacity,
        1,
    )
    _require_int(
        "optimizer.minimum_replay_size",
        optimizer.minimum_replay_size,
        1,
    )
    if optimizer.minimum_replay_size > optimizer.replay_buffer_capacity:
        raise _config_error(
            "optimizer.minimum_replay_size",
            "不能超过 replay_buffer_capacity",
        )
    if optimizer.batch_size > optimizer.replay_buffer_capacity:
        raise _config_error(
            "optimizer.batch_size",
            "不能超过 replay_buffer_capacity",
        )
    _require_int(
        "optimizer.training_steps_per_iteration",
        optimizer.training_steps_per_iteration,
        1,
    )
    _require_positive_number("optimizer.learning_rate", optimizer.learning_rate)
    if _require_number("optimizer.weight_decay", optimizer.weight_decay) < 0:
        raise _config_error("optimizer.weight_decay", "不能小于 0")
    _require_positive_number(
        "optimizer.gradient_clip_norm",
        optimizer.gradient_clip_norm,
    )
    _require_bool(
        "optimizer.use_board_symmetry_augmentation",
        optimizer.use_board_symmetry_augmentation,
    )

    evaluation = config.evaluation
    _require_int("evaluation.games", evaluation.games, 2)
    if evaluation.games % 2:
        raise _config_error("evaluation.games", "必须是偶数，便于双方交换黑白")
    _require_int(
        "evaluation.simulations_per_move",
        evaluation.simulations_per_move,
        1,
    )
    _require_probability("evaluation.promotion_win_rate", evaluation.promotion_win_rate)
    if evaluation.promotion_win_rate <= 0.5:
        raise _config_error("evaluation.promotion_win_rate", "必须大于 0.5")
    for name in ("screen_games", "pool_games"):
        value = getattr(evaluation, name)
        _require_int(f"evaluation.{name}", value, 2)
        if value % 2:
            raise _config_error(f"evaluation.{name}", "必须是偶数，便于双方交换黑白")
    for name in ("screen_simulations_per_move", "pool_simulations_per_move",
                 "full_every_iterations", "pool_every_iterations",
                 "milestone_every_iterations", "position_simulations_per_move"):
        _require_int(f"evaluation.{name}", getattr(evaluation, name), 1)
    _require_probability("evaluation.screen_min_score_rate", evaluation.screen_min_score_rate)
    confidence = _require_number("evaluation.confidence_level", evaluation.confidence_level)
    if not 0 < confidence < 1:
        raise _config_error("evaluation.confidence_level", "必须大于 0 且小于 1")
    for name in ("position_suite_path", "teacher_labels_path"):
        value = getattr(evaluation, name)
        if not isinstance(value, str) or "\x00" in value:
            raise _config_error(f"evaluation.{name}", "必须是有效路径字符串")
    if bool(evaluation.position_suite_path) != bool(evaluation.teacher_labels_path):
        raise _config_error("evaluation", "固定局面与 KataGo 标注必须同时设置或同时留空")
    if game.board_size != 9 and evaluation.position_suite_path:
        raise _config_error("evaluation.position_suite_path", "当前固定局面评测只支持 9×9")

    runtime = config.runtime
    _require_bool(
        "runtime.pause_while_game_is_active",
        runtime.pause_while_game_is_active,
    )
    _require_int(
        "runtime.checkpoint_every_iterations",
        runtime.checkpoint_every_iterations,
        1,
    )
    _require_int("runtime.keep_checkpoints", runtime.keep_checkpoints, 1)
    _require_int(
        "runtime.log_every_training_steps",
        runtime.log_every_training_steps,
        1,
    )
    if (
        not isinstance(runtime.output_directory, str)
        or not runtime.output_directory.strip()
    ):
        raise _config_error("runtime.output_directory", "不能为空")
    if "\x00" in runtime.output_directory:
        raise _config_error("runtime.output_directory", "不能包含空字符")
    _require_int("runtime.random_seed", runtime.random_seed)


def resolve_rl_training_config(
    preset: str = DEFAULT_RL_PRESET,
    overrides: Optional[Mapping[str, Any]] = None,
) -> RLTrainingConfig:
    """Resolve one built-in preset plus optional, strictly checked overrides."""

    if not isinstance(preset, str) or preset not in _PRESET_DATA:
        choices = "、".join(RL_PRESET_NAMES)
        raise _config_error("preset", f"未知预设 {preset!r}；可选值：{choices}")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise _config_error("overrides", "必须是对象")
    merged = _merge_known(_PRESET_DATA[preset], overrides)
    if merged["game"]["board_size"] != 9:
        evaluation_overrides = overrides.get("evaluation", {})
        if not any(key in evaluation_overrides
                   for key in ("position_suite_path", "teacher_labels_path")):
            # A board-size override should retain the working training path.
            # The bundled teacher labels belong specifically to 9x9 positions.
            merged["evaluation"]["position_suite_path"] = ""
            merged["evaluation"]["teacher_labels_path"] = ""
    return _build_config(preset, merged)


def load_rl_training_config(
    path: Union[str, Path] = DEFAULT_RL_CONFIG_PATH,
) -> RLTrainingConfig:
    """Load a small preset/override JSON file and return its resolved values."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RLConfigError(f"找不到强化学习配置文件：{config_path}") from exc
    except json.JSONDecodeError as exc:
        raise RLConfigError(
            f"强化学习配置文件不是有效 JSON：{config_path}（第 {exc.lineno} 行）"
        ) from exc
    except OSError as exc:
        raise RLConfigError(f"无法读取强化学习配置文件：{config_path}：{exc}") from exc

    if not isinstance(raw, Mapping):
        raise _config_error("根节点", "必须是对象")
    allowed_keys = {"schema_version", "preset", "overrides"}
    unknown = set(raw) - allowed_keys
    if unknown:
        key = sorted(str(item) for item in unknown)[0]
        raise _config_error(key, "未知顶层字段")
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != RL_CONFIG_VERSION
    ):
        raise _config_error(
            "schema_version",
            f"必须是 {RL_CONFIG_VERSION}",
        )
    preset = raw.get("preset")
    if not isinstance(preset, str):
        raise _config_error("preset", "必须是字符串")
    overrides = raw.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise _config_error("overrides", "必须是对象")
    return resolve_rl_training_config(preset, overrides)


def save_rl_training_config(
    path: Union[str, Path],
    preset: str = DEFAULT_RL_PRESET,
    overrides: Optional[Mapping[str, Any]] = None,
) -> RLTrainingConfig:
    """Validate and atomically save a compact preset/override JSON file."""

    resolved = resolve_rl_training_config(preset, overrides)
    saved_overrides: Mapping[str, Any] = overrides if overrides is not None else {}
    payload = {
        "schema_version": RL_CONFIG_VERSION,
        "preset": preset,
        "overrides": deepcopy(dict(saved_overrides)),
    }
    config_path = Path(path)
    temporary_path = config_path.with_name(config_path.name + ".tmp")
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(config_path)
    except (OSError, TypeError) as exc:
        raise RLConfigError(f"无法保存强化学习配置文件：{config_path}：{exc}") from exc
    return resolved
