"""Read the two training-model pools for desktop opponent selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .katago import KataGoConfigurationError, KataGoSettings
from .rl_config import load_rl_training_config
from .rl_curriculum import load_curriculum_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingOpponent:
    label: str
    model_name: str
    model_path: Path
    board_size: int
    role: str


def load_training_opponents(board_size: int, *, project_root: Path = PROJECT_ROOT,
                            curriculum_path: Path | None = None) -> dict[str, TrainingOpponent]:
    config = load_curriculum_config(curriculum_path or project_root / "config/rl_curriculum.rtx5070ti.json")
    definition = config.stage(board_size)
    state_path = config.state_root(project_root) / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    stage = state.get("stages", {}).get(definition.key, {})
    champion = stage.get("champion_model") or definition.initial_champion_model
    fixed = stage.get("fixed_baseline_models") or definition.fixed_baseline_models
    profile = load_rl_training_config(definition.config_path(project_root))
    run_root = Path(profile.runtime.output_directory)
    if not run_root.is_absolute():
        run_root = project_root / run_root
    models_root = (run_root / "models").resolve()
    rows = [("champion", champion)] + [("fixed", name) for name in fixed]
    choices = {}
    for role, name in rows:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
            continue
        model_path = (models_root / name / "model.bin.gz").resolve()
        if not model_path.is_relative_to(models_root) or not model_path.is_file():
            continue
        category = "滚动冠军" if role == "champion" else "历史基准"
        if role == "champion" and not stage.get("champion_established", False):
            category += "（暂定）"
        sample_match = re.fullmatch(r"gogogo-s(\d+)-d\d+", name)
        short_name = f"{int(sample_match[1]) / 10000:.1f}万" if sample_match else name
        label = f"训练·{category}·{short_name}"
        if label in choices:
            label += f"·{name}"
        choices[label] = TrainingOpponent(label, name, model_path, board_size, role)
    return choices


def settings_for_training_opponent(settings: KataGoSettings, opponent: TrainingOpponent,
                                   board_size: int) -> KataGoSettings:
    if board_size != opponent.board_size:
        raise KataGoConfigurationError(f"此训练模型仅用于 {opponent.board_size}×{opponent.board_size} 对局")
    selected = replace(settings, model=str(opponent.model_path), human_model="")
    selected.require_valid()
    return selected
