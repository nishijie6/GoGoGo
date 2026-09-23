"""Candidate decisions on a reusable, KataGo-labeled set of 9x9 positions."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..engine import BLACK
from ..rl_config import RLTrainingConfig
from .eval_positions import canonical_hash, read_suite, read_teacher
from .search import choose_action, search
from .state import Position
from .storage import cpu_state, fingerprint


def evaluate_positions(model, runtime, config: RLTrainingConfig, suite_path: Path,
                       teacher_path: Path, *, simulations: int, check=lambda: None,
                       progress=lambda event: None) -> dict:
    if simulations < 1:
        raise ValueError("Position evaluation requires at least one simulation")
    suite = read_suite(suite_path, size=config.game.board_size, komi=config.game.komi)
    teacher = read_teacher(teacher_path, suite)
    predict = runtime.evaluator(model)
    records = []
    for index, record in enumerate(suite["positions"]):
        check()
        position = Position.new(config.game.board_size, config.game.komi)
        for color, action in record["moves"]:
            if position.game.current_player != color:
                raise ValueError(f"Invalid side to move in {record['id']}")
            position = position.play(action)
        rng = np.random.default_rng(20260923 + index)

        def infer(features):
            policies, values = predict(features[None, ...])
            return policies[0], float(values[0])

        visits, _ = search(position, infer, config.search, rng,
                           simulations=simulations, add_noise=False)
        action = choose_action(visits, 0.0, rng)
        label = teacher["positions"][record["id"]]
        probability = float(label["policy"][action])
        if probability < 0:
            raise ValueError(f"Candidate move is illegal according to KataGo in {record['id']}")
        values = label["score_lead_black_by_action"]
        preferred = values.get(str(label["top_action"]))
        chosen = values.get(str(action))
        score_delta = None
        if preferred is not None and chosen is not None:
            sign = 1 if position.game.current_player == BLACK else -1
            score_delta = sign * (preferred - chosen)
        records.append({"id": record["id"], "phase": record["phase"],
                        "action": action, "teacher_probability": probability,
                        "teacher_top_action": label["top_action"],
                        "log_loss": -math.log(max(probability, 1e-12)),
                        "score_delta": score_delta})
        if (index + 1) % 16 == 0:
            progress({"event": "position_progress", "completed": index + 1,
                      "total": len(suite["positions"])})
    groups = {}
    for phase in sorted({record["phase"] for record in records} | {"all"}):
        group = records if phase == "all" else [record for record in records if record["phase"] == phase]
        valid_score = [record["score_delta"] for record in group if record["score_delta"] is not None]
        groups[phase] = {"positions": len(group),
                         "mean_teacher_log_loss": sum(record["log_loss"] for record in group) / len(group),
                         "teacher_top_move_rate": sum(record["action"] == record["teacher_top_action"]
                                                      for record in group) / len(group),
                         "score_delta_coverage": len(valid_score),
                         "mean_score_delta_when_covered": (
                             sum(valid_score) / len(valid_score) if valid_score else None)}
    return {"suite_sha256": canonical_hash(suite),
            "teacher_labels_sha256": canonical_hash(teacher),
            "teacher_model_sha256": teacher["model_sha256"],
            "teacher_engine_sha256": teacher["engine_sha256"],
            "teacher_config_sha256": teacher.get("analysis_config_sha256"),
            "teacher_visits": teacher["visits"], "candidate_sha256": fingerprint(cpu_state(model)),
            "candidate_visits": simulations, "groups": groups, "positions": records,
            "score_delta_note": "Reported only when KataGo searched the candidate's exact move; it is never copied from rootInfo."}
