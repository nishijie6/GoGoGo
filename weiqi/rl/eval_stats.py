"""Color-swapped match reporting; one independent unit is an opening pair."""

from __future__ import annotations

import math

from ..engine import BLACK, WHITE


def paired_confidence(games, *, confidence: float = 0.95) -> dict:
    """Conservative fixed-sample Hoeffding interval on completed pair scores.

    Every pair uses the same opening and one game as each color. Outcomes within
    a pair can be correlated; this only assumes different opening seeds are
    independent. It must not be repeatedly peeked at as an anytime-valid gate.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between zero and one")
    by_index = {game.index: game for game in games}
    if len(by_index) != len(games):
        raise ValueError("Evaluation game indices must be unique")
    scores = []
    incomplete = 0
    for index in range(0, max(by_index, default=-1) + 1, 2):
        black, white = by_index.get(index), by_index.get(index + 1)
        if black is None or white is None:
            raise ValueError("Evaluation is missing a color-swapped opening")
        if black.seed != white.seed or (black.candidate_color, white.candidate_color) != (BLACK, WHITE):
            raise ValueError("Evaluation openings or colors are not paired")
        if black.reason == "length_limit" or white.reason == "length_limit":
            incomplete += 1
            continue
        points = 0.0
        for game in (black, white):
            points += 0.5 if game.winner is None else float(game.winner == game.candidate_color)
        scores.append(points / 2)
    if not scores:
        return {"method": "paired_hoeffding_fixed_sample", "confidence": confidence,
                "complete_pairs": 0, "truncated_pairs": incomplete,
                "mean_score": None, "lower": None, "upper": None}
    mean = sum(scores) / len(scores)
    radius = math.sqrt(math.log(2 / (1 - confidence)) / (2 * len(scores)))
    return {"method": "paired_hoeffding_fixed_sample", "confidence": confidence,
            "complete_pairs": len(scores), "truncated_pairs": incomplete,
            "mean_score": mean, "lower": max(0.0, mean - radius),
            "upper": min(1.0, mean + radius)}


def confirmed_improvement(summary: dict, *, threshold: float) -> bool:
    paired = summary["paired"]
    return bool(summary["truncated"] == 0 and paired["complete_pairs"] > 0
                and summary["score_rate"] >= threshold and paired["lower"] > 0.5)
