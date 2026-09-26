"""Color-swapped match reporting; one independent unit is an opening pair."""

from __future__ import annotations

import math

from ..engine import BLACK, WHITE


def _pair_scores(games):
    """Return one bounded outcome per complete color-swapped opening."""
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
    return scores, incomplete


def paired_confidence(games, *, confidence: float = 0.95) -> dict:
    """Conservative fixed-sample Hoeffding interval on completed pair scores.

    Every pair uses the same opening and one game as each color. Outcomes within
    a pair can be correlated; this only assumes different opening seeds are
    independent. It must not be repeatedly peeked at as an anytime-valid gate.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between zero and one")
    scores, incomplete = _pair_scores(games)
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


def paired_sign_test(games, *, confidence: float = 0.95) -> dict:
    """One-sided exact sign test across independent opening pairs, fixed N."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between zero and one")
    scores, incomplete = _pair_scores(games)
    wins = sum(score > 0.5 for score in scores)
    losses = sum(score < 0.5 for score in scores)
    ties = len(scores) - wins - losses
    decisive = wins + losses
    p_value = (sum(math.comb(decisive, count) for count in range(wins, decisive + 1))
               / 2 ** decisive) if decisive else 1.0
    return {"method": "paired_sign_fixed_sample", "confidence": confidence,
            "complete_pairs": len(scores), "truncated_pairs": incomplete,
            "wins": wins, "losses": losses, "ties": ties, "p_value": p_value}


def confirmed_improvement(summary: dict, *, threshold: float,
                          method: str = "paired_hoeffding") -> bool:
    if summary["truncated"] or summary["score_rate"] < threshold:
        return False
    if method == "paired_sign":
        sign = summary["paired_sign"]
        return bool(sign["complete_pairs"] > 0 and sign["wins"] > 0
                    and sign["p_value"] <= 1 - sign["confidence"])
    if method == "paired_hoeffding":
        paired = summary["paired"]
        return bool(paired["complete_pairs"] > 0 and paired["lower"] > 0.5)
    raise ValueError(f"Unknown promotion test: {method}")
