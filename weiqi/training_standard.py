"""Versioned training rules shared by the native and KataGo adapters.

Hardware profiles choose execution resources. They never choose admission
thresholds. Archived configurations without a standard retain their old rules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping


STANDARD_ID = "unified_v1"


def _rate(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class TrainingStandard:
    name: str = STANDARD_ID
    promotion_win_rate: float = .55
    board_score_floor: float = .5
    require_confidence: bool = False
    evaluation_interval_samples: int = 51_200
    screen_games_per_board: int = 12
    confirmation_games_per_board: int = 40
    evaluation_visits: int = 32
    screen_min_score: float = .5
    health_minimum_games: int = 32
    live_minimum_games: int = 24
    black_score_min: float = .1
    black_score_max: float = .9
    early_double_pass_max: float = .05
    short_game_max: float = .25
    invalid_max: float = .05
    pass_move_max: float = .2
    short_game_area_fraction: float = .1
    short_game_min_moves: int = 4
    extreme_margin_area_fraction: float = .75
    extreme_margin_rate_max: float = .1

    def to_dict(self) -> dict:
        return asdict(self)

    def promotion_reasons(self, *, score: float, by_board: Mapping,
                          complete: bool, confidence_lower=None,
                          health_reasons=()) -> tuple[str, ...]:
        """Admit complete paired evidence only; adapters verify actual pairing."""
        reasons = list(health_reasons)
        if not complete:
            reasons.append("incomplete_match")
        if not _rate(score) or score < self.promotion_win_rate:
            reasons.append("score_rate")
        if not by_board or any(not _rate(value) or value < self.board_score_floor
                               for value in by_board.values()):
            reasons.append("board_score")
        if self.require_confidence and (not _rate(confidence_lower) or confidence_lower <= .5):
            reasons.append("confidence")
        return tuple(dict.fromkeys(reasons))

    def health_reasons(self, *, games: int, black_score_rate,
                       early_double_pass_rate, invalid_rate,
                       short_game_rate=None, pass_move_rate=None,
                       minimum_games=None) -> tuple[str, ...]:
        """Check comparable self-play rates; extreme margins are diagnostic.

        Old metric records may lack short/pass counts. Adapters pass those rates
        when recorded, rather than treating unavailable observations as zero.
        """
        minimum = self.health_minimum_games if minimum_games is None else minimum_games
        reasons = []
        if type(games) is not int or games < minimum:
            reasons.append("insufficient_games")
        if (not _rate(black_score_rate)
                or not self.black_score_min <= black_score_rate <= self.black_score_max):
            reasons.append("black_score_rate")
        for name, value, maximum in (
                ("early_double_pass_rate", early_double_pass_rate, self.early_double_pass_max),
                ("invalid_rate", invalid_rate, self.invalid_max)):
            if not _rate(value) or value > maximum:
                reasons.append(name)
        for name, value, maximum in (
                ("short_game_rate", short_game_rate, self.short_game_max),
                ("pass_move_rate", pass_move_rate, self.pass_move_max)):
            if value is not None and (not _rate(value) or value > maximum):
                reasons.append(name)
        return tuple(reasons)


_UNIFIED = TrainingStandard()


def get_training_standard(name: str | None) -> TrainingStandard | None:
    if name is None:
        return None
    if name != STANDARD_ID:
        raise ValueError(f"未知训练标准：{name!r}")
    return _UNIFIED


def sample_evaluation_schedule(samples, next_samples) -> dict | None:
    """Present persisted actual-sample boundaries without estimating from rounds."""
    if (type(samples) is not int or samples < 0
            or type(next_samples) is not int or next_samples < 0):
        return None
    return {"trained_samples": samples, "next_samples": next_samples,
            "remaining_samples": max(0, next_samples - samples)}
