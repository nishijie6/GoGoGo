"""Opt-in fixed starts from the existing local joseki reference lessons.

These are experimental corner shapes, not optimal whole-board openings.
Replay every move normally to retain turn order, history and superko state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib

from ..engine import BLACK, WHITE
from ..training import JOSEKI_CATEGORY, TRAINING_LESSONS


JOSEKI_SUITE_VERSION = 1


@dataclass(frozen=True)
class TrainingOpening:
    name: str
    actions: tuple[int, ...]


@lru_cache(maxsize=3)
def joseki_openings(board_size: int) -> tuple[TrainingOpening, ...]:
    """Return both reference sequences in all eight square-board symmetries."""
    if board_size not in (9, 13, 19):
        raise ValueError("Joseki starts require a 9, 13 or 19 line board")
    openings = []
    for lesson in TRAINING_LESSONS:
        if lesson.category != JOSEKI_CATEGORY:
            continue
        sequence = (*lesson.initial_stones,
                    *((move.color, move.point) for move in lesson.moves))
        for index, (color, (row, col)) in enumerate(sequence):
            if color != (BLACK if index % 2 == 0 else WHITE):
                raise ValueError(f"Joseki reference does not alternate colors: {lesson.key}")
            if not (0 <= row < board_size and 0 <= col < board_size):
                raise ValueError(f"Joseki reference does not fit this board: {lesson.key}")
        for symmetry in range(8):
            actions = []
            for _, (row, col) in sequence:
                if symmetry >= 4:
                    row, col = col, row
                for _ in range(symmetry % 4):
                    row, col = col, board_size - 1 - row
                actions.append(row * board_size + col)
            openings.append(TrainingOpening(
                f"joseki-v{JOSEKI_SUITE_VERSION}:{lesson.key}:s{symmetry}", tuple(actions)))
    return tuple(openings)


def select_training_opening(board_size: int, seed: int, fraction: float) -> TrainingOpening | None:
    """Sample by game seed, independently of search RNG and learner color."""
    if fraction == 0:
        return None
    digest = hashlib.sha256(f"gogogo-joseki-v{JOSEKI_SUITE_VERSION}:{seed}".encode("ascii")).digest()
    if int.from_bytes(digest[:8], "big") / 2 ** 64 >= fraction:
        return None
    openings = joseki_openings(board_size)
    return openings[int.from_bytes(digest[8:16], "big") % len(openings)]
