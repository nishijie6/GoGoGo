"""Isolated temporary variations for the in-game reasoning mode."""

from __future__ import annotations

from dataclasses import dataclass

from .engine import GoGame


@dataclass(frozen=True)
class ReasoningSession:
    """Keep a formal game untouched while a temporary variation is explored.

    The variation starts with an empty undo stack.  Consequently, undo can
    remove only moves made after reasoning mode was entered and can never cross
    the saved position into the formal game history.
    """

    formal_game: GoGame
    variation: GoGame
    start_move_number: int

    @classmethod
    def start(cls, game: GoGame) -> "ReasoningSession":
        """Create an isolated variation from an unfinished formal game."""

        if game.game_over:
            raise ValueError("已经结束的棋局不能开始推理模式")
        return cls(
            formal_game=game,
            variation=game.clone(),
            start_move_number=game.move_number,
        )

    @property
    def variation_move_count(self) -> int:
        """Return the number of moves currently kept in the variation."""

        return max(0, self.variation.move_number - self.start_move_number)

    def restore_formal_game(self) -> GoGame:
        """Return the exact formal game object saved at the reasoning boundary."""

        return self.formal_game
