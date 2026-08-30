"""Fast, deterministic win-rate estimates for the local Go application.

This module provides a lightweight positional estimate, not a professional Go
engine evaluation.  It combines stone safety, distance-decayed influence,
enclosed-region strength, komi, game progress, and side-to-move uncertainty.
The calculation is deterministic and fast enough to run after every move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import BLACK, EMPTY, WHITE, GoGame, Point


@dataclass(frozen=True)
class WinRateEstimate:
    """Estimated outcome and score balance from Black's perspective."""

    black_win_probability: float
    black_expected_score: float
    white_expected_score: float
    black_lead: float
    phase: str
    final: bool = False

    @property
    def white_win_probability(self) -> float:
        return 1.0 - self.black_win_probability

    @property
    def black_percent(self) -> float:
        return self.black_win_probability * 100.0

    @property
    def white_percent(self) -> float:
        return self.white_win_probability * 100.0


class WinRateEstimator:
    """Estimate win probability without modifying the supplied game state."""

    def estimate(self, game: GoGame) -> WinRateEstimate:
        if game.game_over:
            return self._final_estimate(game)
        if game.consecutive_passes == 1:
            pass_score = game.calculate_score()
            if pass_score.winner == game.current_player:
                return WinRateEstimate(
                    black_win_probability=(
                        0.98 if game.current_player == BLACK else 0.02
                    ),
                    black_expected_score=pass_score.black_total,
                    white_expected_score=pass_score.white_total,
                    black_lead=pass_score.black_total - pass_score.white_total,
                    phase="可终局",
                )

        area = game.size * game.size
        stones = self._stone_data(game)
        stability = self._stone_stability(game)
        regions = self._empty_region_data(game)
        position_evidence = (
            min(1.0, math.sqrt(len(stones) / max(1.0, area * 0.20)))
            if stones
            else 0.0
        )

        black_expected = 0.0
        influence_scale = {9: 2.0, 13: 2.35, 19: 2.7}[game.size]
        for row in range(game.size):
            for col in range(game.size):
                value = game.board[row][col]
                if value == BLACK:
                    # An endangered stone is partly discounted toward neutral,
                    # rather than assumed to be captured with certainty.
                    black_expected += 0.5 + 0.5 * stability[(row, col)]
                elif value == WHITE:
                    black_expected += 0.5 - 0.5 * stability[(row, col)]
                else:
                    black_expected += self._empty_point_ownership(
                        game,
                        (row, col),
                        stones,
                        stability,
                        regions[(row, col)],
                        influence_scale,
                        position_evidence,
                    )

        white_board_area = area - black_expected
        raw_lead = black_expected - (white_board_area + game.komi)
        occupied = len(stones)
        progress = min(
            1.0,
            max(
                occupied / area,
                game.move_number / max(1.0, area * 0.75),
            ),
        )

        # Komi compensates for Black's first-move initiative.  Influence on the
        # board gradually replaces this prior as the game becomes settled.
        initiative_prior = 6.5 * ((1.0 - progress) ** 1.4)
        tempo = (0.45 if game.current_player == BLACK else -0.45) * (
            1.0 - 0.45 * progress
        )
        correction = initiative_prior + tempo
        adjusted_lead = raw_lead + correction
        black_expected_score = black_expected + correction / 2.0
        white_expected_score = white_board_area + game.komi - correction / 2.0

        uncertainty = max(
            2.5,
            math.sqrt(area) * (0.95 - 0.55 * progress),
        )
        probability = 1.0 / (1.0 + math.exp(-adjusted_lead / uncertainty))
        probability = min(0.98, max(0.02, probability))

        if progress < 0.28:
            phase = "开局"
        elif progress < 0.72:
            phase = "中盘"
        else:
            phase = "收官"

        return WinRateEstimate(
            black_win_probability=probability,
            black_expected_score=black_expected_score,
            white_expected_score=white_expected_score,
            black_lead=adjusted_lead,
            phase=phase,
        )

    def _final_estimate(self, game: GoGame) -> WinRateEstimate:
        if game.score_result is not None:
            score = game.score_result
            black_area = score.black_total
            white_area = score.white_total
            black_lead = black_area - white_area
        else:
            score = game.calculate_score()
            black_area = score.black_total
            white_area = score.white_total
            board_lead = black_area - white_area
            if game.winner == BLACK:
                black_lead = max(0.5, abs(board_lead))
            elif game.winner == WHITE:
                black_lead = -max(0.5, abs(board_lead))
            else:
                black_lead = 0.0
            result_correction = black_lead - board_lead
            black_area += result_correction / 2.0
            white_area -= result_correction / 2.0

        if game.winner == BLACK:
            probability = 1.0
        elif game.winner == WHITE:
            probability = 0.0
        else:
            probability = 0.5
        return WinRateEstimate(
            black_win_probability=probability,
            black_expected_score=black_area,
            white_expected_score=white_area,
            black_lead=black_lead,
            phase="终局",
            final=True,
        )

    def _stone_data(self, game: GoGame) -> list[tuple[int, int, int]]:
        return [
            (row, col, game.board[row][col])
            for row in range(game.size)
            for col in range(game.size)
            if game.board[row][col] != EMPTY
        ]

    def _stone_stability(self, game: GoGame) -> dict[Point, float]:
        result: dict[Point, float] = {}
        visited: set[Point] = set()
        for row in range(game.size):
            for col in range(game.size):
                if game.board[row][col] == EMPTY or (row, col) in visited:
                    continue
                group, liberties = game.group_and_liberties(game.board, row, col)
                visited.update(group)
                liberty_count = len(liberties)
                group_color = game.board[row][col]
                eye_like_liberties = sum(
                    all(
                        game.board[neighbor_row][neighbor_col] == group_color
                        for neighbor_row, neighbor_col in game.neighbors(
                            liberty_row, liberty_col
                        )
                    )
                    for liberty_row, liberty_col in liberties
                )
                if eye_like_liberties >= 2:
                    base = 1.0
                elif liberty_count <= 1:
                    base = 0.58
                elif liberty_count == 2:
                    base = 0.78
                elif liberty_count == 3:
                    base = 0.91
                else:
                    base = 1.0
                for point in group:
                    result[point] = base
        return result

    def _empty_region_data(
        self, game: GoGame
    ) -> dict[Point, tuple[int, frozenset[int], int]]:
        """Map empties to region size, bordering colors, and boundary stones."""

        result: dict[Point, tuple[int, frozenset[int], int]] = {}
        visited: set[Point] = set()
        for start_row in range(game.size):
            for start_col in range(game.size):
                start = (start_row, start_col)
                if game.board[start_row][start_col] != EMPTY or start in visited:
                    continue
                region: set[Point] = set()
                boundary: set[Point] = set()
                border_colors: set[int] = set()
                pending = [start]
                while pending:
                    row, col = pending.pop()
                    if (row, col) in region:
                        continue
                    region.add((row, col))
                    visited.add((row, col))
                    for neighbor_row, neighbor_col in game.neighbors(row, col):
                        value = game.board[neighbor_row][neighbor_col]
                        if value == EMPTY:
                            if (neighbor_row, neighbor_col) not in region:
                                pending.append((neighbor_row, neighbor_col))
                        else:
                            boundary.add((neighbor_row, neighbor_col))
                            border_colors.add(value)
                region_data = (
                    len(region),
                    frozenset(border_colors),
                    len(boundary),
                )
                for point in region:
                    result[point] = region_data
        return result

    def _empty_point_ownership(
        self,
        game: GoGame,
        point: Point,
        stones: list[tuple[int, int, int]],
        stability: dict[Point, float],
        region: tuple[int, frozenset[int], int],
        influence_scale: float,
        position_evidence: float,
    ) -> float:
        row, col = point
        black_influence = 0.0
        white_influence = 0.0
        for stone_row, stone_col, color in stones:
            row_distance = abs(row - stone_row)
            col_distance = abs(col - stone_col)
            distance = (
                row_distance
                + col_distance
                + 0.35 * min(row_distance, col_distance)
            )
            influence = math.exp(-distance / influence_scale)
            influence *= stability[(stone_row, stone_col)]
            if color == BLACK:
                black_influence += influence
            else:
                white_influence += influence

        balance = (black_influence - white_influence) / (
            1.2 + black_influence + white_influence
        )
        ownership = 0.5 + 0.48 * balance * position_evidence

        region_size, borders, boundary_count = region
        if len(borders) == 1:
            owner = next(iter(borders))
            closure = boundary_count / (
                boundary_count + 2.0 * math.sqrt(max(1, region_size))
            )
            closure = min(0.88, closure) * position_evidence
            target = 0.98 if owner == BLACK else 0.02
            ownership = ownership * (1.0 - closure) + target * closure

        return min(0.995, max(0.005, ownership))
