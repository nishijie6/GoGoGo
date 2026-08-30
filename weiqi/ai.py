"""A fast, local heuristic opponent for the Go GUI."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from .engine import BLACK, EMPTY, GoGame, MoveAnalysis, Point, ScoreResult, opponent
from .winrate import WinRateEstimator


AI_DIFFICULTIES = (
    "简单",
    "中等",
    "难",
    *(f"业余棋手{dan}段" for dan in range(1, 6)),
    *(f"职业棋手{dan}段" for dan in range(1, 10)),
)


@dataclass(frozen=True)
class AIMove:
    """The AI decision.  ``point`` is ``None`` when it chooses to pass."""

    point: Optional[Point]
    explanation: str


@dataclass(frozen=True)
class DifficultyProfile:
    """Search parameters for one relative strength tier."""

    label: str
    rank: int
    noise: float
    choice_pool: int
    temperature: float
    strategic_candidates: int
    strategic_weight: float
    reply_width: int


@dataclass
class _Candidate:
    score: float
    point: Point
    analysis: MoveAnalysis
    reasons: list[str]
    area_gain: Optional[float]


class GoAI:
    """A lightweight opponent based on tactical and positional heuristics.

    This is deliberately not a heavyweight neural-network engine.  It responds
    quickly on a 19x19 board, prefers captures and saves endangered groups,
    avoids obvious self-atari, and adds a small amount of variety.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        difficulty: str = "中等",
    ) -> None:
        if difficulty not in AI_DIFFICULTIES:
            raise ValueError(f"未知电脑难度：{difficulty}")
        self._random = random.Random(seed)
        self.difficulty = difficulty
        self.profile = self._difficulty_profile(difficulty)
        self._estimator = WinRateEstimator()

    def choose_move(self, game: GoGame) -> AIMove:
        """Choose a legal move from an independent game snapshot."""

        if game.game_over:
            return AIMove(None, "对局已经结束")

        color = game.current_player
        enemy = opponent(color)
        if game.consecutive_passes == 1:
            current_score = game.calculate_score()
            if current_score.winner == color:
                return AIMove(None, "当前盘面领先，同意结束对局")
        board_area = game.size * game.size
        occupied = sum(value != EMPTY for row in game.board for value in row)
        phase = occupied / board_area
        urgent_liberties = self._urgent_liberties(game, color)
        opening_points = set(self._opening_points(game.size))
        empty_regions = self._empty_region_map(game)
        opponent_has_passed = game.consecutive_passes == 1
        baseline_margin = (
            self._score_margin(game.calculate_score(), color)
            if opponent_has_passed
            else 0.0
        )
        scoring_game = GoGame(game.size, game.komi) if opponent_has_passed else None

        candidates: list[_Candidate] = []
        for row in range(game.size):
            for col in range(game.size):
                if game.board[row][col] != EMPTY:
                    continue
                analysis = game.analyze_move(row, col, color)
                if not analysis.legal or analysis.board is None:
                    continue

                score = self._random.uniform(
                    -self.profile.noise,
                    self.profile.noise,
                )
                reasons: list[str] = []

                if analysis.captured:
                    score += 24.0 * analysis.captured
                    reasons.append(f"提掉 {analysis.captured} 子")

                if (row, col) in urgent_liberties:
                    score += 18.0
                    reasons.append("解救己方棋块")

                adjacent_values = [
                    game.board[nr][nc] for nr, nc in game.neighbors(row, col)
                ]
                friendly_neighbors = adjacent_values.count(color)
                enemy_neighbors = adjacent_values.count(enemy)
                empty_neighbors = adjacent_values.count(EMPTY)
                score += friendly_neighbors * 1.2 + enemy_neighbors * 2.2

                if analysis.liberties == 1 and analysis.captured == 0:
                    score -= 22.0
                elif analysis.liberties == 2:
                    score -= 2.0
                else:
                    score += min(analysis.liberties, 5) * 0.45

                atari_count = self._groups_put_in_atari(
                    game, analysis, row, col, enemy
                )
                if atari_count:
                    score += atari_count * 6.0
                    reasons.append("制造打吃")

                region_size, region_borders = empty_regions[(row, col)]
                obvious_dead_invasion = (
                    region_borders == frozenset({enemy})
                    and region_size <= 6
                    and analysis.captured == 0
                    and atari_count == 0
                )
                if obvious_dead_invasion:
                    score -= 40.0

                # Opening play should spread out and prefer established star points.
                center = (game.size - 1) / 2
                distance_from_center = math.hypot(row - center, col - center)
                if phase < 0.28:
                    score -= distance_from_center * 0.12
                    if (row, col) in opening_points:
                        score += 4.5
                        reasons.append("占据要点")
                    if self._nearest_stone_distance(game, row, col) <= 1:
                        score -= 2.0

                # Filling a completely friendly neighborhood is usually wasteful.
                if adjacent_values and empty_neighbors == 0 and enemy_neighbors == 0:
                    score -= 10.0

                # A small edge penalty in the opening, relaxed later in the game.
                edge_distance = min(row, col, game.size - 1 - row, game.size - 1 - col)
                if phase < 0.4 and edge_distance == 0:
                    score -= 3.0

                area_gain: Optional[float] = None
                if scoring_game is not None:
                    scoring_game.board = [list(board_row) for board_row in analysis.board]
                    resulting_margin = self._score_margin(
                        scoring_game.calculate_score(), color
                    )
                    area_gain = resulting_margin - baseline_margin
                    if obvious_dead_invasion:
                        # Static area scoring treats every newly placed stone as
                        # alive.  A lone, non-tactical invasion of a tiny enemy
                        # eye space cannot form two eyes, so do not mistake the
                        # apparent territory reduction for a real gain.
                        area_gain = min(area_gain, 0.0)
                    # Once the opponent passes, prefer moves that still gain
                    # points and recognize filling one's own territory as zero.
                    score += area_gain * 12.0

                candidates.append(
                    _Candidate(score, (row, col), analysis, reasons, area_gain)
                )

        if not candidates:
            return AIMove(None, "没有合法落点")

        # If the opponent has passed and no legal move improves the current area
        # result, agree to finish rather than filling already secured territory.
        if opponent_has_passed:
            improving_candidates = [
                candidate
                for candidate in candidates
                if candidate.area_gain is not None and candidate.area_gain > 0.0
            ]
            if not improving_candidates:
                return AIMove(None, "盘面已收官，同意结束对局")
            candidates = improving_candidates

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        if self.profile.strategic_candidates:
            self._apply_strategic_analysis(game, color, candidates)
            candidates.sort(key=lambda candidate: candidate.score, reverse=True)

        best_score = candidates[0].score

        # The AI may also initiate a pass on an exceptionally full, settled board.
        if phase > 0.72 and best_score < 1.0:
            return AIMove(None, "盘面已接近收官")

        selected = self._select_candidate(candidates)
        if not selected.reasons:
            selected.reasons.append("兼顾棋形与气")
        return AIMove(selected.point, "、".join(selected.reasons[:2]))

    @staticmethod
    def _difficulty_profile(label: str) -> DifficultyProfile:
        rank = AI_DIFFICULTIES.index(label)
        if label == "简单":
            return DifficultyProfile(label, rank, 6.0, 0, 20.0, 0, 0.0, 0)
        if label == "中等":
            return DifficultyProfile(label, rank, 1.2, 8, 3.5, 0, 0.0, 0)
        if label == "难":
            return DifficultyProfile(label, rank, 0.6, 3, 1.4, 0, 0.0, 0)
        if label.startswith("业余"):
            dan = rank - 2
            return DifficultyProfile(
                label=label,
                rank=rank,
                noise=0.45 - 0.07 * (dan - 1),
                choice_pool=3 if dan <= 2 else (2 if dan <= 4 else 1),
                temperature=max(0.35, 1.2 - 0.18 * (dan - 1)),
                strategic_candidates=6 + 2 * dan,
                strategic_weight=0.75 + 0.16 * dan,
                reply_width=0,
            )

        dan = rank - 7
        return DifficultyProfile(
            label=label,
            rank=rank,
            noise=0.12 * (10 - dan) / 9,
            choice_pool=1,
            temperature=0.1,
            strategic_candidates=16 + 2 * dan,
            strategic_weight=1.7 + 0.18 * dan,
            reply_width=1 + (dan - 1) // 3,
        )

    def _select_candidate(self, candidates: list[_Candidate]) -> _Candidate:
        if self.profile.rank == 0:
            # Easy play deliberately considers a broad range of acceptable and
            # mediocre moves, while still excluding the worst third.
            pool_size = min(
                len(candidates),
                max(12, math.ceil(len(candidates) * 0.65)),
            )
            return self._random.choice(candidates[:pool_size])

        pool_size = min(len(candidates), self.profile.choice_pool)
        if pool_size <= 1:
            return candidates[0]
        pool = candidates[:pool_size]
        best_score = pool[0].score
        temperature = max(0.05, self.profile.temperature)
        weights = [
            math.exp(max(-40.0, (candidate.score - best_score) / temperature))
            for candidate in pool
        ]
        return self._random.choices(pool, weights=weights, k=1)[0]

    def _apply_strategic_analysis(
        self,
        game: GoGame,
        color: int,
        candidates: list[_Candidate],
    ) -> None:
        limit = min(len(candidates), self.profile.strategic_candidates)
        for candidate in candidates[:limit]:
            simulated = game.clone()
            result = simulated.play(*candidate.point)
            if not result.legal:
                continue

            estimate = self._estimator.estimate(simulated)
            strategic_lead = self._perspective_lead(estimate.black_lead, color)
            if self.profile.reply_width:
                reply_lead = self._worst_plausible_reply(
                    simulated,
                    color,
                    self.profile.reply_width,
                )
                if reply_lead is not None:
                    strategic_lead = min(strategic_lead, reply_lead)

            vulnerability = self._vulnerable_group_penalty(simulated, color)
            safety_weight = 0.65 + self.profile.rank * 0.06
            candidate.score += (
                strategic_lead * self.profile.strategic_weight
                - vulnerability * safety_weight
            )
            if self.profile.reply_width:
                candidate.reasons.append("预判对手应手")
            else:
                candidate.reasons.append("综合全局判断")

    def _worst_plausible_reply(
        self,
        game: GoGame,
        original_color: int,
        width: int,
    ) -> Optional[float]:
        reply_color = game.current_player
        urgent = self._urgent_liberties(game, reply_color)
        replies: list[tuple[float, Point]] = []
        for row in range(game.size):
            for col in range(game.size):
                if game.board[row][col] != EMPTY:
                    continue
                analysis = game.analyze_move(row, col, reply_color)
                if not analysis.legal:
                    continue
                quick_score = analysis.captured * 28.0
                if (row, col) in urgent:
                    quick_score += 18.0
                if analysis.liberties == 1 and analysis.captured == 0:
                    quick_score -= 18.0
                else:
                    quick_score += min(analysis.liberties, 5) * 0.5
                adjacent = [
                    game.board[nr][nc] for nr, nc in game.neighbors(row, col)
                ]
                quick_score += adjacent.count(original_color) * 2.2
                quick_score += adjacent.count(reply_color) * 1.1
                replies.append((quick_score, (row, col)))

        if not replies:
            return None
        replies.sort(key=lambda item: item[0], reverse=True)
        worst_lead: Optional[float] = None
        for _, point in replies[:width]:
            reply_game = game.clone()
            if not reply_game.play(*point).legal:
                continue
            estimate = self._estimator.estimate(reply_game)
            lead = self._perspective_lead(estimate.black_lead, original_color)
            worst_lead = lead if worst_lead is None else min(worst_lead, lead)
        return worst_lead

    def _vulnerable_group_penalty(self, game: GoGame, color: int) -> float:
        penalty = 0.0
        visited: set[Point] = set()
        for row in range(game.size):
            for col in range(game.size):
                if game.board[row][col] != color or (row, col) in visited:
                    continue
                group, liberties = game.group_and_liberties(game.board, row, col)
                visited.update(group)
                if len(liberties) == 1:
                    penalty += 4.0 * math.sqrt(len(group))
                elif len(liberties) == 2:
                    penalty += 0.6 * math.sqrt(len(group))
        return penalty

    @staticmethod
    def _perspective_lead(black_lead: float, color: int) -> float:
        return black_lead if color == BLACK else -black_lead

    @staticmethod
    def _score_margin(score: ScoreResult, color: int) -> float:
        if color == BLACK:
            return score.black_total - score.white_total
        return score.white_total - score.black_total

    def _urgent_liberties(self, game: GoGame, color: int) -> set[Point]:
        """Find liberties of friendly groups currently in atari."""

        urgent: set[Point] = set()
        visited: set[Point] = set()
        for row in range(game.size):
            for col in range(game.size):
                if game.board[row][col] != color or (row, col) in visited:
                    continue
                group, liberties = game.group_and_liberties(game.board, row, col)
                visited.update(group)
                if len(liberties) == 1:
                    urgent.update(liberties)
        return urgent

    def _groups_put_in_atari(
        self,
        game: GoGame,
        analysis: MoveAnalysis,
        row: int,
        col: int,
        enemy: int,
    ) -> int:
        if analysis.board is None:
            return 0
        resulting_board = [list(board_row) for board_row in analysis.board]
        seen: set[Point] = set()
        count = 0
        for neighbor_row, neighbor_col in game.neighbors(row, col):
            if resulting_board[neighbor_row][neighbor_col] != enemy:
                continue
            if (neighbor_row, neighbor_col) in seen:
                continue
            group, liberties = game.group_and_liberties(
                resulting_board, neighbor_row, neighbor_col
            )
            seen.update(group)
            if len(liberties) == 1:
                count += 1
        return count

    def _nearest_stone_distance(self, game: GoGame, row: int, col: int) -> int:
        nearest = game.size * 2
        for stone_row in range(game.size):
            for stone_col in range(game.size):
                if game.board[stone_row][stone_col] != EMPTY:
                    nearest = min(
                        nearest,
                        abs(row - stone_row) + abs(col - stone_col),
                    )
        return nearest

    def _empty_region_map(
        self, game: GoGame
    ) -> dict[Point, tuple[int, frozenset[int]]]:
        """Map each empty point to its region size and bordering colors."""

        result: dict[Point, tuple[int, frozenset[int]]] = {}
        visited: set[Point] = set()
        for start_row in range(game.size):
            for start_col in range(game.size):
                start = (start_row, start_col)
                if game.board[start_row][start_col] != EMPTY or start in visited:
                    continue
                region: set[Point] = set()
                borders: set[int] = set()
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
                            borders.add(value)
                region_info = (len(region), frozenset(borders))
                for point in region:
                    result[point] = region_info
        return result

    @staticmethod
    def _opening_points(size: int) -> tuple[Point, ...]:
        if size == 9:
            axes = (2, 4, 6)
        elif size == 13:
            axes = (3, 6, 9)
        else:
            axes = (3, 9, 15)
        return tuple((row, col) for row in axes for col in axes)
