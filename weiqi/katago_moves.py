"""Turning a KataGo analysis response into one move for a professional or HumanSL tier."""

from __future__ import annotations

import math
from typing import Any, Optional

from .ai import AIMove
from .engine import BLACK, GoGame, Point
from .katago_protocol import point_to_vertex, vertex_to_point
from .katago_settings import KataGoEngineError, KataGoProfile


class MoveSelection:
    """Mixed into :class:`~weiqi.katago.KataGoEngine`; uses its seeded ``_random``."""

    def _decision_from_response(
        self,
        game: GoGame,
        profile: KataGoProfile,
        response: dict[str, Any],
        human_style_requested: bool,
    ) -> AIMove:
        move_infos = response.get("moveInfos")
        if not isinstance(move_infos, list):
            move_infos = []
        move_infos = [info for info in move_infos if isinstance(info, dict)]
        move_infos.sort(key=lambda info: int(info.get("order", 999999)))
        if profile.selection_mode == "human_rank" and not move_infos:
            raise KataGoEngineError(
                "HumanSL 请求缺少用于判断虚手的普通搜索结果"
            )

        top_vertex = str(move_infos[0].get("move", "pass")) if move_infos else "pass"
        human_policy = response.get("humanPolicy")
        has_human_policy = (
            isinstance(human_policy, list)
            and len(human_policy) == game.size * game.size + 1
        )
        has_human_priors = any(
            (self._optional_float(info.get("humanPrior")) or 0.0) > 0.0
            for info in move_infos
        )
        used_human_style = (
            human_style_requested
            and (has_human_priors or has_human_policy)
        )
        if profile.selection_mode == "human_rank" and not human_style_requested:
            raise KataGoEngineError("HumanSL 人类段位请求缺少人类风格模型")
        if profile.selection_mode == "human_rank" and not has_human_policy:
            raise KataGoEngineError(
                "HumanSL 没有返回长度与棋盘匹配的人类策略"
            )

        if top_vertex.strip().upper() == "PASS":
            point: Optional[Point] = None
        elif used_human_style:
            point = None
            if profile.selection_mode == "human_rank" and has_human_policy:
                assert isinstance(human_policy, list)
                point = self._sample_human_policy(game, human_policy, profile)
                if point is None:
                    raise KataGoEngineError(
                        "HumanSL 没有返回可采样的合法非虚手落点"
                    )
            elif profile.selection_mode == "professional":
                point = self._sample_professional_moves(game, move_infos, profile)
                if point is None and has_human_policy:
                    assert isinstance(human_policy, list)
                    point = self._sample_human_policy(game, human_policy, profile)
            if point is None:
                point = self._first_legal_candidate(game, move_infos)
        else:
            point = self._first_legal_candidate(game, move_infos)

        if point is not None and not game.analyze_move(*point).legal:
            point = self._first_legal_candidate(game, move_infos)
        if point is None and top_vertex.strip().upper() != "PASS":
            # If every returned move conflicts with the local rules, do not
            # disguise the protocol mismatch as an intentional pass.
            raise KataGoEngineError("KataGo 没有返回与本程序规则一致的合法落点")

        root_info = response.get("rootInfo")
        if not isinstance(root_info, dict):
            root_info = {}
        selected_vertex = "pass" if point is None else point_to_vertex(point, game.size)
        selected_info = next(
            (
                info
                for info in move_infos
                if str(info.get("move", "")).upper() == selected_vertex.upper()
            ),
            None,
        )
        if (
            profile.selection_mode == "human_rank"
            and point is not None
            and selected_info is None
        ):
            evaluation: dict[str, Any] = {}
        else:
            evaluation = selected_info if selected_info is not None else root_info
        black_win_probability = self._optional_probability(evaluation.get("winrate"))
        black_lead = self._optional_float(evaluation.get("scoreLead"))
        visits = int(root_info.get("visits", profile.max_visits) or 0)

        if point is None:
            reason = f"KataGo 判断当前应当虚手，完成约 {visits} 次搜索"
        elif profile.selection_mode == "training":
            reason = f"{profile.label}，完成约 {visits} 次搜索并选择最高评价点"
        elif used_human_style and profile.selection_mode == "human_rank":
            reason = (
                f"{profile.label}，按该水平人类棋谱策略采样；"
                f"普通搜索完成约 {visits} 次访问并负责判断虚手"
            )
        elif used_human_style:
            assert profile.dan is not None
            reason = (
                "KataGo 人类风格模型按 2023 职业棋谱风格选点，"
                f"以模拟职业 {profile.dan} 段参数完成约 {visits} 次局面核验"
            )
        else:
            reason = (
                f"KataGo 主网络完成约 {visits} 次搜索并选择最高评价点；"
                "未配置人类风格模型"
            )
        return AIMove(
            point=point,
            explanation=reason,
            black_win_probability=black_win_probability,
            black_lead=black_lead,
            analysis_visits=visits,
        )

    def _sample_professional_moves(
        self,
        game: GoGame,
        move_infos: list[dict[str, Any]],
        profile: KataGoProfile,
    ) -> Optional[Point]:
        """Blend professional-game priors with KataGo's searched evaluation."""

        candidates: list[tuple[Point, float, float]] = []
        for info in move_infos:
            human_prior = self._optional_float(info.get("humanPrior"))
            utility = self._optional_float(info.get("utility"))
            if human_prior is None or human_prior <= 0.0 or utility is None:
                continue
            try:
                point = vertex_to_point(str(info.get("move", "")), game.size)
            except ValueError:
                continue
            if point is None or not game.analyze_move(*point).legal:
                continue
            perspective_utility = utility if game.current_player == BLACK else -utility
            candidates.append((point, human_prior, perspective_utility))
        if not candidates:
            return None

        best_utility = max(utility for _, _, utility in candidates)
        weighted_points: list[tuple[Point, float]] = []
        for point, human_prior, utility in candidates:
            log_weight = (
                math.log(human_prior) / profile.move_temperature
                + (utility - best_utility) / profile.utility_scale
            )
            weighted_points.append((point, math.exp(max(-60.0, log_weight))))
        return self._weighted_choice(weighted_points)

    def _sample_human_policy(
        self,
        game: GoGame,
        policy: list[Any],
        profile: KataGoProfile,
    ) -> Optional[Point]:
        weighted_points: list[tuple[Point, float]] = []
        for index, raw_weight in enumerate(policy[:-1]):
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(weight) or weight <= 0:
                continue
            point = (index // game.size, index % game.size)
            if game.analyze_move(*point).legal:
                weighted_points.append(
                    (point, weight ** (1.0 / profile.move_temperature))
                )
        return self._weighted_choice(weighted_points)

    def _weighted_choice(
        self,
        weighted_points: list[tuple[Point, float]],
    ) -> Optional[Point]:
        total = sum(weight for _, weight in weighted_points)
        if total <= 0:
            return None
        threshold = self._random.random() * total
        cumulative = 0.0
        for point, weight in weighted_points:
            cumulative += weight
            if cumulative >= threshold:
                return point
        return weighted_points[-1][0]

    @staticmethod
    def _first_legal_candidate(
        game: GoGame,
        move_infos: list[dict[str, Any]],
    ) -> Optional[Point]:
        for info in move_infos:
            vertex = str(info.get("move", ""))
            try:
                point = vertex_to_point(vertex, game.size)
            except ValueError:
                continue
            if point is None:
                return None
            if game.analyze_move(*point).legal:
                return point
        return None

    @staticmethod
    def _optional_probability(value: Any) -> Optional[float]:
        number = MoveSelection._optional_float(value)
        if number is None:
            return None
        return max(0.0, min(1.0, number))

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
