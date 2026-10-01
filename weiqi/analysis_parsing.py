"""Validation and parsing of KataGo analysis responses into workbench values."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .analysis_model import (
    AnalysisProtocolError,
    CandidateAnalysis,
    CandidateId,
    IllegalVariationMove,
    NodeId,
    PositionAnalysis,
    VariationMove,
)
from .engine import BLACK, GoGame
from .katago import vertex_to_point


class ResponseParsing:
    """Mixed into :class:`~weiqi.analysis_workbench.AnalysisWorkbench`; uses its
    response adapter and ``_apply_variation_move``."""

    def _parse_analysis(
        self,
        node_id: NodeId,
        game: GoGame,
        position_key: str,
        response: object,
        top_k: int,
    ) -> PositionAnalysis:
        if not isinstance(response, Mapping):
            raise AnalysisProtocolError("KataGo 分析响应必须是 JSON 对象")

        root_info = response.get("rootInfo")
        if not isinstance(root_info, Mapping):
            raise AnalysisProtocolError("KataGo 响应缺少 rootInfo")
        black_winrate = self._finite_number(
            root_info.get("winrate"), "rootInfo.winrate", minimum=0.0, maximum=1.0
        )
        black_score_lead = self._finite_number(
            root_info.get("scoreLead"), "rootInfo.scoreLead"
        )
        visits = self._nonnegative_integer(root_info.get("visits"), "rootInfo.visits")

        current_player = root_info.get("currentPlayer")
        if current_player is not None:
            expected = "B" if game.current_player == BLACK else "W"
            if str(current_player).upper() != expected:
                raise AnalysisProtocolError(
                    "rootInfo.currentPlayer 与本地当前行棋方不一致"
                )

        raw_ownership = response.get("ownership")
        if not isinstance(raw_ownership, (list, tuple)):
            raise AnalysisProtocolError("KataGo 响应缺少 ownership")
        expected_ownership = game.size * game.size
        if len(raw_ownership) != expected_ownership:
            raise AnalysisProtocolError(
                f"ownership 长度应为 {expected_ownership}，实际为 {len(raw_ownership)}"
            )
        ownership = tuple(
            self._finite_number(value, f"ownership[{index}]", minimum=-1.0, maximum=1.0)
            for index, value in enumerate(raw_ownership)
        )

        raw_move_infos = response.get("moveInfos")
        if not isinstance(raw_move_infos, list) or not raw_move_infos:
            raise AnalysisProtocolError("KataGo 响应缺少 moveInfos 候选着")
        move_infos: list[Mapping[str, Any]] = []
        for index, item in enumerate(raw_move_infos):
            if not isinstance(item, Mapping):
                raise AnalysisProtocolError(f"moveInfos[{index}] 必须是对象")
            move_infos.append(item)
        move_infos.sort(
            key=lambda item: self._nonnegative_integer(
                item.get("order"), "moveInfos.order"
            )
        )

        warning_items: list[str] = []
        raw_warnings = response.get("_warnings", ())
        if isinstance(raw_warnings, str):
            warning_items.append(raw_warnings)
        elif isinstance(raw_warnings, (list, tuple)):
            warning_items.extend(str(item) for item in raw_warnings if str(item))

        candidates: list[CandidateAnalysis] = []
        for info in move_infos:
            try:
                candidate, candidate_warnings = self._parse_candidate(
                    node_id,
                    game,
                    info,
                )
            except IllegalVariationMove as error:
                warning_items.append(f"忽略非法候选：{error}")
                continue
            candidates.append(candidate)
            warning_items.extend(candidate_warnings)
            if len(candidates) >= top_k:
                break
        if not candidates:
            raise AnalysisProtocolError("KataGo 没有返回符合本地规则的合法候选着")

        fingerprint = str(
            response.get("engineFingerprint", type(self._response_adapter).__name__)
        ).strip()
        if not fingerprint:
            fingerprint = type(self._response_adapter).__name__

        return PositionAnalysis(
            node_id=node_id,
            position_key=position_key,
            black_winrate=black_winrate,
            black_score_lead=black_score_lead,
            visits=visits,
            ownership=ownership,
            candidates=tuple(candidates),
            warnings=tuple(dict.fromkeys(warning_items)),
            engine_fingerprint=fingerprint,
        )

    def _parse_candidate(
        self,
        node_id: NodeId,
        game: GoGame,
        info: Mapping[str, Any],
    ) -> tuple[CandidateAnalysis, list[str]]:
        order = self._nonnegative_integer(info.get("order"), "moveInfos.order")
        move = self._move_from_vertex(info.get("move"), game.current_player, game.size)
        replay = game.clone()
        self._apply_variation_move(replay, move)

        raw_pv = info.get("pv", [])
        if raw_pv is None:
            raw_pv = []
        if not isinstance(raw_pv, list):
            raise AnalysisProtocolError(f"候选 {order} 的 pv 必须是数组")

        warnings: list[str] = []
        pv: list[VariationMove] = [move]
        remaining_vertices = raw_pv
        if raw_pv:
            first = self._move_from_vertex(raw_pv[0], game.current_player, game.size)
            if first != move:
                raise AnalysisProtocolError(f"候选 {order} 的 pv 第一手与候选着不一致")
            remaining_vertices = raw_pv[1:]

        for ply_index, raw_vertex in enumerate(remaining_vertices, start=2):
            pv_move = self._move_from_vertex(
                raw_vertex,
                replay.current_player,
                replay.size,
            )
            try:
                self._apply_variation_move(replay, pv_move)
            except IllegalVariationMove as error:
                warnings.append(
                    f"候选 {order} 的 PV 在第 {ply_index} 手截断：{error}"
                )
                break
            pv.append(pv_move)

        black_winrate = self._finite_number(
            info.get("winrate"),
            f"moveInfos[{order}].winrate",
            minimum=0.0,
            maximum=1.0,
        )
        black_score_lead = self._finite_number(
            info.get("scoreLead"), f"moveInfos[{order}].scoreLead"
        )
        visits = self._nonnegative_integer(
            info.get("visits"), f"moveInfos[{order}].visits"
        )
        prior_value = info.get("prior")
        prior = None
        if prior_value is not None:
            prior = self._finite_number(
                prior_value,
                f"moveInfos[{order}].prior",
                minimum=0.0,
                maximum=1.0,
            )
        candidate_id = CandidateId(f"{node_id}:{self._move_token(move)}")
        return (
            CandidateAnalysis(
                id=candidate_id,
                order=order,
                move=move,
                pv=tuple(pv),
                black_winrate=black_winrate,
                black_score_lead=black_score_lead,
                visits=visits,
                prior=prior,
            ),
            warnings,
        )

    @staticmethod
    def _move_from_vertex(raw: object, color: int, size: int) -> VariationMove:
        if not isinstance(raw, str) or not raw.strip():
            raise AnalysisProtocolError("候选着坐标必须是非空字符串")
        try:
            point = vertex_to_point(raw, size)
        except (TypeError, ValueError) as error:
            raise AnalysisProtocolError(f"无法解析 KataGo 坐标 {raw!r}") from error
        if point is None:
            return VariationMove("pass", color)
        return VariationMove("play", color, point)

    @staticmethod
    def _finite_number(
        value: object,
        field: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        if isinstance(value, bool):
            raise AnalysisProtocolError(f"{field} 必须是有限数值")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise AnalysisProtocolError(f"{field} 必须是有限数值") from error
        if not math.isfinite(number):
            raise AnalysisProtocolError(f"{field} 必须是有限数值")
        if minimum is not None and number < minimum:
            raise AnalysisProtocolError(f"{field} 不能小于 {minimum}")
        if maximum is not None and number > maximum:
            raise AnalysisProtocolError(f"{field} 不能大于 {maximum}")
        return number

    @staticmethod
    def _nonnegative_integer(value: object, field: str) -> int:
        if isinstance(value, bool):
            raise AnalysisProtocolError(f"{field} 必须是非负整数")
        try:
            integer = int(value)
        except (TypeError, ValueError) as error:
            raise AnalysisProtocolError(f"{field} 必须是非负整数") from error
        if integer < 0 or isinstance(value, float) and not value.is_integer():
            raise AnalysisProtocolError(f"{field} 必须是非负整数")
        return integer

    @staticmethod
    def _move_token(move: VariationMove) -> str:
        if move.kind == "pass":
            return f"{move.color}:pass"
        assert move.point is not None
        return f"{move.color}:{move.point[0]},{move.point[1]}"
