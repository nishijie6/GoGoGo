"""Deep module for KataGo-backed analysis, history, and variation trees.

The public behaviour interface deliberately has only three entry points:
``sync_formal``, ``analyze``, and ``explore``.  Tkinter owns scheduling and
KataGo process lifetime; this module owns detached Go positions and never
mutates the formal ``GoGame`` supplied by the caller.
"""

from __future__ import annotations

import hashlib
import math
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Literal, Mapping, NewType, Optional, Protocol, Union

from .engine import BLACK, WHITE, BoardHash, GoGame, MoveRecord, Point
from .katago import vertex_to_point


NodeId = NewType("NodeId", str)
CandidateId = NewType("CandidateId", str)


class AnalysisWorkbenchError(RuntimeError):
    """Base error exposed at the analysis-workbench seam."""


class WorkbenchStateError(AnalysisWorkbenchError):
    """Raised for an action that is invalid in the current workbench state."""


class ForeignFormalGameError(WorkbenchStateError):
    """Raised when a workbench is reused for another formal ``GoGame``."""


class IllegalVariationMove(AnalysisWorkbenchError):
    """Raised when a variation move conflicts with the local rule engine."""


class AnalysisConfigurationError(AnalysisWorkbenchError):
    """Raised when the configured production response adapter is unavailable."""


class AnalysisEngineError(AnalysisWorkbenchError):
    """Raised when the external analysis dependency fails."""


class AnalysisProtocolError(AnalysisWorkbenchError):
    """Raised when an external response cannot be trusted or normalized."""


class StaleAnalysis(AnalysisWorkbenchError):
    """Raised when a response no longer matches the node that requested it."""


class AnalysisResponsePort(Protocol):
    """Internal seam for the true-external KataGo response dependency."""

    def analyze_position(
        self,
        game: GoGame,
        max_visits: int,
        pv_length: int,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        """Return one final raw KataGo-style response without mutating ``game``."""


class KataGoResponseAdapter:
    """Duck-typed production adapter for a KataGo engine.

    Construction intentionally does not invoke the external engine.  Capability
    is checked when analysis is requested, which keeps setup lazy and produces a
    stable configuration error for an incompatible engine object.
    """

    def __init__(self, engine: object) -> None:
        self._engine = engine

    def analyze_position(
        self,
        game: GoGame,
        max_visits: int,
        pv_length: int,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        method = getattr(self._engine, "analyze_position", None)
        if not callable(method):
            raise AnalysisConfigurationError(
                "当前 KataGo 引擎尚未提供 analyze_position 响应接口"
            )
        response = method(
            game,
            max_visits=max_visits,
            pv_length=pv_length,
            include_ownership=include_ownership,
            cancel_event=cancel_event,
        )
        if not isinstance(response, dict):
            raise AnalysisProtocolError("KataGo 分析响应必须是 JSON 对象")
        return response


@dataclass(frozen=True)
class AnalysisSpec:
    """Search controls that affect a cached position analysis."""

    max_visits: int = 400
    top_k: int = 5
    pv_length: int = 12
    force_refresh: bool = False

    def __post_init__(self) -> None:
        if type(self.max_visits) is not int or self.max_visits < 1:
            raise ValueError("max_visits 必须是正整数")
        if type(self.top_k) is not int or not 1 <= self.top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")
        if type(self.pv_length) is not int or not 1 <= self.pv_length <= 64:
            raise ValueError("pv_length 必须在 1 到 64 之间")
        if type(self.force_refresh) is not bool:
            raise ValueError("force_refresh 必须是布尔值")


@dataclass(frozen=True)
class VariationMove:
    """A locally meaningful move; raw GTP vertices never cross the seam."""

    kind: Literal["play", "pass"]
    color: int
    point: Optional[Point] = None

    def __post_init__(self) -> None:
        if self.color not in (BLACK, WHITE):
            raise ValueError("color 必须是 BLACK 或 WHITE")
        if self.kind == "pass":
            if self.point is not None:
                raise ValueError("虚手不能带坐标")
        elif self.kind == "play":
            if self.point is None:
                raise ValueError("落子必须带坐标")
        else:
            raise ValueError("变化着只支持 play 或 pass")


@dataclass(frozen=True)
class CandidateAnalysis:
    id: CandidateId
    order: int
    move: VariationMove
    pv: tuple[VariationMove, ...]
    black_winrate: float
    black_score_lead: float
    visits: int
    prior: Optional[float]


@dataclass(frozen=True)
class PositionAnalysis:
    node_id: NodeId
    position_key: str
    black_winrate: float
    black_score_lead: float
    visits: int
    ownership: tuple[float, ...]
    candidates: tuple[CandidateAnalysis, ...]
    warnings: tuple[str, ...]
    engine_fingerprint: str


@dataclass(frozen=True)
class HistoryPoint:
    node_id: NodeId
    ply: int
    black_winrate: float
    black_score_lead: float
    visits: int
    source: Literal["heuristic", "katago"]


@dataclass(frozen=True)
class HistorySeries:
    formal: tuple[HistoryPoint, ...]
    active: tuple[HistoryPoint, ...]


@dataclass(frozen=True)
class PositionView:
    size: int
    komi: float
    board: BoardHash
    current_player: int
    move_number: int
    last_move: Optional[Point]
    black_captures: int
    white_captures: int
    game_over: bool


@dataclass(frozen=True)
class TreeNodeView:
    id: NodeId
    parent_id: Optional[NodeId]
    incoming_move: Optional[VariationMove]
    ply: int
    is_formal: bool
    is_selected: bool
    has_analysis: bool
    child_ids: tuple[NodeId, ...]


@dataclass(frozen=True)
class WorkbenchView:
    revision: int
    formal_tip_id: NodeId
    selected_node_id: NodeId
    position: PositionView
    selected_analysis: Optional[PositionAnalysis]
    tree: tuple[TreeNodeView, ...]
    history: HistorySeries


@dataclass(frozen=True)
class SelectNode:
    node_id: NodeId


@dataclass(frozen=True)
class FollowCandidate:
    candidate_id: CandidateId
    pv_plies: int = 1

    def __post_init__(self) -> None:
        if type(self.pv_plies) is not int or self.pv_plies < 1:
            raise ValueError("pv_plies 必须是正整数")


@dataclass(frozen=True)
class PlayMove:
    move: VariationMove


@dataclass(frozen=True)
class Back:
    plies: int = 1

    def __post_init__(self) -> None:
        if type(self.plies) is not int or self.plies < 1:
            raise ValueError("plies 必须是正整数")


ExploreAction = Union[SelectNode, FollowCandidate, PlayMove, Back]


@dataclass
class _Node:
    id: NodeId
    parent_id: Optional[NodeId]
    incoming_move: Optional[VariationMove]
    game: GoGame
    creation_order: int
    is_formal: bool = False
    analysis: Optional[PositionAnalysis] = None
    analysis_spec: Optional[tuple[int, int, int]] = None
    history_estimate: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        self.children: dict[tuple[object, ...], NodeId] = {}


class AnalysisWorkbench:
    """Own detached analysis state behind a three-entry-point interface."""

    def __init__(self, response_adapter: AnalysisResponsePort) -> None:
        self._response_adapter = response_adapter
        self._lock = threading.RLock()
        self._nodes: dict[NodeId, _Node] = {}
        self._formal_path: list[NodeId] = []
        self._formal_tip_id: Optional[NodeId] = None
        self._selected_node_id: Optional[NodeId] = None
        self._formal_game_ref: Optional[weakref.ReferenceType[GoGame]] = None
        self._formal_signature: Optional[tuple[object, ...]] = None
        self._next_node_number = 0
        self._revision = 0
        self._cancel_event = threading.Event()

    def sync_formal(self, formal_game: GoGame) -> WorkbenchView:
        """Clone and mirror the same formal game without ever mutating it."""

        with self._lock:
            if (
                self._formal_game_ref is not None
                and self._formal_game_ref() is not formal_game
            ):
                raise ForeignFormalGameError(
                    "分析工作台只能同步创建它的同一盘正式棋局"
                )

            signature = self._game_signature(formal_game)
            if self._formal_signature == signature:
                return self._view()

            replayed = self._replay_formal(formal_game)
            if self._formal_game_ref is None:
                self._formal_game_ref = weakref.ref(formal_game)
                root_game = replayed[0]
                root_id = self._create_node(
                    parent_id=None,
                    incoming_move=None,
                    game=root_game,
                    is_formal=True,
                )
                self._formal_path = [root_id]
                self._formal_tip_id = root_id
                self._selected_node_id = root_id

            old_formal_tip = self._formal_tip_id
            for node in self._nodes.values():
                node.is_formal = False

            root_id = self._formal_path[0]
            root = self._nodes[root_id]
            root.is_formal = True
            root.game = replayed[0]
            root.history_estimate = None
            new_formal_path = [root_id]
            parent_id = root_id

            for move_record, game_after in zip(formal_game.moves, replayed[1:]):
                move = self._variation_from_record(move_record)
                move_key = self._move_key(move)
                parent = self._nodes[parent_id]
                child_id = parent.children.get(move_key)
                if child_id is None:
                    child_id = self._create_node(
                        parent_id=parent_id,
                        incoming_move=move,
                        game=game_after,
                        is_formal=True,
                    )
                    parent.children[move_key] = child_id
                child = self._nodes[child_id]
                child.game = game_after
                child.is_formal = True
                child.history_estimate = None
                new_formal_path.append(child_id)
                parent_id = child_id

            self._formal_path = new_formal_path
            self._formal_tip_id = new_formal_path[-1]
            if self._selected_node_id is None or self._selected_node_id == old_formal_tip:
                self._selected_node_id = self._formal_tip_id
            self._formal_signature = signature
            self._revision += 1
            return self._view()

    def analyze(
        self,
        node_id: NodeId,
        spec: AnalysisSpec = AnalysisSpec(),
    ) -> WorkbenchView:
        """Analyze one node and atomically attach a normalized response to it.

        This is the only blocking entry point and is intended to run on the
        application's existing single executor.
        """

        with self._lock:
            self._require_synced()
            node = self._nodes.get(node_id)
            if node is None:
                raise WorkbenchStateError("要分析的变化节点不存在")
            if node.game.game_over:
                raise WorkbenchStateError("已经结束的变化节点不能继续分析")
            spec_key = (spec.max_visits, spec.top_k, spec.pv_length)
            if (
                not spec.force_refresh
                and node.analysis is not None
                and node.analysis_spec == spec_key
            ):
                return self._view()
            snapshot = node.game.clone()
            position_key = self._position_key(snapshot)

        if self._cancel_event.is_set():
            raise StaleAnalysis("分析工作台已经关闭")

        try:
            response = self._response_adapter.analyze_position(
                snapshot,
                max_visits=spec.max_visits,
                pv_length=spec.pv_length,
                include_ownership=True,
                cancel_event=self._cancel_event,
            )
        except AnalysisWorkbenchError:
            raise
        except Exception as error:
            raise AnalysisEngineError(f"KataGo 分析失败：{error}") from error

        analysis = self._parse_analysis(
            node_id=node_id,
            game=snapshot,
            position_key=position_key,
            response=response,
            top_k=spec.top_k,
        )

        with self._lock:
            if self._cancel_event.is_set():
                raise StaleAnalysis("分析工作台已经关闭")
            current = self._nodes.get(node_id)
            if current is None or self._position_key(current.game) != position_key:
                raise StaleAnalysis("分析结果对应的变化节点已经失效")
            current.analysis = analysis
            current.analysis_spec = spec_key
            self._revision += 1
            return self._view()

    def cancel(self) -> None:
        """Invalidate pending external work; safe to call repeatedly on close."""

        self._cancel_event.set()

    def explore(self, action: ExploreAction) -> WorkbenchView:
        """Navigate or extend the local variation tree without external I/O."""

        with self._lock:
            self._require_synced()
            assert self._selected_node_id is not None
            selected_before = self._selected_node_id
            node_count_before = len(self._nodes)

            if isinstance(action, SelectNode):
                if action.node_id not in self._nodes:
                    raise WorkbenchStateError("选择的变化节点不存在")
                self._selected_node_id = action.node_id
            elif isinstance(action, Back):
                node_id = self._selected_node_id
                for _ in range(action.plies):
                    parent_id = self._nodes[node_id].parent_id
                    if parent_id is None:
                        break
                    node_id = parent_id
                self._selected_node_id = node_id
            elif isinstance(action, PlayMove):
                self._selected_node_id = self._follow_move(
                    self._selected_node_id,
                    action.move,
                )
            elif isinstance(action, FollowCandidate):
                selected = self._nodes[self._selected_node_id]
                analysis = selected.analysis
                if analysis is None:
                    raise WorkbenchStateError("当前节点尚无可选择的分析候选")
                candidate = next(
                    (
                        item
                        for item in analysis.candidates
                        if item.id == action.candidate_id
                    ),
                    None,
                )
                if candidate is None:
                    raise WorkbenchStateError("候选着不属于当前分析节点")
                node_id = self._selected_node_id
                for move in candidate.pv[: action.pv_plies]:
                    node_id = self._follow_move(node_id, move)
                self._selected_node_id = node_id
            else:
                raise TypeError(f"不支持的推演动作：{type(action).__name__}")

            if (
                self._selected_node_id != selected_before
                or len(self._nodes) != node_count_before
            ):
                self._revision += 1
            return self._view()

    def _require_synced(self) -> None:
        if self._cancel_event.is_set():
            raise WorkbenchStateError("分析工作台已经关闭")
        if self._formal_tip_id is None or self._selected_node_id is None:
            raise WorkbenchStateError("请先同步正式棋局")

    def _create_node(
        self,
        parent_id: Optional[NodeId],
        incoming_move: Optional[VariationMove],
        game: GoGame,
        is_formal: bool,
    ) -> NodeId:
        node_id = NodeId(f"n{self._next_node_number}")
        creation_order = self._next_node_number
        self._next_node_number += 1
        self._nodes[node_id] = _Node(
            id=node_id,
            parent_id=parent_id,
            incoming_move=incoming_move,
            game=game.clone(),
            creation_order=creation_order,
            is_formal=is_formal,
        )
        return node_id

    def _follow_move(self, parent_id: NodeId, move: VariationMove) -> NodeId:
        parent = self._nodes[parent_id]
        move_key = self._move_key(move)
        existing = parent.children.get(move_key)
        if existing is not None:
            return existing
        game_after = parent.game.clone()
        self._apply_variation_move(game_after, move)
        child_id = self._create_node(
            parent_id=parent_id,
            incoming_move=move,
            game=game_after,
            is_formal=False,
        )
        parent.children[move_key] = child_id
        return child_id

    @staticmethod
    def _apply_variation_move(game: GoGame, move: VariationMove) -> None:
        if game.game_over:
            raise IllegalVariationMove("当前变化已经结束")
        if move.color != game.current_player:
            raise IllegalVariationMove("变化着颜色与当前行棋方不一致")
        if move.kind == "pass":
            if not game.pass_turn():
                raise IllegalVariationMove("当前局面不能虚手")
            return
        assert move.point is not None
        row, col = move.point
        analysis = game.analyze_move(row, col)
        if not analysis.legal:
            raise IllegalVariationMove(analysis.reason or "该变化着不合法")
        committed = game.play(row, col)
        if not committed.legal:
            raise IllegalVariationMove(committed.reason or "该变化着不合法")

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
    def _move_key(move: VariationMove) -> tuple[object, ...]:
        return move.kind, move.color, move.point

    @staticmethod
    def _move_token(move: VariationMove) -> str:
        if move.kind == "pass":
            return f"{move.color}:pass"
        assert move.point is not None
        return f"{move.color}:{move.point[0]},{move.point[1]}"

    @staticmethod
    def _variation_from_record(record: MoveRecord) -> VariationMove:
        if record.kind == "pass":
            return VariationMove("pass", record.color)
        if record.kind == "play" and record.row is not None and record.col is not None:
            return VariationMove("play", record.color, (record.row, record.col))
        raise WorkbenchStateError("认输后的终局不能建立可分析变化树")

    def _replay_formal(self, formal_game: GoGame) -> list[GoGame]:
        replay = GoGame(formal_game.size, formal_game.komi)
        positions = [replay.clone()]
        for record in formal_game.moves:
            if record.kind == "resign":
                if record.color != replay.current_player or not replay.resign():
                    raise WorkbenchStateError("无法重放正式棋局的认输记录")
            else:
                self._apply_variation_move(replay, self._variation_from_record(record))
            positions.append(replay.clone())
        if self._game_signature(replay) != self._game_signature(formal_game):
            raise WorkbenchStateError("正式棋局无法从棋谱无损重建")
        return positions

    @staticmethod
    def _record_signature(record: MoveRecord) -> tuple[object, ...]:
        return record.kind, record.color, record.row, record.col, record.captured

    def _game_signature(self, game: GoGame) -> tuple[object, ...]:
        return (
            game.size,
            game.komi,
            game.board_hash(),
            game.current_player,
            tuple(self._record_signature(move) for move in game.moves),
            game.captures[BLACK],
            game.captures[WHITE],
            game.consecutive_passes,
            game.game_over,
            game.winner,
            game.margin,
            game.last_move,
        )

    def _position_key(self, game: GoGame) -> str:
        encoded = repr(self._game_signature(game)).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _view(self) -> WorkbenchView:
        self._require_synced()
        assert self._formal_tip_id is not None
        assert self._selected_node_id is not None
        selected = self._nodes[self._selected_node_id]
        nodes_in_order = sorted(
            self._nodes.values(), key=lambda item: item.creation_order
        )
        tree = tuple(
            TreeNodeView(
                id=node.id,
                parent_id=node.parent_id,
                incoming_move=node.incoming_move,
                ply=node.game.move_number,
                is_formal=node.is_formal,
                is_selected=node.id == self._selected_node_id,
                has_analysis=node.analysis is not None,
                child_ids=tuple(node.children.values()),
            )
            for node in nodes_in_order
        )
        position = PositionView(
            size=selected.game.size,
            komi=selected.game.komi,
            board=selected.game.board_hash(),
            current_player=selected.game.current_player,
            move_number=selected.game.move_number,
            last_move=selected.game.last_move,
            black_captures=selected.game.captures[BLACK],
            white_captures=selected.game.captures[WHITE],
            game_over=selected.game.game_over,
        )
        formal_history = tuple(
            self._history_point(self._nodes[node_id])
            for node_id in self._formal_path
        )
        active_path = self._path_to_root(self._selected_node_id)
        active_history = tuple(
            self._history_point(self._nodes[node_id])
            for node_id in active_path
        )
        return WorkbenchView(
            revision=self._revision,
            formal_tip_id=self._formal_tip_id,
            selected_node_id=self._selected_node_id,
            position=position,
            selected_analysis=selected.analysis,
            tree=tree,
            history=HistorySeries(formal=formal_history, active=active_history),
        )

    def _history_point(self, node: _Node) -> HistoryPoint:
        if node.analysis is None:
            if node.history_estimate is None:
                node.history_estimate = self._fast_history_estimate(node.game)
            probability, lead = node.history_estimate
            return HistoryPoint(
                node_id=node.id,
                ply=node.game.move_number,
                black_winrate=probability,
                black_score_lead=lead,
                visits=0,
                source="heuristic",
            )
        return HistoryPoint(
            node_id=node.id,
            ply=node.game.move_number,
            black_winrate=node.analysis.black_winrate,
            black_score_lead=node.analysis.black_score_lead,
            visits=node.analysis.visits,
            source="katago",
        )

    @staticmethod
    def _fast_history_estimate(game: GoGame) -> tuple[float, float]:
        """Return an inexpensive, labeled baseline for a complete game curve.

        This estimate intentionally uses only stone balance, terminal area,
        komi, initiative, and progress.  It is O(board area), unlike the live estimator's
        distance influence calculation, so replaying a long 19x19 record does
        not freeze Tk.  Exact KataGo results replace these points node by node.
        """

        area = game.size * game.size
        black_stones = sum(row.count(BLACK) for row in game.board)
        white_stones = sum(row.count(WHITE) for row in game.board)
        occupied = black_stones + white_stones
        if game.game_over:
            score = game.calculate_score()
            raw_lead = score.black_total - score.white_total
        else:
            # Mid-game Chinese scoring cannot treat the one huge open region
            # as settled territory merely because it currently touches one
            # color.  Stone balance is deliberately conservative; exact
            # territory enters only through KataGo or terminal scoring.
            raw_lead = float(black_stones - white_stones) - game.komi
        progress = min(
            1.0,
            max(
                occupied / max(1.0, float(area)),
                game.move_number / max(1.0, area * 0.75),
            ),
        )
        initiative = game.komi * ((1.0 - progress) ** 1.4)
        tempo = (0.35 if game.current_player == BLACK else -0.35) * (
            1.0 - 0.45 * progress
        )
        lead = raw_lead + initiative + tempo

        if game.game_over:
            if game.winner == BLACK:
                return 1.0, max(0.5, abs(lead))
            if game.winner == WHITE:
                return 0.0, -max(0.5, abs(lead))
            return 0.5, 0.0

        uncertainty = max(
            2.5,
            math.sqrt(area) * (0.95 - 0.55 * progress),
        )
        probability = 1.0 / (1.0 + math.exp(-lead / uncertainty))
        return min(0.98, max(0.02, probability)), lead

    def _path_to_root(self, node_id: NodeId) -> list[NodeId]:
        reversed_path: list[NodeId] = []
        while True:
            reversed_path.append(node_id)
            parent_id = self._nodes[node_id].parent_id
            if parent_id is None:
                break
            node_id = parent_id
        return list(reversed(reversed_path))
