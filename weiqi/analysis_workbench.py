"""Deep module for KataGo-backed analysis, history, and variation trees.

The public behaviour interface deliberately has only three entry points:
``sync_formal``, ``analyze``, and ``explore``.  Tkinter owns scheduling and
KataGo process lifetime; this module owns detached Go positions and never
mutates the formal ``GoGame`` supplied by the caller.

Its value types, errors and actions live in ``analysis_model`` (re-exported
here) and KataGo response parsing in ``analysis_parsing``.
"""

from __future__ import annotations

import hashlib
import math
import threading
import weakref
from dataclasses import dataclass
from typing import Optional

from .analysis_model import (
    AnalysisConfigurationError,
    AnalysisEngineError,
    AnalysisProtocolError,
    AnalysisResponsePort,
    AnalysisSpec,
    AnalysisWorkbenchError,
    Back,
    CandidateAnalysis,
    CandidateId,
    ExploreAction,
    FollowCandidate,
    ForeignFormalGameError,
    HistoryPoint,
    HistorySeries,
    IllegalVariationMove,
    KataGoResponseAdapter,
    NodeId,
    PlayMove,
    PositionAnalysis,
    PositionView,
    SelectNode,
    StaleAnalysis,
    TreeNodeView,
    VariationMove,
    WorkbenchStateError,
    WorkbenchView,
)
from .analysis_parsing import ResponseParsing
from .engine import BLACK, WHITE, GoGame, MoveRecord


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


class AnalysisWorkbench(ResponseParsing):
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

    @staticmethod
    def _move_key(move: VariationMove) -> tuple[object, ...]:
        return move.kind, move.color, move.point

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


__all__ = [
    "AnalysisConfigurationError",
    "AnalysisEngineError",
    "AnalysisProtocolError",
    "AnalysisResponsePort",
    "AnalysisSpec",
    "AnalysisWorkbench",
    "AnalysisWorkbenchError",
    "Back",
    "CandidateAnalysis",
    "CandidateId",
    "ExploreAction",
    "FollowCandidate",
    "ForeignFormalGameError",
    "HistoryPoint",
    "HistorySeries",
    "IllegalVariationMove",
    "KataGoResponseAdapter",
    "NodeId",
    "PlayMove",
    "PositionAnalysis",
    "PositionView",
    "SelectNode",
    "StaleAnalysis",
    "TreeNodeView",
    "VariationMove",
    "WorkbenchStateError",
    "WorkbenchView",
]
