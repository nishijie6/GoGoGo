"""Data model of the analysis workbench: errors, engine port, values and actions.

Everything here is immutable or a plain exception; ``AnalysisWorkbench`` in
``analysis_workbench`` owns all state and re-exports these names.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal, NewType, Optional, Protocol, Union

from .engine import BLACK, WHITE, BoardHash, GoGame, Point


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


__all__ = [
    "AnalysisConfigurationError",
    "AnalysisEngineError",
    "AnalysisProtocolError",
    "AnalysisResponsePort",
    "AnalysisSpec",
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
