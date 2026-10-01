"""Translation between local games and KataGo's JSON analysis protocol."""

from __future__ import annotations

from typing import Any, Optional

from .engine import BLACK, GoGame, MoveRecord, Point
from .katago_settings import KataGoProfile


GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def point_to_vertex(point: Point, board_size: int) -> str:
    """Convert the local top-left-based point to a GTP vertex."""

    row, col = point
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError("落点超出棋盘")
    return f"{GTP_COLUMNS[col]}{board_size - row}"


def vertex_to_point(vertex: str, board_size: int) -> Optional[Point]:
    """Convert a GTP vertex to a local point; ``pass`` becomes ``None``."""

    normalized = vertex.strip().upper()
    if normalized == "PASS":
        return None
    if len(normalized) < 2 or normalized[0] not in GTP_COLUMNS:
        raise ValueError(f"KataGo 返回了无效坐标：{vertex}")
    try:
        number = int(normalized[1:])
    except ValueError as error:
        raise ValueError(f"KataGo 返回了无效坐标：{vertex}") from error
    col = GTP_COLUMNS.index(normalized[0])
    row = board_size - number
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"KataGo 返回了棋盘外坐标：{vertex}")
    return row, col


def _move_to_protocol(move: MoveRecord, board_size: int) -> list[str]:
    color = "B" if move.color == BLACK else "W"
    if move.kind == "pass":
        return [color, "pass"]
    if move.kind != "play" or move.row is None or move.col is None:
        raise ValueError("认输记录不能发送给 KataGo 继续分析")
    return [color, point_to_vertex((move.row, move.col), board_size)]


def build_analysis_query(
    game: GoGame,
    profile: KataGoProfile,
    request_id: str,
    human_style: bool,
    *,
    max_visits: Optional[int] = None,
    pv_length: Optional[int] = None,
    include_ownership: bool = False,
) -> dict[str, Any]:
    """Build one official KataGo JSON analysis request from a game snapshot."""

    query: dict[str, Any] = {
        "id": request_id,
        "moves": [_move_to_protocol(move, game.size) for move in game.moves],
        "rules": {
            "ko": "POSITIONAL",
            "scoring": "AREA",
            "tax": "NONE",
            "suicide": False,
            "hasButton": False,
            "whiteHandicapBonus": "N",
            "friendlyPassOk": False,
        },
        "komi": game.komi,
        "boardXSize": game.size,
        "boardYSize": game.size,
        "maxVisits": profile.max_visits if max_visits is None else max_visits,
        "analysisPVLen": 8 if pv_length is None else pv_length,
    }
    if include_ownership:
        query["includeOwnership"] = True
    if not game.moves:
        query["initialPlayer"] = "B" if game.current_player == BLACK else "W"
    if human_style:
        query["includePolicy"] = True
        override_settings: dict[str, Any] = {
            "humanSLProfile": profile.human_sl_profile,
            "ignorePreRootHistory": False,
        }
        if profile.selection_mode == "professional":
            override_settings.update(
                {
                    "humanSLRootExploreProbWeightless": 0.5,
                    "humanSLCpuctPermanent": 2.0,
                }
            )
        query["overrideSettings"] = override_settings
    return query


__all__ = [
    "GTP_COLUMNS",
    "build_analysis_query",
    "point_to_vertex",
    "vertex_to_point",
]
