"""Go (Weiqi) rules engine used by the Tkinter application.

The engine intentionally contains no GUI code so that the rules can be tested
and reused independently.  Coordinates are zero based: ``(row, column)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, Optional


EMPTY = 0
BLACK = 1
WHITE = 2

Point = tuple[int, int]
BoardHash = tuple[tuple[int, ...], ...]
MoveKind = Literal["play", "pass", "resign"]


def opponent(color: int) -> int:
    """Return the opposite stone color."""

    if color not in (BLACK, WHITE):
        raise ValueError("棋子颜色必须是 BLACK 或 WHITE")
    return WHITE if color == BLACK else BLACK


def color_name(color: int) -> str:
    """Return a Chinese display name for a stone color."""

    return "黑方" if color == BLACK else "白方"


@dataclass(frozen=True)
class MoveAnalysis:
    """Result of checking a potential move without modifying the game."""

    legal: bool
    reason: str = ""
    board: Optional[BoardHash] = None
    captured: int = 0
    liberties: int = 0


@dataclass(frozen=True)
class MoveRecord:
    """A committed move, pass, or resignation."""

    kind: MoveKind
    color: int
    row: Optional[int] = None
    col: Optional[int] = None
    captured: int = 0


@dataclass(frozen=True)
class ScoreResult:
    """Chinese area-scoring result."""

    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    neutral_points: int
    komi: float
    black_total: float
    white_total: float
    winner: Optional[int]
    margin: float

    @property
    def summary(self) -> str:
        if self.winner is None:
            outcome = "和棋"
        else:
            outcome = f"{color_name(self.winner)}胜 {self.margin:g} 目"
        return (
            f"{outcome}｜黑 {self.black_total:g}（子 {self.black_stones} + 地 "
            f"{self.black_territory}）｜白 {self.white_total:g}（子 "
            f"{self.white_stones} + 地 {self.white_territory} + 贴目 "
            f"{self.komi:g}）"
        )


@dataclass(frozen=True)
class _Snapshot:
    board: BoardHash
    current_player: int
    black_captures: int
    white_captures: int
    consecutive_passes: int
    game_over: bool
    result_text: str
    winner: Optional[int]
    margin: float
    last_move: Optional[Point]
    position_history: frozenset[BoardHash]
    move_count: int
    score_result: Optional[ScoreResult]


class GoGame:
    """A complete local Go game using Chinese area scoring.

    Implemented rules:

    * captures and liberty checking;
    * suicide prohibition;
    * positional superko (a move may not recreate an earlier board);
    * pass, two-pass ending, resignation, and undo;
    * Chinese area scoring with configurable komi.
    """

    SUPPORTED_SIZES = (9, 13, 19)

    def __init__(
        self, size: int = 19, komi: float = 6.5, *, record_undo: bool = True
    ) -> None:
        if size not in self.SUPPORTED_SIZES:
            raise ValueError(f"棋盘大小必须是 {self.SUPPORTED_SIZES} 之一")

        self.size = size
        self._record_undo = record_undo
        self.komi = float(komi)
        self.board: list[list[int]] = [
            [EMPTY for _ in range(size)] for _ in range(size)
        ]
        self.current_player = BLACK
        self.captures = {BLACK: 0, WHITE: 0}
        self.consecutive_passes = 0
        self.game_over = False
        self.result_text = ""
        self.winner: Optional[int] = None
        self.margin = 0.0
        self.last_move: Optional[Point] = None
        self.score_result: Optional[ScoreResult] = None
        self.moves: list[MoveRecord] = []
        self._position_history: set[BoardHash] = {self.board_hash()}
        self._undo_stack: list[_Snapshot] = []

    @property
    def move_number(self) -> int:
        return len(self.moves)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def board_hash(self, board: Optional[Iterable[Iterable[int]]] = None) -> BoardHash:
        source = self.board if board is None else board
        return tuple(tuple(row) for row in source)

    def clone(self) -> "GoGame":
        """Return an independent copy suitable for AI analysis."""

        clone = GoGame(self.size, self.komi, record_undo=self._record_undo)
        clone.board = [row[:] for row in self.board]
        clone.current_player = self.current_player
        clone.captures = dict(self.captures)
        clone.consecutive_passes = self.consecutive_passes
        clone.game_over = self.game_over
        clone.result_text = self.result_text
        clone.winner = self.winner
        clone.margin = self.margin
        clone.last_move = self.last_move
        clone.score_result = self.score_result
        clone.moves = list(self.moves)
        clone._position_history = set(self._position_history)
        # Analysis copies do not need to inherit the user's undo stack.
        clone._undo_stack = []
        return clone

    def neighbors(self, row: int, col: int) -> Iterator[Point]:
        if row > 0:
            yield row - 1, col
        if row + 1 < self.size:
            yield row + 1, col
        if col > 0:
            yield row, col - 1
        if col + 1 < self.size:
            yield row, col + 1

    def group_and_liberties(
        self,
        board: list[list[int]],
        row: int,
        col: int,
    ) -> tuple[set[Point], set[Point]]:
        """Return the connected group and its distinct liberties."""

        color = board[row][col]
        if color == EMPTY:
            return set(), set()

        group: set[Point] = set()
        liberties: set[Point] = set()
        pending = [(row, col)]
        while pending:
            point = pending.pop()
            if point in group:
                continue
            group.add(point)
            point_row, point_col = point
            for neighbor_row, neighbor_col in self.neighbors(point_row, point_col):
                value = board[neighbor_row][neighbor_col]
                if value == EMPTY:
                    liberties.add((neighbor_row, neighbor_col))
                elif value == color and (neighbor_row, neighbor_col) not in group:
                    pending.append((neighbor_row, neighbor_col))
        return group, liberties

    def analyze_move(
        self,
        row: int,
        col: int,
        color: Optional[int] = None,
    ) -> MoveAnalysis:
        """Check a move and return its resulting position without committing it."""

        move_color = self.current_player if color is None else color
        if move_color not in (BLACK, WHITE):
            return MoveAnalysis(False, "无效的棋子颜色")
        if self.game_over:
            return MoveAnalysis(False, "本局已经结束")
        if not (0 <= row < self.size and 0 <= col < self.size):
            return MoveAnalysis(False, "落点超出棋盘")
        if self.board[row][col] != EMPTY:
            return MoveAnalysis(False, "该交叉点已有棋子")

        next_board = [board_row[:] for board_row in self.board]
        next_board[row][col] = move_color
        enemy = opponent(move_color)
        captured_points: set[Point] = set()
        checked_enemy: set[Point] = set()

        for neighbor_row, neighbor_col in self.neighbors(row, col):
            if next_board[neighbor_row][neighbor_col] != enemy:
                continue
            if (neighbor_row, neighbor_col) in checked_enemy:
                continue
            group, liberties = self.group_and_liberties(
                next_board, neighbor_row, neighbor_col
            )
            checked_enemy.update(group)
            if not liberties:
                captured_points.update(group)

        for captured_row, captured_col in captured_points:
            next_board[captured_row][captured_col] = EMPTY

        _, own_liberties = self.group_and_liberties(next_board, row, col)
        if not own_liberties:
            return MoveAnalysis(False, "禁入点：落子后己方棋块没有气")

        next_hash = self.board_hash(next_board)
        if next_hash in self._position_history:
            return MoveAnalysis(False, "劫争禁着：该落子会重复之前的局面")

        return MoveAnalysis(
            True,
            board=next_hash,
            captured=len(captured_points),
            liberties=len(own_liberties),
        )

    def play(self, row: int, col: int) -> MoveAnalysis:
        """Play a stone for the current player if the move is legal."""

        analysis = self.analyze_move(row, col)
        if not analysis.legal or analysis.board is None:
            return analysis

        self._push_snapshot()
        color = self.current_player
        self.board = [list(board_row) for board_row in analysis.board]
        self.captures[color] += analysis.captured
        self.consecutive_passes = 0
        self.last_move = (row, col)
        self.score_result = None
        self.result_text = ""
        self.winner = None
        self.margin = 0.0
        self.moves.append(
            MoveRecord("play", color, row=row, col=col, captured=analysis.captured)
        )
        self._position_history.add(analysis.board)
        self.current_player = opponent(color)
        return analysis

    def pass_turn(self) -> bool:
        """Pass for the current player; two consecutive passes end the game."""

        if self.game_over:
            return False

        self._push_snapshot()
        color = self.current_player
        self.moves.append(MoveRecord("pass", color))
        self.last_move = None
        self.consecutive_passes += 1
        self.current_player = opponent(color)
        if self.consecutive_passes >= 2:
            self.game_over = True
            self.score_result = self.calculate_score()
            self.winner = self.score_result.winner
            self.margin = self.score_result.margin
            self.result_text = self.score_result.summary
        return True

    def resign(self) -> bool:
        """Resign for the current player."""

        if self.game_over:
            return False

        self._push_snapshot()
        color = self.current_player
        self.moves.append(MoveRecord("resign", color))
        self.game_over = True
        self.winner = opponent(color)
        self.margin = 0.0
        self.last_move = None
        self.score_result = None
        self.result_text = f"{color_name(color)}认输，{color_name(self.winner)}胜"
        return True

    def undo(self, steps: int = 1) -> int:
        """Undo up to ``steps`` actions and return the number actually undone."""

        if steps < 1:
            return 0
        undone = 0
        while undone < steps and self._undo_stack:
            snapshot = self._undo_stack.pop()
            self.board = [list(row) for row in snapshot.board]
            self.current_player = snapshot.current_player
            self.captures = {
                BLACK: snapshot.black_captures,
                WHITE: snapshot.white_captures,
            }
            self.consecutive_passes = snapshot.consecutive_passes
            self.game_over = snapshot.game_over
            self.result_text = snapshot.result_text
            self.winner = snapshot.winner
            self.margin = snapshot.margin
            self.last_move = snapshot.last_move
            self._position_history = set(snapshot.position_history)
            self.score_result = snapshot.score_result
            del self.moves[snapshot.move_count :]
            undone += 1
        return undone

    def legal_moves(self, color: Optional[int] = None) -> Iterator[Point]:
        """Yield every legal board move for ``color`` (passes are not included)."""

        move_color = self.current_player if color is None else color
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == EMPTY:
                    if self.analyze_move(row, col, move_color).legal:
                        yield row, col

    def calculate_score(self) -> ScoreResult:
        """Score the current position with Chinese area scoring.

        Empty regions bordered by only one color count as that color's territory.
        Empty regions touching both colors are neutral.  Players should capture
        dead stones before passing because automatic life-and-death adjudication
        is intentionally outside the scope of this local game.
        """

        black_stones = sum(row.count(BLACK) for row in self.board)
        white_stones = sum(row.count(WHITE) for row in self.board)
        black_territory = 0
        white_territory = 0
        neutral_points = 0
        visited: set[Point] = set()

        for start_row in range(self.size):
            for start_col in range(self.size):
                if self.board[start_row][start_col] != EMPTY:
                    continue
                if (start_row, start_col) in visited:
                    continue

                region: set[Point] = set()
                borders: set[int] = set()
                pending = [(start_row, start_col)]
                while pending:
                    row, col = pending.pop()
                    if (row, col) in region:
                        continue
                    region.add((row, col))
                    visited.add((row, col))
                    for neighbor_row, neighbor_col in self.neighbors(row, col):
                        value = self.board[neighbor_row][neighbor_col]
                        if value == EMPTY and (neighbor_row, neighbor_col) not in region:
                            pending.append((neighbor_row, neighbor_col))
                        elif value in (BLACK, WHITE):
                            borders.add(value)

                if borders == {BLACK}:
                    black_territory += len(region)
                elif borders == {WHITE}:
                    white_territory += len(region)
                else:
                    neutral_points += len(region)

        black_total = float(black_stones + black_territory)
        white_total = float(white_stones + white_territory) + self.komi
        if black_total > white_total:
            winner: Optional[int] = BLACK
            margin = black_total - white_total
        elif white_total > black_total:
            winner = WHITE
            margin = white_total - black_total
        else:
            winner = None
            margin = 0.0

        return ScoreResult(
            black_stones=black_stones,
            white_stones=white_stones,
            black_territory=black_territory,
            white_territory=white_territory,
            neutral_points=neutral_points,
            komi=self.komi,
            black_total=black_total,
            white_total=white_total,
            winner=winner,
            margin=margin,
        )

    def _push_snapshot(self) -> None:
        if not self._record_undo:
            return
        self._undo_stack.append(
            _Snapshot(
                board=self.board_hash(),
                current_player=self.current_player,
                black_captures=self.captures[BLACK],
                white_captures=self.captures[WHITE],
                consecutive_passes=self.consecutive_passes,
                game_over=self.game_over,
                result_text=self.result_text,
                winner=self.winner,
                margin=self.margin,
                last_move=self.last_move,
                position_history=frozenset(self._position_history),
                move_count=len(self.moves),
                score_result=self.score_result,
            )
        )
