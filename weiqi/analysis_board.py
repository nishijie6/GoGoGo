"""Board, ownership overlay and board input of the AI analysis window."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from typing import Optional

from .analysis_style import BOARD_COLOR, field, finite_float, ownership_color, sequence
from .engine import BLACK, EMPTY, WHITE, Point
from .gui_constants import COLUMN_NAMES


class AnalysisBoard:
    """Mixed into :class:`~weiqi.analysis_gui.AnalysisWorkbenchWindow`; relies on
    the widgets, variables and view state it creates."""

    def _position_board(self) -> tuple[tuple[int, ...], ...]:
        position = field(self.view, "position")
        board = field(position, "board", "stones", default=())
        rows = sequence(board)
        result: list[tuple[int, ...]] = []
        for row in rows:
            try:
                result.append(tuple(int(value) for value in row))
            except (TypeError, ValueError):
                return ()
        return tuple(result)

    def _ownership(self, size: int) -> tuple[float, ...]:
        analysis = field(self.view, "selected_analysis", "analysis")
        values = field(analysis, "ownership", "black_ownership", default=())
        flat: list[float] = []
        for value in sequence(values):
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for nested in value:
                    number = finite_float(nested)
                    flat.append(0.0 if number is None else number)
            else:
                number = finite_float(value)
                flat.append(0.0 if number is None else number)
        return tuple(flat) if len(flat) == size * size else ()

    def _draw_board(self) -> None:
        if not hasattr(self, "board_canvas"):
            return
        canvas = self.board_canvas
        width = max(100, canvas.winfo_width())
        height = max(100, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=BOARD_COLOR, outline="")
        canvas.create_rectangle(
            5,
            5,
            width - 5,
            height - 5,
            outline="#bd8743",
            width=2,
        )

        board = self._position_board()
        if not board:
            canvas.create_text(
                width / 2,
                height / 2,
                text="尚无可显示的局面",
                fill="#604426",
                font=("Microsoft YaHei UI", 14, "bold"),
            )
            self._board_geometry = None
            return
        size = len(board)
        margin = max(30.0, min(width, height) * 0.064)
        board_side = max(1.0, min(width, height) - margin * 2)
        spacing = board_side / max(1, size - 1)
        actual_side = spacing * (size - 1)
        offset_x = (width - actual_side) / 2
        offset_y = (height - actual_side) / 2
        self._board_geometry = (offset_x, offset_y, spacing)

        ownership = self._ownership(size)
        if ownership:
            heat_radius = spacing * 0.47
            for row in range(size):
                for col in range(size):
                    color = ownership_color(ownership[row * size + col])
                    if color is None:
                        continue
                    x = offset_x + col * spacing
                    y = offset_y + row * spacing
                    canvas.create_rectangle(
                        x - heat_radius,
                        y - heat_radius,
                        x + heat_radius,
                        y + heat_radius,
                        fill=color,
                        outline="",
                    )

        line_color = "#4b3420"
        for index in range(size):
            x = offset_x + index * spacing
            y = offset_y + index * spacing
            canvas.create_line(
                offset_x,
                y,
                offset_x + actual_side,
                y,
                fill=line_color,
            )
            canvas.create_line(
                x,
                offset_y,
                x,
                offset_y + actual_side,
                fill=line_color,
            )

        coordinate_font = ("Microsoft YaHei UI", max(7, int(spacing * 0.26)))
        for index in range(size):
            x = offset_x + index * spacing
            y = offset_y + index * spacing
            if index < len(COLUMN_NAMES):
                canvas.create_text(
                    x,
                    offset_y - max(12, spacing * 0.55),
                    text=COLUMN_NAMES[index],
                    fill="#71502d",
                    font=coordinate_font,
                )
            canvas.create_text(
                offset_x - max(13, spacing * 0.55),
                y,
                text=str(size - index),
                fill="#71502d",
                font=coordinate_font,
            )

        star_radius = max(2.2, min(4.2, spacing * 0.13))
        for row, col in self._star_points(size):
            x = offset_x + col * spacing
            y = offset_y + row * spacing
            canvas.create_oval(
                x - star_radius,
                y - star_radius,
                x + star_radius,
                y + star_radius,
                fill="#3f2c1b",
                outline="",
            )

        radius = max(5.0, spacing * 0.46)
        for row in range(size):
            for col in range(size):
                if board[row][col] in (BLACK, WHITE):
                    self._draw_stone(row, col, board[row][col], radius)

        analysis = field(self.view, "selected_analysis", "analysis")
        for index, candidate in enumerate(self._analysis_candidates(analysis)[:12], start=1):
            point = self._candidate_point(candidate)
            if point is None:
                continue
            row, col = point
            if not (0 <= row < size and 0 <= col < size):
                continue
            x = offset_x + col * spacing
            y = offset_y + row * spacing
            marker_radius = max(7.0, spacing * 0.29)
            canvas.create_oval(
                x - marker_radius,
                y - marker_radius,
                x + marker_radius,
                y + marker_radius,
                fill="#f2b457" if index == 1 else "#e7d39b",
                outline="#5e4325",
                width=1,
            )
            canvas.create_text(
                x,
                y,
                text=str(index),
                fill="#2c261d",
                font=("Microsoft YaHei UI", max(8, int(marker_radius * 0.9)), "bold"),
            )

        if self._hover_point is not None:
            row, col = self._hover_point
            if 0 <= row < size and 0 <= col < size and board[row][col] == EMPTY:
                x = offset_x + col * spacing
                y = offset_y + row * spacing
                hover_radius = max(6.0, spacing * 0.39)
                canvas.create_oval(
                    x - hover_radius,
                    y - hover_radius,
                    x + hover_radius,
                    y + hover_radius,
                    outline="#2d7145",
                    width=2,
                    dash=(3, 3),
                )

    def _draw_stone(self, row: int, col: int, color: int, radius: float) -> None:
        if self._board_geometry is None:
            return
        offset_x, offset_y, spacing = self._board_geometry
        x = offset_x + col * spacing
        y = offset_y + row * spacing
        self.board_canvas.create_oval(
            x - radius + 2,
            y - radius + 3,
            x + radius + 2,
            y + radius + 3,
            fill="#806039",
            outline="",
        )
        if color == BLACK:
            self.board_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#17201a",
                outline="#070a08",
                width=1,
            )
        else:
            self.board_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#f4f0e6",
                outline="#8d8a82",
                width=1,
            )

    def _event_to_point(self, x: float, y: float) -> Optional[Point]:
        if self._board_geometry is None:
            return None
        board = self._position_board()
        if not board:
            return None
        offset_x, offset_y, spacing = self._board_geometry
        col = round((x - offset_x) / spacing)
        row = round((y - offset_y) / spacing)
        size = len(board)
        if not (0 <= row < size and 0 <= col < size):
            return None
        px = offset_x + col * spacing
        py = offset_y + row * spacing
        if (x - px) ** 2 + (y - py) ** 2 > (spacing * 0.43) ** 2:
            return None
        return row, col

    def _on_board_click(self, event: tk.Event) -> None:
        if self._busy:
            return
        if bool(field(field(self.view, "position"), "game_over", default=False)):
            self.status_var.set("当前推演分支已经结束；请撤回或选择其它节点。")
            return
        point = self._event_to_point(event.x, event.y)
        if point is None:
            return
        board = self._position_board()
        if board and board[point[0]][point[1]] != EMPTY:
            self.status_var.set("该交叉点已经有棋子。")
            return
        self._play_variation(point)

    def _on_board_motion(self, event: tk.Event) -> None:
        point = self._event_to_point(event.x, event.y)
        if point != self._hover_point:
            self._hover_point = point
            self._draw_board()

    def _on_board_leave(self, _event: object = None) -> None:
        if self._hover_point is not None:
            self._hover_point = None
            self._draw_board()

    def _coordinate(self, point: Point) -> str:
        board = self._position_board()
        size = len(board) if board else 19
        row, col = point
        if not (0 <= col < len(COLUMN_NAMES)):
            return f"({row}, {col})"
        return f"{COLUMN_NAMES[col]}{size - row}"

    @staticmethod
    def _star_points(size: int) -> tuple[Point, ...]:
        if size == 19:
            axes = (3, 9, 15)
        elif size == 13:
            axes = (3, 6, 9)
        elif size == 9:
            axes = (2, 4, 6)
        else:
            return ()
        return tuple((row, col) for row in axes for col in axes)
