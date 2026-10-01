"""Lesson board drawing and mouse input of the reasoning-trainer window."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from .engine import BLACK, EMPTY, Point
from .gui_constants import COLUMN_NAMES
from .training import position_after


class TrainerBoard:
    """Mixed into :class:`~weiqi.training_gui.ReasoningTrainer`; relies on the
    widgets, variables and lesson state it creates."""

    def _on_board_click(self, event: tk.Event) -> None:
        point = self._event_to_point(event.x, event.y)
        if point is None or self.progress >= len(self.lesson.moves):
            return
        move = self.lesson.moves[self.progress]
        if point != move.point:
            self.wrong_point = point
            self.hint_point = None
            coordinate = self._coordinate(point)
            self.feedback = (
                "再想一步",
                f"{coordinate} 不是本课参考次序。{move.prompt}",
                "warning",
            )
            self.notice_var.set("这手没有落下。可换个方向思考，或点击“提示落点”。")
            self._refresh()
            return
        self._accept_move(move, demonstrated=False)

    def _event_to_point(self, x: float, y: float) -> Optional[Point]:
        if self._board_geometry is None:
            return None
        offset_x, offset_y, spacing = self._board_geometry
        col = round((x - offset_x) / spacing)
        row = round((y - offset_y) / spacing)
        if not (0 <= row < self.lesson.board_size and 0 <= col < self.lesson.board_size):
            return None
        px = offset_x + col * spacing
        py = offset_y + row * spacing
        if (x - px) ** 2 + (y - py) ** 2 > (spacing * 0.43) ** 2:
            return None
        return row, col

    def _on_board_motion(self, event: tk.Event) -> None:
        point = self._event_to_point(event.x, event.y)
        if point != self.hover_point:
            self.hover_point = point
            self.draw_board()

    def _on_board_leave(self, _event: object = None) -> None:
        if self.hover_point is not None:
            self.hover_point = None
            self.draw_board()

    def draw_board(self) -> None:
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        width = max(canvas.winfo_width(), 100)
        height = max(canvas.winfo_height(), 100)
        canvas.delete("all")

        canvas.create_rectangle(0, 0, width, height, fill="#d9a85f", outline="")
        canvas.create_rectangle(
            5,
            5,
            width - 5,
            height - 5,
            outline="#bd8743",
            width=2,
        )

        size = self.lesson.board_size
        margin = max(29.0, min(width, height) * 0.065)
        board_side = max(1.0, min(width, height) - margin * 2)
        spacing = board_side / (size - 1)
        actual_side = spacing * (size - 1)
        offset_x = (width - actual_side) / 2
        offset_y = (height - actual_side) / 2
        self._board_geometry = (offset_x, offset_y, spacing)

        line_color = "#4b3420"
        for index in range(size):
            pos_x = offset_x + index * spacing
            pos_y = offset_y + index * spacing
            canvas.create_line(
                offset_x,
                pos_y,
                offset_x + actual_side,
                pos_y,
                fill=line_color,
                width=1,
            )
            canvas.create_line(
                pos_x,
                offset_y,
                pos_x,
                offset_y + actual_side,
                fill=line_color,
                width=1,
            )

        coordinate_font = ("Microsoft YaHei UI", max(7, int(spacing * 0.28)))
        for index in range(size):
            pos = offset_x + index * spacing
            row_pos = offset_y + index * spacing
            canvas.create_text(
                pos,
                offset_y - max(12, spacing * 0.55),
                text=COLUMN_NAMES[index],
                fill="#71502d",
                font=coordinate_font,
            )
            canvas.create_text(
                offset_x - max(13, spacing * 0.55),
                row_pos,
                text=str(size - index),
                fill="#71502d",
                font=coordinate_font,
            )

        star_radius = max(2.3, min(4.2, spacing * 0.13))
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

        board = position_after(self.lesson, self.progress)
        radius = max(5.5, spacing * 0.46)
        move_labels: dict[Point, tuple[int, int]] = {}
        for number, move in enumerate(self.lesson.moves[: self.progress], start=1):
            if board[move.point[0]][move.point[1]] == move.color:
                move_labels[move.point] = (number, move.color)

        for row in range(size):
            for col in range(size):
                color = board[row][col]
                if color == EMPTY:
                    continue
                self._draw_stone(row, col, color, radius)
                label = move_labels.get((row, col))
                if label is not None:
                    number, label_color = label
                    x = offset_x + col * spacing
                    y = offset_y + row * spacing
                    canvas.create_text(
                        x,
                        y,
                        text=str(number),
                        fill="#f8f5ed" if label_color == BLACK else "#222722",
                        font=(
                            "Microsoft YaHei UI",
                            max(7, int(radius * (0.72 if number < 10 else 0.57))),
                            "bold",
                        ),
                    )

        if self.progress:
            last = self.lesson.moves[self.progress - 1]
            if board[last.point[0]][last.point[1]] == last.color:
                x = offset_x + last.point[1] * spacing
                y = offset_y + last.point[0] * spacing
                marker = max(2.0, radius * 0.15)
                canvas.create_oval(
                    x - marker,
                    y - marker,
                    x + marker,
                    y + marker,
                    fill="#c74e3d",
                    outline="#fff4e2",
                    width=1,
                )

        if self.hint_point is not None:
            self._draw_target(self.hint_point, "#2f8a50", "?")
        if self.wrong_point is not None:
            self._draw_wrong_mark(self.wrong_point)
        if (
            self.hover_point is not None
            and self.progress < len(self.lesson.moves)
            and board[self.hover_point[0]][self.hover_point[1]] == EMPTY
        ):
            self._draw_hover(self.hover_point, self.lesson.moves[self.progress].color)

    def _draw_stone(self, row: int, col: int, color: int, radius: float) -> None:
        if self._board_geometry is None:
            return
        offset_x, offset_y, spacing = self._board_geometry
        x = offset_x + col * spacing
        y = offset_y + row * spacing
        self.canvas.create_oval(
            x - radius + 2,
            y - radius + 3,
            x + radius + 2,
            y + radius + 3,
            fill="#8a6539",
            outline="",
        )
        if color == BLACK:
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#17201a",
                outline="#070a08",
                width=1,
            )
            shine = max(1.5, radius * 0.19)
            self.canvas.create_oval(
                x - radius * 0.52,
                y - radius * 0.58,
                x - radius * 0.52 + shine,
                y - radius * 0.58 + shine,
                fill="#596159",
                outline="",
            )
        else:
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#f4f0e6",
                outline="#8d8a82",
                width=1,
            )
            self.canvas.create_arc(
                x - radius * 0.72,
                y - radius * 0.72,
                x + radius * 0.72,
                y + radius * 0.72,
                start=38,
                extent=105,
                style="arc",
                outline="#ffffff",
                width=2,
            )

    def _draw_target(self, point: Point, color: str, text: str) -> None:
        if self._board_geometry is None:
            return
        offset_x, offset_y, spacing = self._board_geometry
        row, col = point
        x = offset_x + col * spacing
        y = offset_y + row * spacing
        radius = max(7, spacing * 0.40)
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=color,
            width=3,
            dash=(5, 3),
        )
        self.canvas.create_text(
            x,
            y,
            text=text,
            fill=color,
            font=("Microsoft YaHei UI", max(10, int(radius)), "bold"),
        )

    def _draw_wrong_mark(self, point: Point) -> None:
        if self._board_geometry is None:
            return
        offset_x, offset_y, spacing = self._board_geometry
        row, col = point
        x = offset_x + col * spacing
        y = offset_y + row * spacing
        radius = max(6, spacing * 0.32)
        self.canvas.create_line(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="#b8483a",
            width=3,
        )
        self.canvas.create_line(
            x + radius,
            y - radius,
            x - radius,
            y + radius,
            fill="#b8483a",
            width=3,
        )

    def _draw_hover(self, point: Point, color: int) -> None:
        if self._board_geometry is None:
            return
        offset_x, offset_y, spacing = self._board_geometry
        row, col = point
        x = offset_x + col * spacing
        y = offset_y + row * spacing
        radius = max(5, spacing * 0.40)
        outline = "#202720" if color == BLACK else "#f6f2e9"
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=outline,
            width=2,
            dash=(3, 3),
        )

    @staticmethod
    def _star_points(size: int) -> tuple[Point, ...]:
        if size == 19:
            positions = (3, 9, 15)
            return tuple((row, col) for row in positions for col in positions)
        if size == 13:
            positions = (3, 6, 9)
            return tuple((row, col) for row in positions for col in positions)
        positions = (2, 4, 6)
        return tuple((row, col) for row in positions for col in positions)

    def _coordinate(self, point: Point) -> str:
        row, col = point
        return f"{COLUMN_NAMES[col]}{self.lesson.board_size - row}"
