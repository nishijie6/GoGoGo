"""Board drawing and mouse input for :class:`GoApp`."""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from .engine import BLACK, EMPTY, Point, color_name
from .gui_constants import COLUMN_NAMES, MODE_AI


class BoardView:
    """Mixed into :class:`~weiqi.gui.GoApp`; relies on the widgets, Tk
    variables and game state it creates, and on its other mixins."""

    def _on_board_click(self, event: tk.Event) -> None:
        if not self._human_can_act():
            if self.ai_busy:
                self.notice_var.set("电脑正在思考，请稍候…")
            elif self.analysis_active:
                self.notice_var.set(
                    "正式棋局已暂停；请在 AI 分析工作台的棋盘中推演。"
                )
                assert self.analysis_window is not None
                self.analysis_window.lift()
            elif self.game.game_over:
                if self.in_reasoning_mode:
                    self.notice_var.set(
                        "当前推演分支已经结束，可悔棋继续推演，或退出推理模式恢复正式棋局。"
                    )
                elif self.game.result_text:
                    self.notice_var.set(self.game.result_text)
                else:
                    self.notice_var.set("本局已经结束，可悔棋或开始新局。")
            elif self.active_mode == MODE_AI:
                self.notice_var.set("现在轮到电脑落子。")
            return

        point = self._event_to_point(event.x, event.y)
        if point is None:
            return
        row, col = point
        color = self.game.current_player
        analysis = self.game.play(row, col)
        if not analysis.legal:
            self.notice_var.set(f"不能落子：{analysis.reason}")
            self.root.bell()
            self._draw_hover()
            return

        coordinate = self._coordinate(row, col)
        if analysis.captured:
            notice = f"{color_name(color)}于 {coordinate} 落子，提掉 {analysis.captured} 子。"
        else:
            notice = f"{color_name(color)}于 {coordinate} 落子。"
        if self.in_reasoning_mode:
            notice = f"推演：{notice}"
        self.hover_point = None
        self._refresh(notice)
        if self._is_ai_turn():
            self._start_ai_turn()

    def _on_board_motion(self, event: tk.Event) -> None:
        point = self._event_to_point(event.x, event.y)
        if point == self.hover_point:
            return
        self.hover_point = point
        self._draw_hover()

    def _on_board_leave(self, _event: object = None) -> None:
        self.hover_point = None
        self.canvas.delete("hover")

    def _event_to_point(self, x: float, y: float) -> Optional[Point]:
        if self._board_geometry is None:
            return None
        origin_x, origin_y, gap = self._board_geometry
        col = round((x - origin_x) / gap)
        row = round((y - origin_y) / gap)
        if not (0 <= row < self.game.size and 0 <= col < self.game.size):
            return None
        point_x = origin_x + col * gap
        point_y = origin_y + row * gap
        if ((x - point_x) ** 2 + (y - point_y) ** 2) ** 0.5 > gap * 0.45:
            return None
        return row, col

    def draw_board(self) -> None:
        """Redraw the complete board according to the current canvas size."""

        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        self.canvas.delete("all")

        short_side = min(width, height)
        margin = max(40.0, min(62.0, short_side * 0.10))
        board_span = max(100.0, short_side - margin * 2)
        gap = board_span / (self.game.size - 1)
        origin_x = (width - board_span) / 2
        origin_y = (height - board_span) / 2
        stone_radius = min(gap * 0.46, 33.0)
        coordinate_offset = min(
            margin - 9.0,
            max(stone_radius + 14.0, margin * 0.65),
        )
        self._board_geometry = (origin_x, origin_y, gap)

        # Layered board background gives the flat Canvas a little warmth/depth.
        self.canvas.create_rectangle(
            0, 0, width, height, fill="#d9aa63", outline="", tags="board"
        )
        for stripe in range(0, int(height), 24):
            self.canvas.create_line(
                0,
                stripe,
                width,
                stripe + 7,
                fill="#d3a159",
                width=1,
                stipple="gray75",
                tags="board",
            )

        end_x = origin_x + board_span
        end_y = origin_y + board_span
        for index in range(self.game.size):
            position_x = origin_x + index * gap
            position_y = origin_y + index * gap
            line_width = 2 if index in (0, self.game.size - 1) else 1
            self.canvas.create_line(
                origin_x,
                position_y,
                end_x,
                position_y,
                fill="#3d2a17",
                width=line_width,
                tags="grid",
            )
            self.canvas.create_line(
                position_x,
                origin_y,
                position_x,
                end_y,
                fill="#3d2a17",
                width=line_width,
                tags="grid",
            )

            coordinate_font = ("Segoe UI", max(8, min(10, int(gap * 0.24))))
            self.canvas.create_text(
                position_x,
                end_y + coordinate_offset,
                text=COLUMN_NAMES[index],
                fill="#4e361e",
                font=coordinate_font,
                tags="coordinates",
            )
            self.canvas.create_text(
                origin_x - coordinate_offset,
                position_y,
                text=str(self.game.size - index),
                fill="#4e361e",
                font=coordinate_font,
                tags="coordinates",
            )

        star_radius = max(2.5, min(4.2, gap * 0.11))
        for star_row, star_col in self._star_points():
            star_x = origin_x + star_col * gap
            star_y = origin_y + star_row * gap
            self.canvas.create_oval(
                star_x - star_radius,
                star_y - star_radius,
                star_x + star_radius,
                star_y + star_radius,
                fill="#342313",
                outline="",
                tags="stars",
            )

        for row in range(self.game.size):
            for col in range(self.game.size):
                color = self.game.board[row][col]
                if color != EMPTY:
                    self._draw_stone(row, col, color, stone_radius)

        if self.game.last_move is not None:
            row, col = self.game.last_move
            center_x = origin_x + col * gap
            center_y = origin_y + row * gap
            marker_radius = max(2.2, stone_radius * 0.12)
            marker_color = "#f1c65c" if self.game.board[row][col] == BLACK else "#b84334"
            self.canvas.create_oval(
                center_x - marker_radius,
                center_y - marker_radius,
                center_x + marker_radius,
                center_y + marker_radius,
                fill=marker_color,
                outline="",
                tags="last-move",
            )

        self._draw_hover()
        self._draw_reasoning_overlay(stone_radius)

    def _draw_reasoning_overlay(self, stone_radius: float) -> None:
        """Mark temporary stones and keep the variation boundary unmistakable."""

        session = self._reasoning_session
        if session is None or self._board_geometry is None:
            return

        origin_x, origin_y, gap = self._board_geometry
        marked_points: set[Point] = set()
        for move in self.game.moves[session.start_move_number :]:
            if move.kind != "play" or move.row is None or move.col is None:
                continue
            point = (move.row, move.col)
            if point in marked_points or self.game.board[move.row][move.col] != move.color:
                continue
            marked_points.add(point)
            center_x = origin_x + move.col * gap
            center_y = origin_y + move.row * gap
            radius = stone_radius * 0.84
            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                outline="#e47b2d",
                width=max(2, int(stone_radius * 0.10)),
                tags="reasoning-overlay",
            )

        badge_text = f"推理模式 · 临时变化 +{session.variation_move_count} 手"
        self.canvas.create_rectangle(
            13,
            11,
            235,
            42,
            fill="#7d431e",
            outline="#f0a154",
            width=1,
            tags="reasoning-overlay",
        )
        self.canvas.create_text(
            24,
            26,
            text=badge_text,
            anchor="w",
            fill="#fff5e8",
            font=("Microsoft YaHei UI", 10, "bold"),
            tags="reasoning-overlay",
        )

    def _draw_stone(self, row: int, col: int, color: int, radius: float) -> None:
        if self._board_geometry is None:
            return
        origin_x, origin_y, gap = self._board_geometry
        center_x = origin_x + col * gap
        center_y = origin_y + row * gap
        shadow_offset = max(1.5, radius * 0.09)
        self.canvas.create_oval(
            center_x - radius + shadow_offset,
            center_y - radius + shadow_offset,
            center_x + radius + shadow_offset,
            center_y + radius + shadow_offset,
            fill="#6e5034",
            outline="",
            stipple="gray50",
            tags="stones",
        )
        if color == BLACK:
            fill, outline, highlight = "#171b19", "#080a09", "#58605b"
        else:
            fill, outline, highlight = "#f5f2e9", "#a89f8e", "#ffffff"
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=fill,
            outline=outline,
            width=max(1, int(radius * 0.06)),
            tags="stones",
        )
        shine_radius = radius * 0.23
        self.canvas.create_oval(
            center_x - radius * 0.48,
            center_y - radius * 0.5,
            center_x - radius * 0.48 + shine_radius,
            center_y - radius * 0.5 + shine_radius,
            fill=highlight,
            outline="",
            stipple="gray50",
            tags="stones",
        )

    def _draw_hover(self) -> None:
        self.canvas.delete("hover")
        if not self._human_can_act() or self.hover_point is None:
            return
        row, col = self.hover_point
        if self.game.board[row][col] != EMPTY:
            return
        analysis = self.game.analyze_move(row, col)
        if not analysis.legal or self._board_geometry is None:
            return
        origin_x, origin_y, gap = self._board_geometry
        center_x = origin_x + col * gap
        center_y = origin_y + row * gap
        radius = min(gap * 0.43, 31.0)
        fill = "#1d211f" if self.game.current_player == BLACK else "#f7f4ec"
        outline = "#111513" if self.game.current_player == BLACK else "#8f8778"
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=fill,
            outline=outline,
            width=2,
            stipple="gray50",
            tags="hover",
        )

    def _star_points(self) -> tuple[Point, ...]:
        if self.game.size == 9:
            return ((2, 2), (2, 6), (4, 4), (6, 2), (6, 6))
        if self.game.size == 13:
            axes = (3, 6, 9)
        else:
            axes = (3, 9, 15)
        return tuple((row, col) for row in axes for col in axes)
