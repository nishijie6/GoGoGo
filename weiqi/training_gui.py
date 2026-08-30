"""Interactive reasoning trainer for joseki and life-and-death lessons."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import Optional

from .engine import BLACK, EMPTY, WHITE, BoardHash, Point, color_name
from .training import (
    JOSEKI_CATEGORY,
    JOSEKI_NOTICE,
    TRAINING_CATEGORIES,
    TrainingLesson,
    TrainingMove,
    lesson_by_title,
    lessons_for_category,
    position_after,
)


COLUMN_NAMES = "ABCDEFGHJKLMNOPQRST"


class ReasoningTrainer:
    """A non-modal, guided board for explaining curated local sequences."""

    def __init__(
        self,
        parent: tk.Tk,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        self.parent = parent
        self.on_close = on_close
        self.window = tk.Toplevel(parent)
        self.window.title("推理训练 · 定式与死活 · 弈境")
        self.window.configure(bg="#172019")
        self.window.minsize(900, 620)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        self.window.bind("<Control-w>", lambda _event: self.close())
        self.window.bind("<Key-h>", lambda _event: self.show_hint())
        self.window.bind("<Key-H>", lambda _event: self.show_hint())
        self.window.bind("<space>", lambda _event: self.demonstrate_next())

        self.category_var = tk.StringVar(value=TRAINING_CATEGORIES[0])
        first_lesson = lessons_for_category(TRAINING_CATEGORIES[0])[0]
        self.lesson_var = tk.StringVar(value=first_lesson.title)
        self.progress_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.notice_var = tk.StringVar()

        self.lesson: TrainingLesson = first_lesson
        self.progress = 0
        self.hint_point: Optional[Point] = None
        self.wrong_point: Optional[Point] = None
        self.hover_point: Optional[Point] = None
        self.feedback: Optional[tuple[str, str, str]] = None
        self._board_geometry: Optional[tuple[float, float, float]] = None
        self._closed = False

        self._build_layout()
        self._select_lesson(first_lesson)
        self._place_window()

    def _build_layout(self) -> None:
        shell = ttk.Frame(
            self.window,
            style="App.TFrame",
            padding=(18, 14, 18, 18),
        )
        shell.grid(row=0, column=0, sticky="nsew")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="推理训练", style="Title.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            header,
            text="先猜落点，再读目的 · 定式理解与死活要点",
            style="Subtitle.TLabel",
        ).grid(row=0, column=1, sticky="sw", padx=(14, 0), pady=(0, 3))
        ttk.Button(
            header,
            text="关闭  Esc",
            style="Header.TButton",
            command=self.close,
        ).grid(row=0, column=2, sticky="e", padx=(12, 0))

        board_shell = tk.Frame(
            shell,
            bg="#0f1712",
            highlightbackground="#2d3a31",
            highlightthickness=1,
            bd=0,
        )
        board_shell.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        board_shell.rowconfigure(0, weight=1)
        board_shell.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            board_shell,
            bg="#d8a75d",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda _event: self.draw_board())
        self.canvas.bind("<Button-1>", self._on_board_click)
        self.canvas.bind("<Motion>", self._on_board_motion)
        self.canvas.bind("<Leave>", self._on_board_leave)

        panel = ttk.Frame(
            shell,
            style="Panel.TFrame",
            width=340,
            padding=16,
        )
        panel.grid(row=1, column=1, sticky="ns")
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(8, weight=1)

        ttk.Label(panel, text="训练选择", style="PanelTitle.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        selectors = ttk.Frame(panel, style="Panel.TFrame")
        selectors.grid(row=1, column=0, sticky="ew", pady=(7, 8))
        selectors.columnconfigure(1, weight=1)
        ttk.Label(selectors, text="类型", style="Panel.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 7),
        )
        self.category_combo = ttk.Combobox(
            selectors,
            textvariable=self.category_var,
            values=TRAINING_CATEGORIES,
            state="readonly",
            width=12,
        )
        self.category_combo.grid(row=0, column=1, sticky="ew")
        self.category_combo.bind("<<ComboboxSelected>>", self._on_category_selected)
        ttk.Label(selectors, text="课程", style="Panel.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 7),
            pady=(7, 0),
        )
        self.lesson_combo = ttk.Combobox(
            selectors,
            textvariable=self.lesson_var,
            state="readonly",
        )
        self.lesson_combo.grid(row=1, column=1, sticky="ew", pady=(7, 0))
        self.lesson_combo.bind("<<ComboboxSelected>>", self._on_lesson_selected)
        self._update_lesson_choices()

        status = ttk.Frame(panel, style="Panel.TFrame")
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)
        ttk.Label(status, textvariable=self.turn_var, style="Turn.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(status, textvariable=self.progress_var, style="WinRate.TLabel").grid(
            row=0,
            column=1,
            sticky="e",
        )

        tk.Label(
            panel,
            textvariable=self.notice_var,
            bg="#e7e0d2",
            fg="#5b4b35",
            font=("Microsoft YaHei UI", 9),
            padx=9,
            pady=8,
            justify="left",
            anchor="w",
            wraplength=286,
        ).grid(row=3, column=0, sticky="ew", pady=(7, 9))

        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.grid(row=4, column=0, sticky="ew", pady=(0, 9))
        actions.columnconfigure((0, 1), weight=1)
        self.hint_button = ttk.Button(
            actions,
            text="提示落点  H",
            style="Action.TButton",
            command=self.show_hint,
        )
        self.hint_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.demo_button = ttk.Button(
            actions,
            text="演示下一手  Space",
            style="Primary.TButton",
            command=self.demonstrate_next,
        )
        self.demo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.undo_button = ttk.Button(
            actions,
            text="上一步",
            style="Action.TButton",
            command=self.undo,
        )
        self.undo_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(7, 0))
        ttk.Button(
            actions,
            text="重新开始",
            style="Action.TButton",
            command=self.reset,
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(7, 0))

        ttk.Separator(panel).grid(row=5, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(panel, text="行棋说明", style="PanelTitle.TLabel").grid(
            row=6,
            column=0,
            sticky="w",
        )
        ttk.Label(
            panel,
            text="这里展示教学理由，不展示模型隐藏思维过程。",
            style="Estimate.TLabel",
        ).grid(row=7, column=0, sticky="w", pady=(2, 6))

        text_frame = ttk.Frame(panel, style="Panel.TFrame")
        text_frame.grid(row=8, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.explanation = tk.Text(
            text_frame,
            wrap="word",
            bg="#fbf8f1",
            fg="#39463d",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#d0c8b9",
            padx=12,
            pady=10,
            font=("Microsoft YaHei UI", 9),
            cursor="arrow",
        )
        text_scroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.explanation.yview,
        )
        self.explanation.configure(yscrollcommand=text_scroll.set)
        self.explanation.grid(row=0, column=0, sticky="nsew")
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.explanation.tag_configure(
            "heading",
            font=("Microsoft YaHei UI", 11, "bold"),
            foreground="#315e43",
            spacing1=6,
            spacing3=4,
        )
        self.explanation.tag_configure(
            "body",
            font=("Microsoft YaHei UI", 9),
            foreground="#39463d",
            spacing3=7,
        )
        self.explanation.tag_configure(
            "prompt",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground="#76562e",
            lmargin1=8,
            lmargin2=8,
            spacing1=4,
            spacing3=8,
        )
        self.explanation.tag_configure(
            "success",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground="#2f7146",
            spacing1=5,
            spacing3=4,
        )
        self.explanation.tag_configure(
            "warning",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground="#a2513d",
            spacing1=5,
            spacing3=4,
        )
        self.explanation.tag_configure(
            "note",
            font=("Microsoft YaHei UI", 8),
            foreground="#68746c",
            spacing1=4,
            spacing3=7,
        )

    def _place_window(self) -> None:
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        width = min(1120, max(900, screen_width - 80))
        height = min(780, max(620, screen_height - 100))
        self.parent.update_idletasks()
        x = max(
            0,
            self.parent.winfo_rootx()
            + (self.parent.winfo_width() - width) // 2,
        )
        y = max(
            0,
            self.parent.winfo_rooty()
            + (self.parent.winfo_height() - height) // 2,
        )
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.focus_set()

    @property
    def is_alive(self) -> bool:
        if self._closed:
            return False
        try:
            return bool(self.window.winfo_exists())
        except tk.TclError:
            return False

    def lift(self) -> None:
        if not self.is_alive:
            return
        self.window.deiconify()
        self.window.lift()
        self.window.focus_set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        if self.on_close is not None:
            self.on_close()

    def _update_lesson_choices(self) -> None:
        lessons = lessons_for_category(self.category_var.get())
        titles = tuple(lesson.title for lesson in lessons)
        self.lesson_combo.configure(values=titles)
        if self.lesson_var.get() not in titles:
            self.lesson_var.set(titles[0])

    def _on_category_selected(self, _event: object = None) -> None:
        self._update_lesson_choices()
        self._on_lesson_selected()

    def _on_lesson_selected(self, _event: object = None) -> None:
        lesson = lesson_by_title(self.category_var.get(), self.lesson_var.get())
        self._select_lesson(lesson)

    def _select_lesson(self, lesson: TrainingLesson) -> None:
        self.lesson = lesson
        self.progress = 0
        self.hint_point = None
        self.wrong_point = None
        self.hover_point = None
        self.feedback = None
        self.notice_var.set("请先在棋盘上猜下一手；需要时可查看提示或演示答案。")
        self._refresh()

    def reset(self) -> None:
        self.progress = 0
        self.hint_point = None
        self.wrong_point = None
        self.feedback = ("已重置", "从第一手重新思考。", "note")
        self.notice_var.set("课程已重置，请重新寻找第一手。")
        self._refresh()

    def undo(self) -> None:
        if self.progress <= 0:
            return
        self.progress -= 1
        self.hint_point = None
        self.wrong_point = None
        self.feedback = ("已回退", "请重新判断当前这手。", "note")
        self.notice_var.set("已退回上一手。")
        self._refresh()

    def show_hint(self) -> None:
        if self.progress >= len(self.lesson.moves):
            return
        move = self.lesson.moves[self.progress]
        self.hint_point = move.point
        self.wrong_point = None
        self.feedback = ("思考提示", move.hint, "prompt")
        self.notice_var.set("绿色圆圈标出了本题主线落点。")
        self._refresh()

    def demonstrate_next(self) -> None:
        if self.progress >= len(self.lesson.moves):
            return
        move = self.lesson.moves[self.progress]
        self._accept_move(move, demonstrated=True)

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

    def _accept_move(self, move: TrainingMove, demonstrated: bool) -> None:
        expected = self.lesson.moves[self.progress]
        if move is not expected:
            raise ValueError("只能推进当前训练步骤")
        self.progress += 1
        self.hint_point = None
        self.wrong_point = None
        prefix = "演示答案" if demonstrated else "判断正确"
        self.feedback = (prefix, expected.reason, "success")
        if self.progress == len(self.lesson.moves):
            self.notice_var.set("课程完成！可回看次序，或选择另一课继续训练。")
        else:
            self.notice_var.set("这手的理由已显示。请继续判断下一手。")
        self._refresh()

    def _refresh(self) -> None:
        total = len(self.lesson.moves)
        self.progress_var.set(f"{self.progress} / {total} 手")
        if self.progress < total:
            current = self.lesson.moves[self.progress]
            self.turn_var.set(f"轮到{color_name(current.color)}")
        else:
            self.turn_var.set("训练完成")
        self.undo_button.configure(state="normal" if self.progress else "disabled")
        next_state = "normal" if self.progress < total else "disabled"
        self.hint_button.configure(state=next_state)
        self.demo_button.configure(state=next_state)
        self._render_explanation()
        self.draw_board()

    def _render_explanation(self) -> None:
        text = self.explanation
        text.configure(state="normal")
        text.delete("1.0", tk.END)
        text.insert(tk.END, self.lesson.title + "\n", "heading")
        text.insert(tk.END, self.lesson.level + "\n", "note")
        text.insert(tk.END, self.lesson.intro + "\n", "body")
        text.insert(tk.END, "训练目标\n", "heading")
        text.insert(tk.END, self.lesson.objective + "\n", "body")
        if self.lesson.category == JOSEKI_CATEGORY:
            text.insert(tk.END, "定式说明\n", "heading")
            text.insert(tk.END, JOSEKI_NOTICE + "\n", "note")

        if self.feedback is not None:
            title, body, tag = self.feedback
            text.insert(tk.END, title + "\n", tag)
            text.insert(tk.END, body + "\n", "body")

        if self.progress < len(self.lesson.moves):
            move = self.lesson.moves[self.progress]
            text.insert(
                tk.END,
                f"下一步 · 第 {self.progress + 1} 手 · {color_name(move.color)}\n",
                "heading",
            )
            text.insert(tk.END, move.prompt + "\n", "prompt")
        else:
            text.insert(tk.END, "本课结论\n", "heading")
            text.insert(tk.END, self.lesson.conclusion + "\n", "success")
        text.configure(state="disabled")
        text.see("1.0")

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
