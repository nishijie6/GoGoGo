"""Interactive reasoning trainer for joseki and life-and-death lessons."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Optional

from .engine import Point, color_name
from .training import (
    JOSEKI_CATEGORY,
    JOSEKI_NOTICE,
    TRAINING_CATEGORIES,
    TrainingLesson,
    TrainingMove,
    lesson_by_title,
    lessons_for_category,
)
from .training_board import TrainerBoard
from .training_layout import TrainerLayout


class ReasoningTrainer(TrainerLayout, TrainerBoard):
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
