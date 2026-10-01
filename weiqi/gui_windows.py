"""Secondary windows of :class:`GoApp`: trainer, AI analysis, KataGo settings and rules."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

from .ai import is_katago_difficulty, is_training_difficulty
from .analysis_gui import AnalysisWorkbenchWindow
from .gui_constants import MODE_AI
from .katago import (
    KataGoAI,
    KataGoConfigurationError,
    KataGoEngine,
    KataGoError,
    KataGoSettings,
)
from .katago_gui import KataGoSettingsDialog
from .rl_opponents import settings_for_training_opponent
from .rules import RULE_SECTIONS, RULES_INTRO
from .training_gui import ReasoningTrainer


class CompanionWindows:
    """Mixed into :class:`~weiqi.gui.GoApp`; relies on the widgets, Tk
    variables and game state it creates, and on its other mixins."""

    def show_training(self) -> None:
        """Open the guided joseki and life-and-death reasoning trainer."""

        if self.training_window is not None and self.training_window.is_alive:
            self.training_window.lift()
            return
        self.training_window = ReasoningTrainer(
            self.root,
            on_close=self._training_closed,
        )

    def _training_closed(self) -> None:
        self.training_window = None

    @property
    def analysis_active(self) -> bool:
        """Whether an isolated AI analysis workbench is currently visible."""

        window = getattr(self, "analysis_window", None)
        return window is not None and window.is_alive

    def show_analysis(self) -> bool:
        """Open the KataGo analysis workbench for the unchanged formal game."""

        if self.analysis_active:
            assert self.analysis_window is not None
            self.analysis_window.lift()
            return True
        if self.in_reasoning_mode:
            self.notice_var.set(
                "请先退出 F3 推理模式；AI 分析工作台会建立自己的多分支推演树。"
            )
            self.root.bell()
            return False
        if self.game.game_over:
            self.notice_var.set("本局已经结束，不能从终局开启 AI 分析工作台。")
            self.root.bell()
            return False

        settings = KataGoSettings.load()
        try:
            if is_training_difficulty(self.active_difficulty):
                settings = settings_for_training_opponent(
                    settings, self.active_training_opponent, self.game.size,
                )
            settings.require_valid()
        except KataGoConfigurationError as error:
            self._pending_analysis_open = True
            self.notice_var.set(f"AI 分析需要先配置 KataGo：{error}。")
            self.show_katago_settings()
            return False

        old_engine = self.katago_engine
        if (
            old_engine is not None
            and not old_engine.closed
            and old_engine.settings.fingerprint == settings.fingerprint
        ):
            analysis_engine = old_engine
        else:
            try:
                analysis_engine = KataGoEngine(settings)
            except KataGoError as error:
                self.notice_var.set(f"KataGo 配置不可用：{error}。")
                self.show_katago_settings()
                return False

        analysis_ai: Optional[KataGoAI] = None
        if self.active_mode == MODE_AI and is_katago_difficulty(
            self.active_difficulty
        ):
            try:
                analysis_ai = KataGoAI(analysis_engine, self.active_difficulty)
            except KataGoError as error:
                if analysis_engine is not old_engine:
                    analysis_engine.close()
                self._pending_analysis_open = True
                self.notice_var.set(f"当前 KataGo 对局配置不可用：{error}。")
                self.show_katago_settings()
                return False

        self._invalidate_ai()
        if old_engine is not None and old_engine is not analysis_engine:
            old_engine.close()
        self.katago_engine = analysis_engine
        if analysis_ai is not None:
            self.ai = analysis_ai

        try:
            self.analysis_window = AnalysisWorkbenchWindow(
                self.root,
                self.game,
                analysis_engine,
                self.executor,
                on_close=self._analysis_closed,
            )
        except Exception as error:
            self.analysis_window = None
            self.notice_var.set(f"无法打开 AI 分析工作台：{error}")
            if self._is_ai_turn():
                self.root.after(220, self._start_ai_turn)
            return False

        self.hover_point = None
        self._refresh(
            "AI 分析工作台已开启：正式棋局已冻结且不会被推演修改；"
            "关闭窗口后可从原局面继续。"
        )
        return True

    def _analysis_closed(self) -> None:
        """Release analysis work and resume the untouched formal position."""

        self.analysis_window = None
        if self.katago_engine is not None:
            # Future.cancel() cannot stop work that has already entered the
            # external process.  Stopping here also prevents a late response
            # from occupying the shared single-worker executor.
            self.katago_engine.stop()
        if getattr(self, "_closing", False):
            return
        self.hover_point = None
        self._refresh("AI 分析工作台已关闭，正式棋局保持不变，可继续对局。")
        if self._is_ai_turn():
            self.root.after(220, self._start_ai_turn)

    def show_katago_settings(self, pending_new_game: bool = False) -> None:
        """Open the KataGo path configuration dialog."""

        self._pending_katago_new_game = (
            self._pending_katago_new_game or pending_new_game
        )
        if (
            self.katago_settings_window is not None
            and self.katago_settings_window.is_alive
        ):
            self.katago_settings_window.lift()
            return
        self.katago_settings_window = KataGoSettingsDialog(
            self.root,
            on_saved=self._katago_settings_saved,
            on_close=self._katago_settings_closed,
        )

    def _katago_settings_saved(self, settings: KataGoSettings) -> None:
        """Apply new paths and resume the operation that requested setup."""

        should_start_new_game = self._pending_katago_new_game
        should_open_analysis = getattr(self, "_pending_analysis_open", False)
        self._pending_katago_new_game = False
        self._pending_analysis_open = False
        self._invalidate_ai()
        analysis_window = getattr(self, "analysis_window", None)
        if analysis_window is not None:
            analysis_window.close()
        if self.katago_engine is not None:
            self.katago_engine.close()
            self.katago_engine = None

        if should_start_new_game:
            self.root.after(80, self.new_game)
            return

        needs_play_engine = self.active_mode == MODE_AI and is_katago_difficulty(
            self.active_difficulty
        )
        if needs_play_engine or should_open_analysis:
            try:
                if needs_play_engine and is_training_difficulty(self.active_difficulty):
                    settings = settings_for_training_opponent(
                        settings, self.active_training_opponent, self.game.size,
                    )
                self.katago_engine = KataGoEngine(settings)
                if needs_play_engine:
                    self.ai = KataGoAI(
                        self.katago_engine,
                        self.active_difficulty,
                    )
            except KataGoError as error:
                if self.katago_engine is not None:
                    self.katago_engine.close()
                    self.katago_engine = None
                self.notice_var.set(f"KataGo 配置仍不可用：{error}")
                return
            if should_open_analysis:
                self.notice_var.set("KataGo 配置已更新，准备打开 AI 分析工作台。")
                self.root.after(80, self.show_analysis)
            else:
                self.notice_var.set("KataGo 配置已更新，准备继续当前对局。")
            if not should_open_analysis and self._is_ai_turn():
                self.root.after(120, self._start_ai_turn)

    def _katago_settings_closed(self) -> None:
        self.katago_settings_window = None
        self._pending_katago_new_game = False
        self._pending_analysis_open = False

    def show_rules(self) -> None:
        """Open the in-program Go rules reference."""

        if self.rules_window is not None:
            try:
                if self.rules_window.winfo_exists():
                    self.rules_window.deiconify()
                    self.rules_window.lift()
                    self.rules_window.focus_set()
                    return
            except tk.TclError:
                pass

        window = tk.Toplevel(self.root)
        self.rules_window = window
        window.title("围棋规则 · 弈境")
        window.configure(bg="#f2eee5")
        window.minsize(560, 460)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._close_rules)
        window.bind("<Escape>", lambda _event: self._close_rules())
        window.bind("<Control-w>", lambda _event: self._close_rules())
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)

        shell = ttk.Frame(window, style="Panel.TFrame", padding=18)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.rowconfigure(2, weight=1)
        shell.columnconfigure(0, weight=1)

        ttk.Label(shell, text="围棋规则", style="RuleTitle.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            shell,
            text=RULES_INTRO,
            style="RuleIntro.TLabel",
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(9, 12))

        text_frame = ttk.Frame(shell, style="Panel.TFrame")
        text_frame.grid(row=2, column=0, sticky="nsew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        rules_text = tk.Text(
            text_frame,
            wrap="word",
            bg="#fbf8f1",
            fg="#39463d",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#d0c8b9",
            padx=18,
            pady=14,
            font=("Microsoft YaHei UI", 10),
            cursor="arrow",
        )
        rules_scrollbar = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=rules_text.yview,
        )
        rules_text.configure(yscrollcommand=rules_scrollbar.set)
        rules_text.grid(row=0, column=0, sticky="nsew")
        rules_scrollbar.grid(row=0, column=1, sticky="ns")

        rules_text.tag_configure(
            "heading",
            font=("Microsoft YaHei UI", 12, "bold"),
            foreground="#315e43",
            spacing1=12,
            spacing3=6,
        )
        rules_text.tag_configure(
            "body",
            font=("Microsoft YaHei UI", 10),
            foreground="#39463d",
            spacing1=2,
            spacing3=8,
        )
        rules_text.tag_configure(
            "bullet",
            font=("Microsoft YaHei UI", 10),
            foreground="#39463d",
            lmargin1=16,
            lmargin2=30,
            spacing1=2,
            spacing3=6,
        )
        for section in RULE_SECTIONS:
            rules_text.insert(tk.END, section.title + "\n", "heading")
            for paragraph in section.paragraphs:
                tag = "bullet" if paragraph.startswith("•") else "body"
                rules_text.insert(tk.END, paragraph + "\n", tag)
            rules_text.insert(tk.END, "\n", "body")
        rules_text.configure(state="disabled")

        footer = ttk.Frame(shell, style="Panel.TFrame")
        footer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            text="本说明与当前程序采用的规则一致 · Esc 关闭",
            style="Estimate.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            footer,
            text="关闭",
            style="Action.TButton",
            command=self._close_rules,
        ).grid(row=0, column=1, sticky="e")

        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        width = min(760, max(560, screen_width - 80))
        height = min(680, max(460, screen_height - 100))
        self.root.update_idletasks()
        x = max(0, self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2)
        y = max(0, self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.focus_set()

    def _close_rules(self) -> None:
        if self.rules_window is not None:
            try:
                self.rules_window.destroy()
            except tk.TclError:
                pass
        self.rules_window = None
