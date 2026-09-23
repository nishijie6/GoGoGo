"""Tkinter user interface for the local Go game."""

from __future__ import annotations

import threading
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from tkinter import messagebox, ttk
from typing import Optional

from .ai import (
    AI_DIFFICULTIES,
    AIMove,
    GoAI,
    is_human_sl_difficulty,
    is_katago_difficulty,
    is_training_difficulty,
)
from .analysis_gui import AnalysisWorkbenchWindow
from .engine import BLACK, EMPTY, WHITE, GoGame, MoveRecord, Point, color_name
from .katago import (
    KataGoAI,
    KataGoConfigurationError,
    KataGoEngine,
    KataGoError,
    KataGoSettings,
)
from .katago_gui import KataGoSettingsDialog
from .reasoning import ReasoningSession
from .rules import RULE_SECTIONS, RULES_INTRO
from .training_gui import ReasoningTrainer
from .winrate import WinRateEstimate, WinRateEstimator
from .rl_opponents import load_training_opponents, settings_for_training_opponent


MODE_AI = "人机对战"
MODE_LOCAL = "双人对战"
COLUMN_NAMES = "ABCDEFGHJKLMNOPQRST"


class GoApp:
    """A responsive desktop Go board supporting AI and local play."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("弈境 · 围棋")
        self.root.geometry("1080x760")
        self.root.minsize(900, 690)
        self.root.configure(bg="#172019")

        self._configure_styles()

        self.game = GoGame(size=9)
        self._reasoning_session: Optional[ReasoningSession] = None
        self.ai = GoAI()
        self.katago_engine: Optional[KataGoEngine] = None
        self.winrate_estimator = WinRateEstimator()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="go-ai")
        self.ai_future: Optional[Future[AIMove]] = None
        self._ai_cancel_event: Optional[threading.Event] = None
        self.ai_busy = False
        self.generation = 0
        self.active_mode = MODE_AI
        self.active_difficulty = "中等"
        self.human_color = BLACK
        self.ai_color = WHITE
        self.hover_point: Optional[Point] = None
        self._board_geometry: Optional[tuple[float, float, float]] = None
        self._end_dialog_shown = False
        self.rules_window: Optional[tk.Toplevel] = None
        self.training_window: Optional[ReasoningTrainer] = None
        self.analysis_window: Optional[AnalysisWorkbenchWindow] = None
        self.katago_settings_window: Optional[KataGoSettingsDialog] = None
        self._pending_katago_new_game = False
        self._pending_analysis_open = False
        self._closing = False
        self._winrate_cache_key: Optional[tuple[object, ...]] = None
        self._last_winrate: Optional[WinRateEstimate] = None
        self._winrate_source = "启发式估算"
        self._training_opponents = {}
        self.active_training_opponent = None

        self.mode_var = tk.StringVar(value=MODE_AI)
        self.size_var = tk.StringVar(value="9×9")
        self.human_color_var = tk.StringVar(value="黑方（先手）")
        self.difficulty_var = tk.StringVar(value="中等")
        self.turn_var = tk.StringVar()
        self.notice_var = tk.StringVar()
        self.capture_var = tk.StringVar()
        self.move_var = tk.StringVar()
        self.last_var = tk.StringVar()
        self.winrate_var = tk.StringVar()
        self.winlead_var = tk.StringVar()

        self._build_layout()
        self._refresh_training_opponents()
        self._bind_shortcuts()
        self.new_game()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#172019")
        style.configure("Panel.TFrame", background="#f2eee5")
        style.configure(
            "Title.TLabel",
            background="#172019",
            foreground="#f6f0df",
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#172019",
            foreground="#aebbad",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "PanelTitle.TLabel",
            background="#f2eee5",
            foreground="#26332b",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Panel.TLabel",
            background="#f2eee5",
            foreground="#445148",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "WinRate.TLabel",
            background="#f2eee5",
            foreground="#26332b",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.configure(
            "Estimate.TLabel",
            background="#f2eee5",
            foreground="#68746c",
            font=("Microsoft YaHei UI", 8),
        )
        style.configure(
            "Turn.TLabel",
            background="#f2eee5",
            foreground="#172019",
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        style.configure(
            "Notice.TLabel",
            background="#e7e0d2",
            foreground="#5b4b35",
            font=("Microsoft YaHei UI", 9),
            padding=9,
        )
        style.configure(
            "Primary.TButton",
            background="#376348",
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(12, 9),
        )
        style.configure(
            "Header.TButton",
            background="#2b4133",
            foreground="#f2eddf",
            font=("Microsoft YaHei UI", 9),
            padding=(11, 7),
        )
        style.map(
            "Header.TButton",
            background=[("active", "#3b5a47")],
        )
        style.configure(
            "ReasoningActive.TButton",
            background="#9a5728",
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(11, 7),
        )
        style.map(
            "ReasoningActive.TButton",
            background=[("active", "#b86a31")],
        )
        style.configure(
            "RuleTitle.TLabel",
            background="#f2eee5",
            foreground="#1d3024",
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "RuleIntro.TLabel",
            background="#e7e0d2",
            foreground="#4f5a52",
            font=("Microsoft YaHei UI", 9),
            padding=10,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#437756"), ("disabled", "#aab3ac")],
        )
        style.configure(
            "Action.TButton",
            font=("Microsoft YaHei UI", 9),
            padding=(9, 7),
        )
        style.configure(
            "TCombobox",
            font=("Microsoft YaHei UI", 9),
            padding=5,
        )

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(18, 14, 18, 18))
        shell.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="弈境", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="本地围棋 · 中国数子法 · 支持 9×9 / 13×13 / 19×19",
            style="Subtitle.TLabel",
        ).grid(row=0, column=1, sticky="sw", padx=(14, 0), pady=(0, 3))
        ttk.Button(
            header,
            text="KataGo 设置",
            style="Header.TButton",
            command=self.show_katago_settings,
        ).grid(row=0, column=2, sticky="e", padx=(12, 0))
        self.reasoning_button = ttk.Button(
            header,
            text="开启推理  F3",
            style="Header.TButton",
            command=self.toggle_reasoning_mode,
        )
        self.reasoning_button.grid(row=0, column=3, sticky="e", padx=(8, 0))
        self.analysis_button = ttk.Button(
            header,
            text="AI 分析  F4",
            style="Header.TButton",
            command=self.show_analysis,
        )
        self.analysis_button.grid(row=0, column=4, sticky="e", padx=(8, 0))
        ttk.Button(
            header,
            text="推理训练  F2",
            style="Header.TButton",
            command=self.show_training,
        ).grid(row=0, column=5, sticky="e", padx=(8, 0))
        ttk.Button(
            header,
            text="围棋规则  F1",
            style="Header.TButton",
            command=self.show_rules,
        ).grid(row=0, column=6, sticky="e", padx=(8, 0))

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

        panel_shell = ttk.Frame(shell, style="Panel.TFrame", width=286)
        panel_shell.grid(row=1, column=1, sticky="ns")
        panel_shell.grid_propagate(False)
        panel_shell.rowconfigure(0, weight=1)
        panel_shell.columnconfigure(0, weight=1)
        self.panel_canvas = tk.Canvas(
            panel_shell,
            width=268,
            bg="#f2eee5",
            bd=0,
            highlightthickness=0,
            yscrollincrement=24,
        )
        panel_scrollbar = ttk.Scrollbar(
            panel_shell,
            orient="vertical",
            command=self.panel_canvas.yview,
        )
        self.panel_canvas.configure(yscrollcommand=panel_scrollbar.set)
        self.panel_canvas.grid(row=0, column=0, sticky="nsew")
        panel_scrollbar.grid(row=0, column=1, sticky="ns")

        panel = ttk.Frame(self.panel_canvas, style="Panel.TFrame", padding=16)
        panel_window = self.panel_canvas.create_window(
            (0, 0),
            window=panel,
            anchor="nw",
        )
        panel.bind(
            "<Configure>",
            lambda _event: self.panel_canvas.configure(
                scrollregion=self.panel_canvas.bbox("all")
            ),
        )
        self.panel_canvas.bind(
            "<Configure>",
            lambda event: self.panel_canvas.itemconfigure(
                panel_window,
                width=event.width,
            ),
        )
        self.panel_canvas.bind("<MouseWheel>", self._on_panel_mousewheel)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="新局设置", style="PanelTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(panel, text="对战模式", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 2)
        )
        self.mode_combo = ttk.Combobox(
            panel,
            textvariable=self.mode_var,
            values=(MODE_AI, MODE_LOCAL),
            state="readonly",
        )
        self.mode_combo.grid(row=2, column=0, sticky="ew")
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_selected)

        options = ttk.Frame(panel, style="Panel.TFrame")
        options.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        options.columnconfigure((0, 1), weight=1)
        ttk.Label(options, text="棋盘", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(options, text="执子", style="Panel.TLabel").grid(
            row=0, column=1, sticky="w", padx=(7, 0)
        )
        self.size_combo = ttk.Combobox(
            options,
            textvariable=self.size_var,
            values=("9×9", "13×13", "19×19"),
            state="readonly",
            width=8,
        )
        self.size_combo.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        self.size_combo.bind("<<ComboboxSelected>>", self._refresh_training_opponents)
        self.human_combo = ttk.Combobox(
            options,
            textvariable=self.human_color_var,
            values=("黑方（先手）", "白方（后手）"),
            state="readonly",
            width=12,
        )
        self.human_combo.grid(row=1, column=1, sticky="ew", padx=(7, 0), pady=(3, 0))
        ttk.Label(
            options,
            text="电脑难度 / 引擎",
            style="Panel.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.difficulty_combo = ttk.Combobox(
            options,
            textvariable=self.difficulty_var,
            values=AI_DIFFICULTIES,
            state="readonly",
            postcommand=self._refresh_training_opponents,
        )
        self.difficulty_combo.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(3, 0),
        )

        ttk.Button(
            panel,
            text="开始新局",
            style="Primary.TButton",
            command=self.new_game,
        ).grid(row=4, column=0, sticky="ew", pady=(9, 10))

        ttk.Separator(panel).grid(row=5, column=0, sticky="ew", pady=(0, 9))
        ttk.Label(panel, textvariable=self.turn_var, style="Turn.TLabel").grid(
            row=6, column=0, sticky="w"
        )
        ttk.Label(
            panel,
            textvariable=self.notice_var,
            style="Notice.TLabel",
            wraplength=218,
            justify="left",
        ).grid(row=7, column=0, sticky="ew", pady=(6, 7))
        ttk.Label(panel, textvariable=self.capture_var, style="Panel.TLabel").grid(
            row=8, column=0, sticky="w", pady=2
        )
        ttk.Label(panel, textvariable=self.move_var, style="Panel.TLabel").grid(
            row=9, column=0, sticky="w", pady=2
        )
        ttk.Label(panel, textvariable=self.last_var, style="Panel.TLabel").grid(
            row=10, column=0, sticky="w", pady=2
        )

        winrate_frame = ttk.Frame(panel, style="Panel.TFrame")
        winrate_frame.grid(row=11, column=0, sticky="ew", pady=(7, 3))
        winrate_frame.columnconfigure(1, weight=1)
        ttk.Label(
            winrate_frame,
            text="实时胜率",
            style="PanelTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            winrate_frame,
            textvariable=self.winrate_var,
            style="WinRate.TLabel",
        ).grid(row=0, column=1, sticky="e")
        self.winrate_bar = tk.Canvas(
            winrate_frame,
            width=218,
            height=14,
            bg="#ded8ca",
            bd=0,
            highlightthickness=1,
            highlightbackground="#aaa293",
        )
        self.winrate_bar.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(4, 3),
        )
        self.winrate_bar.bind(
            "<Configure>", lambda _event: self._draw_winrate_bar()
        )
        ttk.Label(
            winrate_frame,
            textvariable=self.winlead_var,
            style="Estimate.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w")

        actions = ttk.Frame(panel, style="Panel.TFrame")
        actions.grid(row=12, column=0, sticky="ew", pady=(7, 9))
        actions.columnconfigure((0, 1), weight=1)
        self.undo_button = ttk.Button(
            actions, text="悔棋", style="Action.TButton", command=self.undo
        )
        self.undo_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.pass_button = ttk.Button(
            actions, text="虚手", style="Action.TButton", command=self.pass_turn
        )
        self.pass_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.resign_button = ttk.Button(
            actions, text="认输", style="Action.TButton", command=self.resign
        )
        self.resign_button.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        ttk.Label(panel, text="棋谱", style="PanelTitle.TLabel").grid(
            row=13, column=0, sticky="w", pady=(1, 5)
        )
        log_frame = ttk.Frame(panel, style="Panel.TFrame")
        log_frame.grid(row=14, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        panel.rowconfigure(14, weight=1)
        self.move_log = tk.Listbox(
            log_frame,
            height=5,
            bg="#fbf8f1",
            fg="#39463d",
            selectbackground="#78917e",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground="#d5cdbf",
            font=("Microsoft YaHei UI", 9),
            activestyle="none",
        )
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.move_log.yview)
        self.move_log.configure(yscrollcommand=scrollbar.set)
        self.move_log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            panel,
            text=(
                "快捷键：Ctrl+N 新局 · Ctrl+Z 悔棋 · P 虚手\n"
                "F1 规则 · F2 训练 · F3 推理 · F4 AI 分析。"
            ),
            style="Panel.TLabel",
            wraplength=225,
            justify="left",
        ).grid(row=15, column=0, sticky="w", pady=(7, 0))
        self._bind_panel_mousewheel(panel)

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

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-n>", lambda _event: self.new_game())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Key-p>", lambda _event: self.pass_turn())
        self.root.bind("<Key-P>", lambda _event: self.pass_turn())
        self.root.bind("<F1>", lambda _event: self.show_rules())
        self.root.bind("<F2>", lambda _event: self.show_training())
        self.root.bind("<F3>", lambda _event: self.toggle_reasoning_mode())
        self.root.bind("<F4>", lambda _event: self.show_analysis())

    def _bind_panel_mousewheel(self, widget: tk.Misc) -> None:
        if not isinstance(widget, (tk.Listbox, ttk.Scrollbar)):
            widget.bind("<MouseWheel>", self._on_panel_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_panel_mousewheel(child)

    def _on_panel_mousewheel(self, event: tk.Event) -> str:
        if event.delta:
            direction = -1 if event.delta > 0 else 1
            self.panel_canvas.yview_scroll(direction * 2, "units")
        return "break"

    def _on_mode_selected(self, _event: object = None) -> None:
        state = "readonly" if self.mode_var.get() == MODE_AI else "disabled"
        self.human_combo.configure(state=state)
        self.difficulty_combo.configure(state=state)

    @property
    def in_reasoning_mode(self) -> bool:
        """Whether the board is currently showing an isolated variation."""

        return self._reasoning_session is not None

    def toggle_reasoning_mode(self) -> bool:
        """Enter reasoning mode, or discard the variation and restore the game."""

        if self.in_reasoning_mode:
            return self.exit_reasoning_mode()
        return self.enter_reasoning_mode()

    def enter_reasoning_mode(self) -> bool:
        """Save the formal game and switch the board to a temporary variation."""

        if self.analysis_active:
            assert self.analysis_window is not None
            self.analysis_window.lift()
            self.notice_var.set(
                "AI 分析工作台已经包含多分支推演；请先关闭它再进入 F3 推理模式。"
            )
            self.root.bell()
            return False
        if self.game.game_over:
            self.notice_var.set("本局已经结束，不能从终局开启推理模式。")
            self.root.bell()
            return False

        self._invalidate_ai()
        session = ReasoningSession.start(self.game)
        self._reasoning_session = session
        self.game = session.variation
        self.hover_point = None
        self._end_dialog_shown = False
        self._winrate_cache_key = None
        self._last_winrate = None
        self._winrate_source = "启发式估算"
        self._refresh(
            "推理模式已开启：正式棋局已保存。现在可为黑白双方连续推演，"
            "悔棋只撤回推演着手。"
        )
        return True

    def exit_reasoning_mode(self) -> bool:
        """Discard the temporary variation and resume the saved formal game."""

        session = self._reasoning_session
        if session is None:
            return False

        self._invalidate_ai()
        variation_moves = session.variation_move_count
        self.game = session.restore_formal_game()
        self._reasoning_session = None
        self.hover_point = None
        self._end_dialog_shown = False
        self._winrate_cache_key = None
        self._last_winrate = None
        self._winrate_source = "启发式估算"
        if variation_moves:
            notice = (
                f"已退出推理模式，丢弃当前推演分支的 {variation_moves} 手；"
                "正式棋局已恢复，可继续对局。"
            )
        else:
            notice = "已退出推理模式，正式棋局已恢复，可继续对局。"
        self._refresh(notice)
        if self._is_ai_turn():
            self.root.after(220, self._start_ai_turn)
        return True

    def _refresh_training_opponents(self, _event=None) -> None:
        size = int(self.size_var.get().split("×", maxsplit=1)[0])
        try:
            self._training_opponents = load_training_opponents(size)
        except (OSError, ValueError, KeyError) as error:
            self._training_opponents = {}
            self.notice_var.set(f"训练对手列表暂不可用：{error}")
        self.difficulty_combo.configure(values=(*AI_DIFFICULTIES, *self._training_opponents))
        selected = self.difficulty_var.get()
        if is_training_difficulty(selected) and selected not in self._training_opponents:
            self.difficulty_var.set("中等")

    def new_game(self) -> None:
        """Apply the selected options and replace the current game."""

        size = int(self.size_var.get().split("×", maxsplit=1)[0])
        selected_mode = self.mode_var.get()
        selected_difficulty = self.difficulty_var.get()

        candidate_engine: Optional[KataGoEngine] = None
        training_opponent = None
        if selected_mode == MODE_AI and is_katago_difficulty(selected_difficulty):
            settings = KataGoSettings.load()
            try:
                if is_training_difficulty(selected_difficulty):
                    training_opponent = self._training_opponents.get(selected_difficulty)
                    if training_opponent is None:
                        self.notice_var.set("训练对手已变化，请重新选择。")
                        return
                    settings = settings_for_training_opponent(settings, training_opponent, size)
                settings.require_valid()
            except KataGoConfigurationError as error:
                self.notice_var.set(f"所选电脑难度需要先配置 KataGo：{error}。")
                self.show_katago_settings(pending_new_game=True)
                return
            if (
                is_human_sl_difficulty(selected_difficulty)
                and not settings.human_style_enabled
            ):
                self.notice_var.set(
                    "HumanSL 人类段位需要先配置人类风格模型。"
                )
                self.show_katago_settings(pending_new_game=True)
                return

            if (
                self.katago_engine is not None
                and not self.katago_engine.closed
                and self.katago_engine.settings.fingerprint == settings.fingerprint
            ):
                candidate_engine = self.katago_engine
            else:
                try:
                    candidate_engine = KataGoEngine(settings)
                except KataGoError as error:
                    self.notice_var.set(f"KataGo 配置不可用：{error}。")
                    self.show_katago_settings(pending_new_game=True)
                    return
            candidate_ai = KataGoAI(candidate_engine, selected_difficulty)
        else:
            builtin_difficulty = (
                "中等" if is_katago_difficulty(selected_difficulty) else selected_difficulty
            )
            candidate_ai = GoAI(difficulty=builtin_difficulty)

        old_engine = self.katago_engine
        self._invalidate_ai()
        analysis_window = getattr(self, "analysis_window", None)
        if analysis_window is not None:
            analysis_window.close()
        self._reasoning_session = None
        if old_engine is not None and old_engine is not candidate_engine:
            old_engine.close()
        self.katago_engine = candidate_engine
        self.ai = candidate_ai
        self.active_mode = selected_mode
        self.active_difficulty = selected_difficulty
        self.active_training_opponent = training_opponent
        self.human_color = (
            BLACK if self.human_color_var.get().startswith("黑") else WHITE
        )
        self.ai_color = WHITE if self.human_color == BLACK else BLACK
        self.game = GoGame(size=size, komi=6.5)
        self.hover_point = None
        self._end_dialog_shown = False
        self._winrate_cache_key = None
        self._last_winrate = None
        self._winrate_source = "启发式估算"
        if self.active_mode == MODE_AI:
            engine_note = (
                "；首次启动引擎可能需要调优显卡"
                if is_katago_difficulty(self.active_difficulty)
                else ""
            )
            self.notice_var.set(
                f"新对局已开始，电脑难度：{self.active_difficulty}{engine_note}。"
            )
            if is_human_sl_difficulty(self.active_difficulty) and size != 19:
                self.notice_var.set(
                    self.notice_var.get()
                    + " HumanSL 级段位主要基于 19×19 人类棋谱，"
                    "当前尺寸仅作风格模拟。"
                )
        else:
            self.notice_var.set("新对局已开始，请在棋盘交叉点落子。")
        self._on_mode_selected()
        self._refresh()
        if self._is_ai_turn():
            self.root.after(300, self._start_ai_turn)

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

    def pass_turn(self) -> None:
        if not self._human_can_act():
            if self.ai_busy:
                self.notice_var.set("电脑正在思考，现在不能虚手。")
            return
        color = self.game.current_player
        if not self.game.pass_turn():
            return
        self.hover_point = None
        prefix = "推演：" if self.in_reasoning_mode else ""
        self._refresh(f"{prefix}{color_name(color)}选择虚手。")
        if self.game.game_over:
            if not self.in_reasoning_mode:
                self._show_game_over()
        elif self._is_ai_turn():
            self._start_ai_turn()

    def resign(self) -> None:
        if not self._human_can_act():
            return
        color = self.game.current_player
        reasoning = self.in_reasoning_mode
        title = "确认推演认输" if reasoning else "确认认输"
        prompt = (
            f"确定让{color_name(color)}在当前推演分支认输吗？"
            "这不会影响正式棋局。"
            if reasoning
            else f"确定由{color_name(color)}认输并结束本局吗？"
        )
        if not messagebox.askyesno(
            title,
            prompt,
            parent=self.root,
        ):
            return
        if self.game.resign():
            self.hover_point = None
            if reasoning:
                self._refresh(
                    f"推演分支：{self.game.result_text}。可悔棋继续推演，"
                    "或退出推理模式恢复正式棋局。"
                )
            else:
                self._refresh(self.game.result_text)
                self._show_game_over()

    def undo(self) -> None:
        if self.analysis_active:
            self.notice_var.set(
                "正式棋局已暂停；请使用 AI 分析工作台中的“撤回”浏览推演树。"
            )
            assert self.analysis_window is not None
            self.analysis_window.lift()
            return
        if not self.game.can_undo:
            if self.in_reasoning_mode:
                self.notice_var.set(
                    "推演已经回到保存点，不能撤回正式棋局中的着手。"
                )
            else:
                self.notice_var.set("当前没有可以撤回的着手。")
            return

        was_ai_busy = self.ai_busy
        self._invalidate_ai()
        undone = self.game.undo(1)
        if self.in_reasoning_mode:
            self.hover_point = None
            self._end_dialog_shown = False
            self._refresh(f"推演已撤回 {undone} 手，正式棋局保持不变。")
            return
        if self.active_mode == MODE_AI and not was_ai_busy:
            # Normally take back the AI response and the preceding human action,
            # leaving the human at the decision they wanted to reconsider.
            while self.game.can_undo and self.game.current_player != self.human_color:
                undone += self.game.undo(1)

        self.hover_point = None
        self._end_dialog_shown = False
        self._refresh(f"已撤回 {undone} 手。")
        if self._is_ai_turn():
            self.root.after(220, self._start_ai_turn)

    def _start_ai_turn(self) -> None:
        if not self._is_ai_turn() or self.ai_busy:
            return
        self.ai_busy = True
        request_generation = self.generation
        snapshot = self.game.clone()
        cancel_event = threading.Event()
        self._ai_cancel_event = cancel_event
        if isinstance(self.ai, KataGoAI):
            self.ai_future = self.executor.submit(
                self.ai.choose_move,
                snapshot,
                cancel_event,
            )
        else:
            self.ai_future = self.executor.submit(self.ai.choose_move, snapshot)
        self._refresh(
            f"{color_name(self.ai_color)}电脑正在思考…"
            f"（{self.active_difficulty}）"
        )
        self.root.after(60, lambda: self._poll_ai(request_generation))

    def _poll_ai(self, request_generation: int) -> None:
        if request_generation != self.generation:
            return
        future = self.ai_future
        if future is None:
            return
        if not future.done():
            self.root.after(60, lambda: self._poll_ai(request_generation))
            return

        self.ai_busy = False
        self.ai_future = None
        self._ai_cancel_event = None
        try:
            decision = future.result()
        except Exception as error:  # Keep the GUI usable if an AI bug occurs.
            if isinstance(error, KataGoError) or isinstance(self.ai, KataGoAI):
                self._refresh(
                    f"KataGo 计算失败，棋盘已保留：{error}。"
                    "请检查设置后重试。"
                )
                messagebox.showerror(
                    "KataGo 计算失败",
                    f"{error}\n\n棋盘没有改变。请检查引擎、模型或显卡后端设置，"
                    "保存后程序会尝试继续当前对局。",
                    parent=self.root,
                )
                self.show_katago_settings()
                return
            # A safe automatic pass hands control back instead of leaving the
            # application permanently stuck on the computer's turn.
            if self._is_ai_turn():
                self.game.pass_turn()
            self._refresh(f"电脑计算失败并已自动虚手：{error}")
            if self.game.game_over:
                self._show_game_over()
            return

        if request_generation != self.generation or not self._is_ai_turn():
            self._refresh()
            return

        color = self.game.current_player
        if decision.point is None:
            self.game.pass_turn()
            notice = f"{color_name(color)}电脑选择虚手（{decision.explanation}）。"
        else:
            row, col = decision.point
            analysis = self.game.play(row, col)
            if not analysis.legal:
                # The snapshot and live game should match.  Passing is a safe
                # fallback if they ever do not.
                self.game.pass_turn()
                notice = f"{color_name(color)}电脑选择虚手。"
            else:
                coordinate = self._coordinate(row, col)
                capture_text = (
                    f"，提掉 {analysis.captured} 子" if analysis.captured else ""
                )
                notice = (
                    f"{color_name(color)}电脑于 {coordinate} 落子{capture_text}"
                    f"（{decision.explanation}）。"
                )
        self._store_katago_winrate(decision)
        self._refresh(notice)
        if self.game.game_over:
            self._show_game_over()

    def _invalidate_ai(self) -> None:
        self.generation += 1
        cancel_event = getattr(self, "_ai_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        self._ai_cancel_event = None
        future_running = self.ai_future is not None and not self.ai_future.done()
        if self.ai_future is not None:
            self.ai_future.cancel()
        if future_running and self.katago_engine is not None:
            self.katago_engine.stop()
        self.ai_future = None
        self.ai_busy = False

    def _is_ai_turn(self) -> bool:
        return (
            not self.in_reasoning_mode
            and not self.analysis_active
            and self.active_mode == MODE_AI
            and not self.game.game_over
            and self.game.current_player == self.ai_color
        )

    def _human_can_act(self) -> bool:
        if self.game.game_over or self.ai_busy or self.analysis_active:
            return False
        if self.in_reasoning_mode:
            return True
        return self.active_mode == MODE_LOCAL or self.game.current_player == self.human_color

    def _refresh(self, notice: Optional[str] = None) -> None:
        if self.in_reasoning_mode and self.game.game_over:
            result = self.game.result_text or "当前变化已经结束"
            self.notice_var.set(
                f"推演分支已结束：{result}。可悔棋继续推演，"
                "或退出推理模式恢复正式棋局。"
            )
        elif self.game.game_over and self.game.result_text:
            self.notice_var.set(self.game.result_text)
        elif notice is not None:
            self.notice_var.set(notice)

        if self.in_reasoning_mode:
            if self.game.game_over:
                self.turn_var.set("◆ 推理模式 · 分支结束")
            else:
                self.turn_var.set(
                    f"◆ 推理模式 · {color_name(self.game.current_player)}推演"
                )
        elif self.analysis_active:
            self.turn_var.set("◇ AI 分析工作台 · 正式棋局已暂停")
        elif self.game.game_over:
            self.turn_var.set("对局结束")
        elif self.ai_busy:
            self.turn_var.set(
                f"● {color_name(self.game.current_player)} · 电脑 · "
                f"{self.active_difficulty}"
            )
        elif self.active_mode == MODE_AI:
            role = (
                "玩家"
                if self.game.current_player == self.human_color
                else f"电脑 · {self.active_difficulty}"
            )
            self.turn_var.set(f"● {color_name(self.game.current_player)} · {role}")
        else:
            self.turn_var.set(f"● {color_name(self.game.current_player)}落子")

        self.capture_var.set(
            f"提子：黑 {self.game.captures[BLACK]}  ·  白 {self.game.captures[WHITE]}"
        )
        session = self._reasoning_session
        if session is not None:
            self.move_var.set(
                f"正式 {session.start_move_number} 手  ·  "
                f"推演 +{session.variation_move_count} 手  ·  "
                f"连续虚手 {self.game.consecutive_passes}"
            )
        else:
            self.move_var.set(
                f"手数：{self.game.move_number}  ·  连续虚手：{self.game.consecutive_passes}"
            )
        if self.game.moves:
            last = self.game.moves[-1]
            if last.kind == "play" and last.row is not None and last.col is not None:
                last_text = self._coordinate(last.row, last.col)
            elif last.kind == "pass":
                last_text = "虚手"
            else:
                last_text = "认输"
            self.last_var.set(f"上一手：{color_name(last.color)} {last_text}")
        else:
            self.last_var.set("上一手：—")

        if self.in_reasoning_mode:
            self.reasoning_button.configure(
                text="退出推理  F3",
                style="ReasoningActive.TButton",
                state="normal",
            )
        else:
            self.reasoning_button.configure(
                text="开启推理  F3",
                style="Header.TButton",
                state=(
                    "disabled"
                    if self.game.game_over or self.analysis_active
                    else "normal"
                ),
            )
        if self.analysis_active:
            self.analysis_button.configure(
                text="分析已开启  F4",
                style="ReasoningActive.TButton",
                state="normal",
            )
        else:
            self.analysis_button.configure(
                text="AI 分析  F4",
                style="Header.TButton",
                state=(
                    "disabled"
                    if self.game.game_over or self.in_reasoning_mode
                    else "normal"
                ),
            )
        undo_state = (
            "normal"
            if self.game.can_undo and not self.analysis_active
            else "disabled"
        )
        self.undo_button.configure(state=undo_state)
        action_state = "normal" if self._human_can_act() else "disabled"
        self.pass_button.configure(state=action_state)
        self.resign_button.configure(state=action_state)
        self._update_winrate()
        self._refresh_move_log()
        self.draw_board()

    def _update_winrate(self) -> None:
        cache_key = self._position_cache_key()
        if cache_key != self._winrate_cache_key:
            self._last_winrate = self.winrate_estimator.estimate(self.game)
            self._winrate_cache_key = cache_key
            self._winrate_source = "启发式估算"

        estimate = self._last_winrate
        if estimate is None:
            return
        self.winrate_var.set(
            f"黑 {estimate.black_percent:.1f}%  ·  白 {estimate.white_percent:.1f}%"
        )
        if estimate.final:
            if self.game.winner is None:
                detail = "和棋"
            else:
                detail = f"{color_name(self.game.winner)}胜"
            self.winlead_var.set(f"终局 · {detail}")
        elif abs(estimate.black_lead) < 0.35:
            self.winlead_var.set(
                f"{estimate.phase} · 局势接近均衡（{self._winrate_source}）"
            )
        else:
            leader = "黑" if estimate.black_lead > 0 else "白"
            self.winlead_var.set(
                f"{estimate.phase} · {leader}约领先 {abs(estimate.black_lead):.1f} 目"
                f"（{self._winrate_source}）"
            )
        self._draw_winrate_bar()

    def _position_cache_key(self) -> tuple[object, ...]:
        return (
            self.game.size,
            self.game.komi,
            self.game.board_hash(),
            self.game.current_player,
            self.game.move_number,
            self.game.consecutive_passes,
            self.game.game_over,
            self.game.winner,
        )

    def _store_katago_winrate(self, decision: AIMove) -> None:
        """Cache KataGo's post-move evaluation for the position now on screen."""

        probability = decision.black_win_probability
        if probability is None or self.game.game_over:
            return
        heuristic = self.winrate_estimator.estimate(self.game)
        black_lead = (
            decision.black_lead
            if decision.black_lead is not None
            else heuristic.black_lead
        )
        correction = black_lead - heuristic.black_lead
        self._last_winrate = WinRateEstimate(
            black_win_probability=max(0.0, min(1.0, probability)),
            black_expected_score=heuristic.black_expected_score + correction / 2.0,
            white_expected_score=heuristic.white_expected_score - correction / 2.0,
            black_lead=black_lead,
            phase=heuristic.phase,
        )
        self._winrate_cache_key = self._position_cache_key()
        source = "HumanSL" if is_human_sl_difficulty(self.active_difficulty) else "KataGo"
        self._winrate_source = f"{source} · {decision.analysis_visits} visits"

    def _draw_winrate_bar(self) -> None:
        if not hasattr(self, "winrate_bar"):
            return
        self.winrate_bar.delete("all")
        estimate = self._last_winrate
        if estimate is None:
            return
        width = max(
            10,
            self.winrate_bar.winfo_width(),
            self.winrate_bar.winfo_reqwidth(),
        )
        height = max(10, self.winrate_bar.winfo_height())
        black_width = width * estimate.black_win_probability
        self.winrate_bar.create_rectangle(
            0,
            0,
            black_width,
            height,
            fill="#1b201d",
            outline="",
        )
        self.winrate_bar.create_rectangle(
            black_width,
            0,
            width,
            height,
            fill="#ece7dc",
            outline="",
        )
        self.winrate_bar.create_line(
            width / 2,
            0,
            width / 2,
            height,
            fill="#8e887d",
            width=1,
        )

    def _refresh_move_log(self) -> None:
        self.move_log.delete(0, tk.END)
        session = self._reasoning_session
        for index, move in enumerate(self.game.moves, start=1):
            if session is not None and index == session.start_move_number + 1:
                self.move_log.insert(tk.END, "──── 推理分支起点 ────")
            self.move_log.insert(tk.END, self._format_move(index, move))
        if session is not None and self.game.move_number == session.start_move_number:
            self.move_log.insert(tk.END, "──── 推理分支起点 ────")
        if self.game.moves:
            self.move_log.see(tk.END)

    def _format_move(self, number: int, move: MoveRecord) -> str:
        stone = "●" if move.color == BLACK else "○"
        if move.kind == "play" and move.row is not None and move.col is not None:
            text = self._coordinate(move.row, move.col)
            if move.captured:
                text += f"  提 {move.captured}"
        elif move.kind == "pass":
            text = "虚手"
        else:
            text = "认输"
        return f"{number:>3}. {stone}  {text}"

    def _coordinate(self, row: int, col: int) -> str:
        return f"{COLUMN_NAMES[col]}{self.game.size - row}"

    def _show_game_over(self) -> None:
        if self.in_reasoning_mode or self._end_dialog_shown:
            return
        self._end_dialog_shown = True
        if self.game.score_result is not None:
            score = self.game.score_result
            message = (
                "双方连续虚手，对局结束。按当前盘面采用中国数子法计分：\n\n"
                f"黑方：棋子 {score.black_stones} + 围空 {score.black_territory}"
                f" = {score.black_total:g}\n"
                f"白方：棋子 {score.white_stones} + 围空 {score.white_territory}"
                f" + 贴目 {score.komi:g} = {score.white_total:g}\n\n"
                f"{self.game.result_text}\n\n"
                "提示：程序不自动判定死子，终局前应先提净死子。"
            )
        else:
            message = self.game.result_text
        messagebox.showinfo("对局结果", message, parent=self.root)

    def close(self) -> None:
        self._closing = True
        self._invalidate_ai()
        analysis_window = getattr(self, "analysis_window", None)
        if analysis_window is not None:
            analysis_window.close()
            self.analysis_window = None
        if self.katago_engine is not None:
            self.katago_engine.close()
            self.katago_engine = None
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.training_window is not None:
            self.training_window.close()
            self.training_window = None
        if self.katago_settings_window is not None:
            self.katago_settings_window.close()
            self.katago_settings_window = None
        self._close_rules()
        self.root.destroy()


def run() -> None:
    root = tk.Tk()
    GoApp(root)
    root.mainloop()
