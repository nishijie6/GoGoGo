"""Window styles and the static widget layout of :class:`GoApp`."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ai import AI_DIFFICULTIES
from .gui_constants import MODE_AI, MODE_LOCAL


class GoAppLayout:
    """Mixed into :class:`~weiqi.gui.GoApp`; relies on the widgets, Tk
    variables and game state it creates, and on its other mixins."""

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
