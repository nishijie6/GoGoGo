"""Widget layout and placement of the reasoning-trainer window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .training import TRAINING_CATEGORIES


class TrainerLayout:
    """Mixed into :class:`~weiqi.training_gui.ReasoningTrainer`; relies on the
    widgets, variables and lesson state it creates."""

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
