"""Styles, widget layout and placement of the AI analysis window."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .analysis_style import BOARD_COLOR, MUTED_COLOR, PANEL_COLOR, PANEL_DARK, TEXT_COLOR


class AnalysisLayout:
    """Mixed into :class:`~weiqi.analysis_gui.AnalysisWorkbenchWindow`; relies on
    the widgets, variables and view state it creates."""

    def _configure_styles(self) -> None:
        style = ttk.Style(self.window)
        style.configure("Analysis.TFrame", background=PANEL_COLOR)
        style.configure("AnalysisDark.TFrame", background=PANEL_DARK)
        style.configure(
            "Analysis.TLabel",
            background=PANEL_COLOR,
            foreground=TEXT_COLOR,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "AnalysisTitle.TLabel",
            background=PANEL_COLOR,
            foreground="#f3eadb",
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "AnalysisMuted.TLabel",
            background=PANEL_COLOR,
            foreground=MUTED_COLOR,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Analysis.TLabelframe",
            background=PANEL_COLOR,
            foreground=TEXT_COLOR,
            bordercolor="#39463d",
        )
        style.configure(
            "Analysis.TLabelframe.Label",
            background=PANEL_COLOR,
            foreground="#e8c98f",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Analysis.Treeview",
            background="#131c16",
            fieldbackground="#131c16",
            foreground="#e6ece5",
            rowheight=24,
            borderwidth=0,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Analysis.Treeview.Heading",
            background="#263229",
            foreground="#f2e7d5",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map(
            "Analysis.Treeview",
            background=[("selected", "#496552")],
            foreground=[("selected", "#ffffff")],
        )

    def _build_layout(self) -> None:
        shell = ttk.Frame(
            self.window,
            style="Analysis.TFrame",
            padding=(16, 12, 16, 16),
        )
        shell.grid(row=0, column=0, sticky="nsew")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=3)
        shell.columnconfigure(1, weight=2)

        header = ttk.Frame(shell, style="Analysis.TFrame")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header,
            text="AI 分析工作台",
            style="AnalysisTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            textvariable=self.position_var,
            style="AnalysisMuted.TLabel",
        ).grid(row=0, column=1, sticky="sw", padx=(14, 0), pady=(0, 3))
        ttk.Button(header, text="关闭  Esc", command=self.close).grid(
            row=0, column=2, sticky="e"
        )

        left = ttk.Frame(shell, style="Analysis.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        board_shell = tk.Frame(
            left,
            bg="#0e1510",
            highlightbackground="#354239",
            highlightthickness=1,
            bd=0,
        )
        board_shell.grid(row=0, column=0, sticky="nsew")
        board_shell.rowconfigure(0, weight=1)
        board_shell.columnconfigure(0, weight=1)
        self.board_canvas = tk.Canvas(
            board_shell,
            bg=BOARD_COLOR,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.board_canvas.grid(row=0, column=0, sticky="nsew", padx=7, pady=7)
        self.board_canvas.bind("<Configure>", lambda _event: self._draw_board())
        self.board_canvas.bind("<Button-1>", self._on_board_click)
        self.board_canvas.bind("<Motion>", self._on_board_motion)
        self.board_canvas.bind("<Leave>", self._on_board_leave)

        actions = ttk.Frame(left, style="Analysis.TFrame")
        actions.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column in range(5):
            actions.columnconfigure(column, weight=1)
        self.analyze_button = ttk.Button(
            actions,
            text="分析当前  F5",
            command=self.analyze_current,
        )
        self.analyze_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.expand_pv_button = ttk.Button(
            actions,
            text="展开 PV",
            command=self.expand_selected_pv,
        )
        self.expand_pv_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.back_button = ttk.Button(actions, text="撤回", command=self.go_back)
        self.back_button.grid(row=0, column=2, sticky="ew", padx=4)
        self.pass_button = ttk.Button(actions, text="虚手", command=self.play_pass)
        self.pass_button.grid(row=0, column=3, sticky="ew", padx=4)
        ttk.Button(actions, text="关闭", command=self.close).grid(
            row=0, column=4, sticky="ew", padx=(4, 0)
        )

        ttk.Label(
            left,
            textvariable=self.status_var,
            style="AnalysisMuted.TLabel",
            anchor="w",
            wraplength=680,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            left,
            text=(
                "热图：蓝色偏黑方控制，红色偏白方控制；"
                "候选圆点中的数字对应右侧排名。"
            ),
            style="AnalysisMuted.TLabel",
            anchor="w",
            wraplength=680,
        ).grid(row=3, column=0, sticky="ew", pady=(3, 0))

        right = ttk.Frame(shell, style="Analysis.TFrame")
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(1, weight=3)
        right.rowconfigure(2, weight=2)
        right.rowconfigure(3, weight=2)
        right.columnconfigure(0, weight=1)

        summary = tk.Frame(
            right,
            bg="#233028",
            highlightbackground="#3b4b40",
            highlightthickness=1,
            padx=10,
            pady=8,
        )
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            summary,
            textvariable=self.summary_var,
            bg="#233028",
            fg="#f0e6d6",
            anchor="w",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(fill="x")

        candidates_frame = ttk.LabelFrame(
            right,
            text="候选着 · 双击进入分支",
            style="Analysis.TLabelframe",
            padding=5,
        )
        candidates_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        candidates_frame.rowconfigure(0, weight=1)
        candidates_frame.columnconfigure(0, weight=1)
        candidate_columns = ("move", "winrate", "lead", "visits", "pv")
        self.candidate_tree = ttk.Treeview(
            candidates_frame,
            columns=candidate_columns,
            show="headings",
            style="Analysis.Treeview",
            selectmode="browse",
        )
        headings = {
            "move": ("候选", 62, "center"),
            "winrate": ("黑胜率", 70, "e"),
            "lead": ("目差", 72, "e"),
            "visits": ("访问", 64, "e"),
            "pv": ("PV", 240, "w"),
        }
        for name, (text, width, anchor) in headings.items():
            self.candidate_tree.heading(name, text=text)
            self.candidate_tree.column(
                name,
                width=width,
                minwidth=45,
                stretch=name == "pv",
                anchor=anchor,
            )
        candidate_scroll = ttk.Scrollbar(
            candidates_frame,
            orient="vertical",
            command=self.candidate_tree.yview,
        )
        self.candidate_tree.configure(yscrollcommand=candidate_scroll.set)
        self.candidate_tree.grid(row=0, column=0, sticky="nsew")
        candidate_scroll.grid(row=0, column=1, sticky="ns")
        self.candidate_tree.bind("<Double-1>", self._on_candidate_open)

        branches_frame = ttk.LabelFrame(
            right,
            text="多分支推演树",
            style="Analysis.TLabelframe",
            padding=5,
        )
        branches_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        branches_frame.rowconfigure(0, weight=1)
        branches_frame.columnconfigure(0, weight=1)
        self.branch_tree = ttk.Treeview(
            branches_frame,
            columns=("move", "evaluation"),
            show="tree headings",
            style="Analysis.Treeview",
            selectmode="browse",
        )
        self.branch_tree.heading("#0", text="节点")
        self.branch_tree.heading("move", text="着手")
        self.branch_tree.heading("evaluation", text="评估")
        self.branch_tree.column("#0", width=110, minwidth=80)
        self.branch_tree.column("move", width=70, minwidth=55, anchor="center")
        self.branch_tree.column("evaluation", width=170, minwidth=100)
        branch_scroll = ttk.Scrollbar(
            branches_frame,
            orient="vertical",
            command=self.branch_tree.yview,
        )
        self.branch_tree.configure(yscrollcommand=branch_scroll.set)
        self.branch_tree.grid(row=0, column=0, sticky="nsew")
        branch_scroll.grid(row=0, column=1, sticky="ns")
        self.branch_tree.bind("<<TreeviewSelect>>", self._on_branch_selected)

        history_frame = ttk.LabelFrame(
            right,
            text="历史曲线 · 点击查看局面",
            style="Analysis.TLabelframe",
            padding=5,
        )
        history_frame.grid(row=3, column=0, sticky="nsew")
        history_frame.rowconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)
        self.history_canvas = tk.Canvas(
            history_frame,
            bg="#101713",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
            height=165,
        )
        self.history_canvas.grid(row=0, column=0, sticky="nsew")
        self.history_canvas.bind("<Configure>", lambda _event: self._draw_history())
        self.history_canvas.bind("<Button-1>", self._on_history_click)

    def _place_window(self) -> None:
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        width = min(1380, max(1060, screen_width - 70))
        height = min(900, max(680, screen_height - 90))
        try:
            self.parent.update_idletasks()
            parent_x = self.parent.winfo_rootx()
            parent_y = self.parent.winfo_rooty()
            parent_width = self.parent.winfo_width()
            parent_height = self.parent.winfo_height()
        except tk.TclError:
            parent_x = parent_y = 0
            parent_width = screen_width
            parent_height = screen_height
        x = max(0, parent_x + (parent_width - width) // 2)
        y = max(0, parent_y + (parent_height - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.focus_set()
