"""Tkinter window for exploring rich KataGo position analysis.

The window is deliberately independent from :mod:`weiqi.gui`.  It renders
immutable workbench views, sends intent objects back to ``AnalysisWorkbench``,
and uses the caller-owned executor for every potentially blocking operation.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, Future
from typing import Any, Optional

from tkinter import ttk

from .engine import BLACK, EMPTY, WHITE, GoGame, Point


COLUMN_NAMES = "ABCDEFGHJKLMNOPQRST"
_BOARD_COLOR = "#d9a85f"
_PANEL_COLOR = "#172019"
_PANEL_DARK = "#101713"
_TEXT_COLOR = "#edf2eb"
_MUTED_COLOR = "#a9b5aa"
_ACCENT_COLOR = "#e5a447"
_SUCCESS_COLOR = "#69a979"
_ERROR_COLOR = "#e07a6d"


def _field(value: object, *names: str, default: Any = None) -> Any:
    """Read the first matching attribute/key from a workbench view object."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _sequence(value: object) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def _point_from(value: object) -> Optional[Point]:
    """Best-effort extraction used only for rendering coordinates."""

    if value is None:
        return None
    nested = _field(value, "point", "move", "vertex", default=value)
    if nested is not value:
        return _point_from(nested)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
    row = _field(value, "row")
    col = _field(value, "col", "column")
    if row is None or col is None:
        return None
    try:
        return int(row), int(col)
    except (TypeError, ValueError):
        return None


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_probability(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "—"
    if -1.0 <= number <= 1.0:
        number *= 100.0
    return f"{number:.1f}%"


def _format_lead(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "—"
    if abs(number) < 0.05:
        return "均势"
    return f"黑+{number:.1f}" if number > 0 else f"白+{-number:.1f}"


def _blend(color: str, background: str, strength: float) -> str:
    """Blend two ``#rrggbb`` colors without relying on Canvas alpha support."""

    strength = max(0.0, min(1.0, strength))
    foreground_rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    background_rgb = tuple(
        int(background[index : index + 2], 16) for index in (1, 3, 5)
    )
    mixed = tuple(
        round(base + (front - base) * strength)
        for front, base in zip(foreground_rgb, background_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _ownership_color(value: object) -> Optional[str]:
    number = _finite_float(value)
    if number is None:
        return None
    number = max(-1.0, min(1.0, number))
    if abs(number) < 0.04:
        return None
    foreground = "#315f95" if number > 0 else "#c95e55"
    return _blend(foreground, _BOARD_COLOR, 0.18 + abs(number) * 0.52)


class AnalysisWorkbenchWindow:
    """Non-modal KataGo analysis workbench.

    ``executor`` is owned by the caller.  This class never creates or shuts
    down an executor.  ``on_close`` is invoked exactly once after outstanding
    GUI futures have been invalidated, allowing the host to resume the formal
    game safely.
    """

    def __init__(
        self,
        parent: tk.Misc,
        formal_game: GoGame,
        analyzer: object,
        executor: Executor,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        # Imported lazily so basic helper tests remain useful while the core
        # workbench module is developed independently.
        from .analysis_workbench import AnalysisWorkbench

        self.parent = parent
        self.executor = executor
        self.on_close = on_close
        self.workbench = AnalysisWorkbench(analyzer)
        self.view: Optional[object] = None
        self._future: Optional[Future[object]] = None
        self._future_generation = 0
        self._poll_after_id: Optional[str] = None
        self._closed = False
        self._busy = False
        self._board_geometry: Optional[tuple[float, float, float]] = None
        self._hover_point: Optional[Point] = None
        self._candidate_by_item: dict[str, object] = {}
        self._node_by_item: dict[str, object] = {}
        self._history_hits: list[tuple[float, float, object]] = []

        self.window = tk.Toplevel(parent)
        self.window.title("AI 分析工作台 · 弈境")
        self.window.configure(bg=_PANEL_COLOR)
        self.window.minsize(1060, 680)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        self.window.bind("<Control-w>", lambda _event: self.close())
        self.window.bind("<F5>", lambda _event: self.analyze_current())

        self.status_var = tk.StringVar(value="正在准备正式棋局快照…")
        self.position_var = tk.StringVar(value="局面 —")
        self.summary_var = tk.StringVar(value="等待 KataGo 分析")

        self._configure_styles()
        self._build_layout()
        self._place_window()

        try:
            initial = self.workbench.sync_formal(formal_game)
        except Exception as error:
            self._show_error(f"无法载入正式棋局：{error}")
        else:
            self._render(initial)
            self.status_var.set("正式局面已载入，正在请求 KataGo 分析…")
            self.window.after_idle(self.analyze_current)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.window)
        style.configure("Analysis.TFrame", background=_PANEL_COLOR)
        style.configure("AnalysisDark.TFrame", background=_PANEL_DARK)
        style.configure(
            "Analysis.TLabel",
            background=_PANEL_COLOR,
            foreground=_TEXT_COLOR,
            font=("Microsoft YaHei UI", 10),
        )
        style.configure(
            "AnalysisTitle.TLabel",
            background=_PANEL_COLOR,
            foreground="#f3eadb",
            font=("Microsoft YaHei UI", 18, "bold"),
        )
        style.configure(
            "AnalysisMuted.TLabel",
            background=_PANEL_COLOR,
            foreground=_MUTED_COLOR,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Analysis.TLabelframe",
            background=_PANEL_COLOR,
            foreground=_TEXT_COLOR,
            bordercolor="#39463d",
        )
        style.configure(
            "Analysis.TLabelframe.Label",
            background=_PANEL_COLOR,
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
            bg=_BOARD_COLOR,
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
        self._future_generation += 1
        if self._poll_after_id is not None:
            try:
                self.window.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        if self._future is not None:
            self._future.cancel()
            self._future = None
        cancel = getattr(self.workbench, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass
        closer = getattr(self.workbench, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        if self.on_close is not None:
            callback, self.on_close = self.on_close, None
            callback()

    # Workbench commands are wired after the core types are imported in their
    # individual handlers.  This keeps every public click path one operation.

    def analyze_current(self) -> None:
        from .analysis_workbench import AnalysisSpec

        node_id = _field(self.view, "selected_node_id")
        if node_id is None:
            self.status_var.set("当前没有可以分析的局面。")
            return
        self._submit(
            "正在分析当前局面…",
            lambda: self.workbench.analyze(
                node_id,
                AnalysisSpec(force_refresh=True),
            ),
        )

    def _on_candidate_open(self, _event: object = None) -> None:
        selection = self.candidate_tree.selection()
        if not selection:
            return
        candidate = self._candidate_by_item.get(selection[0])
        if candidate is None:
            return
        self._follow_candidate(candidate)

    def _follow_candidate(self, candidate: object) -> None:
        from .analysis_workbench import FollowCandidate

        reference = _field(candidate, "ref", "candidate_ref", "id", default=candidate)
        self._submit(
            "正在进入候选分支…",
            lambda: self.workbench.explore(FollowCandidate(reference)),
        )

    def go_back(self) -> None:
        from .analysis_workbench import Back

        self._submit("正在撤回推演节点…", lambda: self.workbench.explore(Back()))

    def play_pass(self) -> None:
        self._play_variation(None)

    def _play_variation(self, point: Optional[Point]) -> None:
        from .analysis_workbench import PlayMove, VariationMove

        position = _field(self.view, "position")
        color = _field(position, "current_player", "to_play")
        if color not in (BLACK, WHITE):
            self.status_var.set("无法确定当前行棋方，不能添加推演着手。")
            return
        move = (
            VariationMove("pass", color)
            if point is None
            else VariationMove("play", color, point)
        )
        label = "正在推演虚手…" if point is None else "正在推演所选落点…"
        self._submit(label, lambda: self.workbench.explore(PlayMove(move)))

    def expand_selected_pv(self) -> None:
        selection = self.candidate_tree.selection()
        if selection:
            candidate = self._candidate_by_item.get(selection[0])
        else:
            candidate = next(iter(self._candidate_by_item.values()), None)
        if candidate is None:
            self.status_var.set("当前没有可以展开的候选 PV。")
            return
        pv = self._candidate_pv(candidate)
        if not pv:
            self.status_var.set("所选候选没有返回 PV。")
            return

        def apply_pv() -> object:
            from .analysis_workbench import FollowCandidate

            reference = _field(candidate, "ref", "candidate_ref", "id", default=candidate)
            return self.workbench.explore(
                FollowCandidate(reference, pv_plies=len(pv))
            )

        self._submit(f"正在展开 {len(pv)} 手 PV…", apply_pv)

    def _on_branch_selected(self, _event: object = None) -> None:
        selection = self.branch_tree.selection()
        if not selection or self._busy:
            return
        node = self._node_by_item.get(selection[0])
        if node is None:
            return
        from .analysis_workbench import SelectNode

        reference = _field(node, "ref", "position", "node_id", "id", default=node)
        self._submit(
            "正在切换推演节点…",
            lambda: self.workbench.explore(SelectNode(reference)),
        )

    def _on_history_click(self, event: tk.Event) -> None:
        if not self._history_hits or self._busy:
            return
        _, _, item = min(
            self._history_hits,
            key=lambda entry: (entry[0] - event.x) ** 2
            + (entry[1] - event.y) ** 2,
        )
        from .analysis_workbench import SelectNode

        reference = _field(item, "position", "ref", "node_id", "id", default=item)
        self._submit(
            "正在载入历史局面…",
            lambda: self.workbench.explore(SelectNode(reference)),
        )

    def _submit(self, message: str, operation: Callable[[], object]) -> None:
        if self._closed:
            return
        if self._future is not None and not self._future.done():
            self.status_var.set("上一项 KataGo 计算仍在进行，请稍候或关闭工作台。")
            return
        self._future_generation += 1
        generation = self._future_generation
        self._busy = True
        self.status_var.set(message)
        self._update_action_states()
        try:
            self._future = self.executor.submit(operation)
        except Exception as error:
            self._busy = False
            self._show_error(f"无法提交分析任务：{error}")
            self._update_action_states()
            return
        self._poll_after_id = self.window.after(
            60,
            lambda: self._poll_future(generation),
        )

    def _poll_future(self, generation: int) -> None:
        self._poll_after_id = None
        if self._closed or generation != self._future_generation:
            return
        future = self._future
        if future is None:
            return
        if not future.done():
            self._poll_after_id = self.window.after(
                60,
                lambda: self._poll_future(generation),
            )
            return
        self._future = None
        self._busy = False
        try:
            view = future.result()
        except Exception as error:
            self._show_error(f"KataGo 分析失败：{error}")
        else:
            self._render(view)
            error = _field(view, "error")
            if error:
                self._show_error(str(error))
            elif bool(_field(_field(view, "position"), "game_over", default=False)):
                self.status_var.set(
                    "当前推演分支已经结束；可撤回或选择其它变化节点。"
                )
            else:
                analysis = _field(view, "selected_analysis", "analysis")
                warnings = _sequence(_field(analysis, "warnings", default=()))
                if warnings:
                    self.status_var.set(f"分析结果已更新；提示：{warnings[0]}")
                else:
                    self.status_var.set("分析结果已更新。")
        self._update_action_states()

    def _show_error(self, message: str) -> None:
        self.status_var.set(message)
        if hasattr(self, "summary_var"):
            self.summary_var.set("分析不可用")

    def _render(self, view: object) -> None:
        if view is None:
            return
        self.view = view
        position = _field(view, "position")
        move_number = _field(position, "move_number", "ply", default=0)
        to_play = _field(position, "to_play", "current_player")
        to_play_text = "黑方" if to_play == BLACK else "白方" if to_play == WHITE else "—"
        self.position_var.set(f"第 {move_number} 手 · {to_play_text}行棋")

        analysis = _field(view, "selected_analysis", "analysis")
        evaluation = _field(analysis, "root", "evaluation", default=analysis)
        probability = _field(
            evaluation,
            "black_win_probability",
            "black_winrate",
            "winrate",
            "win_rate",
        )
        lead = _field(evaluation, "black_score_lead", "black_lead", "score_lead")
        visits = _field(evaluation, "visits", "analysis_visits")
        if evaluation is None:
            self.summary_var.set("尚未分析当前局面")
        else:
            visit_text = "—" if visits is None else str(visits)
            self.summary_var.set(
                f"黑胜率 {_format_probability(probability)}  ·  "
                f"{_format_lead(lead)}  ·  {visit_text} visits"
            )

        self._refresh_candidates(analysis)
        self._refresh_branches(_field(view, "tree"))
        self._draw_board()
        self._draw_history()
        self._update_action_states()

    def _analysis_candidates(self, analysis: object) -> tuple[object, ...]:
        return _sequence(_field(analysis, "candidates", "moves", default=()))

    def _candidate_point(self, candidate: object) -> Optional[Point]:
        return _point_from(_field(candidate, "move", "point"))

    def _candidate_pv(self, candidate: object) -> tuple[Optional[Point], ...]:
        result: list[Optional[Point]] = []
        for move in _sequence(_field(candidate, "pv", "principal_variation", default=())):
            kind = str(_field(move, "kind", default="")).lower()
            if kind == "pass" or move is None:
                result.append(None)
            else:
                point = _point_from(move)
                if point is not None:
                    result.append(point)
        return tuple(result)

    def _refresh_candidates(self, analysis: object) -> None:
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        self._candidate_by_item.clear()
        for index, candidate in enumerate(self._analysis_candidates(analysis), start=1):
            point = self._candidate_point(candidate)
            move_text = "虚手" if point is None else self._coordinate(point)
            evaluation = _field(candidate, "evaluation", default=candidate)
            probability = _field(
                evaluation,
                "black_win_probability",
                "black_winrate",
                "winrate",
                "win_rate",
            )
            lead = _field(
                evaluation,
                "black_score_lead",
                "black_lead",
                "score_lead",
            )
            visits = _field(evaluation, "visits", "analysis_visits", default="—")
            pv_text = " ".join(
                "虚手" if move is None else self._coordinate(move)
                for move in self._candidate_pv(candidate)
            )
            item = self.candidate_tree.insert(
                "",
                "end",
                values=(
                    f"{index}. {move_text}",
                    _format_probability(probability),
                    _format_lead(lead),
                    visits,
                    pv_text or "—",
                ),
            )
            self._candidate_by_item[item] = candidate

    def _tree_nodes(self, tree: object) -> tuple[object, ...]:
        if tree is None:
            return ()
        nodes = _field(tree, "nodes")
        if isinstance(nodes, Mapping):
            return tuple(nodes.values())
        if nodes is not None:
            return _sequence(nodes)
        return _sequence(tree)

    def _refresh_branches(self, tree: object) -> None:
        self.branch_tree.delete(*self.branch_tree.get_children())
        self._node_by_item.clear()
        nodes = self._tree_nodes(tree)
        if not nodes:
            return
        remaining = list(nodes)
        inserted: dict[object, str] = {}
        selected_ref = _field(
            self.view,
            "selected_node_id",
            "selected",
            "selected_ref",
        )
        self.branch_tree.tag_configure(
            "selected",
            background="#3e5f4a",
            foreground="#ffffff",
        )
        while remaining:
            progressed = False
            for node in remaining[:]:
                reference = _field(node, "ref", "position", "node_id", "id", default=id(node))
                parent_reference = _field(node, "parent", "parent_ref", "parent_id")
                parent_item = inserted.get(parent_reference, "")
                if parent_reference is not None and not parent_item:
                    continue
                incoming = _field(
                    node,
                    "incoming_move",
                    "move",
                    "move_from_parent",
                )
                point = _point_from(incoming)
                move_text = "根" if parent_reference is None else (
                    "虚手" if point is None else self._coordinate(point)
                )
                evaluation = _field(node, "evaluation", "analysis")
                probability = _field(
                    evaluation,
                    "black_win_probability",
                    "winrate",
                    "win_rate",
                )
                is_formal = bool(_field(node, "is_formal", default=False))
                has_analysis = bool(_field(node, "has_analysis", default=False))
                label = str(
                    _field(
                        node,
                        "label",
                        "name",
                        default=(
                            f"正式 {int(_field(node, 'ply', default=0))}"
                            if is_formal
                            else f"分支 {len(inserted) + 1}"
                        ),
                    )
                )
                item = self.branch_tree.insert(
                    parent_item,
                    "end",
                    text=label,
                    values=(
                        move_text,
                        _format_probability(probability)
                        if evaluation is not None
                        else ("已分析" if has_analysis else "待分析"),
                    ),
                    open=True,
                    tags=("selected",) if reference == selected_ref else (),
                )
                inserted[reference] = item
                self._node_by_item[item] = node
                if selected_ref is not None and reference == selected_ref:
                    self.branch_tree.see(item)
                remaining.remove(node)
                progressed = True
            if not progressed:
                # Malformed/cyclic input should remain visible rather than
                # hanging the Tk event loop.
                for node in remaining:
                    item = self.branch_tree.insert(
                        "",
                        "end",
                        text="孤立节点",
                        values=("—", "—"),
                    )
                    self._node_by_item[item] = node
                break

    def _position_board(self) -> tuple[tuple[int, ...], ...]:
        position = _field(self.view, "position")
        board = _field(position, "board", "stones", default=())
        rows = _sequence(board)
        result: list[tuple[int, ...]] = []
        for row in rows:
            try:
                result.append(tuple(int(value) for value in row))
            except (TypeError, ValueError):
                return ()
        return tuple(result)

    def _ownership(self, size: int) -> tuple[float, ...]:
        analysis = _field(self.view, "selected_analysis", "analysis")
        values = _field(analysis, "ownership", "black_ownership", default=())
        flat: list[float] = []
        for value in _sequence(values):
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for nested in value:
                    number = _finite_float(nested)
                    flat.append(0.0 if number is None else number)
            else:
                number = _finite_float(value)
                flat.append(0.0 if number is None else number)
        return tuple(flat) if len(flat) == size * size else ()

    def _draw_board(self) -> None:
        if not hasattr(self, "board_canvas"):
            return
        canvas = self.board_canvas
        width = max(100, canvas.winfo_width())
        height = max(100, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_rectangle(0, 0, width, height, fill=_BOARD_COLOR, outline="")
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
                    color = _ownership_color(ownership[row * size + col])
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

        analysis = _field(self.view, "selected_analysis", "analysis")
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
        if bool(_field(_field(self.view, "position"), "game_over", default=False)):
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

    def _history_items(self) -> tuple[object, ...]:
        history = _field(self.view, "history")
        points = _field(history, "points", "items")
        if points is not None:
            return _sequence(points)
        formal = _sequence(_field(history, "formal", default=()))
        active = _sequence(_field(history, "active", default=()))
        if formal or active:
            selected_id = _field(self.view, "selected_node_id")
            selected_node = next(
                (
                    node
                    for node in self._tree_nodes(_field(self.view, "tree"))
                    if _field(node, "id", "node_id") == selected_id
                ),
                None,
            )
            if bool(_field(selected_node, "is_formal", default=False)):
                return formal
            return active or formal
        return _sequence(history)

    def _draw_history(self) -> None:
        if not hasattr(self, "history_canvas"):
            return
        canvas = self.history_canvas
        width = max(100, canvas.winfo_width())
        height = max(80, canvas.winfo_height())
        canvas.delete("all")
        self._history_hits.clear()
        items = self._history_items()
        if not items:
            canvas.create_text(
                width / 2,
                height / 2,
                text="分析过的历史局面会显示在这里",
                fill="#7f9083",
                font=("Microsoft YaHei UI", 9),
            )
            return

        left, right, top, bottom = 38.0, width - 14.0, 14.0, height - 25.0
        canvas.create_line(left, top, left, bottom, fill="#58665c")
        canvas.create_line(left, bottom, right, bottom, fill="#58665c")
        canvas.create_line(left, (top + bottom) / 2, right, (top + bottom) / 2, fill="#344238", dash=(3, 3))

        move_numbers = [
            int(_field(item, "move_number", "ply", "index", default=index))
            for index, item in enumerate(items)
        ]
        minimum_move = min(move_numbers)
        maximum_move = max(move_numbers)
        span = max(1, maximum_move - minimum_move)
        win_points: list[tuple[float, float]] = []
        win_markers: list[tuple[float, float, str]] = []
        lead_points: list[tuple[float, float]] = []
        lead_values = [
            _finite_float(
                _field(
                    _field(item, "evaluation", "analysis", default=item),
                    "black_score_lead",
                    "black_lead",
                    "score_lead",
                )
            )
            for item in items
        ]
        lead_scale = max(5.0, max((abs(value) for value in lead_values if value is not None), default=5.0))

        for item, move_number, lead_value in zip(items, move_numbers, lead_values):
            evaluation = _field(item, "evaluation", "analysis", default=item)
            probability = _finite_float(
                _field(
                    evaluation,
                    "black_win_probability",
                    "black_winrate",
                    "winrate",
                    "win_rate",
                )
            )
            x = left + (move_number - minimum_move) / span * (right - left)
            if probability is not None:
                if probability > 1.0:
                    probability /= 100.0
                probability = max(0.0, min(1.0, probability))
                y = bottom - probability * (bottom - top)
                win_points.append((x, y))
                win_markers.append(
                    (x, y, str(_field(item, "source", default="katago")))
                )
                self._history_hits.append((x, y, item))
            if lead_value is not None:
                normalized = max(-1.0, min(1.0, lead_value / lead_scale))
                y = (top + bottom) / 2 - normalized * (bottom - top) * 0.45
                lead_points.append((x, y))
                self._history_hits.append((x, y, item))

        if len(win_points) > 1:
            canvas.create_line(*[coordinate for point in win_points for coordinate in point], fill="#e5ad52", width=2, smooth=True)
        if len(lead_points) > 1:
            canvas.create_line(*[coordinate for point in lead_points for coordinate in point], fill="#6fa6d8", width=2, smooth=True)
        for x, y, source in win_markers:
            exact = source == "katago"
            canvas.create_oval(
                x - 3,
                y - 3,
                x + 3,
                y + 3,
                fill="#e5ad52" if exact else "#101713",
                outline="#e5ad52" if exact else "#a69678",
            )

        canvas.create_text(4, top, text="100%", anchor="w", fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(8, bottom, text="0%", anchor="w", fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(left, height - 10, text=str(minimum_move), fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(right, height - 10, text=str(maximum_move), fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(
            right - 150,
            top + 7,
            text="胜率（实心=KataGo）",
            fill="#e5ad52",
            font=("Microsoft YaHei UI", 8),
        )
        canvas.create_text(right - 58, top + 7, text="目差", fill="#6fa6d8", font=("Microsoft YaHei UI", 8))

    def _update_action_states(self) -> None:
        if not hasattr(self, "analyze_button"):
            return
        unavailable = self._busy or self._closed
        position = _field(self.view, "position")
        game_over = bool(_field(position, "game_over", default=False))
        analysis = _field(self.view, "selected_analysis", "analysis")
        has_candidates = bool(self._analysis_candidates(analysis))
        selected_id = _field(self.view, "selected_node_id")
        selected_node = next(
            (
                node
                for node in self._tree_nodes(_field(self.view, "tree"))
                if _field(node, "id", "node_id") == selected_id
            ),
            None,
        )
        can_go_back = selected_node is not None and _field(
            selected_node,
            "parent_id",
            "parent",
            "parent_ref",
        ) is not None
        self.analyze_button.configure(
            state="disabled" if unavailable or game_over else "normal"
        )
        self.expand_pv_button.configure(
            state="disabled" if unavailable or not has_candidates else "normal"
        )
        self.back_button.configure(
            state="disabled" if unavailable or not can_go_back else "normal"
        )
        self.pass_button.configure(
            state="disabled" if unavailable or game_over else "normal"
        )

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


__all__ = ["AnalysisWorkbenchWindow"]
