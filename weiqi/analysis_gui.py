"""Tkinter window for exploring rich KataGo position analysis.

The window is deliberately independent from :mod:`weiqi.gui`.  It renders
immutable workbench views, sends intent objects back to ``AnalysisWorkbench``,
and uses the caller-owned executor for every potentially blocking operation.

This module keeps the window's lifecycle, user actions and async plumbing; it
combines layout (``analysis_layout``), board (``analysis_board``) and side
panels (``analysis_panels``), with shared colors and formatting in
``analysis_style``.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Optional

from .analysis_board import AnalysisBoard
from .analysis_layout import AnalysisLayout
from .analysis_panels import AnalysisPanels
from .analysis_style import PANEL_COLOR, field, format_lead, format_probability, sequence
from .engine import BLACK, WHITE, GoGame, Point


class AnalysisWorkbenchWindow(AnalysisLayout, AnalysisBoard, AnalysisPanels):
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
        self.window.configure(bg=PANEL_COLOR)
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

        node_id = field(self.view, "selected_node_id")
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

        reference = field(candidate, "ref", "candidate_ref", "id", default=candidate)
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

        position = field(self.view, "position")
        color = field(position, "current_player", "to_play")
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

            reference = field(candidate, "ref", "candidate_ref", "id", default=candidate)
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

        reference = field(node, "ref", "position", "node_id", "id", default=node)
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

        reference = field(item, "position", "ref", "node_id", "id", default=item)
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
            error = field(view, "error")
            if error:
                self._show_error(str(error))
            elif bool(field(field(view, "position"), "game_over", default=False)):
                self.status_var.set(
                    "当前推演分支已经结束；可撤回或选择其它变化节点。"
                )
            else:
                analysis = field(view, "selected_analysis", "analysis")
                warnings = sequence(field(analysis, "warnings", default=()))
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
        position = field(view, "position")
        move_number = field(position, "move_number", "ply", default=0)
        to_play = field(position, "to_play", "current_player")
        to_play_text = "黑方" if to_play == BLACK else "白方" if to_play == WHITE else "—"
        self.position_var.set(f"第 {move_number} 手 · {to_play_text}行棋")

        analysis = field(view, "selected_analysis", "analysis")
        evaluation = field(analysis, "root", "evaluation", default=analysis)
        probability = field(
            evaluation,
            "black_win_probability",
            "black_winrate",
            "winrate",
            "win_rate",
        )
        lead = field(evaluation, "black_score_lead", "black_lead", "score_lead")
        visits = field(evaluation, "visits", "analysis_visits")
        if evaluation is None:
            self.summary_var.set("尚未分析当前局面")
        else:
            visit_text = "—" if visits is None else str(visits)
            self.summary_var.set(
                f"黑胜率 {format_probability(probability)}  ·  "
                f"{format_lead(lead)}  ·  {visit_text} visits"
            )

        self._refresh_candidates(analysis)
        self._refresh_branches(field(view, "tree"))
        self._draw_board()
        self._draw_history()
        self._update_action_states()


__all__ = ["AnalysisWorkbenchWindow"]
