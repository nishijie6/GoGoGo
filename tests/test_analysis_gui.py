"""Focused tests for the AI analysis window's rendering and async flow."""

from __future__ import annotations

import time
import tkinter as tk
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import Mock

from weiqi.analysis_gui import (
    AnalysisWorkbenchWindow,
    _format_lead,
    _format_probability,
    _ownership_color,
    _point_from,
)
from weiqi.engine import GoGame


class _ImmediateAnalyzer:
    def analyze_position(
        self,
        game: GoGame,
        max_visits: int,
        pv_length: int,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        del max_visits, pv_length, include_ownership
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled")
        return {
            "rootInfo": {
                "winrate": 0.55,
                "scoreLead": 1.5,
                "visits": 40,
                "currentPlayer": "B",
            },
            "ownership": [0.25] * (game.size * game.size),
            "moveInfos": [
                {
                    "move": "D4",
                    "order": 0,
                    "pv": ["D4", "E4"],
                    "winrate": 0.57,
                    "scoreLead": 2.0,
                    "visits": 24,
                    "prior": 0.2,
                }
            ],
        }


class _PrestartAnalyzer:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.external_started = False

    def analyze_position(
        self,
        game: GoGame,
        max_visits: int,
        pv_length: int,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        del game, max_visits, pv_length, include_ownership
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test barrier timed out")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled before external start")
        self.external_started = True
        raise AssertionError("cancelled work unexpectedly reached the external start")


class _PassAnalyzer:
    def analyze_position(
        self,
        game: GoGame,
        max_visits: int,
        pv_length: int,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        del max_visits, pv_length, include_ownership, cancel_event
        return {
            "rootInfo": {
                "winrate": 0.3,
                "scoreLead": -4.0,
                "visits": 20,
                "currentPlayer": "B" if game.current_player == 1 else "W",
            },
            "ownership": [0.0] * (game.size * game.size),
            "moveInfos": [
                {
                    "move": "pass",
                    "order": 0,
                    "pv": ["pass"],
                    "winrate": 0.3,
                    "scoreLead": -4.0,
                    "visits": 20,
                    "prior": 0.1,
                }
            ],
        }


class AnalysisGuiHelperTests(unittest.TestCase):
    def test_display_helpers_use_consistent_black_perspective(self) -> None:
        self.assertEqual(_format_probability(0.571), "57.1%")
        self.assertEqual(_format_lead(2.25), "黑+2.2")
        self.assertEqual(_format_lead(-1.75), "白+1.8")
        self.assertIsNotNone(_ownership_color(0.5))
        self.assertIsNotNone(_ownership_color(-0.5))
        self.assertIsNone(_ownership_color(0.0))
        self.assertEqual(_point_from((3, 4)), (3, 4))

    def test_history_hit_testing_honors_the_lead_curve_y_coordinate(self) -> None:
        window = AnalysisWorkbenchWindow.__new__(AnalysisWorkbenchWindow)
        first = SimpleNamespace(node_id="first")
        second = SimpleNamespace(node_id="second")
        window._busy = False
        window._history_hits = [
            (0.0, 0.0, first),
            (0.0, 100.0, first),
            (2.0, 100.0, second),
        ]
        window.workbench = Mock()
        window._submit = lambda _message, operation: operation()

        window._on_history_click(SimpleNamespace(x=0.0, y=100.0))

        action = window.workbench.explore.call_args.args[0]
        self.assertEqual(action.node_id, "first")

    def test_hidden_window_auto_analyzes_without_mutating_formal_game(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"当前环境没有 Tk 显示服务：{error}")
        root.withdraw()
        executor = ThreadPoolExecutor(max_workers=1)
        formal = GoGame(9)
        before = (formal.board_hash(), tuple(formal.moves), formal.current_player)
        closed: list[bool] = []
        window: Optional[AnalysisWorkbenchWindow] = None
        try:
            window = AnalysisWorkbenchWindow(
                root,
                formal,
                _ImmediateAnalyzer(),
                executor,
                on_close=lambda: closed.append(True),
            )
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                root.update()
                if (
                    window.view is not None
                    and window.view.selected_analysis is not None
                    and not window._busy
                ):
                    break
                time.sleep(0.01)
            else:
                self.fail(f"分析窗口未完成自动分析：{window.status_var.get()}")

            self.assertEqual(len(window.view.selected_analysis.candidates), 1)
            self.assertEqual(len(window.view.selected_analysis.ownership), 81)
            self.assertEqual(
                (formal.board_hash(), tuple(formal.moves), formal.current_player),
                before,
            )
        finally:
            if window is not None:
                window.close()
            executor.shutdown(wait=True, cancel_futures=True)
            root.destroy()
        self.assertEqual(closed, [True])

    def test_close_cancels_a_worker_that_has_not_started_external_io(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"当前环境没有 Tk 显示服务：{error}")
        root.withdraw()
        executor = ThreadPoolExecutor(max_workers=1)
        analyzer = _PrestartAnalyzer()
        window: Optional[AnalysisWorkbenchWindow] = None
        try:
            window = AnalysisWorkbenchWindow(
                root,
                GoGame(9),
                analyzer,
                executor,
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not analyzer.entered.is_set():
                root.update()
                time.sleep(0.01)
            self.assertTrue(analyzer.entered.is_set())
            running_future = window._future
            self.assertIsNotNone(running_future)

            window.close()
            analyzer.release.set()
            assert running_future is not None
            with self.assertRaisesRegex(Exception, "cancelled before external start"):
                running_future.result(timeout=2.0)
            self.assertFalse(analyzer.external_started)
            self.assertEqual(executor.submit(lambda: "free").result(timeout=1.0), "free")
        finally:
            analyzer.release.set()
            if window is not None:
                window.close()
            executor.shutdown(wait=True, cancel_futures=True)
            root.destroy()

    def test_terminal_variation_disables_analysis_and_pass_but_keeps_back(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"当前环境没有 Tk 显示服务：{error}")
        root.withdraw()
        executor = ThreadPoolExecutor(max_workers=1)
        formal = GoGame(9)
        self.assertTrue(formal.pass_turn())
        window: Optional[AnalysisWorkbenchWindow] = None
        try:
            window = AnalysisWorkbenchWindow(
                root,
                formal,
                _PassAnalyzer(),
                executor,
            )

            def wait_until(predicate: Any) -> None:
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    root.update()
                    if predicate():
                        return
                    time.sleep(0.01)
                self.fail(f"分析窗口状态未按时更新：{window.status_var.get()}")

            wait_until(
                lambda: window.view is not None
                and window.view.selected_analysis is not None
                and not window._busy
            )
            candidate = window.view.selected_analysis.candidates[0]
            window._follow_candidate(candidate)
            wait_until(
                lambda: not window._busy and window.view.position.game_over
            )

            self.assertEqual(str(window.analyze_button.cget("state")), "disabled")
            self.assertEqual(str(window.pass_button.cget("state")), "disabled")
            self.assertEqual(str(window.back_button.cget("state")), "normal")
            self.assertIn("已经结束", window.status_var.get())
        finally:
            if window is not None:
                window.close()
            executor.shutdown(wait=True, cancel_futures=True)
            root.destroy()


if __name__ == "__main__":
    unittest.main()
