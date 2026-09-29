"""Integration tests for the main-window AI analysis lifecycle."""

from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from weiqi.ai import KATAGO_DIFFICULTIES
from weiqi.engine import BLACK, WHITE, GoGame
from weiqi.gui import MODE_AI, MODE_LOCAL, GoApp
from weiqi.katago import KataGoAI


class AnalysisLifecycleTests(unittest.TestCase):
    @staticmethod
    def variable(value: str) -> Mock:
        variable = Mock()
        variable.get.return_value = value
        return variable

    def test_open_workbench_pauses_formal_human_and_ai_actions(self) -> None:
        app = GoApp.__new__(GoApp)
        app.game = GoGame(9)
        app._reasoning_session = None
        app.analysis_window = Mock(is_alive=True)
        app.active_mode = MODE_AI
        app.human_color = WHITE
        app.ai_color = BLACK
        app.ai_busy = False

        self.assertTrue(app.analysis_active)
        self.assertFalse(app._is_ai_turn())
        self.assertFalse(app._human_can_act())

    def test_show_analysis_passes_exact_formal_game_and_shared_executor(self) -> None:
        app = GoApp.__new__(GoApp)
        formal = GoGame(9)
        executor = Mock()
        root = Mock()
        settings = Mock()
        settings.fingerprint = ("exe", "model", "")
        settings.require_valid.return_value = None
        engine = Mock()
        engine.closed = False
        engine.settings = settings
        window = Mock(is_alive=True)

        app.root = root
        app.game = formal
        app.executor = executor
        app.analysis_window = None
        app._reasoning_session = None
        app.katago_engine = None
        app.active_mode = MODE_LOCAL
        app.active_difficulty = "中等"
        app.notice_var = Mock()
        app.hover_point = None
        app._invalidate_ai = Mock()
        app._refresh = Mock()

        with patch("weiqi.gui.KataGoSettings.load", return_value=settings):
            with patch("weiqi.gui.KataGoEngine", return_value=engine):
                with patch(
                    "weiqi.gui.AnalysisWorkbenchWindow",
                    return_value=window,
                ) as window_type:
                    opened = app.show_analysis()

        self.assertTrue(opened)
        self.assertIs(app.analysis_window, window)
        self.assertIs(app.katago_engine, engine)
        app._invalidate_ai.assert_called_once_with()
        window_type.assert_called_once_with(
            root,
            formal,
            engine,
            executor,
            on_close=app._analysis_closed,
        )

    def test_analysis_reuses_the_game_pinned_training_model(self) -> None:
        app = GoApp.__new__(GoApp)
        app.root = Mock()
        app.game = GoGame(9)
        app.executor = Mock()
        app.analysis_window = None
        app._reasoning_session = None
        app.active_mode = MODE_AI
        app.active_difficulty = "训练·滚动冠军"
        app.active_training_opponent = object()
        app.notice_var = Mock()
        app._invalidate_ai = Mock()
        app._refresh = Mock()

        saved_settings = Mock()
        selected_settings = Mock()
        selected_settings.fingerprint = ("exe", "selected-training-model", "")
        selected_settings.require_valid.return_value = None
        engine = Mock(closed=False)
        engine.settings = selected_settings
        app.katago_engine = engine

        with patch("weiqi.gui.KataGoSettings.load", return_value=saved_settings):
            with patch(
                "weiqi.gui.settings_for_training_opponent",
                return_value=selected_settings,
            ) as select_model:
                with patch("weiqi.gui.KataGoEngine") as engine_type:
                    with patch("weiqi.gui.AnalysisWorkbenchWindow", return_value=Mock(is_alive=True)):
                        self.assertTrue(app.show_analysis())

        select_model.assert_called_once_with(
            saved_settings, app.active_training_opponent, 9,
        )
        engine_type.assert_not_called()
        self.assertIs(app.katago_engine, engine)
        self.assertIs(app.ai.engine, engine)

    def test_saved_settings_keep_the_training_model_for_analysis(self) -> None:
        app = GoApp.__new__(GoApp)
        app.root = Mock()
        app.game = GoGame(9)
        app._pending_katago_new_game = False
        app._pending_analysis_open = True
        app.analysis_window = None
        app.katago_engine = None
        app.active_mode = MODE_AI
        app.active_difficulty = "训练·滚动冠军"
        app.active_training_opponent = object()
        app.notice_var = Mock()
        app._invalidate_ai = Mock()
        selected_settings = Mock()
        engine = Mock()
        engine.settings = selected_settings

        with patch(
            "weiqi.gui.settings_for_training_opponent",
            return_value=selected_settings,
        ) as select_model:
            with patch("weiqi.gui.KataGoEngine", return_value=engine):
                saved_settings = Mock()
                app._katago_settings_saved(saved_settings)

        select_model.assert_called_once_with(
            saved_settings, app.active_training_opponent, 9,
        )
        self.assertIs(app.ai.engine, engine)
        app.root.after.assert_called_once_with(80, app.show_analysis)

    def test_close_workbench_preserves_formal_state_and_resumes_ai(self) -> None:
        formal = GoGame(9)
        self.assertTrue(formal.play(4, 4).legal)
        before = (
            formal.board_hash(),
            formal.current_player,
            tuple(formal.moves),
            dict(formal.captures),
            formal.can_undo,
        )
        app = GoApp.__new__(GoApp)
        app.root = Mock()
        app.game = formal
        app._reasoning_session = None
        app.analysis_window = Mock(is_alive=True)
        app.katago_engine = Mock()
        app.active_mode = MODE_AI
        app.human_color = BLACK
        app.ai_color = WHITE
        app.ai_busy = False
        app.hover_point = None
        app._closing = False
        app._refresh = Mock()
        app._start_ai_turn = Mock()

        app._analysis_closed()

        self.assertIsNone(app.analysis_window)
        app.katago_engine.stop.assert_called_once_with()
        self.assertEqual(
            (
                formal.board_hash(),
                formal.current_player,
                tuple(formal.moves),
                dict(formal.captures),
                formal.can_undo,
            ),
            before,
        )
        app.root.after.assert_called_once_with(220, app._start_ai_turn)

    def test_katago_ai_turn_receives_a_cancel_event_before_submission(self) -> None:
        app = GoApp.__new__(GoApp)
        engine = Mock()
        engine.settings.human_style_enabled = False
        app.ai = KataGoAI(engine, KATAGO_DIFFICULTIES[0])
        app.game = GoGame(9)
        app._reasoning_session = None
        app.analysis_window = None
        app.active_mode = MODE_AI
        app.active_difficulty = KATAGO_DIFFICULTIES[0]
        app.human_color = WHITE
        app.ai_color = BLACK
        app.ai_busy = False
        app.generation = 7
        app.executor = Mock()
        app.root = Mock()
        app._refresh = Mock()

        app._start_ai_turn()

        submitted = app.executor.submit.call_args.args
        self.assertEqual(len(submitted), 3)
        self.assertEqual(submitted[0], app.ai.choose_move)
        self.assertIsInstance(submitted[1], GoGame)
        self.assertIsNot(submitted[1], app.game)
        self.assertEqual(submitted[1].board_hash(), app.game.board_hash())
        self.assertIs(submitted[2], app._ai_cancel_event)
        self.assertFalse(app._ai_cancel_event.is_set())

    def test_successful_new_game_closes_an_open_analysis_workbench(self) -> None:
        app = GoApp.__new__(GoApp)
        analysis_window = Mock(is_alive=True)
        app.analysis_window = analysis_window
        app.game = GoGame(9)
        app._reasoning_session = None
        app.size_var = self.variable("13×13")
        app.mode_var = self.variable(MODE_LOCAL)
        app.difficulty_var = self.variable("中等")
        app.human_color_var = self.variable("黑方（先手）")
        app.notice_var = Mock()
        app.katago_engine = None
        app._invalidate_ai = Mock()
        app._on_mode_selected = Mock()
        app._refresh = Mock()
        app._is_ai_turn = Mock(return_value=False)
        app.root = Mock()

        app.new_game()

        analysis_window.close.assert_called_once_with()
        self.assertEqual(app.game.size, 13)

    def test_f3_visibly_disables_f4_until_reasoning_exits(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"当前环境没有 Tk 显示服务：{error}")
        root.withdraw()
        app = GoApp(root)
        try:
            self.assertTrue(app.enter_reasoning_mode())
            root.update_idletasks()
            self.assertEqual(str(app.analysis_button.cget("state")), "disabled")
            self.assertTrue(app.exit_reasoning_mode())
            root.update_idletasks()
            self.assertEqual(str(app.analysis_button.cget("state")), "normal")
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
