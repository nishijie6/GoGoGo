"""Tests for the isolated in-game reasoning variation."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from weiqi.ai import KATAGO_DIFFICULTIES
from weiqi.engine import BLACK, EMPTY, GoGame
from weiqi.gui import MODE_AI, MODE_LOCAL, GoApp
from weiqi.katago import KataGoConfigurationError
from weiqi.reasoning import ReasoningSession


class ReasoningSessionTests(unittest.TestCase):
    @staticmethod
    def variable(value: str) -> Mock:
        variable = Mock()
        variable.get.return_value = value
        return variable

    def play_ok(self, game: GoGame, row: int, col: int) -> None:
        result = game.play(row, col)
        self.assertTrue(result.legal, result.reason)

    def test_variation_is_isolated_and_cannot_undo_past_saved_position(self) -> None:
        formal_game = GoGame(9)
        self.play_ok(formal_game, 4, 4)
        self.play_ok(formal_game, 3, 3)
        formal_state = (
            formal_game.board_hash(),
            formal_game.current_player,
            tuple(formal_game.moves),
            dict(formal_game.captures),
        )

        session = ReasoningSession.start(formal_game)

        self.assertIsNot(session.variation, formal_game)
        self.assertEqual(session.variation.board_hash(), formal_game.board_hash())
        self.assertFalse(session.variation.can_undo)
        self.play_ok(session.variation, 2, 2)
        self.play_ok(session.variation, 6, 6)
        self.assertEqual(session.variation_move_count, 2)
        self.assertEqual(session.variation.undo(3), 2)
        self.assertEqual(session.variation_move_count, 0)
        self.assertFalse(session.variation.can_undo)
        self.assertEqual(
            (
                formal_game.board_hash(),
                formal_game.current_player,
                tuple(formal_game.moves),
                dict(formal_game.captures),
            ),
            formal_state,
        )

    def test_restore_discards_variation_and_formal_game_continues_normally(self) -> None:
        formal_game = GoGame(9)
        self.play_ok(formal_game, 4, 4)
        self.play_ok(formal_game, 3, 3)
        session = ReasoningSession.start(formal_game)
        self.play_ok(session.variation, 2, 2)

        restored = session.restore_formal_game()

        self.assertIs(restored, formal_game)
        self.assertEqual(restored.board[2][2], EMPTY)
        self.assertEqual(restored.current_player, BLACK)
        self.play_ok(restored, 5, 5)
        self.assertEqual(restored.move_number, 3)
        self.assertEqual(restored.undo(), 1)
        self.assertEqual(restored.move_number, 2)
        self.assertTrue(restored.can_undo)

    def test_branch_end_and_undo_do_not_end_formal_game(self) -> None:
        formal_game = GoGame(9)
        self.play_ok(formal_game, 4, 4)
        session = ReasoningSession.start(formal_game)

        self.assertTrue(session.variation.pass_turn())
        self.assertTrue(session.variation.pass_turn())
        self.assertTrue(session.variation.game_over)
        self.assertFalse(formal_game.game_over)
        self.assertEqual(session.variation.undo(), 1)
        self.assertFalse(session.variation.game_over)
        self.assertFalse(formal_game.game_over)

    def test_finished_game_rejects_new_reasoning_session(self) -> None:
        game = GoGame(9)
        self.assertTrue(game.pass_turn())
        self.assertTrue(game.pass_turn())

        with self.assertRaisesRegex(ValueError, "已经结束"):
            ReasoningSession.start(game)

    def test_reasoning_mode_suppresses_ai_and_allows_both_sides_to_move(self) -> None:
        formal_game = GoGame(9)
        self.play_ok(formal_game, 4, 4)
        app = GoApp.__new__(GoApp)
        app.game = formal_game
        app._reasoning_session = None
        app.active_mode = MODE_AI
        app.human_color = BLACK
        app.ai_color = formal_game.current_player
        app.ai_busy = False

        self.assertTrue(app._is_ai_turn())
        self.assertFalse(app._human_can_act())

        app._reasoning_session = ReasoningSession.start(formal_game)
        app.game = app._reasoning_session.variation

        self.assertFalse(app._is_ai_turn())
        self.assertTrue(app._human_can_act())

    def test_enter_and_exit_invalidate_ai_and_resume_saved_ai_turn(self) -> None:
        formal_game = GoGame(9)
        self.play_ok(formal_game, 4, 4)
        app = GoApp.__new__(GoApp)
        app.game = formal_game
        app._reasoning_session = None
        app.active_mode = MODE_AI
        app.human_color = BLACK
        app.ai_color = formal_game.current_player
        app.ai_busy = False
        app._invalidate_ai = Mock()
        app._refresh = Mock()
        app.root = Mock()
        app.hover_point = None
        app._end_dialog_shown = False
        app._winrate_cache_key = None
        app._last_winrate = None
        app._winrate_source = "启发式估算"

        self.assertTrue(app.enter_reasoning_mode())
        self.assertIsNot(app.game, formal_game)
        self.play_ok(app.game, 3, 3)
        self.assertTrue(app.exit_reasoning_mode())

        self.assertIs(app.game, formal_game)
        self.assertEqual(app._invalidate_ai.call_count, 2)
        app.root.after.assert_called_once_with(220, app._start_ai_turn)

    def test_failed_katago_new_game_keeps_reasoning_restore_point(self) -> None:
        formal_game = GoGame(9)
        session = ReasoningSession.start(formal_game)
        app = GoApp.__new__(GoApp)
        app.game = session.variation
        app._reasoning_session = session
        app.size_var = self.variable("9×9")
        app.mode_var = self.variable(MODE_AI)
        app.difficulty_var = self.variable(KATAGO_DIFFICULTIES[0])
        app.notice_var = Mock()
        app.show_katago_settings = Mock()
        invalid_settings = Mock()
        invalid_settings.require_valid.side_effect = KataGoConfigurationError("测试无效配置")

        with patch("weiqi.gui.KataGoSettings.load", return_value=invalid_settings):
            app.new_game()

        self.assertIs(app._reasoning_session, session)
        self.assertIs(app.game, session.variation)
        app.show_katago_settings.assert_called_once_with(pending_new_game=True)

    def test_successful_new_game_discards_reasoning_session(self) -> None:
        formal_game = GoGame(9)
        session = ReasoningSession.start(formal_game)
        app = GoApp.__new__(GoApp)
        app.game = session.variation
        app._reasoning_session = session
        app.size_var = self.variable("9×9")
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

        self.assertIsNone(app._reasoning_session)
        self.assertIsNot(app.game, session.variation)
        self.assertEqual(app.game.move_number, 0)
        self.assertEqual(app.active_mode, MODE_LOCAL)


if __name__ == "__main__":
    unittest.main()
