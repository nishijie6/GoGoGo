"""Tests for captures, legality, ko, scoring, undo, and the AI contract."""

from __future__ import annotations

import unittest

from weiqi.ai import AI_DIFFICULTIES, GoAI
from weiqi.engine import BLACK, EMPTY, WHITE, GoGame


class GoGameTests(unittest.TestCase):
    def play_ok(self, game: GoGame, row: int, col: int) -> None:
        result = game.play(row, col)
        self.assertTrue(result.legal, result.reason)

    def test_supported_board_sizes_and_initial_state(self) -> None:
        for size in (9, 13, 19):
            game = GoGame(size)
            self.assertEqual(len(game.board), size)
            self.assertTrue(all(len(row) == size for row in game.board))
            self.assertEqual(game.current_player, BLACK)
            self.assertEqual(game.move_number, 0)
        with self.assertRaises(ValueError):
            GoGame(11)

    def test_illegal_moves_are_atomic(self) -> None:
        game = GoGame(9)
        self.play_ok(game, 4, 4)
        before = (
            game.board_hash(),
            game.current_player,
            dict(game.captures),
            game.move_number,
        )
        result = game.play(4, 4)
        self.assertFalse(result.legal)
        after = (
            game.board_hash(),
            game.current_player,
            dict(game.captures),
            game.move_number,
        )
        self.assertEqual(before, after)

    def test_single_stone_capture(self) -> None:
        game = GoGame(9)
        sequence = [
            (0, 1),  # Black
            (1, 1),  # White stone to surround
            (1, 0),
            (8, 8),
            (2, 1),
            (8, 7),
        ]
        for point in sequence:
            self.play_ok(game, *point)
        result = game.play(1, 2)
        self.assertTrue(result.legal)
        self.assertEqual(result.captured, 1)
        self.assertEqual(game.board[1][1], EMPTY)
        self.assertEqual(game.captures[BLACK], 1)

    def test_suicide_is_rejected(self) -> None:
        game = GoGame(9)
        sequence = [
            (8, 8),
            (0, 1),
            (8, 7),
            (1, 0),
            (7, 8),
            (2, 1),
            (7, 7),
            (1, 2),
        ]
        for point in sequence:
            self.play_ok(game, *point)
        before = game.board_hash()
        result = game.play(1, 1)
        self.assertFalse(result.legal)
        self.assertIn("没有气", result.reason)
        self.assertEqual(game.board_hash(), before)
        self.assertEqual(game.current_player, BLACK)

    def test_ko_recapture_is_rejected(self) -> None:
        game = GoGame(9)
        sequence = [
            (0, 1),  # B
            (1, 1),  # W: ko stone
            (1, 0),  # B
            (0, 2),  # W
            (2, 1),  # B
            (2, 2),  # W
        ]
        for point in sequence:
            self.play_ok(game, *point)
        self.assertTrue(game.pass_turn())  # B balances the setup move count.
        self.play_ok(game, 1, 3)  # W completes the ko shape.
        capture = game.play(1, 2)
        self.assertTrue(capture.legal)
        self.assertEqual(capture.captured, 1)

        recapture = game.play(1, 1)
        self.assertFalse(recapture.legal)
        self.assertIn("劫争", recapture.reason)

    def test_pass_end_score_and_undo(self) -> None:
        game = GoGame(9, komi=6.5)
        self.assertTrue(game.pass_turn())
        self.assertFalse(game.game_over)
        self.assertTrue(game.pass_turn())
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, WHITE)
        self.assertAlmostEqual(game.margin, 6.5)

        self.assertEqual(game.undo(), 1)
        self.assertFalse(game.game_over)
        self.assertEqual(game.consecutive_passes, 1)
        self.assertEqual(game.current_player, WHITE)

    def test_chinese_area_score_finds_enclosed_point(self) -> None:
        game = GoGame(9, komi=0)
        for row, col in ((3, 4), (5, 4), (4, 3), (4, 5)):
            game.board[row][col] = BLACK
        game.board[0][0] = WHITE
        score = game.calculate_score()
        self.assertEqual(score.black_stones, 4)
        self.assertEqual(score.white_stones, 1)
        self.assertEqual(score.black_territory, 1)
        self.assertEqual(score.white_territory, 0)

    def test_clone_is_independent(self) -> None:
        game = GoGame(9)
        self.play_ok(game, 4, 4)
        clone = game.clone()
        self.play_ok(clone, 3, 3)
        self.assertEqual(game.board[3][3], EMPTY)
        self.assertEqual(game.move_number, 1)


class GoAITests(unittest.TestCase):
    def test_difficulty_levels_are_complete_and_strictly_ordered(self) -> None:
        expected = (
            "简单",
            "中等",
            "难",
            *(f"业余棋手{dan}段" for dan in range(1, 6)),
            *(f"职业棋手{dan}段" for dan in range(1, 10)),
        )
        self.assertEqual(AI_DIFFICULTIES, expected)
        profiles = [GoAI(seed=1, difficulty=label).profile for label in expected]
        self.assertEqual([profile.rank for profile in profiles], list(range(17)))
        self.assertEqual(
            [profile.strategic_candidates for profile in profiles],
            sorted(profile.strategic_candidates for profile in profiles),
        )
        self.assertEqual(
            [profile.reply_width for profile in profiles],
            sorted(profile.reply_width for profile in profiles),
        )
        self.assertEqual(
            [profile.noise for profile in profiles],
            sorted((profile.noise for profile in profiles), reverse=True),
        )

    def test_every_difficulty_returns_a_legal_move_without_mutation(self) -> None:
        game = GoGame(9)
        before = (game.board_hash(), game.current_player, game.move_number)
        for label in AI_DIFFICULTIES:
            decision = GoAI(seed=11, difficulty=label).choose_move(game)
            self.assertIsNotNone(decision.point, label)
            assert decision.point is not None
            self.assertTrue(game.analyze_move(*decision.point).legal, label)
            self.assertEqual(
                (game.board_hash(), game.current_player, game.move_number),
                before,
            )

    def test_unknown_difficulty_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GoAI(difficulty="神秘难度")

    def test_ai_returns_legal_move_without_mutating_game(self) -> None:
        for size in (9, 13, 19):
            game = GoGame(size)
            before = (game.board_hash(), game.current_player, game.move_number)
            decision = GoAI(seed=7).choose_move(game)
            self.assertIsNotNone(decision.point)
            assert decision.point is not None
            self.assertTrue(game.analyze_move(*decision.point).legal)
            self.assertEqual(
                (game.board_hash(), game.current_player, game.move_number), before
            )

    def test_ai_prefers_immediate_capture(self) -> None:
        game = GoGame(9)
        for point in ((0, 1), (1, 1), (1, 0), (8, 8), (2, 1), (8, 7)):
            result = game.play(*point)
            self.assertTrue(result.legal, result.reason)
        decision = GoAI(seed=1).choose_move(game)
        self.assertEqual(decision.point, (1, 2))

    def test_ai_passes_after_opponent_when_only_territory_fills_remain(self) -> None:
        game = GoGame(9)
        result = game.play(4, 4)
        self.assertTrue(result.legal)
        self.assertTrue(game.pass_turn())
        decision = GoAI(seed=3).choose_move(game)
        self.assertIsNone(decision.point)

    def test_ai_takes_a_guaranteed_win_after_opponent_passes(self) -> None:
        game = GoGame(9, komi=6.5)
        self.assertTrue(game.pass_turn())
        decision = GoAI(seed=9).choose_move(game)
        self.assertIsNone(decision.point)

    def test_ai_continues_after_pass_when_a_scoring_move_remains(self) -> None:
        game = GoGame(9, komi=0)
        self.assertTrue(game.play(4, 4).legal)
        self.assertTrue(game.play(0, 0).legal)
        self.assertTrue(game.pass_turn())
        decision = GoAI(seed=4).choose_move(game)
        self.assertIsNotNone(decision.point)
        assert decision.point is not None
        self.assertTrue(game.analyze_move(*decision.point).legal)

    def test_ai_does_not_invade_a_small_settled_enemy_eye_after_pass(self) -> None:
        game = GoGame(9)
        for row in range(game.size):
            for col in range(game.size):
                game.board[row][col] = BLACK if col <= 3 else WHITE
        for row, col in (
            (3, 1),
            (3, 2),
            (4, 1),
            (4, 2),
            (3, 6),
            (3, 7),
            (4, 6),
            (4, 7),
        ):
            game.board[row][col] = EMPTY
        game._position_history = {game.board_hash()}
        game.current_player = BLACK
        self.assertTrue(game.pass_turn())

        # Static area scoring alone values a dead white invasion in Black's
        # 2x2 eye.  The AI should recognize this small, non-tactical invasion
        # as futile and agree to finish instead.
        self.assertTrue(game.analyze_move(3, 1).legal)
        decision = GoAI(seed=5).choose_move(game)
        self.assertIsNone(decision.point)


if __name__ == "__main__":
    unittest.main()
