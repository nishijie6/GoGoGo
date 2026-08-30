"""Tests for the lightweight real-time win-rate estimator."""

from __future__ import annotations

import unittest

from weiqi.engine import BLACK, EMPTY, WHITE, GoGame
from weiqi.winrate import WinRateEstimator


class WinRateEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = WinRateEstimator()

    def test_empty_boards_start_near_even(self) -> None:
        for size in (9, 13, 19):
            estimate = self.estimator.estimate(GoGame(size))
            self.assertGreater(estimate.black_win_probability, 0.48)
            self.assertLess(estimate.black_win_probability, 0.53)
            self.assertAlmostEqual(
                estimate.black_percent + estimate.white_percent,
                100.0,
            )

    def test_first_black_move_shifts_estimate_without_extreme_jump(self) -> None:
        for size in (9, 13, 19):
            game = GoGame(size)
            baseline = self.estimator.estimate(game)
            self.assertTrue(game.play(size // 2, size // 2).legal)
            after_move = self.estimator.estimate(game)
            self.assertGreater(
                after_move.black_win_probability,
                baseline.black_win_probability,
            )
            self.assertLess(after_move.black_win_probability, 0.65)

    def test_rotation_keeps_the_same_estimate(self) -> None:
        game = GoGame(9, komi=6.5)
        rotated = GoGame(9, komi=6.5)
        stones = {
            (2, 2): BLACK,
            (2, 3): BLACK,
            (5, 6): WHITE,
            (6, 6): WHITE,
            (7, 7): WHITE,
        }
        for (row, col), color in stones.items():
            game.board[row][col] = color
            rotated.board[8 - row][8 - col] = color
        game.current_player = BLACK
        rotated.current_player = BLACK

        estimate = self.estimator.estimate(game)
        rotated_estimate = self.estimator.estimate(rotated)
        self.assertAlmostEqual(
            estimate.black_win_probability,
            rotated_estimate.black_win_probability,
            places=10,
        )
        self.assertAlmostEqual(
            estimate.black_lead,
            rotated_estimate.black_lead,
            places=10,
        )

    def test_more_komi_reduces_black_lead_and_win_rate(self) -> None:
        games = [GoGame(9, komi=komi) for komi in (0.0, 6.5, 7.5)]
        sequence = ((2, 2), (6, 6), (2, 6), (6, 2))
        for game in games:
            for point in sequence:
                self.assertTrue(game.play(*point).legal)
        estimates = [self.estimator.estimate(game) for game in games]
        self.assertGreater(
            estimates[0].black_win_probability,
            estimates[1].black_win_probability,
        )
        self.assertGreater(
            estimates[1].black_win_probability,
            estimates[2].black_win_probability,
        )
        self.assertAlmostEqual(
            estimates[0].black_lead - estimates[1].black_lead,
            6.5,
        )
        self.assertAlmostEqual(
            estimates[1].black_lead - estimates[2].black_lead,
            1.0,
        )

    def test_capture_counter_is_not_counted_twice(self) -> None:
        game = GoGame(9)
        same_board = game.clone()
        same_board.captures[BLACK] = 12
        same_board.captures[WHITE] = 7
        estimate = self.estimator.estimate(game)
        with_captures = self.estimator.estimate(same_board)
        self.assertEqual(
            estimate.black_win_probability,
            with_captures.black_win_probability,
        )

    def test_clear_capture_improves_capturing_side_estimate(self) -> None:
        game = GoGame(9)
        for point in ((0, 1), (1, 1), (1, 0), (8, 8), (2, 1), (8, 7)):
            self.assertTrue(game.play(*point).legal)
        before = self.estimator.estimate(game)
        capture = game.play(1, 2)
        self.assertTrue(capture.legal)
        self.assertEqual(capture.captured, 1)
        after = self.estimator.estimate(game)
        self.assertGreater(after.black_win_probability, before.black_win_probability)

    def test_estimation_does_not_mutate_game(self) -> None:
        game = GoGame(9)
        self.assertTrue(game.play(4, 4).legal)
        before = (
            game.board_hash(),
            game.current_player,
            game.move_number,
            dict(game.captures),
            game.consecutive_passes,
            frozenset(game._position_history),
            len(game._undo_stack),
        )
        self.estimator.estimate(game)
        after = (
            game.board_hash(),
            game.current_player,
            game.move_number,
            dict(game.captures),
            game.consecutive_passes,
            frozenset(game._position_history),
            len(game._undo_stack),
        )
        self.assertEqual(before, after)

    def test_finished_game_uses_exact_result(self) -> None:
        game = GoGame(9, komi=6.5)
        self.assertTrue(game.pass_turn())
        self.assertTrue(game.pass_turn())
        estimate = self.estimator.estimate(game)
        self.assertTrue(estimate.final)
        self.assertEqual(estimate.black_win_probability, 0.0)
        self.assertEqual(estimate.white_win_probability, 1.0)
        self.assertAlmostEqual(estimate.black_lead, -6.5)

    def test_resignation_uses_declared_winner(self) -> None:
        game = GoGame(9)
        self.assertTrue(game.resign())
        estimate = self.estimator.estimate(game)
        self.assertTrue(estimate.final)
        self.assertEqual(estimate.black_win_probability, 0.0)
        self.assertEqual(estimate.white_win_probability, 1.0)
        self.assertAlmostEqual(
            estimate.black_expected_score - estimate.white_expected_score,
            estimate.black_lead,
        )

    def test_player_with_a_winning_pass_available_is_strong_favorite(self) -> None:
        game = GoGame(9, komi=6.5)
        self.assertTrue(game.pass_turn())
        estimate = self.estimator.estimate(game)
        self.assertFalse(estimate.final)
        self.assertEqual(estimate.phase, "可终局")
        self.assertEqual(estimate.black_win_probability, 0.02)
        self.assertAlmostEqual(estimate.black_lead, -6.5)

    def test_pass_does_not_flip_a_settled_two_eye_position(self) -> None:
        game = GoGame(9, komi=6.5)
        for row in range(game.size):
            for col in range(game.size):
                game.board[row][col] = BLACK if col <= 4 else WHITE
        for row, col in ((3, 2), (5, 2), (3, 7), (5, 7)):
            game.board[row][col] = EMPTY
        game._position_history = {game.board_hash()}

        before_pass = self.estimator.estimate(game)
        self.assertTrue(game.pass_turn())
        after_pass = self.estimator.estimate(game)
        self.assertGreater(before_pass.black_win_probability, 0.5)
        self.assertGreater(after_pass.black_win_probability, 0.5)
        self.assertLess(
            abs(
                before_pass.black_win_probability
                - after_pass.black_win_probability
            ),
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
