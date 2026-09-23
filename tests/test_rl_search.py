"""Search correctness: rule legality, player perspective, truncation and symmetry."""

import importlib.util
import unittest

AVAILABLE = importlib.util.find_spec("numpy") is not None
if AVAILABLE:
    import numpy as np
    from weiqi.rl.search import search
    from weiqi.rl.selfplay import GameJob, evaluation_summary, play_game
    from weiqi.rl.state import Position, augment

from weiqi.engine import BLACK, WHITE, GoGame
from weiqi.rl_config import resolve_rl_training_config


@unittest.skipUnless(AVAILABLE, "Optional training dependency NumPy is not installed")
class SearchTests(unittest.TestCase):
    def setUp(self):
        self.config = resolve_rl_training_config(overrides={"search": {"simulations_per_move": 8}})

    @staticmethod
    def pass_evaluator(features):
        logits = np.full(features.shape[-1] ** 2 + 1, -100.0)
        logits[-1] = 100.0
        return logits, 0.0

    def test_terminal_win_and_loss_are_backed_up_in_root_perspective(self):
        # White can end an empty board and win on komi.
        white = Position.new(9, 6.5).play(81)
        policy, value = search(white, self.pass_evaluator, self.config.search, np.random.default_rng(0), simulations=1)
        self.assertEqual(policy[-1], 1)
        self.assertEqual(value, 1)
        # Black is behind after B A9, W J1, B B9, W pass; ending loses.
        black = Position.new(9, 6.5)
        for action in (0, 80, 1, 81):
            black = black.play(action)
        _, value = search(black, self.pass_evaluator, self.config.search, np.random.default_rng(0), simulations=1)
        self.assertEqual(value, -1)

    def test_illegal_policy_prior_is_masked_and_search_does_not_mutate_game(self):
        state = Position.new(9, 6.5).play(40)
        before = state.game.board_hash(), list(state.game.moves)
        def evaluator(features):
            logits = np.zeros(82)
            logits[40] = 1000
            return logits, 0.0
        policy, _ = search(state, evaluator, self.config.search, np.random.default_rng(2))
        self.assertEqual(policy[40], 0)
        self.assertAlmostEqual(float(policy.sum()), 1)
        self.assertEqual(before, (state.game.board_hash(), list(state.game.moves)))
        for action in np.flatnonzero(policy):
            state.play(int(action))

    def test_features_follow_player_history_pass_and_legal_mask(self):
        state = Position.new(9, 6.5).play(0)
        features = state.features()
        self.assertEqual(features.shape, (20, 9, 9))
        self.assertEqual(features[1, 0, 0], 1)
        self.assertEqual(features[0].sum(), 0)
        self.assertEqual(features[16].sum(), 0)
        self.assertAlmostEqual(float(features[17, 0, 0]), 6.5 / 20)
        self.assertEqual(features[19, 0, 0], 0)
        passed = state.play(81).features()
        self.assertEqual(passed[0, 0, 0], 1)
        self.assertEqual(passed[2, 0, 0], 1)
        self.assertTrue((passed[18] == 1).all())

    def test_all_symmetries_keep_policy_aligned_and_pass_unchanged(self):
        features = np.zeros((20, 9, 9), dtype=np.float32)
        features[0, 1, 3] = 1
        policy = np.zeros(82, dtype=np.float32)
        policy[12], policy[-1] = 0.75, 0.25
        seen = set()
        for symmetry in range(8):
            x, pi = augment(features, policy, symmetry)
            self.assertEqual(int(x[0].argmax()), int(pi[:-1].argmax()))
            self.assertEqual(pi[-1], 0.25)
            self.assertTrue(x.flags.c_contiguous)
            seen.add(int(pi[:-1].argmax()))
        self.assertEqual(len(seen), 8)

    def test_real_two_pass_episode_labels_both_players_correctly(self):
        config = resolve_rl_training_config(overrides={
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0},
        })
        game = play_game(config, GameJob(0, 3), lambda _, x: self.pass_evaluator(x), training=True)
        self.assertEqual(game.reason, "two_passes")
        self.assertEqual(game.winner, WHITE)
        self.assertEqual([row[2] for row in game.examples], [-1, 1])
        replay = GoGame(9)
        for color, action in game.moves:
            self.assertEqual(color, replay.current_player)
            replay.pass_turn() if action == 81 else replay.play(*divmod(action, 9))
        self.assertTrue(replay.game_over)
        self.assertEqual(replay.winner, game.winner)

    def test_move_limit_does_not_create_fake_winner_or_replay_targets(self):
        config = resolve_rl_training_config(overrides={
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0},
            "self_play": {"max_game_length_factor": 1.0},
        })
        def no_pass(_, features):
            logits = np.arange(82, dtype=np.float32)
            logits[-1] = -1000
            return logits, 0.0
        game = play_game(config, GameJob(0, 7), no_pass, training=True)
        self.assertEqual(game.reason, "length_limit")
        self.assertEqual(len(game.moves), 81)
        self.assertEqual(game.examples, [])
        self.assertIsNone(game.winner)
        summary = evaluation_summary([game])
        self.assertEqual(summary["truncated"], 1)
        self.assertEqual(summary["draws"], 0)
        self.assertEqual(summary["score_rate"], 0)

    def test_paired_evaluation_uses_the_same_opening_for_both_colors(self):
        first = play_game(self.config, GameJob(0, 19, BLACK), lambda _, x: self.pass_evaluator(x), training=False)
        second = play_game(self.config, GameJob(1, 19, WHITE), lambda _, x: self.pass_evaluator(x), training=False)
        self.assertEqual(first.moves[:2], second.moves[:2])


class UndoFreeGameTests(unittest.TestCase):
    def test_training_games_keep_superko_but_do_not_store_undo_snapshots(self):
        game = GoGame(9, record_undo=False)
        for point in [(0, 1), (1, 1), (1, 0), (0, 2), (2, 1), (2, 2), (8, 8), (1, 3), (1, 2)]:
            self.assertTrue(game.play(*point).legal)
        self.assertFalse(game.analyze_move(1, 1).legal)
        self.assertFalse(game.can_undo)
        clone = game.clone()
        self.assertFalse(clone.analyze_move(1, 1).legal)
        clone.pass_turn()
        self.assertFalse(clone.can_undo)
        normal = GoGame(9)
        normal.play(0, 0)
        self.assertTrue(normal.can_undo)
