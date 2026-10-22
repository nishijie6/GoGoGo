"""Fixed joseki starts must preserve legal history and independent evaluation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from weiqi.engine import BLACK, WHITE
from weiqi.rl_config import RLConfigError, resolve_rl_training_config

AVAILABLE = importlib.util.find_spec("numpy") is not None


class OpeningConfigTests(unittest.TestCase):
    def test_old_configs_keep_empty_board_training(self):
        self.assertEqual(resolve_rl_training_config().self_play.joseki_fraction, 0.0)

    def test_opening_fraction_is_validated_and_can_be_enabled_on_resume(self):
        from train import with_training_policy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps({"self_play": {"joseki_fraction": 0.25}}))
            config = with_training_policy(resolve_rl_training_config(), path)
        self.assertEqual(config.self_play.joseki_fraction, 0.25)
        for value in (-0.1, 1.1, True, "0.25", float("nan")):
            with self.subTest(value=value), self.assertRaises(RLConfigError):
                resolve_rl_training_config(overrides={"self_play": {"joseki_fraction": value}})

    def test_katago_does_not_silently_ignore_the_native_experiment_setting(self):
        from weiqi.katago_rl import KataGoRLRunnerError, build_selfplay_overrides
        config = resolve_rl_training_config(overrides={"self_play": {"joseki_fraction": 0.25}})
        with self.assertRaisesRegex(KataGoRLRunnerError, "joseki_fraction"):
            build_selfplay_overrides(config)


@unittest.skipUnless(AVAILABLE, "Optional NumPy dependency is not installed")
class JosekiSelfPlayTests(unittest.TestCase):
    @staticmethod
    def pass_evaluator(_, features):
        import numpy as np
        logits = np.full(features.shape[-1] ** 2 + 1, -100.0)
        logits[-1] = 100.0
        return logits, 0.0

    def config(self, fraction=1.0):
        return resolve_rl_training_config(overrides={
            "search": {"simulations_per_move": 1, "full_search_probability": 1.0,
                       "dirichlet_epsilon": 0.0},
            "evaluation": {"simulations_per_move": 1},
            "self_play": {"joseki_fraction": fraction},
        })

    def test_every_board_and_symmetry_replays_with_alternating_legal_moves(self):
        from weiqi.rl.openings import joseki_openings
        from weiqi.rl.state import Position
        for size in (9, 13, 19):
            openings = joseki_openings(size)
            self.assertEqual(len({opening.actions for opening in openings}), 16)
            for opening in openings:
                with self.subTest(size=size, opening=opening.name):
                    state = Position.new(size, 6.5)
                    for action in opening.actions:
                        self.assertLess(action, size * size)
                        self.assertTrue(state.legal[action])
                        state = state.play(action)
                    self.assertFalse(state.game.game_over)
                    self.assertEqual(len(state.game.moves), len(opening.actions))

    def test_forced_moves_are_in_sgf_history_but_never_policy_targets(self):
        from weiqi.rl.selfplay import GameJob, play_game
        game = play_game(self.config(), GameJob(0, 3), self.pass_evaluator, training=True)
        self.assertTrue(game.opening_id)
        self.assertIn(game.opening_moves, (6, 13))
        self.assertEqual(game.reason, "two_passes")
        self.assertEqual(len(game.moves), game.opening_moves + 2)
        self.assertEqual(len(game.examples), 2)
        self.assertGreater(float(game.examples[0][0][:2].sum()), 0)
        self.assertEqual(game.record(self.config())["opening"]["moves"], game.opening_moves)

    def test_color_swapped_training_games_share_the_same_opening(self):
        from weiqi.rl.selfplay import GameJob, play_game
        black = play_game(self.config(), GameJob(0, 9, BLACK), self.pass_evaluator, training=True)
        white = play_game(self.config(), GameJob(1, 9, WHITE), self.pass_evaluator, training=True)
        self.assertEqual(black.opening_id, white.opening_id)
        self.assertEqual(black.moves, white.moves)

    def test_evaluation_is_unchanged_when_training_openings_are_enabled(self):
        from weiqi.rl.selfplay import GameJob, play_game
        job = GameJob(0, 17)
        original = play_game(self.config(0), job, self.pass_evaluator, training=False)
        enabled = play_game(self.config(1), job, self.pass_evaluator, training=False)
        self.assertEqual(enabled.moves, original.moves)
        self.assertFalse(enabled.opening_id)
        self.assertEqual(enabled.opening_moves, 0)

    def test_disabled_openings_preserve_empty_board_and_sampling_is_repeatable(self):
        from weiqi.rl.openings import select_training_opening
        from weiqi.rl.selfplay import GameJob, play_game
        game = play_game(self.config(0), GameJob(0, 3), self.pass_evaluator, training=True)
        self.assertEqual(game.moves, [(BLACK, 81), (WHITE, 81)])
        self.assertNotIn("opening", game.record(self.config(0)))
        first = [select_training_opening(9, seed, 0.25) for seed in range(100)]
        self.assertEqual(first, [select_training_opening(9, seed, 0.25) for seed in range(100)])
        self.assertTrue(any(opening is None for opening in first))
        self.assertTrue(any(opening is not None for opening in first))

    def test_spawned_workers_keep_opening_metadata_and_exclude_forced_targets(self):
        import numpy as np
        from weiqi.rl.selfplay import GameJob, run_games
        def predict(batch):
            logits = np.full((len(batch), 82), -100.0, dtype=np.float32)
            logits[:, -1] = 100.0
            return logits, np.zeros(len(batch))
        games = run_games(self.config(), [GameJob(0, 3), GameJob(1, 9)],
                          {0: predict}, training=True)
        for game in games:
            self.assertTrue(game.opening_id)
            self.assertEqual(len(game.moves) - game.opening_moves, 2)
            self.assertEqual(len(game.examples), 2)


if __name__ == "__main__":
    unittest.main()
