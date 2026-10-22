"""Search correctness: rule legality, player perspective, truncation and symmetry."""

import importlib.util
import unittest

AVAILABLE = importlib.util.find_spec("numpy") is not None
if AVAILABLE:
    import numpy as np
    from weiqi.rl.search import search
    from weiqi.rl.selfplay import GameJob, evaluation_summary, play_game
    from weiqi.rl.state import Position, augment, augment_batch

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

    @staticmethod
    def seeded_evaluator(features):
        generator = np.random.default_rng(int(abs(features.sum() * 1000)) % 9973)
        return generator.normal(size=features.shape[-1] ** 2 + 1), float(generator.uniform(-0.9, 0.9))

    def batch_evaluator(self, batch):
        rows = [self.seeded_evaluator(features) for features in batch]
        return np.array([row[0] for row in rows]), np.array([row[1] for row in rows])

    def test_single_leaf_batches_match_the_sequential_search(self):
        position = Position.new(9, 6.5)
        expected = search(position, self.seeded_evaluator, self.config.search,
                          np.random.default_rng(4), simulations=32, add_noise=True)
        actual = search(position, self.seeded_evaluator, self.config.search,
                        np.random.default_rng(4), simulations=32, add_noise=True,
                        evaluate_batch=self.batch_evaluator)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])

    def test_batched_leaves_use_every_simulation_and_leave_no_virtual_loss(self):
        from dataclasses import replace
        config = replace(self.config.search, leaf_batch_size=8)
        for simulations in (1, 7, 33):
            policy, value = search(Position.new(9, 6.5), self.seeded_evaluator, config,
                                   np.random.default_rng(1), simulations=simulations,
                                   evaluate_batch=self.batch_evaluator)
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=5)
            self.assertLessEqual(abs(value), 1.0)
            self.assertAlmostEqual(float((policy * simulations).sum()), simulations, places=3)

    def test_evaluation_uses_each_players_leaf_batch_after_swapping_colors(self):
        from dataclasses import replace
        config = replace(self.config, evaluation=replace(self.config.evaluation,
                                                        simulations_per_move=8),
                         self_play=replace(self.config.self_play, max_game_length_factor=1.0))
        for color in (BLACK, WHITE):
            with self.subTest(candidate_color=color):
                try:
                    job = GameJob(0, 71, color, candidate_leaf_batch_size=4,
                                  opponent_leaf_batch_size=1)
                except TypeError as error:
                    self.fail(f"Per-player search batches are not supported: {error}")
                calls = []

                def many(model_id, features):
                    calls.append((model_id, len(features)))
                    rows = [self.seeded_evaluator(row) for row in features]
                    return np.array([row[0] for row in rows]), np.array([row[1] for row in rows])

                def one(model_id, features):
                    policies, values = many(model_id, features[None])
                    return policies[0], float(values[0])

                result = play_game(config, job, one, training=False, evaluate_batch=many)
                self.assertTrue(any(model_id == 1 and size > 1 for model_id, size in calls))
                self.assertTrue(any(model_id == 0 for model_id, _ in calls))
                self.assertTrue(all(size == 1 for model_id, size in calls if model_id == 0))
                self.assertEqual(result.examples, [])
                self.assertEqual(result.candidate_color, color)

    def test_player_batch_overrides_are_validated_and_evaluation_only(self):
        try:
            job = GameJob(0, 71, candidate_leaf_batch_size=4, opponent_leaf_batch_size=1)
        except TypeError as error:
            self.fail(f"Per-player search batches are not supported: {error}")
        with self.assertRaisesRegex(ValueError, "evaluation"):
            play_game(self.config, job, self.pass_evaluator, training=True)
        from dataclasses import replace
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                play_game(self.config, replace(job, candidate_leaf_batch_size=size),
                          self.pass_evaluator, training=False)

    def test_direct_comparison_replays_an_interrupted_trailing_opening_pair(self):
        import json
        from pathlib import Path
        import tempfile
        from scripts import compare_search_batching
        from weiqi.rl.selfplay import GameResult

        checksum = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory = output / "games"
            directory.mkdir()
            for index in range(3):
                game = GameResult(index, 500 + index // 2,
                                  BLACK if index % 2 == 0 else WHITE,
                                  WHITE, "two_passes", 1.0, 7.5,
                                  [(BLACK, 0), (WHITE, 1), (BLACK, 81), (WHITE, 81)], 0.1)
                row = game.record(self.config)
                row.update(model_sha256=checksum,
                           search_leaf_batches={"candidate": 4, "opponent": 1})
                (directory / f"game_{index:04d}.json").write_text(json.dumps(row))
                (directory / f"game_{index:04d}.sgf").write_text(f"saved-game-{index}")
            first_pair = (directory / "game_0000.json").read_bytes()
            try:
                restored = compare_search_batching.load_saved_games(output, 500, 4, 4, 1, checksum)
            except AttributeError as error:
                self.fail(f"Interrupted paired matches cannot resume: {error}")
            self.assertEqual([game.index for game in restored], [0, 1])
            self.assertEqual((directory / "game_0000.json").read_bytes(), first_pair)
            self.assertFalse((directory / "game_0002.json").exists())
            self.assertEqual(len(list((output / "interrupted").rglob("game_0002.json"))), 1)
            self.assertEqual(len(list((output / "interrupted").rglob("game_0002.sgf"))), 1)

    def test_exported_match_records_keep_search_metadata_with_actual_game_fields(self):
        import json
        from pathlib import Path
        import tempfile
        from weiqi.rl.selfplay import GameResult
        from weiqi.rl.storage import write_games
        game = GameResult(0, 71, BLACK, WHITE, "two_passes", 1.0, 7.5,
                          [(BLACK, 81), (WHITE, 81)], 0.1)
        metadata = {"search_leaf_batches": {"candidate": 4, "opponent": 1},
                    "model_sha256": "a" * 64, "index": 999}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            try:
                write_games([game], self.config, directory, metadata=metadata)
            except TypeError as error:
                self.fail(f"Search metadata cannot be saved with the game: {error}")
            row = json.loads((directory / "game_0000.json").read_text())
            self.assertEqual(row["index"], 0)
            self.assertEqual(row["search_leaf_batches"], {"candidate": 4, "opponent": 1})
            self.assertEqual(row["model_sha256"], "a" * 64)
            self.assertTrue((directory / "game_0000.sgf").is_file())

    def test_batch_augmentation_matches_row_wise_augmentation(self):
        generator = np.random.default_rng(0)
        features = generator.random((32, 20, 9, 9)).astype(np.float32)
        policies = generator.random((32, 82)).astype(np.float32)
        symmetries = np.arange(32) % 8
        batch_features, batch_policies = augment_batch(features, policies, symmetries)
        for row, symmetry in enumerate(symmetries):
            expected = augment(features[row], policies[row], int(symmetry))
            np.testing.assert_array_equal(batch_features[row], expected[0])
            np.testing.assert_array_equal(batch_policies[row], expected[1])

    def test_replay_sampling_returns_float32_batches_without_a_dataloader(self):
        from weiqi.rl.storage import ReplayBuffer
        generator = np.random.default_rng(1)
        replay = ReplayBuffer(100, 9, True)
        policy = np.full(82, 1 / 82, dtype=np.float32)
        replay.extend([(generator.random((20, 9, 9)).astype(np.float32), policy, 1.0)
                       for _ in range(10)])
        features, policies, values, ownership, scores, weights = replay.sample(16)
        self.assertEqual(tuple(ownership.shape), (16, 81))
        self.assertEqual(float(weights.sum()), 0.0)  # legacy 3-tuples carry no aux targets
        self.assertEqual(tuple(features.shape), (16, 20, 9, 9))
        self.assertEqual(tuple(policies.shape), (16, 82))
        self.assertEqual(tuple(values.shape), (16,))
        self.assertAlmostEqual(float(policies.sum(dim=1).min()), 1.0, places=5)

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
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0, "full_search_probability": 1.0},
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
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0, "full_search_probability": 1.0},
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
