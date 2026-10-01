"""Training-quality features: playout caps, auxiliary targets, replay ratio, and SPRT."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weiqi.engine import BLACK, EMPTY, WHITE, GoGame
from weiqi.rl.eval_stats import confirmed_improvement, paired_sprt
from weiqi.rl_config import RLConfigError, resolve_rl_training_config

NUMPY = importlib.util.find_spec("numpy") is not None
TORCH = NUMPY and importlib.util.find_spec("torch") is not None
if NUMPY:
    import numpy as np
    from weiqi.rl.selfplay import GameJob, GameResult, play_game
    from weiqi.rl.state import augment_batch
if TORCH:
    import torch
    from weiqi.rl.network import PolicyValueNet
    from weiqi.rl.runner import Trainer
    from weiqi.rl.storage import ReplayBuffer, checkpoint_config


@dataclass
class MatchRecord:
    index: int
    seed: int
    candidate_color: int
    winner: int | None
    reason: str


def pairs(outcomes: str) -> list[MatchRecord]:
    """``W`` candidate wins both colors, ``L`` loses both, ``T`` splits the pair."""
    games = []
    for pair, outcome in enumerate(outcomes):
        for color in (BLACK, WHITE):
            other = WHITE if color == BLACK else BLACK
            wins = outcome == "W" or (outcome == "T" and color == BLACK)
            games.append(MatchRecord(len(games), pair, color, color if wins else other, "two_passes"))
    return games


class AreaOwnershipTests(unittest.TestCase):
    def test_ownership_agrees_with_area_score(self):
        game = GoGame(9)
        for row in range(9):
            game.board[row][3] = BLACK
            game.board[row][5] = WHITE
        game.board[4][4] = BLACK  # the middle column touches both colors except here
        owners = game.area_ownership()
        self.assertEqual(owners[0][0], BLACK)
        self.assertEqual(owners[0][8], WHITE)
        self.assertEqual(owners[0][4], EMPTY)
        score = game.calculate_score()
        self.assertEqual(sum(row.count(BLACK) for row in owners),
                         score.black_stones + score.black_territory)
        self.assertEqual(sum(row.count(EMPTY) for row in owners), score.neutral_points)


class SprtTests(unittest.TestCase):
    def test_decisions_follow_the_likelihood_ratio(self):
        options = {"alpha": 0.05, "beta": 0.10, "pair_win_rate": 0.65}
        self.assertEqual(paired_sprt(pairs("W" * 12), **options)["decision"], "accept")
        self.assertEqual(paired_sprt(pairs("L" * 7), **options)["decision"], "reject")
        undecided = paired_sprt(pairs("WLWT"), **options)
        self.assertEqual(undecided["decision"], "continue")
        self.assertEqual((undecided["wins"], undecided["losses"], undecided["ties"]), (2, 1, 1))

    def test_ties_carry_no_evidence(self):
        test = paired_sprt(pairs("T" * 30), alpha=0.05, beta=0.10, pair_win_rate=0.65)
        self.assertEqual((test["llr"], test["decision"]), (0.0, "continue"))

    def test_promotion_requires_an_accepted_sprt(self):
        options = {"alpha": 0.05, "beta": 0.10, "pair_win_rate": 0.65}
        summary = {"truncated": 0, "score_rate": 1.0, "paired_sprt": paired_sprt(pairs("W" * 12), **options)}
        self.assertTrue(confirmed_improvement(summary, threshold=0.55, method="paired_sprt"))
        summary["paired_sprt"] = paired_sprt(pairs("W" * 4), **options)
        self.assertFalse(confirmed_improvement(summary, threshold=0.55, method="paired_sprt"))

    def test_invalid_sprt_settings_fail_early(self):
        with self.assertRaises(RLConfigError):
            resolve_rl_training_config(overrides={"evaluation": {"sprt_pair_win_rate": 0.5}})
        with self.assertRaises(RLConfigError):
            resolve_rl_training_config(overrides={"search": {"full_search_probability": 0.0}})


@unittest.skipUnless(NUMPY, "Optional training dependency NumPy is not installed")
class PlayoutCapTests(unittest.TestCase):
    @staticmethod
    def passing(_, features):
        logits = np.full(features.shape[-1] ** 2 + 1, -100.0)
        logits[-1] = 100.0
        return logits, 0.0

    def config(self, probability):
        return resolve_rl_training_config(overrides={"search": {
            "simulations_per_move": 8, "fast_simulations_per_move": 2,
            "dirichlet_epsilon": 0.0, "full_search_probability": probability}})

    def test_only_full_searches_become_samples_and_use_the_full_budget(self):
        budgets = []

        def recording_search(state, evaluate, config, rng, *, simulations, add_noise, evaluate_batch):
            budgets.append((simulations, add_noise))
            policy = np.zeros(len(state.legal), dtype=np.float32)
            policy[-1] = 1.0
            return policy, 0.0

        with patch("weiqi.rl.selfplay.search", recording_search):
            seeds = range(40)
            games = [play_game(self.config(0.5), GameJob(0, seed), self.passing, training=True)
                     for seed in seeds]
        self.assertEqual(sum(len(game.examples) for game in games),
                         sum(budget == 8 for budget, _ in budgets))
        self.assertTrue({budget for budget, _ in budgets} == {2, 8})
        # Root noise only on full searches.
        self.assertTrue(all(noise == (budget == 8) for budget, noise in budgets))

    def test_disabled_cap_records_every_move(self):
        game = play_game(self.config(1.0), GameJob(0, 3), self.passing, training=True)
        self.assertEqual(len(game.examples), len(game.moves))

    def test_played_out_games_carry_consistent_ownership_and_score_targets(self):
        config = self.config(1.0)
        game = play_game(config, GameJob(0, 3), self.passing, training=True)
        self.assertEqual(game.reason, "two_passes")
        for features, policy, value, ownership, score, weight in game.examples:
            self.assertEqual(weight, 1.0)
            self.assertEqual(ownership.shape, (81,))
        # Black's perspective: owned points minus opponent's points minus komi.
        black = game.examples[0]
        self.assertAlmostEqual(black[4] * 81, black[3].sum() - config.game.komi)
        self.assertAlmostEqual(game.examples[1][4], -black[4])

    def test_resigned_games_mask_auxiliary_targets(self):
        config = resolve_rl_training_config(overrides={
            "search": {"simulations_per_move": 2, "full_search_probability": 1.0},
            "self_play": {"resign_threshold": -0.5, "resign_min_move": 1}})
        # Black-to-move positions look lost, so Black resigns at its second move.
        losing_black = lambda _, x: (np.zeros(82), -0.9 if x[16, 0, 0] else 0.9)
        game = play_game(config, GameJob(0, 5), losing_black, training=True)
        self.assertEqual(game.reason, "resign")
        self.assertTrue(game.examples)
        self.assertTrue(all(row[5] == 0.0 and not row[3].any() for row in game.examples))

    def test_ownership_follows_the_feature_symmetry(self):
        generator = np.random.default_rng(4)
        features = generator.random((8, 20, 9, 9)).astype(np.float32)
        ownership = features[:, 0].reshape(8, 81).copy()
        policies = np.full((8, 82), 1 / 82, dtype=np.float32)
        moved, _, moved_ownership = augment_batch(features, policies, np.arange(8), ownership)
        np.testing.assert_array_equal(moved[:, 0].reshape(8, 81), moved_ownership)


@unittest.skipUnless(TORCH, "Optional training dependency PyTorch is not installed")
class TrainerQualityTests(unittest.TestCase):
    def config(self, **optimizer):
        return resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32"},
            "network": {"channels": 8, "residual_blocks": 1},
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0},
            "optimizer": {"batch_size": 4, "minimum_replay_size": 1,
                          "training_steps_per_iteration": 50, **optimizer},
            "evaluation": {"games": 40, "sprt_batch_pairs": 2},
            "runtime": {"pause_while_game_is_active": False},
        })

    @staticmethod
    def samples(count, weight=1.0):
        rows = []
        for index in range(count):
            policy = np.full(82, 1 / 82, dtype=np.float32)
            ownership = np.where(np.arange(81) % 2, 1.0, -1.0).astype(np.float32)
            rows.append((np.zeros((20, 9, 9), dtype=np.float32), policy, float(index % 2 * 2 - 1),
                         ownership, 0.1, weight))
        return rows

    def test_auxiliary_heads_are_trained_and_reported(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(directory))
            self.assertTrue(trainer.model.auxiliary)
            trainer.replay.extend(self.samples(16))
            trainer.untrained_samples = 16
            result = trainer.train_updates()
            self.assertGreater(result["ownership_loss"], 0)
            self.assertGreater(result["score_loss"], 0)
            policy, value, ownership, score = trainer.model(torch.zeros((2, 20, 9, 9)), auxiliary=True)
            self.assertEqual((tuple(ownership.shape), tuple(score.shape)), ((2, 81), (2,)))
            self.assertEqual(len(trainer.model(torch.zeros((2, 20, 9, 9)))), 2)

    def test_replay_ratio_sets_steps_from_untrained_samples(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(target_sample_reuse=2.0), Path(directory))
            trainer.replay.extend(self.samples(16))
            trainer.untrained_samples = 10
            self.assertEqual(trainer.training_step_count(), 5)  # ceil(10 * 2 / 4)
            trainer.untrained_samples = 1000
            self.assertEqual(trainer.training_step_count(), 50)  # capped
            trainer.untrained_samples = 10
            result = trainer.train_updates()
            self.assertEqual((result["steps"], result["sample_reuse"]), (5, 2.0))
            self.assertEqual(trainer.untrained_samples, 0)
            trainer.save(archive=False)
            payload = torch.load(Path(directory) / "latest.pt", weights_only=True)
            self.assertEqual(payload["untrained_samples"], 0)

    def test_sprt_confirmation_stops_at_the_first_decisive_batch(self):
        calls = []

        def winning(config, jobs, evaluators, **_):
            calls.append(len(jobs))
            return [GameResult(job.index, job.seed, job.candidate_color, job.candidate_color,
                               "two_passes", 0.0, 0.0, [], 0.0) for job in jobs]

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(directory))
            with patch("weiqi.rl.runner.run_games", winning):
                results = trainer.confirmation_games(trainer.config, {})
        # 2 pairs per batch; 12 straight pair wins cross the 0.05/0.10 bound.
        self.assertEqual(calls, [4] * 6)
        self.assertEqual(trainer.sprt(results)["decision"], "accept")
        self.assertEqual([game.index for game in results], list(range(24)))

    def test_replay_round_trip_and_legacy_replay_without_auxiliary_targets(self):
        replay = ReplayBuffer(100, 9)
        replay.extend(self.samples(3))
        restored = ReplayBuffer(100, 9)
        restored.restore(replay.state())
        self.assertEqual([row[5] for row in restored.samples], [1.0, 1.0, 1.0])
        legacy = {key: value for key, value in replay.state().items()
                  if key in ("features", "policies", "values")}
        restored.restore(legacy)
        self.assertEqual([row[5] for row in restored.samples], [0.0, 0.0, 0.0])

    def test_legacy_checkpoints_keep_their_original_training_behavior(self):
        from weiqi.rl.state import FEATURE_VERSION
        from weiqi.rl.storage import CHECKPOINT_VERSION

        raw = self.config().to_dict()
        raw["network"].pop("auxiliary_heads")
        raw["search"].pop("full_search_probability")
        raw["optimizer"].pop("target_sample_reuse")
        restored = checkpoint_config({"checkpoint_version": CHECKPOINT_VERSION,
                                      "feature_version": FEATURE_VERSION, "config": raw})
        self.assertFalse(restored.network.auxiliary_heads)
        self.assertEqual(restored.search.full_search_probability, 1.0)
        self.assertIsNone(restored.optimizer.target_sample_reuse)
        self.assertFalse(PolicyValueNet(restored).auxiliary)


if __name__ == "__main__":
    unittest.main()
