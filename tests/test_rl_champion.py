"""Regression coverage for using frozen accepted models during training."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.engine import BLACK, WHITE
from weiqi.rl_config import resolve_rl_training_config


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.runner import Trainer
    from weiqi.rl.selfplay import GameJob, play_game
    from weiqi.rl.storage import cpu_state, fingerprint


@unittest.skipUnless(AVAILABLE, "Optional training dependencies are absent")
class ChampionSelfPlayTests(unittest.TestCase):
    def test_training_pair_uses_a_frozen_champion_on_both_colors(self):
        config = resolve_rl_training_config(overrides={
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0},
        })
        selected = []

        def predict(model_id, features):
            selected.append(model_id)
            logits = np.full(config.action_size, -100.0)
            logits[-1] = 100.0
            return logits, 0.0

        black = play_game(config, GameJob(0, 7, BLACK, opponent_id=1), predict,
                          training=True)
        # Each player uses its own model throughout its search, including leaf
        # positions. The opponent model takes over on the next actual turn.
        self.assertEqual(selected, [0, 0, 1])
        selected.clear()
        white = play_game(config, GameJob(1, 7, WHITE, opponent_id=1), predict,
                          training=True)
        self.assertEqual(selected, [1, 1, 0])
        self.assertEqual(black.reason, white.reason)
        self.assertEqual(len(black.examples), 2)
        self.assertEqual(len(white.examples), 2)

    def test_iteration_rotates_accepted_champion_in_balanced_pairs(self):
        config = resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "data_loader_workers": 0},
            "network": {"channels": 8, "residual_blocks": 1},
            "self_play": {"games_per_iteration": 4,
                          "champion_fraction": 0.5},
            "optimizer": {"minimum_replay_size": 100},
            "runtime": {"pause_while_game_is_active": False},
        })
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(config, Path(temporary))
            champion_sha = fingerprint(cpu_state(trainer.best))
            with torch.no_grad():
                next(trainer.model.parameters()).add_(0.25)
            seen = []

            def capture(_config, jobs, evaluators, *, training, **_kwargs):
                self.assertTrue(training)
                seen.append((jobs, evaluators))
                return []

            with patch("weiqi.rl.runner.run_games", side_effect=capture):
                trainer.run_iteration()

        self.assertEqual(len(seen), 1)
        jobs, evaluators = seen[0]
        champion_games = [job for job in jobs if job.opponent_id != 0]
        self.assertEqual(len(champion_games), 2)
        self.assertEqual({job.candidate_color for job in champion_games},
                         {BLACK, WHITE})
        self.assertEqual(champion_games[0].seed, champion_games[1].seed)
        self.assertTrue(all(job.opponent_sha256 == champion_sha
                            for job in champion_games))
        self.assertIn(0, evaluators)
        self.assertIn(champion_games[0].opponent_id, evaluators)

    def test_multiple_accepted_champions_rotate_across_iterations(self):
        from copy import deepcopy
        from weiqi.rl.eval_pool import save_anchor

        config = resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "data_loader_workers": 0},
            "network": {"channels": 8, "residual_blocks": 1},
            "self_play": {"games_per_iteration": 4,
                          "champion_fraction": 0.5},
            "runtime": {"pause_while_game_is_active": False},
        })
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(config, Path(temporary))
            with torch.no_grad():
                next(trainer.model.parameters()).add_(0.25)
            for number in (2, 4):
                champion = deepcopy(trainer.best)
                with torch.no_grad():
                    next(champion.parameters()).add_(float(number))
                save_anchor(trainer.anchors, champion, config,
                            kind="accepted", iteration=number)
            sequence = []
            for iteration in range(1, 5):
                jobs, evaluators, _ = trainer.selfplay_jobs(iteration)
                paired = [job for job in jobs if job.opponent_id]
                self.assertEqual(len(paired), 2)
                self.assertEqual({job.candidate_color for job in paired},
                                 {BLACK, WHITE})
                self.assertIn(paired[0].opponent_id, evaluators)
                sequence.append(paired[0].opponent_name)
        self.assertEqual(sequence, ["accepted_000000", "accepted_000002",
                                    "accepted_000004", "accepted_000000"])
