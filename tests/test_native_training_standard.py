"""Shared native admission, health and hardware-independent evaluation cadence."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from train import with_training_policy
from weiqi.engine import BLACK, WHITE
from weiqi.rl.control import TrainingStopped
from weiqi.rl_config import resolve_rl_training_config

AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.runner import Trainer
    from weiqi.rl.selfplay import GameResult
    from weiqi.rl.state import Position
    from weiqi.rl.storage import load_checkpoint


def games(count, *, wins=None, collapsed=False, opponent="candidate_self"):
    rows = []
    for index in range(count):
        color = BLACK if index % 2 == 0 else WHITE
        winner = (BLACK if index % 2 == 0 else WHITE) if wins is None else (
            color if index < wins else WHITE if color == BLACK else BLACK)
        moves = [(BLACK if move % 2 == 0 else WHITE, move) for move in range(20)]
        if collapsed:
            moves = [(BLACK, 81), (WHITE, 81)]
        rows.append(GameResult(index, index // 2, color, winner, "two_passes", 40, 41,
                               moves, .01, opponent_name=opponent))
    return rows


@unittest.skipUnless(AVAILABLE, "Optional training dependencies are absent")
class NativeTrainingStandardTests(unittest.TestCase):
    def config(self, **extra):
        sections = {
            "hardware": {"device": "cpu", "precision": "float32", "data_loader_workers": 0},
            "network": {"channels": 8, "residual_blocks": 1},
            "optimizer": {"batch_size": 4, "minimum_replay_size": 1,
                          "training_steps_per_iteration": 2, "target_sample_reuse": None},
            "evaluation": {"standard": "unified_v1", "position_suite_path": "",
                           "teacher_labels_path": ""},
            "runtime": {"pause_while_game_is_active": False},
        }
        for name, values in extra.items():
            sections.setdefault(name, {}).update(values)
        return resolve_rl_training_config(overrides=sections)

    def test_selected_standard_resolves_same_budget_and_resume_policy(self):
        config = self.config(evaluation={"games": 100, "simulations_per_move": 900})
        self.assertEqual((config.evaluation.games, config.evaluation.screen_games,
                          config.evaluation.simulations_per_move,
                          config.evaluation.screen_simulations_per_move), (40, 12, 32, 32))
        legacy = resolve_rl_training_config()
        self.assertIsNone(legacy.evaluation.standard)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps({"evaluation": {"standard": "unified_v1"}}))
            resumed = with_training_policy(legacy, path)
            self.assertEqual(resumed.evaluation.games, 40)
        with self.assertRaises(ValueError):
            self.config(evaluation={"standard": "unknown"})

    def test_actual_sample_draws_persist_and_survive_batch_change(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            trainer = Trainer(self.config(), root)
            state = Position.new(9, 6.5)
            policy = state.legal.astype(np.float32)
            policy /= policy.sum()
            trainer.replay.extend([(state.features(), policy, 1.)] * 8)
            trainer.train_updates()
            trainer.save(archive=False)
            payload, config = load_checkpoint(root / "latest.pt")
            self.assertEqual(payload["training_samples"], 8)
            self.assertEqual(payload["next_evaluation_sample"], 51_200)
            self.assertEqual(payload["training_standard"]["name"], "unified_v1")
            config = replace(config, optimizer=replace(config.optimizer, batch_size=2))
            restored = Trainer(config, root, payload)
            restored.train_updates()
            self.assertEqual(restored.training_samples, 12)
            restored.save(archive=False)
            payload, _ = load_checkpoint(root / "latest.pt")
            payload.pop("training_samples")
            payload["config"]["optimizer"]["batch_size"] = 4
            legacy = Trainer(config, root, payload)
            self.assertEqual(legacy.training_samples, 16)

    def test_overflowed_update_does_not_advance_trained_sample_cadence(self):
        class SkipFirstUpdate:
            def __init__(self):
                self.attempts, self.value = 0, 2.
            def is_enabled(self):
                return False
            def get_scale(self):
                return self.value
            def scale(self, loss):
                return loss
            def unscale_(self, optimizer):
                pass
            def step(self, optimizer):
                if self.attempts:
                    optimizer.step()
            def update(self):
                if not self.attempts:
                    self.value /= 2
                self.attempts += 1
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            state = Position.new(9, 6.5)
            policy = state.legal.astype(np.float32)
            policy /= policy.sum()
            trainer.replay.extend([(state.features(), policy, 1.)] * 8)
            trainer.scaler = SkipFirstUpdate()
            result = trainer.train_updates()
            self.assertEqual(result["steps"], 1)
            self.assertEqual(trainer.training_samples, 4)

    def test_selecting_standard_from_legacy_counters_starts_next_sample_interval(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            legacy = self.config(evaluation={"standard": None})
            trainer = Trainer(legacy, root)
            trainer.training_steps = 12_802
            trainer.training_samples = 51_208
            trainer.save(archive=False)
            payload, _ = load_checkpoint(root / "latest.pt")
            restored = Trainer(self.config(), root, payload)
            self.assertEqual(restored.training_samples, 51_208)
            self.assertEqual(restored.next_evaluation_sample, 102_400)

    def test_promotion_uses_sample_boundary_and_advances_failed_screen(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            trainer.training_samples = 51_199
            with patch("weiqi.rl.runner.run_games", side_effect=AssertionError("premature match")):
                result, accepted = trainer._promotion(3, Path(temporary), {},
                                                     {"truncated": 0, "score_rate": 1.})
            self.assertFalse(accepted)
            self.assertEqual(result["status"], "scheduled")
            trainer.training_samples = 51_200
            result, accepted = trainer._promotion(4, Path(temporary), {},
                                                 {"truncated": 0, "score_rate": .25})
            self.assertEqual(result["status"], "screen_rejected")
            self.assertEqual((trainer.last_evaluation_sample, trainer.next_evaluation_sample),
                             (51_200, 102_400))
            trainer.save(archive=False)
            payload, config = load_checkpoint(Path(temporary) / "latest.pt")
            restored = Trainer(config, Path(temporary), payload)
            self.assertEqual(restored.next_evaluation_sample, 102_400)

    def test_non_due_iteration_does_not_run_a_screen(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            trainer.replay.extend([(np.zeros((20, 9, 9)), np.zeros(82), 0.)])
            with patch.object(trainer, "_selfplay_phase", return_value={}), \
                 patch.object(trainer, "train_updates", return_value={}), \
                 patch.object(trainer, "_position_quality", return_value={}), \
                 patch.object(trainer, "_opponent_pool", return_value={}), \
                 patch.object(trainer, "_screening", side_effect=AssertionError("premature screen")), \
                 patch.object(trainer, "_commit_iteration", side_effect=lambda n, d, s, t: s):
                result = trainer.run_iteration()
            self.assertEqual(result["evaluation"]["status"], "scheduled")

    def test_fixed_confirmation_promotes_55_percent_without_confidence_gate(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            trainer.training_samples = 51_200
            trainer._record_selfplay_health(games(32))
            with patch("weiqi.rl.runner.run_games", return_value=games(40, wins=22)):
                result, accepted = trainer._promotion(1, Path(temporary), {},
                                                     {"truncated": 0, "score_rate": .5})
            self.assertTrue(accepted)
            self.assertLess(result["paired"]["lower"], .5)
            self.assertEqual(result["games"], 40)
            self.assertEqual(result["training_standard"]["name"], "unified_v1")
            self.assertEqual(result["rejection_reasons"], [])

    def test_incomplete_confirmation_and_missing_health_cannot_promote(self):
        for count, enough_health in ((38, True), (40, False)):
            with self.subTest(count=count, enough_health=enough_health), \
                    tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                trainer = Trainer(self.config(), Path(temporary))
                trainer.training_samples = 51_200
                if enough_health:
                    trainer._record_selfplay_health(games(32))
                with patch("weiqi.rl.runner.run_games", return_value=games(count, wins=count)):
                    result, accepted = trainer._promotion(1, Path(temporary), {},
                                                         {"truncated": 0, "score_rate": 1.})
                self.assertFalse(accepted)
                self.assertIn("incomplete_match" if enough_health else "insufficient_games",
                              result["rejection_reasons"])
                self.assertEqual(trainer.best_iteration, 0)
                self.assertEqual(trainer.next_evaluation_sample, 102_400)

    def test_confirmation_does_not_stop_early_for_diagnostic_sprt(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            def complete(_config, jobs, _evaluators, **_kwargs):
                rows = games(len(jobs), wins=len(jobs))
                for row, job in zip(rows, jobs):
                    row.index, row.seed = job.index, job.seed
                return rows
            with patch("weiqi.rl.runner.run_games", side_effect=complete):
                result = trainer.confirmation_games(trainer.config, {})
            self.assertEqual(len(result), 40)

    def test_health_blocks_promotion_but_mixed_opponents_do_not_bias_color(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            trainer.training_samples = 51_200
            opponents = games(224, opponent="accepted_000001")
            for row in opponents:
                row.winner = BLACK
            trainer._record_selfplay_health(games(32) + opponents)
            self.assertEqual(trainer._selfplay_health()["candidate_self_games"], 32)
            self.assertEqual(trainer._selfplay_health()["rejection_reasons"], [])
            trainer._record_selfplay_health(games(32, collapsed=True))
            with patch("weiqi.rl.runner.run_games", return_value=games(40, wins=40)):
                result, accepted = trainer._promotion(1, Path(temporary), {},
                                                     {"truncated": 0, "score_rate": 1.})
            self.assertFalse(accepted)
            self.assertIn("early_double_pass_rate", result["rejection_reasons"])

    def test_health_reports_short_invalid_pass_and_color_failures(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            cases = ("short", "invalid", "pass", "color", "margin")
            for case in cases:
                with self.subTest(case=case):
                    rows = games(32)
                    for row in rows:
                        if case == "short":
                            row.moves = row.moves[:4]
                        elif case == "invalid":
                            row.reason = "length_limit"
                        elif case == "pass":
                            row.moves = row.moves[:3] + [(WHITE, 81)] * 17
                        elif case == "color":
                            row.winner = BLACK
                        else:
                            row.black_score, row.white_score = 81., 0.
                    trainer._record_selfplay_health(rows)
                    report = trainer._selfplay_health()
                    expected = {"short": "short_game_rate", "invalid": "invalid_rate",
                                "pass": "pass_move_rate", "color": "black_score_rate"}
                    if case == "margin":
                        self.assertTrue(report["passed"])
                        self.assertEqual(report["extreme_margin_rate"], 1.)
                    else:
                        self.assertIn(expected[case], report["rejection_reasons"])

    def test_live_adversarial_sparring_does_not_halt_on_missing_self_color_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(self_play={"games_per_iteration": 24}), Path(temporary))
            rows = games(24, opponent="accepted_000001")
            for row in rows:
                row.winner = BLACK
            with patch("weiqi.rl.runner.run_games", return_value=rows):
                result = trainer._selfplay_phase(1, Path(temporary), time.monotonic())
            self.assertEqual(result["selfplay_finished"], 24)
            self.assertIn("insufficient_candidate_self_games",
                          trainer._selfplay_health()["rejection_reasons"])

    def test_color_guard_waits_for_24_self_games_and_admission_requires_32(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            trainer._record_selfplay_health(games(16))
            trainer._record_selfplay_health(games(16, opponent="accepted_000001"))
            self.assertIn("insufficient_candidate_self_games",
                          trainer._selfplay_health()["rejection_reasons"])
            black_only = games(24)
            for row in black_only:
                row.winner = BLACK
            trainer.recent_selfplay_health = []
            trainer._record_selfplay_health(black_only)
            live = trainer._selfplay_health(minimum_games=24, require_color_evidence=False)
            self.assertIn("black_score_rate", live["rejection_reasons"])

    def test_live_health_halts_before_collapsed_games_enter_replay(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(self_play={"games_per_iteration": 24}), Path(temporary))
            with patch("weiqi.rl.runner.run_games", return_value=games(24, collapsed=True)):
                with self.assertRaisesRegex(TrainingStopped, "health"):
                    trainer._selfplay_phase(1, Path(temporary), time.monotonic())
            self.assertEqual(len(trainer.replay), 0)

    def test_health_window_and_thresholds_survive_resume(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.config(), Path(temporary))
            rows = games(32)
            rows[0].reason = "length_limit"
            trainer._record_selfplay_health(rows)
            self.assertEqual(trainer._selfplay_health()["invalid_rate"], 1 / 32)
            trainer.save(archive=False)
            payload, config = load_checkpoint(Path(temporary) / "latest.pt")
            restored = Trainer(config, Path(temporary), payload)
            self.assertEqual(restored._selfplay_health(), trainer._selfplay_health())
            restored._record_selfplay_health(games(32))
            self.assertEqual(restored._selfplay_health()["invalid_rate"], 0)


if __name__ == "__main__":
    if AVAILABLE:
        torch.set_num_threads(1)
    unittest.main()
