"""Safety checks for rescheduling and game-disjoint replay experiments."""

import tempfile
import unittest
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scripts.benchmark_schedule_replay import (
    ArrayWriter, analyze_holdouts, benchmark_schedule, game_id, held_out_game,
    prepare_pool, read_arrays, row_digest,
)
from weiqi.rl_curriculum_evaluation import EvaluationSteps


class EvaluationRescheduleTests(unittest.TestCase):
    def state(self, samples=8_200_000):
        return SimpleNamespace(
            phase="evaluating", warnings=[], active=SimpleNamespace(
                pending_champion_evaluation=True, samples=samples, next_evaluation_sample=10_000_000,
                latest_model="candidate", champion_model="champion", evaluations=[{"passed": False}],
            ))

    def runtime(self, every_cycle=False):
        return SimpleNamespace(config=SimpleNamespace(champion_evaluation_every_cycle=every_cycle),
                               store=Mock())

    def test_only_optional_pending_match_is_deferred_without_promotion(self):
        state, runtime = self.state(), self.runtime()
        self.assertTrue(EvaluationSteps._defer_disabled_champion_evaluation(runtime, state))
        self.assertEqual(state.phase, "training")
        self.assertFalse(state.active.pending_champion_evaluation)
        self.assertEqual(state.active.champion_model, "champion")
        self.assertEqual(state.active.evaluations, [{"passed": False}])
        self.assertEqual(state.warnings[0]["kind"], "evaluation_deferred")
        self.assertEqual(state.warnings[0]["candidate_model"], "candidate")
        runtime.store.save.assert_called_once_with(state)
        self.assertFalse(EvaluationSteps._defer_disabled_champion_evaluation(runtime, state))

    def test_due_fixed_gate_and_enabled_per_cycle_match_cannot_be_skipped(self):
        for samples, enabled, pending in ((10_000_000, False, True), (8_200_000, True, True),
                                           (8_200_000, False, False)):
            with self.subTest(samples=samples, enabled=enabled, pending=pending):
                state, runtime = self.state(samples), self.runtime(enabled)
                state.active.pending_champion_evaluation = pending
                self.assertFalse(EvaluationSteps._defer_disabled_champion_evaluation(runtime, state))
                self.assertEqual(state.phase, "evaluating")
                runtime.store.save.assert_not_called()


class ReplayExperimentDataTests(unittest.TestCase):
    def test_timing_cannot_reuse_completed_or_partial_match_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "schedule/match").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "new matches"):
                benchmark_schedule(root, 8)

    def test_pool_cannot_overwrite_previous_experiment_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pool/history").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "fresh experiment"):
                prepare_pool(root)

    def test_deduplication_checks_targets_and_ignores_npz_key_order(self):
        arrays = {"features": np.array([[1, 2], [1, 2]], dtype=np.uint8),
                  "targets": np.array([[0.5], [0.5]], dtype=np.float32)}
        self.assertEqual(row_digest(arrays, 0), row_digest(arrays, 1))
        self.assertEqual(row_digest(arrays, 0), row_digest(dict(reversed(list(arrays.items()))), 0))
        arrays["targets"][1, 0] = 0.75
        self.assertNotEqual(row_digest(arrays, 0), row_digest(arrays, 1))

    def test_validation_split_uses_the_entire_game_not_individual_turns(self):
        train_games = holdout_games = 0
        for identity in range(256):
            target = np.zeros(80, dtype=np.float32)
            target[41] = identity
            key, split = game_id(target), held_out_game(target, 26)
            for turn in (0, 10, 150):
                target[51] = turn
                self.assertEqual(game_id(target), key)
                self.assertEqual(held_out_game(target, 26), split)
            holdout_games += int(split)
            train_games += int(not split)
        self.assertGreater(train_games, 0)
        self.assertGreater(holdout_games, 0)

    def test_pool_writer_preserves_selected_rows_without_aliasing(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArrayWriter(Path(temporary), rows_per_file=3)
            arrays = {"x": np.arange(12, dtype=np.int16).reshape(6, 2)}
            writer.add(arrays, [4, 1])
            arrays["x"][4] = -1
            writer.add(arrays, [5, 0])
            writer.flush()
            restored = np.concatenate([read_arrays(path)["x"] for path in sorted(Path(temporary).glob("*.npz"))])
            np.testing.assert_array_equal(restored, [[8, 9], [2, 3], [10, 11], [0, 1]])
            self.assertEqual(writer.rows, 4)

    def test_game_bootstrap_compares_paired_weighted_losses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = {}
            for name in ("500000", "500000-replica2", "1000000", "1000000-replica2"):
                factor = 0.9 if name.startswith("1000000") else 1.0
                cohort = {"rows": 4, "policy_ce": factor, "value_ce": factor / 2,
                    "per_game_policy_value_sums_weights": {
                        "a": [factor, 1.0, factor / 2, 1.0],
                        "b": [factor * 3, 3.0, factor * 1.5, 3.0],
                    }}
                results[name] = {"fresh-holdout": cohort, "historical-holdout": cohort}
            (root / "holdout-results.json").write_text(json.dumps(results))
            with redirect_stdout(io.StringIO()):
                analyze_holdouts(root)
            comparison = json.loads((root / "holdout-comparison.json").read_text())
            for cohort in comparison["cohorts"].values():
                for metric in ("policy_ce", "value_ce"):
                    self.assertAlmostEqual(cohort[metric]["relative_change_percent"], -10.0)
                    np.testing.assert_allclose(cohort[metric]["game_bootstrap_95_percent"], [-10, -10])


if __name__ == "__main__":
    unittest.main()
