"""Seed-level inference and durable isolation for joseki replication studies."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "Optional training dependencies are absent")
class OpeningStudyTests(unittest.TestCase):
    @staticmethod
    def result(seed, deltas):
        return {"training_seed": seed, "evaluation_seed": 100000 + seed * 1000,
                "pair_deltas": deltas, "complete_planned_match": True,
                "excluded_truncated_pairs": 0, "planned_games": 2 * len(deltas),
                "training": {"control": {"training_steps": 100, "new_samples": 20,
                                          "joseki_samples": 0, "joseki_games": 0},
                             "joseki": {"training_steps": 100, "new_samples": 20,
                                         "joseki_samples": 10, "joseki_games": 5}}}

    def test_exact_sign_test_uses_training_seeds_and_requires_all_positive(self):
        from weiqi.rl.opening_study import aggregate_replicates

        rows = [self.result(seed, [.5, 1]) for seed in range(5)]
        result = aggregate_replicates(rows, planned_seeds=list(range(5)), bootstrap_draws=1000)
        self.assertEqual(result["training_seeds"], 5)
        self.assertEqual(result["seed_sign_p_value"], 1 / 32)
        self.assertEqual(result["conclusion"], "consistent_empirical_improvement")
        rows[-1] = self.result(4, [-1, -.5])
        result = aggregate_replicates(rows, planned_seeds=list(range(5)), bootstrap_draws=1000)
        self.assertEqual(result["seed_sign_p_value"], 6 / 32)
        self.assertEqual(result["conclusion"], "inconclusive")
        rows[-1] = self.result(4, [0, 0])
        result = aggregate_replicates(rows, planned_seeds=list(range(5)), bootstrap_draws=1000)
        self.assertEqual(result["seed_sign_p_value"], 1 / 16)
        self.assertEqual(result["conclusion"], "inconclusive")

    def test_opening_count_cannot_substitute_for_independent_training_seeds(self):
        from weiqi.rl.opening_study import aggregate_replicates

        row = self.result(1, [1] * 1000)
        result = aggregate_replicates([row], planned_seeds=[1], bootstrap_draws=100)
        self.assertEqual(result["seed_sign_p_value"], .5)
        self.assertEqual(result["conclusion"], "inconclusive")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_replicates([row, row], planned_seeds=[1, 2], bootstrap_draws=100)

    def test_bootstrap_is_reproducible_and_incomplete_or_truncated_work_blocks_label(self):
        from weiqi.rl.opening_study import aggregate_replicates

        rows = [self.result(seed, [0, .5, 1]) for seed in range(5)]
        first = aggregate_replicates(rows, planned_seeds=list(range(5)), bootstrap_draws=1000)
        second = aggregate_replicates(rows, planned_seeds=list(range(5)), bootstrap_draws=1000)
        self.assertEqual(first["hierarchical_bootstrap"], second["hierarchical_bootstrap"])
        reordered = aggregate_replicates(list(reversed(rows)), planned_seeds=list(range(5)),
                                         bootstrap_draws=1000)
        self.assertEqual(first["hierarchical_bootstrap"], reordered["hierarchical_bootstrap"])
        interval = first["hierarchical_bootstrap"]
        self.assertLess(interval["lower"], .5)
        self.assertGreater(interval["upper"], .5)
        self.assertTrue(interval["approximate"])
        for field, value in (("complete_planned_match", False), ("excluded_truncated_pairs", 1)):
            changed = deepcopy(rows)
            changed[0][field] = value
            result = aggregate_replicates(changed, planned_seeds=list(range(5)), bootstrap_draws=100)
            self.assertEqual(result["conclusion"], "inconclusive")
        for field in ("training_steps", "new_samples", "joseki_samples"):
            changed = deepcopy(rows)
            changed[0]["training"]["joseki"][field] = 0
            self.assertEqual(aggregate_replicates(changed, planned_seeds=list(range(5)),
                                                 bootstrap_draws=100)["conclusion"], "inconclusive")
        changed = deepcopy(rows)
        changed[0]["pair_deltas"] = [float("nan")]
        with self.assertRaises(ValueError):
            aggregate_replicates(changed, planned_seeds=list(range(5)), bootstrap_draws=100)

    def test_unequal_opening_counts_keep_equal_seed_weight_and_partial_plan_cannot_pass(self):
        from weiqi.rl.opening_study import aggregate_replicates

        rows = [self.result(1, [1] * 100), self.result(2, [-1])]
        result = aggregate_replicates(rows, planned_seeds=[1, 2], bootstrap_draws=100)
        self.assertEqual(result["mean_score_delta"], 0)
        rows = [self.result(seed, [1]) for seed in range(5)]
        result = aggregate_replicates(rows, planned_seeds=list(range(6)), bootstrap_draws=100)
        self.assertEqual(result["conclusion"], "inconclusive")
        rows[0]["pair_deltas"] = []
        result = aggregate_replicates(rows, planned_seeds=list(range(5)), bootstrap_draws=100)
        self.assertEqual(result["uninformative_training_seeds"], 1)
        self.assertEqual(result["conclusion"], "inconclusive")
        rows = [self.result(seed, [1]) for seed in range(5)]
        rows[0]["training"]["control"]["joseki_games"] = 1
        self.assertEqual(aggregate_replicates(rows, planned_seeds=list(range(5)),
                                             bootstrap_draws=100)["conclusion"], "inconclusive")

    @staticmethod
    def records(seed, treatment=False, truncate=False):
        from weiqi.engine import BLACK, WHITE
        return [{"index": index, "seed": seed, "candidate_color": color,
                 "winner": color if treatment else 3 - color,
                 "reason": "length_limit" if truncate and index == 0 else "two_passes",
                 "moves": [[BLACK, 0], [WHITE, 5]], "board_size": 9}
                for index, color in enumerate((BLACK, WHITE))]

    def test_saved_pair_deltas_require_declared_seeds_and_remove_truncated_quartets(self):
        from weiqi.rl.opening_study import pair_deltas

        result = pair_deltas(self.records(810000001), self.records(810000001, treatment=True),
                             planned_games=2, evaluation_seed=810000001)
        self.assertEqual(result["deltas"], [1])
        result = pair_deltas(self.records(810000001),
                             self.records(810000001, treatment=True, truncate=True),
                             planned_games=2, evaluation_seed=810000001)
        self.assertEqual(result["deltas"], [])
        self.assertEqual(result["excluded_truncated_pairs"], 1)
        with self.assertRaises(ValueError):
            pair_deltas(self.records(5), self.records(5), planned_games=2,
                        evaluation_seed=810000001)

    def test_invalid_plans_are_rejected_before_reading_source_or_creating_output(self):
        from weiqi.rl.opening_study import run_study

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for changes in ({"evaluation_length_factor": .5}, {"bootstrap_seed": -1},
                            {"bootstrap_seed": True}, {"training_seeds": [1, 1]},
                            {"evaluation_seed_base": 2026}, {"games": 20002}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    run_study(root / "source/missing.pt", root / "output", **changes)
                self.assertFalse((root / "output").exists())

    def test_study_preserves_source_and_continuation_reuses_only_verified_completed_attempt(self):
        from weiqi.fileio import sha256_file
        from weiqi.rl.opening_study import run_study
        from weiqi.rl.runner import Trainer
        from weiqi.rl.storage import atomic_json
        from weiqi.rl_config import resolve_rl_training_config

        config = resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32"},
            "network": {"channels": 8, "residual_blocks": 1},
            "runtime": {"pause_while_game_is_active": False}})
        captured_sources, attempts = [], []
        wrong_seed = False

        def comparison(resume, output, **kwargs):
            captured_sources.append(sha256_file(resume))
            attempts.append(output)
            output.mkdir(parents=True)
            arguments = {**kwargs, "resume": str(resume), "output": str(output)}
            if wrong_seed:
                arguments["training_seed"] = 999
            atomic_json({"arguments": arguments,
                         "source": {"sha256": sha256_file(resume)}}, output / "manifest.json")
            report = {"status": "complete", "source_sha256": sha256_file(resume),
                      "training": self.result(kwargs["training_seed"], [1])["training"],
                      "matched": {"mean_score_delta": 1},
                      "evaluation": {arm: {"games": 2, "truncated": 0}
                                     for arm in ("control", "joseki")}}
            for arm in ("control", "joseki"):
                for record in self.records(kwargs["evaluation_seed"], arm == "joseki"):
                    atomic_json(record, output / f"evaluate_{arm}/games/game_{record['index']:04d}.json")
            atomic_json(report, output / "report.json")
            return report

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source = root / "original"
            source.mkdir()
            Trainer(config, source).save(archive=False)
            before = {str(p.relative_to(source)): sha256_file(p)
                      for p in source.rglob("*") if p.is_file()}
            kwargs = {"training_seeds": [1, 2], "iterations": 1, "games": 2,
                      "training_games": 2, "training_steps": 100, "bootstrap_draws": 100}
            with patch("weiqi.rl.opening_study.compare_training", side_effect=comparison):
                report = run_study(source / "latest.pt", root / "study", **kwargs)
                self.assertEqual(report["status"], "complete")
                self.assertEqual(captured_sources, [before["latest.pt"]] * 2)
                manifest = json.loads((root / "study/manifest.json").read_text())
                self.assertEqual(manifest["plan"]["training_seeds"], [1, 2])
                self.assertTrue(manifest["plan"]["freeze_training_opponents"])
                self.assertEqual(manifest["plan"]["games"], 2)
                run_study(source / "latest.pt", root / "study", continue_study=True, **kwargs)
                self.assertEqual(len(attempts), 2)
                changed_game_path = attempts[0] / "evaluate_control/games/game_0000.json"
                original_game = json.loads(changed_game_path.read_text())
                changed_game = deepcopy(original_game)
                changed_game["moves"].append([1, 9])
                atomic_json(changed_game, changed_game_path)
                with self.assertRaisesRegex(ValueError, "records"):
                    run_study(source / "latest.pt", root / "study", continue_study=True, **kwargs)
                atomic_json(original_game, changed_game_path)
                # Receipts from the already-running schema-1 study can be
                # explicitly revalidated and certified with the records digest.
                legacy_path = attempts[0] / "study_replicate.json"
                legacy = json.loads(legacy_path.read_text())
                legacy.pop("evaluation_records_sha256")
                atomic_json(legacy, legacy_path)
                run_study(source / "latest.pt", root / "study", continue_study=True, **kwargs)
                recertified = json.loads(legacy_path.read_text())
                self.assertIn("evaluation_records_sha256", recertified)
                self.assertIn("legacy_receipt_certification", recertified)
                # An interrupted attempt is preserved and retried in a new directory.
                receipt = attempts[1] / "study_replicate.json"
                receipt.unlink()
                run_study(source / "latest.pt", root / "study", continue_study=True, **kwargs)
                self.assertEqual(len(attempts), 3)
                self.assertNotEqual(attempts[1], attempts[2])
                self.assertTrue((attempts[1] / "report.json").is_file())
                # Actual saved RNG provenance must match its declared replicate.
                changed = json.loads((attempts[2] / "manifest.json").read_text())
                changed["arguments"]["training_seed"] = 999
                atomic_json(changed, attempts[2] / "manifest.json")
                with self.assertRaises(ValueError):
                    run_study(source / "latest.pt", root / "study", continue_study=True, **kwargs)
                with self.assertRaises(ValueError):
                    run_study(source / "latest.pt", root / "study", **kwargs)
                with self.assertRaisesRegex(ValueError, "plan"):
                    run_study(source / "latest.pt", root / "study", continue_study=True,
                              **dict(kwargs, iterations=2))
                wrong_seed = True
                with self.assertRaisesRegex(ValueError, "seed"):
                    run_study(source / "latest.pt", root / "study_wrong_seed", **kwargs)
            self.assertEqual(before, {str(p.relative_to(source)): sha256_file(p)
                                      for p in source.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
