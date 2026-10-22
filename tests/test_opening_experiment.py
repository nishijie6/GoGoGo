"""Fair, isolated A/B training and matched evaluation reports."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.engine import BLACK, WHITE


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "Optional training dependencies are absent")
class OpeningExperimentTests(unittest.TestCase):
    def test_training_seed_replaces_checkpoint_rng_without_mutating_source_or_global_rng(self):
        import numpy as np
        import torch
        from weiqi.rl.opening_experiment import seeded_resume

        payload = {"numpy_rng": np.random.default_rng(99).bit_generator.state,
                   "torch_rng": torch.Generator().manual_seed(99).get_state(),
                   "cuda_rng": [], "model": {"weight": torch.ones(1)}}
        before = deepcopy(payload)
        global_rng = torch.get_rng_state().clone()
        def draws(seed):
            result = seeded_resume(payload, seed)
            numpy_rng = np.random.default_rng()
            numpy_rng.bit_generator.state = result["numpy_rng"]
            torch_rng = torch.Generator().set_state(result["torch_rng"])
            return numpy_rng.integers(0, 2 ** 32, size=5).tolist(), torch.rand(5, generator=torch_rng).tolist()
        self.assertEqual(draws(1234), draws(1234))
        self.assertNotEqual(draws(1234), draws(5678))
        self.assertEqual(payload["numpy_rng"], before["numpy_rng"])
        self.assertTrue(torch.equal(payload["torch_rng"], before["torch_rng"]))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))
        for seed in (-1, True, 2 ** 32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seeded_resume(payload, seed)

    def test_evaluation_controls_change_real_play_limits_and_keep_training_config_intact(self):
        from weiqi.rl.opening_experiment import make_evaluation_config
        from weiqi.rl_config import resolve_rl_training_config
        original = resolve_rl_training_config()
        result = make_evaluation_config(original, games=200, visits=32, seed=810000001,
                                        length_factor=6, leaf_batch_size=4)
        self.assertEqual(result.runtime.random_seed, 810000001)
        self.assertEqual(result.self_play.max_game_length_factor, 6)
        self.assertEqual(result.search.leaf_batch_size, 4)
        self.assertEqual(result.self_play.joseki_fraction, 0)
        self.assertEqual(original.self_play.max_game_length_factor, 2.5)
        for kwargs in ({"length_factor": .5}, {"leaf_batch_size": 0}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_evaluation_config(original, games=200, visits=32, **kwargs)

    def test_arms_change_only_fraction_and_output_and_reject_invalid_budgets(self):
        from weiqi.rl.opening_experiment import arm_configs
        from weiqi.rl_config import resolve_rl_training_config

        original = resolve_rl_training_config(overrides={"optimizer": {"target_sample_reuse": 2.0}})
        before = original.to_dict()
        configs = arm_configs(original, Path("/tmp/experiment"), fraction=.25,
                              training_games=16, training_steps=100, training_visits=64,
                              training_seed=1234, training_leaf_batch_size=8,
                              freeze_training_opponents=True, checkpoint_retention=2)
        control, treatment = (deepcopy(configs[name].to_dict()) for name in ("control", "joseki"))
        self.assertEqual(control["self_play"].pop("joseki_fraction"), 0)
        self.assertEqual(treatment["self_play"].pop("joseki_fraction"), .25)
        self.assertEqual(control["runtime"].pop("output_directory"), "/tmp/experiment/control")
        self.assertEqual(treatment["runtime"].pop("output_directory"), "/tmp/experiment/joseki")
        self.assertEqual(control, treatment)
        self.assertIsNone(configs["control"].optimizer.target_sample_reuse)
        self.assertIsNone(configs["joseki"].optimizer.target_sample_reuse)
        self.assertEqual(configs["control"].runtime.random_seed, 1234)
        self.assertEqual(configs["control"].search.leaf_batch_size, 8)
        self.assertEqual(configs["control"].runtime.keep_checkpoints, 2)
        self.assertGreater(configs["control"].evaluation.milestone_every_iterations, 1000000)
        self.assertEqual(original.to_dict(), before)
        for kwargs in ({"fraction": 0}, {"fraction": 1.1}, {"fraction": float("nan")},
                       {"fraction": .25, "training_steps": 0},
                       {"fraction": .25, "training_seed": -1},
                       {"fraction": .25, "training_leaf_batch_size": 0},
                       {"fraction": .25, "training_games": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                arm_configs(original, Path("/tmp/experiment"), **kwargs)

    @staticmethod
    def records(pair_scores):
        records = []
        for pair, wins in enumerate(pair_scores):
            for color_index, color in enumerate((BLACK, WHITE)):
                records.append({"index": 2 * pair + color_index, "seed": 100 + pair,
                                "candidate_color": color,
                                "winner": color if color_index < wins else 3 - color,
                                "reason": "two_passes", "moves": [[BLACK, pair], [WHITE, 5]]})
        return records

    def test_matched_delta_discards_whole_truncated_pairs(self):
        from weiqi.rl.opening_experiment import matched_delta

        control, treatment = self.records([0, 2, 1]), self.records([2, 1, 1])
        treatment[2]["reason"] = "length_limit"
        treatment[2]["winner"] = None
        result = matched_delta(control, treatment, planned_games=6)
        self.assertEqual(result["complete_pairs"], 2)
        self.assertEqual(result["excluded_truncated_pairs"], 1)
        self.assertEqual(result["mean_score_delta"], .5)
        self.assertEqual((result["positive_pairs"], result["negative_pairs"], result["tied_pairs"]),
                         (1, 0, 1))
        self.assertEqual(result["one_sided_p_value"], .5)
        self.assertFalse(result["evidence_of_improvement"])

    def test_matched_delta_validates_alignment_and_requires_complete_large_match(self):
        from weiqi.rl.opening_experiment import matched_delta

        control, treatment = self.records([0] * 20), self.records([2] * 20)
        result = matched_delta(control, treatment, planned_games=40)
        self.assertEqual(result["one_sided_p_value"], 1 / 2 ** 20)
        self.assertTrue(result["evidence_of_improvement"])
        self.assertFalse(matched_delta(control[:4], treatment[:4], planned_games=40)
                         ["evidence_of_improvement"])
        for changed in ("seed", "candidate_color", "index", "moves"):
            bad = deepcopy(treatment)
            bad[0][changed] = [[BLACK, 99], [WHITE, 5]] if changed == "moves" else 999
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                matched_delta(control, bad, planned_games=40)

    def test_positive_sign_test_does_not_establish_positive_mean_win_rate_gain(self):
        from weiqi.rl.opening_experiment import matched_delta

        control = self.records([0] * 15 + [2] * 5)
        treatment = self.records([1] * 15 + [0] * 5)
        result = matched_delta(control, treatment, planned_games=40)
        self.assertAlmostEqual(result["mean_score_delta"], .125)
        self.assertLess(result["one_sided_p_value"], .05)
        self.assertFalse(result["evidence_of_improvement"])

    def test_changing_source_during_snapshot_fails_before_either_arm_runs(self):
        from weiqi.rl.opening_experiment import compare_training
        from weiqi.rl.runner import Trainer
        from weiqi.rl.storage import atomic_torch_save, load_checkpoint
        from weiqi.rl_config import resolve_rl_training_config

        config = resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32"},
            "network": {"channels": 8, "residual_blocks": 1},
            "runtime": {"pause_while_game_is_active": False}})
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            Trainer(config, source).save(archive=False)
            checkpoint = source / "latest.pt"
            original_load = load_checkpoint
            mutated = False

            def load_then_replace(path):
                nonlocal mutated
                payload, loaded_config = original_load(path)
                if path.resolve() == checkpoint.resolve() and not mutated:
                    mutated = True
                    changed = dict(payload, iteration=1)
                    atomic_torch_save(changed, checkpoint)
                return payload, loaded_config

            output = root / "experiment"
            with patch("weiqi.rl.opening_experiment.load_checkpoint", side_effect=load_then_replace):
                with self.assertRaisesRegex(ValueError, "changed"):
                    compare_training(checkpoint, output, iterations=1, games=2)
            self.assertFalse((output / "control").exists())
            self.assertFalse((output / "joseki").exists())
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")

    def test_comparison_freezes_state_keeps_source_unchanged_and_writes_results(self):
        import numpy as np
        from weiqi.fileio import sha256_file
        from weiqi.rl.opening_experiment import compare_training
        from weiqi.rl.runner import Trainer
        from weiqi.rl.selfplay import GameResult
        from weiqi.rl.state import INPUT_PLANES
        from weiqi.rl.storage import load_checkpoint
        from weiqi.rl_config import resolve_rl_training_config

        config = resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32"},
            "network": {"channels": 8, "residual_blocks": 1},
            "self_play": {"games_per_iteration": 2, "workers": 1},
            "optimizer": {"batch_size": 2, "minimum_replay_size": 1,
                          "training_steps_per_iteration": 1},
            "evaluation": {"milestone_every_iterations": 1},
            "runtime": {"pause_while_game_is_active": False}})

        training_streams = []
        def games(config, jobs, evaluators, *, training, **kwargs):
            if training:
                training_streams.append([job.seed for job in jobs])
            example = (np.zeros((INPUT_PLANES, 9, 9), dtype=np.float32),
                       np.ones(82, dtype=np.float32) / 82, 1.0)
            return [GameResult(job.index, job.seed, job.candidate_color, BLACK,
                               "two_passes", 10, 0, [(BLACK, 0), (WHITE, 5)], 0,
                               [example] if training else []) for job in jobs]

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            trainer = Trainer(config, source)
            trainer.save(archive=False)
            checkpoint = source / "latest.pt"
            before = {str(p.relative_to(source)): sha256_file(p)
                      for p in source.rglob("*") if p.is_file()}
            with patch("weiqi.rl.runner.run_games", side_effect=games), \
                 patch("weiqi.rl.assessment.run_games", side_effect=games):
                report = compare_training(checkpoint, root / "experiment", iterations=1,
                                          games=2, visits=1, skip_training_evaluation=True,
                                          training_seed=1234, evaluation_seed=810000001,
                                          training_leaf_batch_size=8, evaluation_leaf_batch_size=4,
                                          evaluation_length_factor=6, freeze_training_opponents=True,
                                          checkpoint_retention=2)
            after = {str(p.relative_to(source)): sha256_file(p)
                     for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["conclusion"], "inconclusive")
            self.assertIn("no_treatment_games", report["limitations"])
            self.assertIn("no_treatment_samples", report["limitations"])
            self.assertEqual(report["source_sha256"], before["latest.pt"])
            self.assertEqual(report["training"]["control"]["training_steps"], 1)
            self.assertEqual(report["training"]["joseki"]["new_samples"], 2)
            self.assertEqual(report["matched"]["complete_pairs"], 1)
            self.assertEqual(training_streams, [[4206279610, 4194893555]] * 2)
            self.assertTrue((root / "experiment/report.json").is_file())
            manifest = json.loads((root / "experiment/manifest.json").read_text())
            self.assertTrue(manifest["joseki_suite"]["openings"])
            for arm in ("control", "joseki"):
                self.assertEqual(list((root / f"experiment/{arm}/anchors").glob("milestone_*.pt")), [])
                self.assertEqual(sha256_file(root / f"experiment/{arm}/anchors/accepted_000000.pt"),
                                 before["anchors/accepted_000000.pt"])
                _, evaluation_config = load_checkpoint(root / f"experiment/evaluation_{arm}.pt")
                self.assertEqual(evaluation_config.self_play.joseki_fraction, 0)
                self.assertEqual(evaluation_config.evaluation.simulations_per_move, 1)
                self.assertEqual(evaluation_config.search.leaf_batch_size, 4)
                self.assertEqual(evaluation_config.self_play.max_game_length_factor, 6)
                self.assertEqual(evaluation_config.runtime.random_seed, 810000001)
            with self.assertRaisesRegex(ValueError, "fresh"):
                compare_training(checkpoint, root / "experiment", iterations=1, games=2)
            with self.assertRaisesRegex(ValueError, "source"):
                compare_training(checkpoint, source / "nested", iterations=1, games=2)
            self.assertFalse((source / "nested").exists())
            with self.assertRaises(ValueError):
                compare_training(checkpoint, root / "invalid", iterations=0, games=3)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
