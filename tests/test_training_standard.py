"""Shared admission rules must be independent of trainer and device."""
import unittest
from pathlib import Path

from weiqi.training_standard import get_training_standard


class TrainingStandardTests(unittest.TestCase):
    def setUp(self):
        self.standard = get_training_standard("unified_v1")

    def test_55_percent_with_low_confidence_passes_but_weak_board_does_not(self):
        self.assertEqual(self.standard.promotion_reasons(
            score=.55, by_board={9: .55, 19: .55}, complete=True,
            confidence_lower=.30), ())
        self.assertIn("board_score", self.standard.promotion_reasons(
            score=.60, by_board={9: .45, 19: .75}, complete=True))

    def test_incomplete_match_and_unhealthy_selfplay_cannot_promote(self):
        self.assertIn("incomplete_match", self.standard.promotion_reasons(
            score=1., by_board={9: 1.}, complete=False))
        self.assertIn("black_score_rate", self.standard.promotion_reasons(
            score=1., by_board={9: 1.}, complete=True,
            health_reasons=("black_score_rate",)))

    def test_color_collapse_is_rejected_even_when_games_are_long(self):
        for score in (0., 1.):
            self.assertIn("black_score_rate", self.standard.health_reasons(
                games=32, black_score_rate=score, early_double_pass_rate=0.,
                short_game_rate=0., invalid_rate=0., pass_move_rate=.01))

    def test_healthy_sample_passes_and_missing_or_short_evidence_is_rejected(self):
        observations = dict(games=32, black_score_rate=.5,
                            early_double_pass_rate=0., short_game_rate=0.,
                            invalid_rate=0., pass_move_rate=.01)
        self.assertEqual(self.standard.health_reasons(**observations), ())
        self.assertIn("insufficient_games", self.standard.health_reasons(
            **{**observations, "games": 31}))
        self.assertIn("short_game_rate", self.standard.health_reasons(
            **{**observations, "short_game_rate": .5}))
        self.assertIn("black_score_rate", self.standard.health_reasons(
            **{**observations, "black_score_rate": None}))

    def test_nonfinite_or_empty_match_cannot_promote(self):
        for score in (float("nan"), float("inf"), -.1, 1.1):
            self.assertTrue(self.standard.promotion_reasons(
                score=score, by_board={9: .6}, complete=True))
        self.assertTrue(self.standard.promotion_reasons(
            score=.6, by_board={}, complete=True))

    def test_unknown_standard_is_rejected_and_legacy_is_explicit(self):
        self.assertIsNone(get_training_standard(None))
        with self.assertRaises(ValueError):
            get_training_standard("unified_v2_typo")

    def test_launch_profiles_apply_same_admission_rule_on_different_hardware(self):
        from weiqi.rl_config import load_rl_training_config
        from weiqi.rl_curriculum import load_curriculum_config
        from weiqi.katago_mixed import load_mixed_config
        root = Path(__file__).resolve().parents[1]
        normal = load_rl_training_config(root / "config/rl_training.json")
        rtx = load_curriculum_config(root / "config/rl_curriculum.rtx5070ti.json")
        mixed = load_mixed_config(root / "config/katago_mixed.m4.json")
        for name in (normal.evaluation.standard, rtx.training_standard, mixed.promotion.standard):
            with self.subTest(standard=name):
                policy = get_training_standard(name)
                self.assertIsNotNone(policy)
                self.assertEqual(policy.promotion_reasons(
                    score=.55, by_board={9: .55}, complete=True,
                    confidence_lower=.30), ())


if __name__ == "__main__":
    unittest.main()
