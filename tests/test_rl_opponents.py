"""Training model selection must pin an actual pool model and its board."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_rl_curriculum_runtime import _write_profiles, _make_model
from weiqi.ai import is_katago_difficulty
from weiqi.katago import KataGoConfigurationError, KataGoSettings, profile_for_difficulty
from weiqi.rl_opponents import load_training_opponents, settings_for_training_opponent


class TrainingOpponentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config, _ = _write_profiles(self.root)
        for name in ("champion", "history", "unselected"):
            _make_model(self.root / "9x9", name)
        state = self.root / "curriculum/state.json"
        state.parent.mkdir()
        state.write_text(json.dumps({"stages": {"9x9": {
            "champion_model": "champion", "champion_established": False,
            "fixed_baseline_models": ["history", "../../outside", "missing"],
        }}}), encoding="utf-8")

    def choices(self, board=9):
        return load_training_opponents(board, project_root=self.root, curriculum_path=self.config)

    def test_lists_only_existing_selected_pool_models_on_this_board(self):
        choices = self.choices()
        self.assertEqual({item.model_name for item in choices.values()}, {"champion", "history"})
        self.assertEqual(self.choices(13), {})
        self.assertTrue(any("暂定" in label for label in choices))

    def test_settings_disable_human_model_and_do_not_overwrite_saved_settings(self):
        opponent = next(iter(self.choices().values()))
        original = KataGoSettings("engine", "original", "human")
        with patch.object(KataGoSettings, "require_valid"):
            selected = settings_for_training_opponent(original, opponent, 9)
        self.assertEqual(selected.model, str(opponent.model_path))
        self.assertEqual(selected.human_model, "")
        self.assertEqual(original.model, "original")
        with self.assertRaises(KataGoConfigurationError):
            settings_for_training_opponent(original, opponent, 19)

    def test_training_profile_has_no_rank_label_and_uses_fixed_search_budget(self):
        label = next(iter(self.choices()))
        profile = profile_for_difficulty(label)
        self.assertTrue(is_katago_difficulty(label))
        self.assertEqual(profile.max_visits, 200)
        self.assertIsNone(profile.dan)
        self.assertEqual(profile.selection_mode, "training")


if __name__ == "__main__":
    unittest.main()
