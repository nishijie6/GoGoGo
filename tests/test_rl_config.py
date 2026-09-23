"""Tests for reinforcement-learning presets and configuration safety."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from weiqi.rl_config import (
    DEFAULT_RL_CONFIG_PATH,
    RLConfigError,
    load_rl_training_config,
    resolve_rl_training_config,
    save_rl_training_config,
)


class RLTrainingConfigTests(unittest.TestCase):
    def test_repository_default_is_safe_balanced_9x9(self) -> None:
        config = load_rl_training_config()

        self.assertEqual(DEFAULT_RL_CONFIG_PATH.name, "rl_training.json")
        self.assertEqual(config.preset, "balanced")
        self.assertEqual(config.game.board_size, 9)
        self.assertEqual(config.action_size, 82)
        self.assertEqual(config.self_play.workers, 2)
        self.assertEqual(config.self_play.champion_fraction, 0.5)
        self.assertEqual(config.network.channels, 64)
        self.assertLessEqual(config.hardware.gpu_memory_fraction, 0.60)
        self.assertTrue(config.runtime.pause_while_game_is_active)
        self.assertIsNone(config.self_play.resign_threshold)

    def test_high_performance_preset_keeps_large_gpu_path_available(self) -> None:
        balanced = resolve_rl_training_config("balanced")
        high = resolve_rl_training_config("high_performance")

        self.assertEqual(high.game.board_size, 19)
        self.assertEqual(high.action_size, 362)
        self.assertGreater(high.network.channels, balanced.network.channels)
        self.assertGreater(
            high.network.residual_blocks,
            balanced.network.residual_blocks,
        )
        self.assertGreater(
            high.search.simulations_per_move,
            balanced.search.simulations_per_move,
        )
        self.assertGreater(high.optimizer.batch_size, balanced.optimizer.batch_size)
        self.assertGreater(
            high.hardware.gpu_memory_fraction,
            balanced.hardware.gpu_memory_fraction,
        )

        example_path = (
            DEFAULT_RL_CONFIG_PATH.parent
            / "rl_training.high_performance.example.json"
        )
        example = load_rl_training_config(example_path)
        self.assertEqual(example.preset, "high_performance")
        self.assertEqual(example.hardware.device, "cuda:0")

    def test_nested_overrides_do_not_mutate_the_builtin_preset(self) -> None:
        custom = resolve_rl_training_config(
            "balanced",
            {
                "game": {"board_size": 13},
                "hardware": {"device": "cuda:1", "gpu_memory_fraction": 0.75},
                "network": {"channels": 96},
                "self_play": {"workers": 4},
            },
        )
        untouched = resolve_rl_training_config("balanced")

        self.assertEqual(custom.game.board_size, 13)
        self.assertEqual(custom.action_size, 170)
        self.assertEqual(custom.hardware.device, "cuda:1")
        self.assertEqual(custom.network.channels, 96)
        self.assertEqual(custom.self_play.workers, 4)
        self.assertEqual(custom.evaluation.position_suite_path, "")
        self.assertEqual(custom.evaluation.teacher_labels_path, "")
        self.assertEqual(untouched.game.board_size, 9)
        self.assertEqual(untouched.network.channels, 64)

    def test_unknown_or_invalid_values_fail_early(self) -> None:
        with self.assertRaisesRegex(RLConfigError, "未知字段"):
            resolve_rl_training_config(
                "balanced",
                {"hardware": {"gpu_memroy_fraction": 0.5}},
            )
        with self.assertRaisesRegex(RLConfigError, "9、13 或 19"):
            resolve_rl_training_config(
                "balanced",
                {"game": {"board_size": 10}},
            )
        with self.assertRaisesRegex(RLConfigError, "8 的倍数"):
            resolve_rl_training_config(
                "balanced",
                {"network": {"channels": 65}},
            )
        with self.assertRaisesRegex(RLConfigError, "不超过 1"):
            resolve_rl_training_config(
                "balanced",
                {"hardware": {"gpu_memory_fraction": 1.1}},
            )
        with self.assertRaisesRegex(RLConfigError, "有限数字"):
            resolve_rl_training_config(
                "balanced",
                {"optimizer": {"learning_rate": float("inf")}},
            )
        with self.assertRaisesRegex(RLConfigError, "0 到 1"):
            resolve_rl_training_config(
                "balanced", {"self_play": {"champion_fraction": 1.1}}
            )
        with self.assertRaisesRegex(RLConfigError, "只支持 9×9"):
            resolve_rl_training_config("balanced", {
                "game": {"board_size": 13},
                "evaluation": {"position_suite_path": "config/rl_eval_positions_9x9.json",
                               "teacher_labels_path": "config/rl_eval_teacher_9x9.json"},
            })

    def test_schema_version_must_be_an_integer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": True,
                        "preset": "balanced",
                        "overrides": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RLConfigError, "schema_version"):
                load_rl_training_config(path)

    def test_resignation_is_opt_in_and_checked(self) -> None:
        enabled = resolve_rl_training_config(
            "balanced",
            {
                "self_play": {
                    "resign_threshold": -0.95,
                    "resign_min_move": 80,
                }
            },
        )
        self.assertEqual(enabled.self_play.resign_threshold, -0.95)
        self.assertEqual(enabled.self_play.resign_min_move, 80)

        with self.assertRaisesRegex(RLConfigError, "-1 到 0"):
            resolve_rl_training_config(
                "balanced",
                {"self_play": {"resign_threshold": 0.1}},
            )

    def test_save_then_load_round_trip_preserves_compact_overrides(self) -> None:
        overrides = {
            "game": {"board_size": 13},
            "optimizer": {"batch_size": 192},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "training.json"
            saved = save_rl_training_config(path, "balanced", overrides)
            loaded = load_rl_training_config(path)
            raw = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved, loaded)
        self.assertEqual(raw["preset"], "balanced")
        self.assertEqual(raw["overrides"], overrides)
        self.assertEqual(loaded.game.board_size, 13)
        self.assertEqual(loaded.optimizer.batch_size, 192)

    def test_config_can_be_serialized_as_fully_resolved_values(self) -> None:
        config = resolve_rl_training_config("balanced")
        resolved = config.to_dict()

        self.assertEqual(resolved["schema_version"], 1)
        self.assertEqual(resolved["game"]["board_size"], 9)
        self.assertEqual(resolved["hardware"]["device"], "auto")


if __name__ == "__main__":
    unittest.main()
