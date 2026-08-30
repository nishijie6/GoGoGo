"""Tests for the local official KataGo reinforcement-learning pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from weiqi.katago_rl import (
    DEFAULT_RTX_PROFILE,
    DEFAULT_CURRICULUM_PROFILE,
    KataGoRLRunnerError,
    build_selfplay_overrides,
    format_katago_overrides,
    model_kind_for_config,
    KataGoRLRunner,
    build_parser,
)
from weiqi.rl_config import load_rl_training_config


class KataGoRLConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_rl_training_config(DEFAULT_RTX_PROFILE)

    def test_active_rtx_profile_starts_with_9x9_b10c128(self) -> None:
        self.assertEqual(model_kind_for_config(self.config), "b10c128")
        self.assertEqual(self.config.game.board_size, 9)
        self.assertEqual(self.config.hardware.device, "cuda:0")
        self.assertEqual(self.config.hardware.precision, "amp_bfloat16")
        self.assertFalse(self.config.hardware.compile_model)
        self.assertEqual(self.config.optimizer.batch_size, 1024)
        self.assertEqual(self.config.optimizer.training_steps_per_iteration, 4)
        self.assertEqual(self.config.self_play.games_per_iteration, 128)
        self.assertEqual(
            self.config.runtime.output_directory,
            "training_runs/9x9",
        )

    def test_saved_19x19_stage_remains_available(self) -> None:
        config = load_rl_training_config(
            DEFAULT_RTX_PROFILE.with_name("rl_training.rtx5070ti.19x19.json")
        )
        self.assertEqual(config.game.board_size, 19)
        self.assertEqual(model_kind_for_config(config), "b15c192")
        self.assertEqual(
            config.runtime.output_directory,
            "training_runs/high_performance",
        )

    def test_curriculum_profiles_keep_one_network_and_separate_data(self) -> None:
        stage_cases = (
            (
                "rl_training.rtx5070ti.13x13.json",
                13,
                300,
                0.064,
                20,
                "training_runs/13x13",
            ),
            (
                "rl_training.rtx5070ti.curriculum.19x19.json",
                19,
                400,
                0.03,
                30,
                "training_runs/19x19",
            ),
        )
        stage_directories = {self.config.runtime.output_directory}

        for (
            filename,
            board_size,
            visits,
            dirichlet_alpha,
            temperature_halflife,
            output_directory,
        ) in stage_cases:
            with self.subTest(board_size=board_size):
                config = load_rl_training_config(
                    DEFAULT_RTX_PROFILE.with_name(filename)
                )
                overrides = build_selfplay_overrides(config)

                self.assertEqual(model_kind_for_config(config), "b10c128")
                self.assertEqual(config.game.board_size, board_size)
                self.assertEqual(config.search.simulations_per_move, visits)
                self.assertAlmostEqual(
                    config.search.dirichlet_alpha,
                    dirichlet_alpha,
                )
                self.assertEqual(
                    config.search.temperature_moves,
                    temperature_halflife,
                )
                self.assertEqual(config.self_play.games_per_iteration, 128)
                self.assertIsNone(config.self_play.resign_threshold)
                self.assertEqual(config.runtime.output_directory, output_directory)

                self.assertEqual(overrides["dataBoardLen"], board_size)
                self.assertEqual(overrides["bSizes"], board_size)
                self.assertEqual(overrides["maxVisits"], visits)
                self.assertEqual(overrides["numGameThreads"], 128)
                self.assertEqual(overrides["nnMaxBatchSize"], 128)
                self.assertAlmostEqual(
                    overrides["rootDirichletNoiseTotalConcentration"],
                    dirichlet_alpha * board_size * board_size,
                )
                self.assertEqual(
                    overrides["chosenMoveTemperatureHalflife"],
                    temperature_halflife,
                )
                self.assertFalse(overrides["initGamesWithPolicy"])

                self.assertNotIn(output_directory, stage_directories)
                stage_directories.add(output_directory)

    def test_full_selfplay_overrides_match_game_and_gpu(self) -> None:
        overrides = build_selfplay_overrides(self.config)

        self.assertEqual(overrides["dataBoardLen"], 9)
        self.assertEqual(overrides["bSizes"], 9)
        self.assertEqual(overrides["koRules"], "POSITIONAL")
        self.assertEqual(overrides["scoringRules"], "AREA")
        self.assertEqual(overrides["komiMean"], 6.5)
        self.assertFalse(overrides["komiAuto"])
        self.assertFalse(overrides["fancyKomiVarying"])
        self.assertFalse(overrides["initGamesWithPolicy"])
        self.assertEqual(overrides["maxVisits"], 200)
        self.assertEqual(overrides["numGameThreads"], 128)
        self.assertEqual(overrides["nnMaxBatchSize"], 128)
        self.assertEqual(overrides["numNNServerThreadsPerModel"], 2)
        self.assertTrue(overrides["cudaUseFP16"])
        self.assertTrue(overrides["cudaUseNHWC"])
        self.assertAlmostEqual(
            overrides["rootDirichletNoiseTotalConcentration"],
            9.72,
        )

    def test_random_bootstrap_smoke_profile_avoids_model_only_option(self) -> None:
        overrides = build_selfplay_overrides(
            self.config,
            visits=8,
            smoke=True,
            has_model=False,
        )

        self.assertEqual(overrides["numGameThreads"], 8)
        self.assertEqual(overrides["nnMaxBatchSize"], 16)
        self.assertEqual(overrides["maxVisits"], 8)
        self.assertNotIn("cudaUseNHWC", overrides)

    def test_invalid_zero_visits_is_rejected(self) -> None:
        with self.assertRaisesRegex(KataGoRLRunnerError, "至少为 1"):
            build_selfplay_overrides(self.config, visits=0)

    def test_override_formatter_uses_katago_boolean_spelling(self) -> None:
        formatted = format_katago_overrides(
            {"cudaUseFP16": True, "komiAuto": False, "maxVisits": 400}
        )
        self.assertEqual(
            formatted,
            "cudaUseFP16=true,komiAuto=false,maxVisits=400",
        )

    def test_train_rejects_invalid_batch_override_before_launch(self) -> None:
        runner = KataGoRLRunner(DEFAULT_RTX_PROFILE)
        with patch.object(runner, "_require_runtime"), patch.object(
            runner, "_ensure_directories"
        ), self.assertRaisesRegex(KataGoRLRunnerError, "批次"):
            runner.train(batch_size=0)

    def test_train_uses_one_epoch_with_monotonic_row_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = KataGoRLRunner(
                DEFAULT_RTX_PROFILE, run_root_override=Path(temporary)
            )
            data = runner.shuffled_dir / "latest" / "train" / "data0.npz"
            data.parent.mkdir(parents=True)
            data.write_bytes(b"test")
            commands = []

            def capture(command, **_kwargs):
                commands.append([str(value) for value in command])

            with patch.object(runner, "_require_runtime"), patch.object(
                runner, "_run", side_effect=capture
            ):
                runner.train()
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("-max-epochs-this-instance") + 1], "1")
        self.assertEqual(
            command[command.index("-max-train-bucket-per-new-data") + 1], "4"
        )
        self.assertEqual(command[command.index("-max-train-bucket-size") + 1], "4096")
        self.assertIn("-stop-when-train-bucket-limited", command)

    def test_shuffle_passes_persisted_deleted_row_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = KataGoRLRunner(
                DEFAULT_RTX_PROFILE, run_root_override=Path(temporary)
            )
            model = "gogogo-s100-d1000"
            (runner.models_dir / model).mkdir(parents=True)
            old = runner.selfplay_dir / "older" / "tdata" / "old.npz"
            new = runner.selfplay_dir / model / "tdata" / "new.npz"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            np.savez_compressed(
                old, globalInputNC=np.zeros((800, 1), dtype=np.float32)
            )
            np.savez_compressed(
                new, globalInputNC=np.zeros((100, 1), dtype=np.float32)
            )
            commands = []

            def capture(command, **_kwargs):
                commands.append([str(value) for value in command])

            with patch.object(runner, "_require_runtime"), patch.object(
                runner, "_run", side_effect=capture
            ):
                runner.shuffle(min_rows=1)

            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertEqual(command.count("-add-to-data-rows"), 1)
            self.assertEqual(command[command.index("-add-to-data-rows") + 1], "200")
            self.assertEqual(command[2], str(runner.selfplay_dir))

    def test_cycle_rejects_zero_step_training_without_new_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = KataGoRLRunner(
                DEFAULT_RTX_PROFILE, run_root_override=Path(temporary)
            )
            with patch.object(runner, "selfplay", return_value={}), patch.object(
                runner, "shuffle"
            ), patch.object(runner, "train"), patch.object(
                runner, "export_new_models", return_value=[]
            ), self.assertRaisesRegex(KataGoRLRunnerError, "静默空转"):
                runner.cycle()

    def test_curriculum_commands_have_a_dedicated_config(self) -> None:
        args = build_parser().parse_args(["curriculum-status"])
        self.assertEqual(args.command, "curriculum-status")
        self.assertEqual(args.curriculum_config, DEFAULT_CURRICULUM_PROFILE)


if __name__ == "__main__":
    unittest.main()
