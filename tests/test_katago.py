"""Tests for KataGo configuration, protocol conversion, and rank profiles."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from weiqi.ai import HUMANSL_DIFFICULTIES, KATAGO_DIFFICULTIES
from weiqi.engine import GoGame
from weiqi.katago import (
    HUMANSL_PROFILE_NAMES,
    HUMANSL_PROFILES,
    KATAGO_PROFILES,
    KataGoAI,
    KataGoConfigurationError,
    KataGoEngine,
    KataGoEngineError,
    KataGoSettings,
    build_analysis_query,
    point_to_vertex,
    profile_for_difficulty,
    vertex_to_point,
)


class KataGoProtocolTests(unittest.TestCase):
    def test_professional_profiles_are_strictly_increasing(self) -> None:
        self.assertEqual(
            tuple(profile.label for profile in KATAGO_PROFILES),
            KATAGO_DIFFICULTIES,
        )
        self.assertEqual(
            [profile.dan for profile in KATAGO_PROFILES],
            list(range(1, 10)),
        )
        visits = [profile.max_visits for profile in KATAGO_PROFILES]
        self.assertEqual(visits, sorted(visits))
        self.assertEqual(len(visits), len(set(visits)))
        self.assertEqual(
            [profile.human_sl_profile for profile in KATAGO_PROFILES],
            ["proyear_2023"] * 9,
        )
        self.assertEqual(
            [profile.move_temperature for profile in KATAGO_PROFILES],
            sorted(
                (profile.move_temperature for profile in KATAGO_PROFILES),
                reverse=True,
            ),
        )
        self.assertEqual(
            [profile.utility_scale for profile in KATAGO_PROFILES],
            sorted(
                (profile.utility_scale for profile in KATAGO_PROFILES),
                reverse=True,
            ),
        )

    def test_gtp_coordinates_round_trip_for_all_board_sizes(self) -> None:
        for size in (9, 13, 19):
            for point in ((0, 0), (size // 2, size // 2), (size - 1, size - 1)):
                vertex = point_to_vertex(point, size)
                self.assertEqual(vertex_to_point(vertex, size), point)
        self.assertEqual(point_to_vertex((18, 18), 19), "T1")
        self.assertIsNone(vertex_to_point("pass", 19))
        with self.assertRaises(ValueError):
            vertex_to_point("I9", 19)

    def test_analysis_query_preserves_history_rules_and_rank(self) -> None:
        game = GoGame(9, komi=6.5)
        self.assertTrue(game.play(8, 0).legal)
        self.assertTrue(game.pass_turn())
        profile = profile_for_difficulty(KATAGO_DIFFICULTIES[4])
        query = build_analysis_query(game, profile, "request-1", True)

        self.assertEqual(query["id"], "request-1")
        self.assertEqual(query["moves"], [["B", "A1"], ["W", "pass"]])
        self.assertNotIn("initialPlayer", query)
        self.assertEqual(query["boardXSize"], 9)
        self.assertEqual(query["boardYSize"], 9)
        self.assertEqual(query["komi"], 6.5)
        self.assertEqual(query["maxVisits"], profile.max_visits)
        self.assertEqual(query["rules"]["ko"], "POSITIONAL")
        self.assertEqual(query["rules"]["scoring"], "AREA")
        self.assertFalse(query["rules"]["suicide"])
        self.assertTrue(query["includePolicy"])
        self.assertEqual(
            query["overrideSettings"]["humanSLProfile"],
            "proyear_2023",
        )
        self.assertFalse(query["overrideSettings"]["ignorePreRootHistory"])
        self.assertEqual(
            query["overrideSettings"]["humanSLRootExploreProbWeightless"],
            0.5,
        )

    def test_human_rank_profiles_cover_20k_through_9d(self) -> None:
        self.assertEqual(
            tuple(profile.label for profile in HUMANSL_PROFILES),
            HUMANSL_DIFFICULTIES,
        )
        self.assertEqual(
            tuple(profile.human_sl_profile for profile in HUMANSL_PROFILES),
            HUMANSL_PROFILE_NAMES,
        )
        self.assertEqual(HUMANSL_PROFILE_NAMES[0], "rank_20k")
        self.assertEqual(HUMANSL_PROFILE_NAMES[-1], "rank_9d")
        self.assertTrue(
            all(profile.selection_mode == "human_rank" for profile in HUMANSL_PROFILES)
        )
        self.assertTrue(all(profile.max_visits == 64 for profile in HUMANSL_PROFILES))

    def test_human_rank_query_omits_professional_blend_overrides(self) -> None:
        profile = HUMANSL_PROFILES[17]
        query = build_analysis_query(GoGame(19), profile, "human-rank", True)

        self.assertTrue(query["includePolicy"])
        self.assertEqual(query["maxVisits"], 64)
        self.assertEqual(
            query["overrideSettings"],
            {
                "humanSLProfile": profile.human_sl_profile,
                "ignorePreRootHistory": False,
            },
        )

    def test_empty_game_query_sets_initial_player(self) -> None:
        game = GoGame(13)
        query = build_analysis_query(game, KATAGO_PROFILES[0], "empty", False)
        self.assertEqual(query["moves"], [])
        self.assertEqual(query["initialPlayer"], "B")
        self.assertNotIn("includePolicy", query)

    def test_workbench_query_options_override_play_profile_defaults(self) -> None:
        query = build_analysis_query(
            GoGame(19),
            KATAGO_PROFILES[0],
            "workbench",
            False,
            max_visits=777,
            pv_length=19,
            include_ownership=True,
        )

        self.assertEqual(query["maxVisits"], 777)
        self.assertEqual(query["analysisPVLen"], 19)
        self.assertTrue(query["includeOwnership"])
        self.assertNotIn("includePolicy", query)


class KataGoSettingsAndDecisionTests(unittest.TestCase):
    def _settings(self, folder: Path, with_human: bool = False) -> KataGoSettings:
        executable = folder / "katago.exe"
        model = folder / "main.bin.gz"
        executable.write_bytes(b"fake executable")
        model.write_bytes(b"fake model")
        human = folder / "b18c384nbt-humanv0.bin.gz"
        if with_human:
            human.write_bytes(b"fake human model")
        return KataGoSettings(
            executable=str(executable),
            model=str(model),
            human_model=str(human) if with_human else "",
        )

    def test_settings_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            settings = self._settings(folder, with_human=True)
            self.assertEqual(settings.validation_errors(), ())
            self.assertTrue(settings.human_style_enabled)

            settings_path = folder / "settings.json"
            settings.save(settings_path)
            loaded = KataGoSettings.load(settings_path)
            self.assertEqual(loaded.fingerprint, settings.fingerprint)

    def test_local_discovery_reuses_legacy_install_but_prefers_standard_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standard = root / "katago"
            legacy = root / "vendor" / "katago"
            (legacy / "runtime").mkdir(parents=True)
            (legacy / "models").mkdir(parents=True)
            legacy_executable = legacy / "runtime" / "katago.exe"
            legacy_model = legacy / "models" / "legacy.bin.gz"
            legacy_human = legacy / "models" / "legacy-human.bin.gz"
            for path in (legacy_executable, legacy_model, legacy_human):
                path.write_bytes(b"test")

            with patch("weiqi.katago.KATAGO_FOLDER", standard):
                with patch("weiqi.katago.LEGACY_KATAGO_FOLDER", legacy):
                    discovered = KataGoSettings._discover_local_files()
                    self.assertEqual(discovered.executable, str(legacy_executable))
                    self.assertEqual(discovered.model, str(legacy_model))
                    self.assertEqual(discovered.human_model, str(legacy_human))

                    standard.mkdir()
                    standard_executable = standard / "katago.exe"
                    standard_model = standard / "standard.bin.gz"
                    standard_human = standard / "standard-human.bin.gz"
                    for path in (
                        standard_executable,
                        standard_model,
                        standard_human,
                    ):
                        path.write_bytes(b"test")

                    preferred = KataGoSettings._discover_local_files()
                    self.assertEqual(preferred.executable, str(standard_executable))
                    self.assertEqual(preferred.model, str(standard_model))
                    self.assertEqual(preferred.human_model, str(standard_human))

    def test_main_network_decision_uses_selected_move_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary))
            engine = KataGoEngine(settings, seed=1)
            game = GoGame(9)
            response = {
                "moveInfos": [
                    {
                        "move": "D4",
                        "order": 0,
                        "winrate": 0.61,
                        "scoreLead": 2.75,
                    }
                ],
                "rootInfo": {"winrate": 0.55, "scoreLead": 1.0, "visits": 24},
            }
            decision = engine._decision_from_response(
                game,
                KATAGO_PROFILES[0],
                response,
                human_style_requested=False,
            )
            self.assertEqual(decision.point, (5, 3))
            self.assertAlmostEqual(decision.black_win_probability or 0.0, 0.61)
            self.assertAlmostEqual(decision.black_lead or 0.0, 2.75)
            self.assertEqual(decision.analysis_visits, 24)
            self.assertIn("未配置人类风格模型", decision.explanation)
            engine.close()

    def test_analyze_position_returns_raw_response_with_workbench_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary))
            engine = KataGoEngine(settings)
            process = unittest.mock.Mock()
            response = {
                "id": "response-id",
                "rootInfo": {"winrate": 0.58, "scoreLead": 1.75},
                "moveInfos": [{"move": "D4", "pv": ["D4", "E4"]}],
                "ownership": [0.0] * 81,
            }

            with patch.object(engine, "_ensure_started", return_value=process):
                with patch.object(
                    engine,
                    "_wait_for_response",
                    return_value=response,
                ) as wait_for_response:
                    result = engine.analyze_position(
                        GoGame(9),
                        max_visits=321,
                        pv_length=15,
                        include_ownership=True,
                    )

            self.assertIs(result, response)
            serialized = process.stdin.write.call_args.args[0]
            query = json.loads(serialized)
            self.assertEqual(query["maxVisits"], 321)
            self.assertEqual(query["analysisPVLen"], 15)
            self.assertTrue(query["includeOwnership"])
            self.assertNotIn("includePolicy", query)
            wait_for_response.assert_called_once_with(query["id"])
            process.stdin.flush.assert_called_once_with()
            engine.close()

    def test_stop_detaches_process_and_reaps_it_off_the_calling_thread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary))
            engine = KataGoEngine(settings)
            process = unittest.mock.Mock()
            process.poll.return_value = None
            engine._process = process

            with patch("weiqi.katago.threading.Thread") as thread_type:
                engine.stop()

            self.assertIsNone(engine._process)
            process.terminate.assert_called_once_with()
            process.wait.assert_not_called()
            thread_type.assert_called_once()
            thread_type.return_value.start.assert_called_once_with()
            engine.close()

    def test_cancelled_workbench_request_never_starts_katago(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary))
            engine = KataGoEngine(settings)
            cancel_event = threading.Event()
            cancel_event.set()

            with patch("weiqi.katago.subprocess.Popen") as popen:
                with self.assertRaisesRegex(KataGoEngineError, "取消"):
                    engine.analyze_position(
                        GoGame(9),
                        max_visits=8,
                        pv_length=4,
                        cancel_event=cancel_event,
                    )

            popen.assert_not_called()
            self.assertFalse(engine.running)
            engine.close()

    def test_human_policy_selects_rank_style_legal_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings, seed=3)
            game = GoGame(9)
            policy = [0.0] * 82
            target = (2, 6)
            policy[target[0] * 9 + target[1]] = 1.0
            response = {
                "moveInfos": [
                    {
                        "move": "D4",
                        "order": 0,
                        "winrate": 0.52,
                        "scoreLead": 0.4,
                        "humanPrior": 0.0,
                        "utility": 0.3,
                    },
                    {
                        "move": point_to_vertex(target, 9),
                        "order": 1,
                        "winrate": 0.51,
                        "scoreLead": 0.2,
                        "humanPrior": 1.0,
                        "utility": 0.2,
                    }
                ],
                "humanPolicy": policy,
                "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "visits": 120},
            }
            decision = engine._decision_from_response(
                game,
                KATAGO_PROFILES[4],
                response,
                human_style_requested=True,
            )
            self.assertEqual(decision.point, target)
            self.assertIn("2023 职业棋谱风格", decision.explanation)
            self.assertIn("职业 5 段参数", decision.explanation)
            self.assertTrue(game.analyze_move(*target).legal)
            engine.close()

    def test_human_rank_samples_policy_without_faking_post_move_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings, seed=11)
            game = GoGame(9)
            target = (2, 6)
            policy = [0.0] * 82
            policy[target[0] * game.size + target[1]] = 1.0
            response = {
                "moveInfos": [{"move": "D4", "order": 0}],
                "humanPolicy": policy,
                "rootInfo": {"winrate": 0.77, "scoreLead": 8.0, "visits": 64},
            }

            decision = engine._decision_from_response(
                game,
                HUMANSL_PROFILES[17],
                response,
                human_style_requested=True,
            )

            self.assertEqual(decision.point, target)
            self.assertIsNone(decision.black_win_probability)
            self.assertIsNone(decision.black_lead)
            self.assertIn("人类棋谱策略采样", decision.explanation)
            engine.close()

    def test_human_rank_uses_selected_move_evaluation_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings, seed=13)
            game = GoGame(9)
            target = (2, 6)
            policy = [0.0] * 82
            policy[target[0] * game.size + target[1]] = 1.0
            response = {
                "moveInfos": [
                    {
                        "move": point_to_vertex(target, game.size),
                        "order": 0,
                        "winrate": 0.63,
                        "scoreLead": 2.25,
                    }
                ],
                "humanPolicy": policy,
                "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "visits": 64},
            }

            decision = engine._decision_from_response(
                game,
                HUMANSL_PROFILES[0],
                response,
                human_style_requested=True,
            )

            self.assertEqual(decision.point, target)
            self.assertAlmostEqual(decision.black_win_probability or 0.0, 0.63)
            self.assertAlmostEqual(decision.black_lead or 0.0, 2.25)
            engine.close()

    def test_human_rank_rejects_request_without_enabled_human_style(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings)
            policy = [0.0] * 82
            policy[0] = 1.0
            response = {
                "moveInfos": [{"move": "A9", "order": 0}],
                "humanPolicy": policy,
                "rootInfo": {"visits": 64},
            }

            with self.assertRaisesRegex(KataGoEngineError, "缺少人类风格模型"):
                engine._decision_from_response(
                    GoGame(9),
                    HUMANSL_PROFILES[0],
                    response,
                    human_style_requested=False,
                )
            engine.close()

    def test_human_rank_rejects_policy_whose_weighted_points_are_illegal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings)
            game = GoGame(9)
            self.assertTrue(game.play(0, 0).legal)
            policy = [0.0] * 82
            policy[0] = 1.0
            response = {
                "moveInfos": [{"move": "D4", "order": 0}],
                "humanPolicy": policy,
                "rootInfo": {"visits": 64},
            }

            with self.assertRaisesRegex(KataGoEngineError, "可采样"):
                engine._decision_from_response(
                    game,
                    HUMANSL_PROFILES[0],
                    response,
                    human_style_requested=True,
                )
            engine.close()

    def test_human_rank_rejects_missing_or_empty_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings, seed=5)
            game = GoGame(9)
            profile = HUMANSL_PROFILES[0]
            missing_policy = {
                "moveInfos": [{"move": "D4", "order": 0}],
                "rootInfo": {"visits": 64},
            }
            with self.assertRaisesRegex(KataGoEngineError, "长度与棋盘匹配"):
                engine._decision_from_response(
                    game,
                    profile,
                    missing_policy,
                    human_style_requested=True,
                )

            invalid_weights = [0.0] * 82
            invalid_weights[0] = float("nan")
            invalid_weights[1] = float("inf")
            empty_policy = dict(missing_policy, humanPolicy=invalid_weights)
            with self.assertRaisesRegex(KataGoEngineError, "可采样"):
                engine._decision_from_response(
                    game,
                    profile,
                    empty_policy,
                    human_style_requested=True,
                )
            engine.close()

    def test_human_rank_allows_normal_search_to_choose_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings, seed=2)
            game = GoGame(9)
            policy = [0.0] * 82
            policy[0] = 1.0
            response = {
                "moveInfos": [{"move": "pass", "order": 0}],
                "humanPolicy": policy,
                "rootInfo": {"winrate": 0.6, "scoreLead": 1.5, "visits": 64},
            }

            decision = engine._decision_from_response(
                game,
                HUMANSL_PROFILES[0],
                response,
                human_style_requested=True,
            )

            self.assertIsNone(decision.point)
            self.assertIsNone(decision.black_win_probability)
            self.assertIsNone(decision.black_lead)
            self.assertIn("虚手", decision.explanation)
            engine.close()

    def test_human_rank_rejects_policy_without_normal_search_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings)
            policy = [0.0] * 82
            policy[0] = 1.0
            response = {
                "moveInfos": [],
                "humanPolicy": policy,
                "rootInfo": {"visits": 64},
            }

            with self.assertRaisesRegex(KataGoEngineError, "普通搜索结果"):
                engine._decision_from_response(
                    GoGame(9),
                    HUMANSL_PROFILES[0],
                    response,
                    human_style_requested=True,
                )
            engine.close()

    def test_human_rank_requires_a_configured_human_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=False)
            engine = KataGoEngine(settings)
            with self.assertRaisesRegex(KataGoConfigurationError, "人类风格模型"):
                KataGoAI(engine, HUMANSL_DIFFICULTIES[0])
            engine.close()

    def test_settings_load_precedence_covers_current_and_legacy_environment_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_path = Path(temporary) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "executable": "saved.exe",
                        "model": "saved-model.bin.gz",
                        "human_model": "saved-human.bin.gz",
                    }
                ),
                encoding="utf-8",
            )
            discovered = KataGoSettings(
                executable="discovered.exe",
                model="discovered-model.bin.gz",
                human_model="discovered-human.bin.gz",
            )
            environment = {
                "KATAGO_EXE": "current.exe",
                "WEIQI_KATAGO_EXE": "legacy.exe",
                "WEIQI_KATAGO_MODEL": "legacy-model.bin.gz",
            }

            with patch.dict("os.environ", environment, clear=True):
                with patch.object(
                    KataGoSettings,
                    "_discover_local_files",
                    return_value=discovered,
                ):
                    loaded = KataGoSettings.load(settings_path)

            self.assertEqual(loaded.executable, "current.exe")
            self.assertEqual(loaded.model, "legacy-model.bin.gz")
            self.assertEqual(loaded.human_model, "saved-human.bin.gz")

    def test_discovery_can_mix_standard_executable_with_legacy_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            standard = root / "katago"
            legacy = root / "vendor" / "katago"
            standard.mkdir()
            (legacy / "models").mkdir(parents=True)
            executable = standard / "katago.exe"
            model = legacy / "models" / "legacy.bin.gz"
            human = legacy / "models" / "legacy-human.bin.gz"
            for path in (executable, model, human):
                path.write_bytes(b"test")

            with patch("weiqi.katago.KATAGO_FOLDER", standard):
                with patch("weiqi.katago.LEGACY_KATAGO_FOLDER", legacy):
                    discovered = KataGoSettings._discover_local_files()

            self.assertEqual(discovered.executable, str(executable))
            self.assertEqual(discovered.model, str(model))
            self.assertEqual(discovered.human_model, str(human))

    def test_professional_blend_uses_side_to_move_utility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings = self._settings(Path(temporary), with_human=True)
            engine = KataGoEngine(settings, seed=7)
            game = GoGame(9)
            self.assertTrue(game.play(4, 4).legal)  # White is now to move.
            white_favored = (2, 2)
            black_favored = (6, 6)
            move_infos = [
                {
                    "move": point_to_vertex(black_favored, 9),
                    "humanPrior": 0.5,
                    "utility": 0.8,
                },
                {
                    "move": point_to_vertex(white_favored, 9),
                    "humanPrior": 0.5,
                    "utility": -0.8,
                },
            ]
            selected = engine._sample_professional_moves(
                game,
                move_infos,
                KATAGO_PROFILES[-1],
            )
            self.assertEqual(selected, white_favored)
            engine.close()


if __name__ == "__main__":
    unittest.main()
