"""Tests for KataGo configuration, protocol conversion, and rank profiles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from weiqi.ai import KATAGO_DIFFICULTIES
from weiqi.engine import GoGame
from weiqi.katago import (
    KATAGO_PROFILES,
    KataGoEngine,
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
            [f"rank_{dan}d" for dan in range(1, 10)],
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
            "rank_5d",
        )
        self.assertFalse(query["overrideSettings"]["ignorePreRootHistory"])

    def test_empty_game_query_sets_initial_player(self) -> None:
        game = GoGame(13)
        query = build_analysis_query(game, KATAGO_PROFILES[0], "empty", False)
        self.assertEqual(query["moves"], [])
        self.assertEqual(query["initialPlayer"], "B")
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
            self.assertIn("5 段棋谱分布", decision.explanation)
            self.assertTrue(game.analyze_move(*target).legal)
            engine.close()


if __name__ == "__main__":
    unittest.main()
