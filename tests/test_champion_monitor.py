"""Champion labels follow promotions and current rates never reuse old opponents."""

import json
from pathlib import Path
import tempfile
import unittest
import os
from unittest.mock import patch

from weiqi.monitor_legacy import champion_lineage
from weiqi.training_monitor import MonitorStore


def challenge(candidate, before, after, promoted=False, wins=120, draws=0):
    return {"candidate_model": candidate, "champion_before": before,
            "champion_after": after, "champion_promoted": promoted,
            "training_cycle": 8, "stage_samples": 1000,
            "opponents": [{"model": before, "roles": ["champion"], "match": {
                "games": 200, "candidate_wins": wins, "baseline_wins": 200 - wins - draws,
                "draws": draws, "win_rate": (wins + draws / 2) / 200}}]}


class ChampionMonitorTests(unittest.TestCase):
    def test_numbering_only_advances_on_unique_promotions(self):
        promotion = challenge("b", "a", "b", True)
        stage = {"entry_model": "seed", "champion_model": "c", "evaluations": [
            challenge("failed", "a", "a"), promotion, promotion,
            challenge("lost", "b", "b"), challenge("c", "b", "c", True)]}
        champions, versions = champion_lineage(stage)
        self.assertEqual([(row["label"], row["model"]) for row in champions],
                         [("T0", "a"), ("T1", "b"), ("T2", "c")])
        self.assertEqual(versions[-1]["b"], 1)
        self.assertNotIn("c", versions[-1])
        self.assertEqual(champion_lineage({"champion_model": "new-stage"})[0][0]["label"], "T0")

    def test_current_model_requires_match_against_current_champion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "curriculum"
            run.mkdir()
            state = {"active_board_size": 19, "stages": {"19x19": {
                "champion_model": "b", "latest_model": "new", "evaluations": [
                    challenge("b", "a", "b", True)]}}}
            path = run / "state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            store = MonitorStore(root)
            championship = store.snapshot("curriculum")["championship"]
            self.assertEqual(championship["current"]["label"], "T1")
            self.assertIsNone(championship["match"])
            self.assertEqual(championship["rounds"][0]["champion_label"], "T0")
            state["stages"]["19x19"]["evaluations"].append(challenge("new", "b", "b", wins=90, draws=10))
            path.write_text(json.dumps(state), encoding="utf-8")
            championship = store.snapshot("curriculum")["championship"]
            self.assertEqual(championship["match"]["win_rate"], 0.45)
            self.assertEqual(championship["match"]["score_rate"], 0.475)
            self.assertEqual(championship["match"]["champion_label"], "T1")
            self.assertEqual(championship["rounds"][-1]["champion_round"], 2)


class LiveChampionMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "curriculum"
        self.run.mkdir()
        self.state = {"phase": "evaluating", "active_board_size": 19, "stages": {"19x19": {
            "samples": 1000, "champion_model": "champ", "latest_model": "candidate-model",
            "pending_champion_evaluation": True, "evaluations": []}}}
        (self.run / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.opponent = self.run / "evaluations" / "19x19" / "s1000-candidate-model-20261021-120000" / "opponent-00-champ"
        self.opponent.mkdir(parents=True)
        self.manifest = {"candidate_model": "candidate-model", "baseline_model": "champ",
                         "board_size": 19, "roles": ["champion"], "games": 200}
        self.manifest_path = self.opponent / "evaluation_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.store = MonitorStore(self.root)
        self.win = "(;PB[candidate]PW[baseline]RE[B+1.5])\n"
        self.loss = "(;PB[baseline]PW[candidate]RE[B+2.5])\n"

    def pair(self, index, content, retry=None, config="same opening config"):
        path = self.opponent / "pairs" / f"opening-{index:03d}"
        if retry:
            path = path / retry
        (path / "sgfs").mkdir(parents=True, exist_ok=True)
        (path / "match.cfg").write_text(config, encoding="utf-8")
        sgf = path / "sgfs" / "games.sgfs"
        sgf.write_text(content, encoding="utf-8")
        return sgf

    def live(self):
        return self.store.snapshot("curriculum")["championship"]["live_match"]

    def test_single_finished_game_updates_before_pair_and_ignores_incomplete_tail(self):
        sgf = self.pair(0, self.win + "(;PB[baseline]PW[candidate]RE[")
        first = self.live()
        self.assertEqual((first["games"], first["wins"], first["completed_pairs"]), (1, 1, 0))
        self.assertEqual(first["win_rate"], 1)
        sgf.write_text(self.win + self.loss, encoding="utf-8")
        final = self.live()
        self.assertEqual((final["games"], final["wins"], final["losses"]), (2, 1, 1))
        self.assertEqual(final["win_rate"], 0.5)
        self.assertEqual(final["completed_pairs"], 1)
        self.assertEqual(self.store.snapshot("curriculum")["championship"]["rounds"], [])

    def test_retries_are_not_double_counted_and_new_partial_attempt_replaces_abandoned_one(self):
        self.pair(0, self.win + self.loss)
        self.pair(0, self.win + self.loss, retry="retry-complete")
        self.pair(1, self.win)
        retried = self.pair(1, self.loss, retry="retry-active")
        # Force a deterministic later retry start even on coarse filesystems.
        started = (retried.parent.parent / "match.cfg")
        os.utime(started, ns=(started.stat().st_atime_ns, started.stat().st_mtime_ns + 10000000))
        live = self.live()
        self.assertEqual((live["games"], live["wins"], live["losses"]), (3, 1, 2))
        self.assertEqual(live["completed_pairs"], 1)

    def test_wrong_model_and_no_games_never_reuse_old_winrate(self):
        self.assertIsNone(self.live()["win_rate"])
        self.pair(0, self.win)
        self.manifest["candidate_model"] = "old-model"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.assertIsNone(self.live())
        self.manifest["candidate_model"] = "candidate-model"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.state["phase"] = "training"
        (self.run / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        self.assertIsNone(self.live())

    def test_draws_and_cached_reading_leave_training_files_unchanged(self):
        self.pair(0, self.win + "(;PB[baseline]PW[candidate]RE[0])\n")
        before = {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        result = self.live()
        self.assertEqual(result["win_rate"], 0.5)
        self.assertEqual(result["score_rate"], 0.75)
        with patch("weiqi.monitor_match.parse_match_game", side_effect=AssertionError("unchanged SGFs should use cache")):
            self.assertEqual(self.live(), result)
        self.assertEqual(before, {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()})

    def test_invalid_results_and_incompatible_retry_are_excluded(self):
        self.pair(0, "(;PB[candidate]PW[baseline]RE[Void])\n")
        self.pair(0, self.win + self.loss, retry="retry-wrong-config", config="different opening")
        self.pair(1, "(;PB[unrelated]PW[baseline]RE[B+1.5])\n")
        self.assertEqual(self.live()["games"], 0)

    def test_live_curve_orders_finished_games_and_rebuilds_after_restart(self):
        first = self.pair(0, self.win + self.loss)
        second = self.pair(1, "(;PB[candidate]PW[baseline]RE[W+1.5])\n" + self.loss)
        os.utime(first, ns=(2000000000, 2000000000))
        os.utime(second, ns=(1000000000, 1000000000))
        curve = self.live()["points"]
        self.assertEqual([point["completed"] for point in curve], [1, 2, 3, 4])
        self.assertEqual([point["win_rate"] for point in curve], [0, 0, 1 / 3, 0.25])
        self.store = MonitorStore(self.root)
        self.assertEqual(self.live()["points"], curve)

    def test_finished_curve_survives_new_candidate_and_monitor_restart(self):
        self.pair(0, self.win + self.loss)
        self.pair(1, self.win + self.loss)
        live_points = self.live()["points"]
        self.manifest["games"] = 4
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.state["phase"] = "training"
        stage = self.state["stages"]["19x19"]
        record = challenge("candidate-model", "champ", "champ", wins=2)
        record["opponents"][0]["match"].update(games=4, baseline_wins=2, win_rate=0.5)
        stage["evaluations"] = [record]
        stage["latest_model"] = "next-candidate"
        stage["samples"] = 2000
        (self.run / "state.json").write_text(json.dumps(self.state), encoding="utf-8")
        before = {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        curve = self.store.champion_curve("curriculum", 1)
        self.assertFalse(curve["live"])
        self.assertEqual(curve["champion_label"], "T0")
        self.assertEqual(curve["points"], live_points)
        self.store = MonitorStore(self.root)
        self.assertEqual(self.store.champion_curve("curriculum", 1)["points"], live_points)
        self.assertEqual(before, {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()})
        with self.assertRaises(ValueError):
            self.store.champion_curve("../curriculum", 1)
        with self.assertRaises(ValueError):
            self.store.champion_curve("curriculum", "not-a-round")
        with self.assertRaises(FileNotFoundError):
            self.store.champion_curve("curriculum", 1000)
