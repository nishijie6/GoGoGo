"""Exercise the read-only dashboard against real incremental files and HTTP."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from monitor_training import make_server
from weiqi.training_monitor import EventTail, MonitorStore


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def append_events(path, *events):
    with path.open("ab") as stream:
        for event in events:
            stream.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))


def batch(phase="selfplay", opponents=("champion_4",), iteration=5):
    return [
        {"event": "iteration_started", "iteration": iteration, "phase": phase},
        {"event": "games_started", "phase": phase, "total": len(opponents),
         "jobs": [{"index": index, "opponent_name": opponent,
                   "opponent_sha256": "a" * 64 if opponent != "candidate_self" else None}
                  for index, opponent in enumerate(opponents)]},
    ]


class EventTailTests(unittest.TestCase):
    def test_incomplete_utf8_jsonl_record_is_retried_without_duplicate_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            first = {"event": "ready", "iteration": 4, "phase": "startup"}
            second = {"event": "training_step", "phase": "training", "step": 1,
                      "total": 8, "message": "训练", "loss": 0.75}
            append_events(path, first)
            encoded = (json.dumps(second, ensure_ascii=False) + "\n").encode("utf-8")
            split = encoded.index("训练".encode("utf-8")) + 1
            with path.open("ab") as stream:
                stream.write(encoded[:split])
            tail = EventTail()
            tail.update(path)
            tail.update(path)
            self.assertEqual(list(tail.events), [first])
            self.assertFalse(tail.malformed)
            with path.open("ab") as stream:
                stream.write(encoded[split:])
            tail.update(path)
            tail.update(path)
            self.assertEqual(list(tail.events), [first, second])
            self.assertEqual(tail.progress, {"completed": 1, "total": 8, "unit": "steps"})
            self.assertEqual(tail.offset, path.stat().st_size)

    def test_malformed_records_do_not_prevent_later_valid_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_bytes(b'{bad json}\n{"event":"training_step","loss":NaN}\n\xff\n')
            append_events(path, {"event": "ready", "iteration": 7})
            tail = EventTail()
            tail.update(path)
            self.assertTrue(tail.malformed)
            self.assertEqual(tail.iteration, 7)
            self.assertEqual([row["event"] for row in tail.events], ["ready"])

    def test_new_batch_clears_results_and_candidate_self_is_not_candidate_winrate(self):
        tail = EventTail()
        for event in batch(opponents=("candidate_self", "champion_4")):
            tail.accept(event)
        for index, opponent in enumerate(("candidate_self", "champion_4")):
            tail.accept({"event": "game", "phase": "selfplay", "index": index,
                         "opponent_name": opponent, "result": "win", "completed": index + 1,
                         "total": 2})
        match = tail.current_match("selfplay")
        self.assertEqual((match["games"], match["wins"], match["total"], match["batch_total"]), (1, 1, 1, 2))
        self.assertEqual(match["opponent"], "champion_4")
        self.assertEqual(match["opponent_sha256"], "a" * 64)
        self.assertEqual(match["win_rate"], 1)
        tail.accept(batch(opponents=("candidate_self",))[1])
        match = tail.current_match("selfplay")
        self.assertEqual(match["games"], 0)
        self.assertIsNone(match["win_rate"])
        self.assertIsNone(match["score_rate"])
        self.assertEqual(tail.progress["completed"], 0)

    def test_mixed_frozen_opponents_keep_separate_live_winrates(self):
        tail = EventTail()
        for event in batch(opponents=("accepted_000000", "milestone_000003", "candidate_self")):
            tail.accept(event)
        tail.accept({"event": "game", "phase": "selfplay", "index": 0,
                     "opponent_name": "accepted_000000", "result": "win",
                     "completed": 1, "total": 3})
        tail.accept({"event": "game", "phase": "selfplay", "index": 1,
                     "opponent_name": "milestone_000003", "result": "loss",
                     "completed": 2, "total": 3})
        self.assertIsNone(tail.current_match("selfplay"))
        matches = {row["opponent"]: row for row in tail.current_matches("selfplay")}
        self.assertEqual(matches["accepted_000000"]["win_rate"], 1.0)
        self.assertEqual(matches["milestone_000003"]["win_rate"], 0.0)
        self.assertNotIn("candidate_self", matches)

    def test_all_completed_outcomes_include_truncations_in_the_denominator(self):
        tail = EventTail()
        for event in batch(phase="evaluation", opponents=("best",) * 4):
            tail.accept(event)
        for index, result in enumerate(("win", "loss", "draw", "truncated")):
            event = {"event": "game", "phase": "evaluation", "index": index,
                     "opponent_name": "best", "result": result, "completed": index + 1, "total": 4}
            tail.accept(event)
        tail.accept(event)  # Replayed completion must not count the same game twice.
        match = tail.current_match("evaluation")
        self.assertEqual([match[key] for key in ("games", "wins", "losses", "draws", "truncated")],
                         [4, 1, 1, 1, 1])
        self.assertEqual(match["win_rate"], 0.25)
        self.assertEqual(match["score_rate"], 0.375)
        tail.accept({"event": "game", "phase": "evaluation", "index": 4,
                     "opponent_name": "best", "completed": 5, "total": 5})
        self.assertIsNone(tail.current_match("evaluation")["win_rate"])

    def test_phase_and_new_session_do_not_reuse_previous_match_progress(self):
        tail = EventTail()
        for event in batch():
            tail.accept(event)
        tail.accept({"event": "game", "phase": "selfplay", "index": 0,
                     "opponent_name": "champion_4", "result": "win", "completed": 1, "total": 1})
        tail.accept({"event": "training_started", "phase": "training"})
        self.assertIsNone(tail.current_match("training"))
        self.assertIsNone(tail.progress)
        tail.accept({"event": "training_step", "phase": "training", "step": 3, "total": 5})
        tail.accept({"event": "run_started", "phase": "startup", "session_id": "new"})
        self.assertIsNone(tail.current_match("selfplay"))
        self.assertIsNone(tail.progress)
        self.assertEqual(tail.step, {})
        self.assertEqual(tail.schedule, {})
        self.assertEqual(tail.session, "new")
        tail.accept({"event": "ready", "phase": "startup", "iteration": 8})
        self.assertEqual(tail.iteration, 8)

    def test_truncated_log_restarts_reader_without_previous_session_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append_events(path, *batch(opponents=("old", "old")))
            tail = EventTail()
            tail.update(path)
            path.write_text('{"event":"ready","iteration":1}\n', encoding="utf-8")
            tail.update(path)
            self.assertEqual(tail.iteration, 1)
            self.assertIsNone(tail.current_match("selfplay"))
            self.assertEqual(len(tail.events), 1)


class MonitorStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "run_a"
        self.run.mkdir()
        self.store = MonitorStore(self.root, preferred_run="run_a")

    def test_heartbeat_distinguishes_live_stale_and_terminal_states(self):
        for state, age, expected in (("running", 2, "running"), ("paused", 3, "paused"),
                                     ("running", 16, "stale"), ("paused", 100, "stale"),
                                     ("completed", 1000, "completed"), ("failed", 1000, "failed"),
                                     ("stopped", 1000, "stopped")):
            with self.subTest(state=state, age=age):
                write_json(self.run / "monitor.json", {"status": state, "phase": "evaluation",
                                                       "updated_at_epoch": 2000 - age})
                with patch("weiqi.training_monitor.time.time", return_value=2000):
                    status = self.store.snapshot("run_a")["status"]
                self.assertEqual(status["state"], expected)
                self.assertEqual(status["age_seconds"], age)

    def test_old_session_sidecar_cannot_mark_new_session_completed(self):
        write_json(self.run / "monitor.json", {"session_id": "session_a", "status": "completed",
                                               "phase": "evaluation", "updated_at_epoch": 1998})
        # The retained tail can start after run_started, so every event's session matters.
        append_events(self.run / "events.jsonl", {"event": "training_step", "session_id": "session_b",
                                                 "phase": "training", "time": "1970-01-01T00:33:19Z",
                                                 "step": 2, "total": 5, "training_steps": 32})
        with patch("weiqi.training_monitor.time.time", return_value=2000):
            snapshot = self.store.snapshot("run_a")
        self.assertEqual(snapshot["status"]["state"], "stale")
        self.assertEqual(snapshot["status"]["phase"], "training")
        self.assertEqual(snapshot["current"]["progress"]["completed"], 2)

    def test_same_session_terminal_event_overrides_unwritten_final_sidecar(self):
        for state in ("completed", "stopped", "failed"):
            with self.subTest(state=state):
                run = self.root / state
                session = "session_" + state
                write_json(run / "monitor.json", {"session_id": session, "status": "running",
                                                   "phase": "selfplay", "updated_at_epoch": 1900})
                append_events(run / "events.jsonl",
                              {"event": "run_started", "session_id": session, "phase": "startup"},
                              {"event": "run_" + state, "session_id": session, "phase": "evaluation",
                               "time": "1970-01-01T00:33:19Z"})
                with patch("weiqi.training_monitor.time.time", return_value=2000):
                    snapshot = self.store.snapshot(state)
                self.assertEqual(snapshot["status"]["state"], state)
                self.assertEqual(snapshot["status"]["phase"], "evaluation")

    def test_iteration_completion_uses_final_summary_after_unlogged_training_updates(self):
        write_json(self.run / "summary.json", {"iteration": 3, "training_steps": 100,
                                               "training": {"loss": 1.5}})
        append_events(self.run / "events.jsonl",
                      {"event": "iteration_started", "iteration": 4, "phase": "training"},
                      {"event": "training_step", "phase": "training", "step": 38, "total": 40,
                       "training_steps": 138, "loss": 0.9})
        self.assertEqual(self.store.snapshot("run_a")["current"]["training_steps"], 138)
        # Step 39 succeeds without a log; AMP skips step 40 and its final progress event.
        summary = {"iteration": 4, "training_steps": 139, "training": {"loss": 0.4}}
        write_json(self.run / "summary.json", summary)
        append_events(self.run / "events.jsonl", {"event": "iteration_completed", "phase": "evaluation", **summary})
        current = self.store.snapshot("run_a")["current"]
        self.assertEqual(current["training_steps"], 139)
        self.assertEqual(current["loss"], 0.4)
        self.assertEqual(current["completed_iteration"], 4)

    def test_recent_legacy_log_is_historical_and_never_invents_live_winrate(self):
        append_events(self.run / "events.jsonl", {"event": "game", "phase": "selfplay",
                                                 "time": "2026-09-26T10:00:00Z",
                                                 "completed": 1, "total": 10})
        snapshot = self.store.snapshot("run_a")
        self.assertEqual(snapshot["status"]["state"], "historical")
        self.assertIsNone(snapshot["current"]["match"])
        self.assertTrue(snapshot["warnings"])
        self.assertEqual(snapshot["current"]["progress"]["completed"], 1)

    def test_scheduled_milestone_is_visible_before_its_first_game(self):
        append_events(self.run / "events.jsonl",
                      {"event": "iteration_started", "phase": "selfplay", "iteration": 5},
                      {"event": "selfplay_schedule", "phase": "selfplay",
                       "champions": [{"name": "accepted_000000", "sha256": "best"}],
                       "milestones": [{"name": "milestone_000003", "sha256": "older"}],
                       "candidate_self_games": 8})
        snapshot = self.store.snapshot("run_a")
        self.assertEqual({row["name"] for row in snapshot["current"]["opponents"]},
                         {"accepted_000000", "milestone_000003", "candidate_self"})

    def test_live_snapshot_lists_frozen_opponents_without_merging_their_results(self):
        write_json(self.run / "monitor.json", {"status": "running", "phase": "selfplay",
                                               "updated_at_epoch": time.time()})
        append_events(self.run / "events.jsonl",
                      *batch(opponents=("accepted_000000", "milestone_000003", "candidate_self")),
                      {"event": "game", "phase": "selfplay", "index": 0,
                       "opponent_name": "accepted_000000", "result": "win",
                       "completed": 1, "total": 3},
                      {"event": "game", "phase": "selfplay", "index": 1,
                       "opponent_name": "milestone_000003", "result": "loss",
                       "completed": 2, "total": 3})
        snapshot = self.store.snapshot("run_a")
        self.assertIsNone(snapshot["current"]["match"])
        rows = [row for row in snapshot["opponents"] if row.get("live")]
        self.assertEqual({row["name"]: (row["win_rate"], row["games"], row["total"])
                          for row in rows},
                         {"accepted_000000": (1.0, 1, 1),
                          "milestone_000003": (0.0, 1, 1)})

    def test_history_keeps_new_promotion_test_separate_from_old_interval(self):
        sign = {"wins": 15, "losses": 5, "ties": 0, "p_value": 0.0207}
        write_json(self.run / "summary.json", {"iteration": 6, "evaluation": {
            "games": 40, "wins": 30, "score_rate": 0.75,
            "promotion_test": "paired_sign", "paired_sign": sign,
            "paired": {"lower": 0.44, "upper": 1.0}}})
        row = self.store.snapshot("run_a")["history"][0]["evaluation"]
        self.assertEqual(row["promotion_test"], "paired_sign")
        self.assertEqual(row["paired_sign"], sign)
        self.assertEqual(row["win_rate"], 0.75)

    def test_history_is_sorted_and_latest_summary_does_not_duplicate_iteration(self):
        for iteration in (2, 1):
            write_json(self.run / "iterations" / f"{iteration:06d}" / "summary.json",
                       {"iteration": iteration, "training": {"loss": 1 / iteration},
                        "evaluation": {"games": 4, "wins": 1, "losses": 1, "draws": 1,
                                       "truncated": 1, "score_rate": 0.375}, "promoted": iteration == 2})
        write_json(self.run / "summary.json", {"iteration": 2, "best_iteration": 2})
        snapshot = self.store.snapshot("run_a")
        self.assertEqual([row["iteration"] for row in snapshot["history"]], [1, 2])
        self.assertEqual(snapshot["history"][1]["evaluation"]["win_rate"], 0.25)
        self.assertEqual(snapshot["history"][1]["evaluation"]["score_rate"], 0.375)
        self.assertTrue(snapshot["history"][1]["promoted"])
        self.assertEqual(snapshot["current"]["completed_iteration"], 2)
        self.assertEqual(self.store.runs()["default_run"], "run_a")

    def test_saved_selfplay_games_reconstruct_exact_candidate_results(self):
        write_json(self.run / "summary.json", {
            "iteration": 3,
            "selfplay_opponents": [
                {"name": "champion_1", "sha256": "abc", "games": 4, "finished": 3},
                {"name": "candidate_self", "games": 2, "finished": 1, "wins": 1, "score_rate": 1},
            ],
        })
        for index, (color, winner, reason) in enumerate(((1, 1, "passes"), (2, 1, "passes"),
                                                         (1, None, "passes"), (2, None, "length_limit"))):
            write_json(self.run / "iterations" / "000003" / "selfplay" / f"game_{index:06d}.json",
                       {"candidate_color": color, "winner": winner, "reason": reason,
                        "training_opponent": {"name": "champion_1", "sha256": "abc"}})
        rows = {row["name"]: row for row in self.store.snapshot("run_a")["opponents"]}
        measured = rows["champion_1"]
        self.assertEqual([measured[key] for key in ("wins", "losses", "draws", "truncated")], [1, 1, 1, 1])
        self.assertEqual(measured["win_rate"], 0.25)
        self.assertEqual(measured["score_rate"], 0.375)
        self.assertEqual(measured["sha256"], "abc")
        self.assertIsNone(rows["candidate_self"]["win_rate"])
        self.assertIsNone(rows["candidate_self"]["score_rate"])
        self.assertEqual(rows["candidate_self"]["truncated"], 1)

    def test_partial_historical_game_records_do_not_claim_a_winrate(self):
        write_json(self.run / "summary.json", {"iteration": 1, "selfplay_opponents": [
            {"name": "champion_0", "games": 3, "finished": 3}]})
        write_json(self.run / "iterations" / "000001" / "selfplay" / "game_000000.json",
                   {"candidate_color": 1, "winner": 1, "reason": "passes",
                    "training_opponent": {"name": "champion_0"}})
        row = self.store.snapshot("run_a")["opponents"][0]
        self.assertEqual(row["games"], 3)
        self.assertIsNone(row["win_rate"])

    def test_malformed_json_recovers_after_atomic_replacement(self):
        path = self.run / "summary.json"
        path.write_text('{"iteration":', encoding="utf-8")
        self.assertEqual(self.store.snapshot("run_a")["latest"], {})
        replacement = self.run / "summary.tmp"
        write_json(replacement, {"iteration": 9, "training": {"loss": 0.25}})
        replacement.replace(path)
        self.assertEqual(self.store.snapshot("run_a")["current"]["completed_iteration"], 9)
        path.write_text('{"iteration":10,"loss":Infinity}', encoding="utf-8")
        self.assertEqual(self.store.snapshot("run_a")["latest"], {})


class MonitorHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "runs" / "train_测试"
        write_json(self.run / "summary.json", {"iteration": 2})
        web = self.root / "web"
        web.mkdir()
        (web / "index.html").write_text("<!doctype html><title>Training monitor</title>", encoding="utf-8")
        (web / "app.js").write_text("'use strict';", encoding="utf-8")
        (web / "style.css").write_text("body { color: #123; }", encoding="utf-8")
        (web / "private.txt").write_text("do not expose", encoding="utf-8")
        self.web_patch = patch("monitor_training.WEB", web)
        self.web_patch.start()
        self.addCleanup(self.web_patch.stop)
        self.server = make_server(self.run.parent, port=0, preferred_run=self.run.name)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, method="GET", body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=body)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_real_api_polls_show_newly_appended_training_results(self):
        status, headers, body = self.request("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["default_run"], self.run.name)
        self.assertEqual(headers["Cache-Control"], "no-store")
        url = "/api/snapshot?" + urlencode({"run": self.run.name})
        self.assertIsNone(json.loads(self.request(url)[2])["current"]["match"])
        append_events(self.run / "events.jsonl", *batch(), {
            "event": "game", "phase": "selfplay", "index": 0, "opponent_name": "champion_4",
            "result": "loss", "completed": 1, "total": 1})
        status, _, body = self.request(url)
        self.assertEqual(status, 200)
        snapshot = json.loads(body)
        self.assertEqual(snapshot["current"]["match"]["losses"], 1)
        self.assertEqual(snapshot["current"]["match"]["win_rate"], 0)
        self.assertEqual(json.loads(self.request("/api/health")[2]), {"ok": True})

    def test_run_path_traversal_is_rejected_and_missing_runs_are_404(self):
        for run in ("../web", "..\\web", "E:\\outside", "/absolute", ".", "..", "x\0y"):
            with self.subTest(run=run):
                status, _, body = self.request("/api/snapshot?" + urlencode({"run": run}))
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))
        self.assertEqual(self.request("/api/snapshot")[0], 400)
        self.assertEqual(self.request("/api/snapshot?run=missing")[0], 404)

    def test_static_allowlist_and_post_cannot_read_private_files_or_mutate_training(self):
        before = {path.relative_to(self.run): path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        for path, mime in (("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css")):
            with self.subTest(path=path):
                status, headers, _ = self.request(path)
                self.assertEqual(status, 200)
                self.assertTrue(headers["Content-Type"].startswith(mime))
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        for path in ("/private.txt", "/../private.txt", "/%2e%2e/private.txt", "/api/stop"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)
        for path in ("/api/stop", "/api/pause", "/api/snapshot?" + urlencode({"run": self.run.name})):
            with self.subTest(post=path):
                self.assertEqual(self.request(path, "POST", '{"stop":true}')[0], 501)
        after = {path.relative_to(self.run): path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
