"""Training telemetry stays dependency-free and truthful across run lifecycles."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from weiqi.rl.control import Progress, TrainingControl, TrainingStopped


class ProgressTests(unittest.TestCase):
    def test_session_heartbeat_and_successful_completion_are_persisted(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with Progress(root) as progress:
                progress.phase = "selfplay"
                progress({"event": "iteration_started", "iteration": 7})
                live = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
                self.assertEqual((live["status"], live["phase"], live["iteration"]),
                                 ("running", "selfplay", 7))
            final = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(final["status"], "completed")
            self.assertEqual([event["event"] for event in events],
                             ["run_started", "iteration_started", "run_completed"])
            self.assertEqual({event["session_id"] for event in events}, {final["session_id"]})
            self.assertEqual(list(root.glob("monitor.*.tmp")), [])

    def test_stop_and_failure_preserve_original_exception_and_terminal_status(self):
        for error, status in ((TrainingStopped("STOP file detected"), "stopped"),
                              (KeyboardInterrupt(), "stopped"),
                              (ValueError("broken evaluator"), "failed")):
            with self.subTest(status=status, error=type(error).__name__), \
                    tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                root = Path(directory)
                with self.assertRaises(type(error)) as caught:
                    with Progress(root, operation="evaluate"):
                        raise error
                self.assertIs(caught.exception, error)
                final = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
                self.assertEqual(final["status"], status)
                self.assertEqual(final["operation"], "evaluate")
                self.assertEqual(final["last_event"], "run_" + status)

    def test_resume_uses_a_distinct_session_without_erasing_event_history(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with Progress(root) as first:
                pass
            with Progress(root) as second:
                second({"event": "ready", "resumed": True, "iteration": 3})
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertNotEqual(first.session_id, second.session_id)
            self.assertEqual(len(events), 5)
            self.assertEqual(events[-1]["session_id"], second.session_id)

    def test_pause_keeps_refreshing_heartbeat_until_resume(self):
        now, observed = [0.0], []
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            pause = root / "PAUSE"
            pause.touch()

            def advance(_seconds):
                observed.append(json.loads((root / "monitor.json").read_text(encoding="utf-8")))
                now[0] += 2.5
                if len(observed) == 2:
                    pause.unlink()

            with patch("weiqi.rl.control.time.monotonic", side_effect=lambda: now[0]), \
                    patch("weiqi.rl.control.time.time", side_effect=lambda: now[0]), \
                    patch("weiqi.rl.control.time.sleep", side_effect=advance):
                with Progress(root) as progress:
                    TrainingControl(root, False, progress)()
                    live = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
            self.assertEqual([row["status"] for row in observed], ["paused", "paused"])
            self.assertEqual([row["updated_at_epoch"] for row in observed], [0.0, 2.5])
            self.assertEqual(live["status"], "running")
            self.assertNotIn("reason", live)

    def test_sidecar_write_failure_does_not_break_training_event_log(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with patch("weiqi.rl.control.os.replace", side_effect=PermissionError("busy reader")):
                with Progress(root):
                    pass
            self.assertEqual(len((root / "events.jsonl").read_text().splitlines()), 2)
            self.assertEqual(list(root.glob("monitor.*.tmp")), [])


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Optional NumPy is absent")
class GameOutcomeTests(unittest.TestCase):
    def test_truncation_and_selfplay_cannot_be_candidate_wins(self):
        from weiqi.engine import BLACK, WHITE
        from weiqi.rl.selfplay import game_outcome

        game = SimpleNamespace(reason="length_limit", winner=BLACK, candidate_color=BLACK)
        self.assertEqual(game_outcome(game, training=False, opponent_name="champion"), "truncated")
        game.reason = "two_passes"
        self.assertEqual(game_outcome(game, training=True, opponent_name="candidate_self"), "black_win")
        self.assertEqual(game_outcome(game, training=True, opponent_name="champion"), "win")
        game.candidate_color = WHITE
        self.assertEqual(game_outcome(game, training=False, opponent_name="champion"), "loss")
        game.winner = None
        self.assertEqual(game_outcome(game, training=False, opponent_name="champion"), "draw")


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "torch")),
                     "Optional training dependencies are absent")
class RunnerLifecycleTests(unittest.TestCase):
    def test_training_constructor_failure_is_recorded(self):
        from weiqi.rl.runner import train

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with patch("weiqi.rl.runner.Trainer", side_effect=RuntimeError("device unavailable")):
                with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                    train(None, root, 1)
            final = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "failed")
            self.assertIn("device unavailable", final["error"])

    def test_evaluation_and_benchmark_runtime_failure_is_recorded(self):
        from weiqi.rl.runner import benchmark, evaluate

        config = SimpleNamespace(runtime=SimpleNamespace(pause_while_game_is_active=False))
        for operation in ("evaluate", "benchmark"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory, \
                    redirect_stdout(io.StringIO()):
                root = Path(directory)
                with patch("weiqi.rl.runner.load_checkpoint", return_value=({}, config)), \
                        patch("weiqi.rl.runner.Runtime", side_effect=RuntimeError("device unavailable")):
                    with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                        if operation == "evaluate":
                            evaluate(Path("unused.pt"), None, root)
                        else:
                            benchmark(Path("unused.pt"), root)
                final = json.loads((root / "monitor.json").read_text(encoding="utf-8"))
                self.assertEqual(final["status"], "failed")
                self.assertEqual(final["operation"], operation)


if __name__ == "__main__":
    unittest.main()
