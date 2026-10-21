"""Exercise local-model selection through the real desktop game lifecycle."""

from functools import partial
from concurrent.futures import CancelledError
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from test_native_opponents import AVAILABLE, write_model_run
from weiqi import rl_opponents
from weiqi.gui import GoApp


@unittest.skipUnless(AVAILABLE, "Optional training dependencies are not installed")
class NativeOpponentGUITests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(rl_opponents, "load_native_training_opponents"))
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"No Tk display service: {error}")
        self.root.withdraw()
        self.addCleanup(self.close_root)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name)
        self.run = write_model_run(self.project)
        loader = partial(rl_opponents.load_native_training_opponents, project_root=self.project)
        patcher = patch("weiqi.gui.load_native_training_opponents", loader)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = GoApp(self.root)
        self.addCleanup(self.app.close)

    def close_root(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def select_latest(self):
        choices = rl_opponents.load_native_training_opponents(9, project_root=self.project)
        label = next(item.label for item in choices.values() if item.role == "latest")
        self.app.difficulty_var.set(label)
        self.app.new_game()
        return label

    def finish_ai_turn(self):
        deadline = time.monotonic() + 5
        while self.app.ai_busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.app.ai_busy, "The AI turn did not complete")

    def test_difficulty_list_removes_humansl_and_native_model_replies(self):
        values = self.app.difficulty_combo.cget("values")
        self.assertFalse(any("HumanSL" in label for label in values))
        self.assertIn("本地训练·最新模型", values)
        self.assertIn("本地训练·最佳模型", values)
        label = self.select_latest()
        self.assertEqual(self.app.active_difficulty, label)
        self.assertIsNone(self.app.katago_engine)
        self.assertTrue(self.app.game.play(4, 4).legal)
        self.app._start_ai_turn()
        self.finish_ai_turn()
        self.assertEqual(self.app.game.move_number, 2)
        self.assertEqual((self.app.game.moves[-1].row, self.app.game.moves[-1].col), (0, 0))
        self.assertIn("本地模型", self.app._winrate_source)

    def test_changing_board_removes_incompatible_model_without_changing_game(self):
        self.select_latest()
        self.app.size_var.set("13×13")
        self.app._refresh_training_opponents()
        self.assertEqual(self.app.difficulty_var.get(), "中等")
        self.assertFalse(any(label.startswith("本地训练·")
                             for label in self.app.difficulty_combo.cget("values")))
        self.assertEqual(self.app.game.size, 9)

    def test_failed_model_load_preserves_the_formal_game(self):
        self.assertTrue(self.app.game.play(4, 4).legal)
        before = (self.app.game.board_hash(), tuple(self.app.game.moves))
        (self.run / "candidate.pt").write_bytes(b"invalid checkpoint")
        with patch("weiqi.gui.messagebox.showerror"):
            self.select_latest()
        self.assertEqual((self.app.game.board_hash(), tuple(self.app.game.moves)), before)
        self.assertIn("本地模型", self.app.notice_var.get())

    def test_new_game_cancels_old_inference_and_rejects_its_result(self):
        self.select_latest()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        predict = self.app.ai.predict
        def slow_prediction(batch):
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Test did not release inference")
            return predict(batch)
        self.app.ai.predict = slow_prediction
        self.app.game.play(4, 4)
        self.app._start_ai_turn()
        deadline = time.monotonic() + 5
        while not entered.is_set() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(entered.is_set())
        future, cancellation = self.app.ai_future, self.app._ai_cancel_event
        self.app.difficulty_var.set("中等")
        self.app.new_game()
        release.set()
        deadline = time.monotonic() + 5
        while not future.done() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(cancellation.is_set())
        self.assertTrue(future.done())
        self.assertIsInstance(future.exception(), CancelledError)
        self.assertEqual(self.app.game.move_number, 0)
        self.assertFalse(self.app.ai_busy)

    def test_failed_inference_preserves_board_without_passing(self):
        self.select_latest()
        self.app.game.play(4, 4)
        before = (self.app.game.board_hash(), tuple(self.app.game.moves), self.app.game.current_player)
        with patch.object(self.app.ai, "predict", side_effect=ValueError("inference failed")):
            with patch("weiqi.gui.messagebox.showerror"):
                self.app._start_ai_turn()
                self.finish_ai_turn()
        self.assertEqual((self.app.game.board_hash(), tuple(self.app.game.moves),
                          self.app.game.current_player), before)
        self.assertIn("本地模型计算失败", self.app.notice_var.get())


if __name__ == "__main__":
    unittest.main()
