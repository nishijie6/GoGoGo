"""Local checkpoints must become usable, board-matched desktop opponents."""

from concurrent.futures import CancelledError
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from weiqi import rl_opponents
from weiqi.engine import GoGame
from weiqi.rl_config import resolve_rl_training_config


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.network import PolicyValueNet
    from weiqi.rl.state import Position
    from weiqi.rl.storage import atomic_torch_save, cpu_state


def write_model_run(root, name="current", *, board_size=9):
    """Make real, deterministic exports with different candidate/best moves."""
    run = root / "training_runs" / name
    run.mkdir(parents=True, exist_ok=True)
    config = resolve_rl_training_config(overrides={
        "game": {"board_size": board_size},
        "hardware": {"device": "cpu", "precision": "float32"},
        "network": {"channels": 8, "residual_blocks": 1},
        "evaluation": {"simulations_per_move": 1},
        "runtime": {"pause_while_game_is_active": False},
    })
    (run / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    base = {"checkpoint_version": 1, "feature_version": 1, "config": config.to_dict(),
            "iteration": 12, "best_iteration": 8}
    states = {}
    for role, action in (("candidate", 0), ("best", 4 * board_size + 4)):
        model = PolicyValueNet(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy[-1].bias[action] = 100
        states[role] = cpu_state(model)
        atomic_torch_save({**base, "kind": "model", "model_role": role,
                           "model": states[role]}, run / f"{role}.pt")
    atomic_torch_save({**base, "kind": "training", "model": states["candidate"],
                       "best_model": states["best"]}, run / "latest.pt")
    return run


class NativeOpponentDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_directory(self, name, *, board=9, modified=1):
        run = self.root / "training_runs" / name
        run.mkdir(parents=True)
        (run / "config.resolved.json").write_text(json.dumps({"game": {"board_size": board}}))
        latest = run / "latest.pt"
        latest.touch()
        os.utime(latest, (modified, modified))
        return run

    def choices(self, board=9):
        self.assertTrue(hasattr(rl_opponents, "load_native_training_opponents"),
                        "The desktop needs a native-checkpoint opponent loader")
        return rl_opponents.load_native_training_opponents(board, project_root=self.root)

    def test_uses_newest_matching_run_including_imported_run(self):
        self.run_directory("old", modified=1)
        newest = self.run_directory("imported/run", modified=2)
        self.run_directory("other_board", board=13, modified=3)
        choices = self.choices()
        self.assertEqual({item.run_directory for item in choices.values()}, {newest})
        self.assertEqual({item.role for item in choices.values()}, {"latest", "best"})
        self.assertEqual({item.board_size for item in self.choices(13).values()}, {13})
        self.assertEqual(self.choices(19), {})

    def test_prefers_exports_and_falls_back_to_full_checkpoint(self):
        run = self.run_directory("current")
        candidate = run / "candidate.pt"
        candidate.touch()
        choices = {item.role: item for item in self.choices().values()}
        self.assertEqual(choices["latest"].checkpoint_path, candidate)
        self.assertEqual(choices["best"].checkpoint_path, run / "latest.pt")
        (run / "best.pt").touch()
        choices = {item.role: item for item in self.choices().values()}
        self.assertEqual(choices["best"].checkpoint_path, run / "best.pt")

    def test_refresh_finds_new_run_and_ignores_incomplete_metadata(self):
        old = self.run_directory("old", modified=1)
        self.assertEqual({item.run_directory for item in self.choices().values()}, {old})
        malformed = self.run_directory("incomplete", modified=4)
        (malformed / "config.resolved.json").write_text("{")
        newest = self.run_directory("new", modified=3)
        self.assertEqual({item.run_directory for item in self.choices().values()}, {newest})

    def test_gui_import_does_not_require_training_dependencies(self):
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run([
            sys.executable, "-S", "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import weiqi.gui; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
            str(project_root),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(AVAILABLE, "Optional training dependencies are not installed")
class NativeAITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = write_model_run(self.root)

    def ai(self, role="latest"):
        self.assertIsNotNone(importlib.util.find_spec("weiqi.native_ai"),
                             "Native checkpoints need a playable AI adapter")
        module = importlib.import_module("weiqi.native_ai")
        choices = rl_opponents.load_native_training_opponents(9, project_root=self.root)
        opponent = next(item for item in choices.values() if item.role == role)
        return module.NativeTrainingAI(opponent, simulations=1, device="cpu")

    def test_latest_and_best_load_distinct_weights_and_do_not_mutate_game(self):
        game = GoGame(9)
        before = (game.board_hash(), game.current_player, tuple(game.moves))
        for role, expected in (("latest", (0, 0)), ("best", (4, 4))):
            with self.subTest(role=role):
                decision = self.ai(role).choose_move(game)
                self.assertEqual(decision.point, expected)
                self.assertEqual(decision.analysis_visits, 1)
                self.assertEqual(decision.black_win_probability, 0.5)
        self.assertEqual((game.board_hash(), game.current_player, tuple(game.moves)), before)

    def test_full_checkpoint_fallback_uses_best_weights_and_refreshes_exports(self):
        (self.run / "candidate.pt").unlink()
        (self.run / "best.pt").unlink()
        self.assertEqual(self.ai("best").choose_move(GoGame(9)).point, (4, 4))
        self.assertEqual(self.ai("latest").choose_move(GoGame(9)).point, (0, 0))
        write_model_run(self.root)
        payload = torch.load(self.run / "best.pt", weights_only=True)
        atomic_torch_save(payload, self.run / "candidate.pt")
        self.assertEqual(self.ai("latest").choose_move(GoGame(9)).point, (4, 4))

    def test_history_matches_training_features_after_pass_and_undo(self):
        adapter = self.ai()
        module = importlib.import_module("weiqi.native_ai")
        game = GoGame(9)
        expected = Position.new(9, 6.5)
        for action in (0, 10, 81, 20, 30, 40, 50, 60, 70):
            if action == 81:
                game.pass_turn()
            else:
                self.assertTrue(game.play(*divmod(action, 9)).legal)
            expected = expected.play(action)
        self.assertTrue(game.play(8, 8).legal)
        game.undo()
        actual = module.position_from_game(game)
        np.testing.assert_array_equal(actual.features(), expected.features())
        self.assertEqual(actual.game._position_history, game._position_history)
        decision = adapter.choose_move(game)
        self.assertTrue(decision.point is None or game.analyze_move(*decision.point).legal)

    def test_rejects_board_mismatch_and_damaged_checkpoint(self):
        adapter = self.ai()
        with self.assertRaisesRegex(ValueError, "9.*9"):
            adapter.choose_move(GoGame(13))
        (self.run / "candidate.pt").write_bytes(b"damaged checkpoint")
        with self.assertRaisesRegex(ValueError, "模型|检查点"):
            self.ai()

    def test_inference_ignores_training_only_configuration(self):
        payload = torch.load(self.run / "candidate.pt", weights_only=True)
        config = payload["config"]
        config["self_play"]["future_opening_policy"] = {"fraction": 0.25}
        config["hardware"]["device"] = "another-computer-accelerator"
        config["optimizer"]["future_optimizer"] = "training only"
        config["evaluation"]["future_tournament"] = True
        config["runtime"]["future_scheduler"] = True
        atomic_torch_save(payload, self.run / "candidate.pt")
        try:
            decision = self.ai().choose_move(GoGame(9))
        except ValueError as error:
            self.fail(f"Inference must not validate training-only configuration: {error}")
        self.assertEqual(decision.point, (0, 0))

    def test_cancels_before_and_during_search(self):
        adapter = self.ai()
        event = threading.Event()
        event.set()
        with self.assertRaises(CancelledError):
            adapter.choose_move(GoGame(9), event)
        event.clear()
        predict = adapter.predict
        def cancel_after_prediction(batch):
            result = predict(batch)
            event.set()
            return result
        adapter.predict = cancel_after_prediction
        with self.assertRaises(CancelledError):
            adapter.choose_move(GoGame(9), event)


if __name__ == "__main__":
    unittest.main()
