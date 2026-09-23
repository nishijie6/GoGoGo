"""Game heartbeats work across processes and expire after an unclean exit."""

import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

from weiqi.rl_activity import GameActivity, HEARTBEAT_TTL, game_is_active


class GameActivityTests(unittest.TestCase):
    def test_multiple_games_expiry_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = GameActivity(root), GameActivity(root)
            self.assertFalse(game_is_active(root))
            first.update(True)
            second.update(True)
            self.assertTrue(game_is_active(root))
            first.close()
            self.assertTrue(game_is_active(root))
            old = time.time() - HEARTBEAT_TTL - 1
            os.utime(second.path, (old, old))
            self.assertFalse(game_is_active(root))
            second.update(True)
            self.assertTrue(game_is_active(root))
            second.update(False)
            self.assertFalse(game_is_active(root))

    def test_gui_heartbeat_reflects_finished_and_reasoning_games(self):
        try:
            from weiqi.gui import GoApp
        except ImportError:
            self.skipTest("Tkinter is not installed in this training-only environment")
        from weiqi.engine import GoGame
        app = GoApp.__new__(GoApp)
        app.root = Mock()
        app._closing = False
        app._reasoning_session = None
        app._training_activity = Mock()
        app.game = GoGame(9)
        app._update_training_activity()
        app._training_activity.update.assert_called_with(True)
        app.game.pass_turn()
        app.game.pass_turn()
        app._update_training_activity()
        app._training_activity.update.assert_called_with(False)
        app._reasoning_session = Mock()
        app._update_training_activity()
        app._training_activity.update.assert_called_with(True)
