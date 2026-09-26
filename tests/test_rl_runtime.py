"""Checkpoint/resume and real multiprocessing inference lifecycle tests."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))
if AVAILABLE:
    import numpy as np
    import torch
    from weiqi.rl.control import RunLock, TrainingControl, TrainingStopped
    from weiqi.rl.network import PolicyValueNet, Runtime
    from weiqi.rl.runner import Trainer, train
    from weiqi.rl.selfplay import GameJob, run_games
    from weiqi.rl.state import Position
    from weiqi.rl.storage import ReplayBuffer, checkpoint_config, cpu_state, fingerprint, load_checkpoint
    from weiqi.rl.eval_positions import file_hash

from weiqi.rl_config import resolve_rl_training_config


@unittest.skipUnless(AVAILABLE, "Optional training dependencies PyTorch/NumPy are not installed")
class RuntimeTests(unittest.TestCase):
    def config(self):
        return resolve_rl_training_config(overrides={
            "hardware": {"device": "cpu", "precision": "float32", "data_loader_workers": 0},
            "network": {"channels": 8, "residual_blocks": 1},
            "search": {"simulations_per_move": 1, "dirichlet_epsilon": 0.0},
            "self_play": {"games_per_iteration": 2},
            "optimizer": {"batch_size": 4, "minimum_replay_size": 1,
                          "training_steps_per_iteration": 2, "use_board_symmetry_augmentation": False},
            "runtime": {"pause_while_game_is_active": False},
        })

    def test_saved_optimizer_replay_rng_and_weights_resume_identical_updates(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer = Trainer(self.config(), root)
            state = Position.new(9, 6.5)
            pi = state.legal.astype(np.float32)
            pi /= pi.sum()
            trainer.replay.extend([(state.features(), pi.copy(), float(index % 2 * 2 - 1)) for index in range(16)])
            result = trainer.train_updates()
            self.assertTrue(result["weights_updated"])
            trainer.save(archive=False)
            payload, config = load_checkpoint(root / "latest.pt")
            saved = fingerprint(payload["model"])
            self.assertEqual(payload["training_steps"], 2)
            self.assertEqual(len(payload["replay"]["values"]), 16)
            trainer.train_updates()
            expected = fingerprint(cpu_state(trainer.model))
            restored = Trainer(config, root, payload)
            self.assertEqual(fingerprint(cpu_state(restored.model)), saved)
            restored.train_updates()
            self.assertEqual(restored.training_steps, 4)
            self.assertEqual(fingerprint(cpu_state(restored.model)), expected)

    def test_two_spawned_actors_return_actual_finished_games(self):
        config = self.config()
        def predict(batch):
            policy = np.full((len(batch), 82), -100.0, dtype=np.float32)
            policy[:, -1] = 100
            return policy, np.zeros(len(batch))
        games = run_games(config, [GameJob(0, 7), GameJob(1, 8)], {0: predict}, training=True)
        self.assertEqual([game.index for game in games], [0, 1])
        self.assertTrue(all(game.reason == "two_passes" and len(game.examples) == 2 for game in games))

    def test_evaluator_failure_reaps_all_owned_processes(self):
        import multiprocessing as mp
        before = {child.pid for child in mp.active_children()}
        def fail(batch):
            raise ValueError("deliberate evaluator failure")
        with self.assertRaisesRegex(ValueError, "deliberate"):
            run_games(self.config(), [GameJob(0, 2), GameJob(1, 3)], {0: fail}, training=True)
        self.assertEqual({child.pid for child in mp.active_children()}, before)

    def test_cpu_forward_dimensions_and_rejected_cuda_request(self):
        config = self.config()
        model = PolicyValueNet(config)
        p, v = model(torch.zeros((2, 20, 9, 9)))
        self.assertEqual(tuple(p.shape), (2, 82))
        self.assertEqual(tuple(v.shape), (2,))
        from unittest.mock import patch
        config = replace(config, hardware=replace(config.hardware, device="cuda:0"))
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                Runtime(config)

    def test_model_export_cannot_be_used_as_a_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer = Trainer(self.config(), root)
            trainer.save(archive=False)
            payload, config = load_checkpoint(root / "best.pt")
            with self.assertRaisesRegex(ValueError, "full checkpoint"):
                Trainer(config, root, payload)

    def test_stop_file_interrupts_before_more_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "STOP").touch()
            with self.assertRaises(TrainingStopped):
                TrainingControl(root, False, lambda _: None)()

    def test_output_lock_prevents_a_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunLock(Path(directory)):
                with self.assertRaisesRegex(ValueError, "Another trainer"):
                    with RunLock(Path(directory)):
                        pass

    def test_resume_refuses_to_overwrite_another_existing_run(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run_a, run_b = Path(directory) / "a", Path(directory) / "b"
            Trainer(self.config(), run_a).save(archive=False)
            Trainer(self.config(), run_b).save(archive=False)
            source = run_a / "latest.pt"
            target = run_b / "latest.pt"
            before = file_hash(target)
            payload, config = load_checkpoint(source)
            with self.assertRaisesRegex(ValueError, "Occupied output"):
                train(config, run_b, 1, payload, resume_path=source,
                      resume_checksum=file_hash(source))
            self.assertEqual(file_hash(target), before)
            with self.assertRaisesRegex(ValueError, "Occupied output"):
                train(config, run_a, 1, payload, resume_path=source,
                      resume_checksum="wrong digest")
            with patch("weiqi.rl.runner.Trainer.run_iteration", return_value={}):
                resumed = train(config, run_a, 1, payload, resume_path=source,
                                resume_checksum=file_hash(source))
            self.assertEqual(resumed.iteration, payload["iteration"])
            self.assertEqual(file_hash(source), file_hash(run_a / "latest.pt"))

    def test_resume_only_restores_scheduled_milestone_anchors(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = self.config()
            trainer = Trainer(config, root)
            with torch.no_grad():
                next(trainer.model.parameters()).add_(0.25)
            trainer.iteration = 2
            trainer.save(archive=False)
            payload, _ = load_checkpoint(root / "latest.pt")
            resumed = Trainer(config, root, payload)
            self.assertFalse((root / "anchors" / "milestone_000002.pt").exists())
            resumed.iteration = 5
            resumed.save(archive=False)
            payload, _ = load_checkpoint(root / "latest.pt")
            Trainer(config, root, payload)
            self.assertTrue((root / "anchors" / "milestone_000005.pt").is_file())

    def test_resume_policy_seeds_previous_candidate_from_archived_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = self.config()
            config = replace(config, self_play=replace(config.self_play, games_per_iteration=16))
            trainer = Trainer(config, root)
            with torch.no_grad():
                next(trainer.model.parameters()).add_(0.25)
            trainer.iteration = 3
            trainer.save()
            previous_sha = fingerprint(cpu_state(trainer.model))
            with torch.no_grad():
                next(trainer.model.parameters()).add_(0.25)
            trainer.iteration = 4
            trainer.save()
            payload, loaded = load_checkpoint(root / "latest.pt")
            resumed = Trainer(loaded, root, payload)
            anchor, _ = load_checkpoint(root / "anchors" / "milestone_000003.pt")
            self.assertEqual(fingerprint(anchor["model"]), previous_sha)
            jobs, _, _ = resumed.selfplay_jobs(5)
            self.assertTrue(any(job.opponent_name == "milestone_000003" for job in jobs))

    def test_legacy_checkpoint_keeps_its_original_opponent_and_promotion_policy(self):
        from weiqi.rl.state import FEATURE_VERSION
        from weiqi.rl.storage import CHECKPOINT_VERSION

        raw = self.config().to_dict()
        raw["self_play"]["champion_fraction"] = 0.5
        raw["self_play"].pop("milestone_fraction")
        raw["evaluation"]["games"] = 20
        raw["evaluation"].pop("promotion_test")
        raw["evaluation"].pop("confirmation_max_game_length_factor")
        payload = {"checkpoint_version": CHECKPOINT_VERSION,
                   "feature_version": FEATURE_VERSION, "config": raw}
        restored = checkpoint_config(payload)
        self.assertEqual((restored.self_play.champion_fraction,
                          restored.self_play.milestone_fraction), (0.5, 0.0))
        self.assertEqual((restored.evaluation.games,
                          restored.evaluation.promotion_test,
                          restored.evaluation.confirmation_max_game_length_factor),
                         (20, "paired_hoeffding", raw["self_play"]["max_game_length_factor"]))
        self.assertNotIn("milestone_fraction", raw["self_play"])

    def test_pause_waits_and_reports_resume(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "PAUSE"
            marker.touch()
            events = []
            with patch("weiqi.rl.control.time.sleep", side_effect=lambda _: marker.unlink()):
                TrainingControl(root, False, events.append)()
            self.assertEqual([event["event"] for event in events], ["paused", "resumed"])

    def test_stop_is_honored_while_paused(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "PAUSE").touch()
            with patch("weiqi.rl.control.time.sleep", side_effect=lambda _: (root / "STOP").touch()):
                with self.assertRaises(TrainingStopped):
                    TrainingControl(root, False, lambda _: None)()

    def test_malformed_replay_value_shape_and_unknown_schema_are_rejected(self):
        replay = ReplayBuffer(100, 9)
        state = replay.state()
        state["values"] = torch.empty((0, 1))
        with self.assertRaisesRegex(ValueError, "shapes"):
            replay.restore(state)
        config = self.config().to_dict()
        config["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "configuration version"):
            checkpoint_config({"checkpoint_version": 1, "feature_version": 1, "config": config})
