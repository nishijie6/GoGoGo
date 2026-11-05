"""Resource candidates preserve the frozen state and change one factor each."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from weiqi.katago_mixed import MixedBoardRunner
from weiqi.katago_mixed_experiments import prepare_mixed_experiment
from weiqi.replay_accounting import ReplayRowLedger


def records(root):
    return {path.relative_to(root).as_posix(): (hashlib.sha256(path.read_bytes()).hexdigest(),
                                               path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def differences(left, right, prefix=""):
    changed = set()
    for key in left.keys() | right.keys():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(left.get(key), dict) and isinstance(right.get(key), dict):
            changed.update(differences(left[key], right[key], path))
        elif left.get(key) != right.get(key):
            changed.add(path)
    return changed


class MixedBudgetProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.base = self.root / "sgf-experiment"
        self.output = self.root / "budget-candidates"
        template = Path(__file__).resolve().parents[1] / "config/katago_mixed.m4.json"
        self.config = json.loads(template.read_text())
        overrides = self.config["training"]["overrides"]
        overrides["runtime"]["output_directory"] = str(self.source)
        self.config_path = self.root / "source.json"
        self.config_path.write_text(json.dumps(self.config))
        MixedBoardRunner(self.config_path).ensure_manifest()
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.03, momentum=0.9)
        parameter.grad = torch.tensor([0.2, -0.2])
        optimizer.step()
        checkpoint = self.source / "train/gogogo/checkpoint.ckpt"
        checkpoint.parent.mkdir(parents=True)
        self.state = {"model": {"weight": parameter.detach().clone()},
                      "optimizer": optimizer.state_dict(), "config": {"test_model": True},
                      "train_state": {"global_step_samples": 512, "total_num_data_rows": 50000,
                                      "train_bucket_level_at_row": 50000}}
        torch.save(self.state, checkpoint)
        os.utime(checkpoint, ns=(200_000_000_000, 200_000_000_000))
        for name, rows, seconds in (("old", 30000, 100), ("new", 2, 300)):
            path = self.source / f"selfplay/{name}/tdata/{name}.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, globalInputNC=np.zeros((rows, 1), dtype=np.float32))
            os.utime(path, ns=(seconds * 10**9, seconds * 10**9))
        self.original = prepare_mixed_experiment(self.source, self.config_path, self.base, wall_seconds=180)

    def prepare(self):
        module = importlib.import_module("weiqi.katago_mixed_budget_profiles")
        return module.prepare_budget_candidates(self.base, self.output)

    def test_all_candidates_copy_identical_full_state_replay_and_counter_coordinates(self):
        before = records(self.base)

        report = self.prepare()

        self.assertEqual(records(self.base), before)
        self.assertEqual(set(report["profiles"]), {"control", "search64", "cycle96", "replay25000", "more19"})
        common = self.base / "common"
        for name, profile in report["profiles"].items():
            with self.subTest(profile=name):
                root = Path(profile["run_root"])
                self.assertTrue((root / "mixed_manifest.json").is_file())
                MixedBoardRunner(Path(profile["config_path"])).ensure_manifest()
                for relative in ("train/gogogo/checkpoint.ckpt", "selfplay/old/tdata/old.npz", "selfplay/new/tdata/new.npz"):
                    original, copied = common / relative, root / relative
                    self.assertEqual(copied.read_bytes(), original.read_bytes())
                    self.assertEqual(copied.stat().st_mtime_ns, original.stat().st_mtime_ns)
                    self.assertFalse(copied.is_symlink())
                    self.assertNotEqual(copied.stat().st_ino, original.stat().st_ino)
                loaded = torch.load(root / "train/gogogo/checkpoint.ckpt", map_location="cpu", weights_only=False)
                torch.testing.assert_close(loaded["optimizer"], self.state["optimizer"])
                self.assertEqual(loaded["train_state"], self.state["train_state"])
                ledger = ReplayRowLedger(root).sync()
                self.assertEqual((ledger.physical_rows, ledger.deleted_rows_offset, ledger.logical_rows),
                                 (30002, 20000, 50002))
        self.assertEqual(report["initial_counters"]["global_step_samples"], 512)
        self.assertEqual(report["initial_counters"]["logical_replay_rows"], 50002)
        saved = json.loads((self.output / "budget_candidates_manifest.json").read_text())
        self.assertEqual(saved, report)

    def test_configs_change_exactly_one_scientific_factor_plus_output(self):
        report = self.prepare()
        expected = {
            "control": {},
            "search64": {"training.overrides.search.simulations_per_move": 64},
            "cycle96": {"training.overrides.self_play.games_per_iteration": 96},
            "replay25000": {"training.overrides.optimizer.replay_buffer_capacity": 25000},
            "more19": {"board_weights": [2, 2, 3, 2, 2]},
        }
        runtime_path = "training.overrides.runtime.output_directory"
        for name, changes in expected.items():
            with self.subTest(profile=name):
                profile = report["profiles"][name]
                config = json.loads(Path(profile["config_path"]).read_text())
                self.assertEqual(differences(self.config, config), set(changes) | {runtime_path})
                self.assertEqual(config["training"]["overrides"]["runtime"]["output_directory"], profile["run_root"])
                self.assertEqual(config["training"]["overrides"]["search"]["fast_simulations_per_move"], 8)
                self.assertEqual({item["path"]: item["after"] for item in profile["mutations"]}, changes)
        replay = report["profiles"]["replay25000"]
        self.assertEqual(replay["initial_replay_policy"], "preserve_full_window_then_apply_capacity")
        self.assertEqual(ReplayRowLedger(Path(replay["run_root"])).sync().physical_rows, 30002)

    def test_identity_and_budget_instructions_are_recorded_without_running_candidates(self):
        report = self.prepare()
        identities = set()
        for profile in report["profiles"].values():
            execution = profile["execution"]
            self.assertEqual(execution["api"], "weiqi.katago_mixed_trials.run_budgeted_process")
            self.assertEqual(execution["wall_seconds"], 180)
            self.assertIn("mixed", execution["command"])
            self.assertIn(profile["config_path"], execution["command"])
            self.assertFalse(Path(execution["log_path"]).exists())
            hardware = execution["command"].index("--hardware")
            self.assertEqual(execution["command"][hardware + 1], "mps")
            self.assertEqual(len(profile["profile_identity"]), 64)
            identities.add(profile["profile_identity"])
        self.assertEqual(len(identities), 5)
        self.assertFalse(report["automatic_promotion"])
        self.assertEqual(report["budget"]["wall_seconds"], 180)
        self.assertEqual(report["source"]["files"]["train/gogogo/checkpoint.ckpt"]["sha256"],
                         self.original["source"]["checkpoint"]["sha256"])

    def test_cycle96_explicitly_bypasses_the_mps_iteration_cap_without_changing_other_profiles(self):
        report = self.prepare()
        for name, profile in report["profiles"].items():
            with self.subTest(profile=name):
                execution = profile["execution"]
                if name == "cycle96":
                    index = execution["command"].index("--games")
                    self.assertEqual(execution["command"][index + 1], "96")
                    self.assertEqual(execution["effective_overrides"], {"games": 96})
                else:
                    self.assertNotIn("--games", execution["command"])
                    self.assertEqual(execution["effective_overrides"], {})

    def test_active_sgf_arms_can_change_while_the_common_snapshot_remains_frozen(self):
        checkpoint = self.base / "arms/sgf_starts/run/train/gogogo/checkpoint.ckpt"
        checkpoint.write_bytes(b"active pilot now owns its arm checkpoint")

        report = self.prepare()

        self.assertEqual(len(report["profiles"]), 5)
        self.assertEqual(checkpoint.read_bytes(), b"active pilot now owns its arm checkpoint")

    def test_changed_common_files_or_missing_original_inventory_are_rejected_before_output(self):
        for relative in ("source_config.json", "train/gogogo/checkpoint.ckpt", "selfplay/old/tdata/old.npz"):
            with self.subTest(relative=relative):
                source = self.base / "common" / relative
                source.write_bytes(b"changed frozen artifact")
                with self.assertRaisesRegex(ValueError, "changed|snapshot|hash|inventory|artifact"):
                    self.prepare()
                self.assertFalse(self.output.exists())
                original = self.config_path if relative == "source_config.json" else self.source / relative
                shutil.copy2(original, source)
        manifest = deepcopy(self.original)
        del manifest["prepared_files"]["common/train/gogogo/checkpoint.ckpt"]
        (self.base / "experiment_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "inventory|manifest"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_overlapping_paths_are_rejected_without_modification(self):
        self.output.mkdir()
        marker = self.output / "unidentified.json"
        marker.write_text("preserve me")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(marker.read_text(), "preserve me")
        for output in (self.base, self.base / "candidates", self.root, self.source / "candidates"):
            self.output = output
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.prepare()

    def test_common_checkpoint_cannot_be_reidentified_by_changing_only_prepared_inventory(self):
        checkpoint = self.base / "common/train/gogogo/checkpoint.ckpt"
        checkpoint.write_bytes(b"different checkpoint")
        stat = checkpoint.stat()
        manifest = deepcopy(self.original)
        manifest["prepared_files"]["common/train/gogogo/checkpoint.ckpt"] = {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        (self.base / "experiment_manifest.json").write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "inventory|artifact"):
            self.prepare()

        self.assertFalse(self.output.exists())

    def test_new_replay_cannot_silently_change_the_shared_initial_window(self):
        path = self.base / "common/selfplay/new/tdata/extra.npz"
        np.savez_compressed(path, globalInputNC=np.zeros((3, 1), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "replay|inventory"):
            self.prepare()

        self.assertFalse(self.output.exists())

    def test_more19_rejects_a_different_valid_board_family_before_creating_output(self):
        config = deepcopy(self.config)
        config["board_sizes"] = [7, 9, 13, 19]
        config["board_weights"] = [2, 3, 2, 1]
        config["promotion"]["evaluation_boards"] = [9, 13, 19]
        self.config_path.write_text(json.dumps(config))
        runner = MixedBoardRunner(self.config_path)
        manifest_path = self.source / "mixed_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"] = runner.manifest_identity()
        manifest_path.write_text(json.dumps(manifest))
        self.base = self.root / "other-board-experiment"
        prepare_mixed_experiment(self.source, self.config_path, self.base)

        with self.assertRaisesRegex(ValueError, "default|five boards"):
            self.prepare()

        self.assertFalse(self.output.exists())

    def test_exploration_source_is_rejected_before_creating_candidate_output(self):
        starts = self.root / "external-starts"
        starts.mkdir()
        (starts / "opening.sgf").write_text("(;GM[1]SZ[7]KM[6.5];B[bb];W[dd])")
        config = deepcopy(self.config)
        config["exploration"] = {"start_positions_dir": str(starts), "start_probability": 0.1,
                                 "early_fork_probability": 0.0, "fork_probability": 0.0,
                                 "policy_init": False}
        # Model a legacy frozen experiment without asking today's narrower
        # preparation API to accept an exploration-bearing source.
        frozen_config = self.base / "common/source_config.json"
        frozen_config.write_text(json.dumps(config))
        stat = frozen_config.stat()
        record = {"sha256": hashlib.sha256(frozen_config.read_bytes()).hexdigest(),
                  "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        manifest = deepcopy(self.original)
        manifest["source"]["config"].update(record)
        manifest["prepared_files"]["common/source_config.json"] = record
        runner = MixedBoardRunner(frozen_config, run_root_override=self.base / "common")
        manifest["source"]["mixed_manifest"]["identity"] = runner.manifest_identity()
        (self.base / "experiment_manifest.json").write_text(json.dumps(manifest))

        with self.assertRaisesRegex(ValueError, "exploration"):
            self.prepare()

        self.assertFalse(self.output.exists())

    def test_source_change_during_copy_keeps_partial_output_unidentified(self):
        module = importlib.import_module("weiqi.katago_mixed_budget_profiles")
        copy = module._copy_checked
        changed = False

        def copy_then_change(source, target, expected):
            nonlocal changed
            copy(source, target, expected)
            if Path(source).name == "checkpoint.ckpt" and not changed:
                changed = True
                stat = Path(source).stat()
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        with patch.object(module, "_copy_checked", side_effect=copy_then_change):
            with self.assertRaisesRegex(ValueError, "changed|snapshot|artifact|time"):
                self.prepare()
        self.assertTrue(changed)
        self.assertFalse((self.output / "budget_candidates_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
