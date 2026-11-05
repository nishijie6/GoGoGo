"""Fixed-budget experiments preserve full learner state and retained replay."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from weiqi.katago_mixed import MixedBoardRunner
from weiqi.replay_accounting import ReplayRowLedger


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _hash(path)
            for path in root.rglob("*") if path.is_file()}


class MixedExperimentPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "experiment"
        self.config_path = self.root / "mixed.json"
        self.config = {
            "schema_version": 1,
            "board_sizes": [7, 8, 9, 13, 19],
            "board_weights": [2, 2, 3, 2, 1],
            "shuffle_rows_per_file": 256,
            "training": {"schema_version": 1, "preset": "balanced", "overrides": {
                "game": {"board_size": 19, "komi": 6.5},
                "network": {"channels": 128, "residual_blocks": 10},
                "hardware": {"device": "cpu", "precision": "float32"},
                "runtime": {"output_directory": str(self.source), "random_seed": 17},
            }},
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        MixedBoardRunner(self.config_path).ensure_manifest()
        self.checkpoint = self.source / "train/gogogo/checkpoint.ckpt"
        self.checkpoint.parent.mkdir(parents=True)
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.03, momentum=0.9)
        parameter.grad = torch.tensor([0.25, -0.25])
        optimizer.step()
        self.state = {
            "model": {"weight": parameter.detach().clone()},
            "optimizer": optimizer.state_dict(),
            "train_state": {"global_step_samples": 512, "total_num_data_rows": 1000,
                            "train_bucket_level_at_row": 1000},
            "config": {"test_model": True},
        }
        torch.save(self.state, self.checkpoint)
        os.utime(self.checkpoint, ns=(200_000_000_000, 200_000_000_000))
        for name, rows, seconds in (("old", 3, 100), ("new", 2, 300)):
            path = self.source / f"selfplay/{name}/tdata/{name}.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, globalInputNC=np.arange(rows, dtype=np.float32)[:, None])
            os.utime(path, ns=(seconds * 10**9, seconds * 10**9))
        # Export-only checkpoints and the source ledger must never seed an arm.
        exported = self.source / "torchmodels_toexport/gogogo-s512-d1000/model.ckpt"
        exported.parent.mkdir(parents=True)
        torch.save({"model": {"weight": torch.zeros(2)}}, exported)
        (self.source / "replay_rows.json").write_text('{"source_only": true}', encoding="utf-8")

    def prepare(self, **kwargs) -> dict:
        self.assertIsNotNone(importlib.util.find_spec("weiqi.katago_mixed_experiments"),
                             "mixed experiment preparation module is missing")
        module = importlib.import_module("weiqi.katago_mixed_experiments")
        return module.prepare_mixed_experiment(
            self.source, self.config_path, self.output, **kwargs)

    def enable_source_exploration(self) -> None:
        starts = self.source / "original-training-starts"
        starts.mkdir()
        (starts / "position.sgf").write_text(
            "(;GM[1]FF[4]SZ[7]KM[6.5]PL[B]AB[bb]AW[dd])\n", encoding="utf-8")
        self.config["exploration"] = {
            "start_positions_dir": str(starts), "start_probability": .2,
            "early_fork_probability": .1, "fork_probability": .1, "policy_init": True,
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        path = self.source / "mixed_manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["identity"] = MixedBoardRunner(self.config_path).manifest_identity()
        path.write_text(json.dumps(document), encoding="utf-8")

    def test_active_source_exploration_is_rejected_before_any_experiment_write(self) -> None:
        self.enable_source_exploration()
        before = _hashes(self.source)
        config_before = self.config_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "baseline|exploration"):
            self.prepare()

        self.assertFalse(self.output.exists())
        self.assertEqual(_hashes(self.source), before)
        self.assertEqual(self.config_path.read_bytes(), config_before)

    def test_frozen_verification_rejects_legacy_source_exploration(self) -> None:
        manifest = self.prepare()
        self.enable_source_exploration()
        frozen_config = self.output / "common/source_config.json"
        shutil.copy2(self.config_path, frozen_config)
        config_record = {"sha256": _hash(frozen_config), "size": frozen_config.stat().st_size,
                         "mtime_ns": frozen_config.stat().st_mtime_ns}
        manifest["source"]["config"].update(config_record)
        source_manifest = self.source / "mixed_manifest.json"
        manifest["source"]["mixed_manifest"] = {
            "identity": json.loads(source_manifest.read_text(encoding="utf-8"))["identity"],
            "sha256": _hash(source_manifest), "size": source_manifest.stat().st_size,
            "mtime_ns": source_manifest.stat().st_mtime_ns,
        }
        manifest["prepared_files"]["common/source_config.json"] = config_record
        Path(manifest["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
        before = _hashes(self.output)
        verify = importlib.import_module("weiqi.katago_mixed_experiments").verify_prepared_experiment

        with self.assertRaisesRegex(ValueError, "baseline|exploration"):
            verify(self.output)

        self.assertEqual(_hashes(self.output), before)

    def test_arms_start_from_identical_full_state_and_retained_replay_without_source_writes(self) -> None:
        before = _hashes(self.source)
        config_before = self.config_path.read_bytes()

        manifest = self.prepare(wall_seconds=180)

        self.assertEqual(_hashes(self.source), before)
        self.assertEqual(self.config_path.read_bytes(), config_before)
        self.assertEqual(manifest["budget"]["wall_seconds"], 180)
        self.assertEqual(manifest["source"]["checkpoint"]["sha256"], _hash(self.checkpoint))
        counters = manifest["initial_counters"]
        self.assertEqual(counters["global_step_samples"], 512)
        self.assertEqual(counters["total_num_data_rows"], 1000)
        self.assertEqual(counters["retained_replay_rows"], 5)
        self.assertEqual(counters["deleted_rows_offset"], 997)
        self.assertEqual(counters["logical_replay_rows"], 1002)
        for arm in manifest["arms"].values():
            root = Path(arm["run_root"])
            checkpoint = root / "train/gogogo/checkpoint.ckpt"
            self.assertEqual(_hash(checkpoint), _hash(self.checkpoint))
            self.assertFalse(checkpoint.is_symlink())
            self.assertEqual(checkpoint.stat().st_mtime_ns, self.checkpoint.stat().st_mtime_ns)
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            torch.testing.assert_close(loaded["model"], self.state["model"])
            torch.testing.assert_close(loaded["optimizer"], self.state["optimizer"])
            self.assertEqual(loaded["train_state"], self.state["train_state"])
            for source in self.source.glob("selfplay/**/tdata/*.npz"):
                copied = root / source.relative_to(self.source)
                self.assertEqual(_hash(copied), _hash(source))
                self.assertEqual(copied.stat().st_mtime_ns, source.stat().st_mtime_ns)
                self.assertFalse(copied.is_symlink())
            ledger = ReplayRowLedger(root).sync()
            self.assertEqual((ledger.deleted_rows_offset, ledger.logical_rows), (997, 1002))
            self.assertNotEqual((root / "mixed_manifest.json").read_bytes(),
                                (self.source / "mixed_manifest.json").read_bytes())
            MixedBoardRunner(Path(arm["config_path"])).ensure_manifest()
        snapshot = Path(manifest["snapshot"]["run_root"])
        self.assertEqual(_hash(snapshot / "train/gogogo/checkpoint.ckpt"), _hash(self.checkpoint))

    def test_arm_configs_change_only_output_and_training_exploration(self) -> None:
        manifest = self.prepare()
        control = json.loads(Path(manifest["arms"]["control"]["config_path"]).read_text(encoding="utf-8"))
        variant = json.loads(Path(manifest["arms"]["sgf_starts"]["config_path"]).read_text(encoding="utf-8"))
        for name, config in (("control", control), ("sgf_starts", variant)):
            run_root = manifest["arms"][name]["run_root"]
            self.assertEqual(config["training"]["overrides"]["runtime"]["output_directory"], run_root)
            normalized = deepcopy(config)
            normalized.pop("exploration", None)
            normalized["training"]["overrides"]["runtime"]["output_directory"] = str(self.source)
            self.assertEqual(normalized, self.config)
        self.assertNotIn("exploration", control)
        self.assertEqual(variant["exploration"], {
            "start_positions_dir": manifest["training_starts"]["directory"],
            "start_probability": 0.1, "early_fork_probability": 0.0,
            "fork_probability": 0.0, "policy_init": False,
        })
        self.assertNotEqual(manifest["arms"]["control"]["run_root"],
                            manifest["arms"]["sgf_starts"]["run_root"])

    def test_export_checkpoint_without_optimizer_is_rejected_before_output_creation(self) -> None:
        del self.state["optimizer"]
        torch.save(self.state, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "optimizer|优化器|完整"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_repeat_validates_existing_preparation_and_preserves_results(self) -> None:
        first = self.prepare()
        result = self.output / "arms/control/results.json"
        result.write_text('{"completed": true}', encoding="utf-8")
        before = _hashes(self.output)

        second = self.prepare()

        self.assertEqual(second, first)
        self.assertEqual(_hashes(self.output), before)
        with self.assertRaisesRegex(ValueError, "budget|预算|不兼容|不一致"):
            self.prepare(wall_seconds=181)
        self.assertEqual(_hashes(self.output), before)

    def test_training_starts_are_unique_deterministic_legal_root_setups_with_weighted_exposure(self) -> None:
        first = self.prepare()
        directory = Path(first["training_starts"]["directory"])
        first_hashes = _hashes(directory)
        observed = set()
        positions = set()
        boards = Counter()
        depths = Counter()
        for path in sorted(directory.glob("*.sgf")):
            sgf = path.read_text(encoding="utf-8")
            size = int(re.search(r"SZ\[(\d+)\]", sgf).group(1))
            # One root node prevents the upstream traversal from exposing an
            # empty board or a different number of move prefixes per file.
            self.assertEqual(sgf.count(";"), 1)
            self.assertEqual(sgf.count("("), 1)
            self.assertNotRegex(sgf, r";[BW]\[")
            properties = {key: re.findall(r"\[([^]]*)\]", values)
                          for key, values in re.findall(r"([A-Z]+)((?:\[[^]]*\])+)", sgf)}
            self.assertEqual(properties["PL"], ["B"])
            black, white = properties["AB"], properties["AW"]
            self.assertEqual(len(black), len(white))
            stones = len(black) + len(white)
            self.assertIn(stones, (2, 4, 6))
            observed.add((size, stones))
            boards[size] += 1
            depths[size, stones] += 1
            position = (size, tuple(sorted(black)), tuple(sorted(white)))
            self.assertNotIn(position, positions, "upstream global dedup would change board sampling weights")
            positions.add(position)
            occupied = {}
            moves = [move for pair in zip(black, white)
                     for move in (("B", pair[0]), ("W", pair[1]))]
            for color, coordinate in moves:
                self.assertRegex(coordinate, r"^[a-s]{2}$")
                point = tuple(ord(c) - 97 for c in coordinate)
                self.assertTrue(all(1 <= value < size - 1 for value in point))
                self.assertEqual(sum(point) % 2, 0)
                self.assertNotIn(point, occupied)
                occupied[point] = color
                # Check every group independently after every move. An enemy
                # group without liberties would be a capture; our own would
                # be suicide. No production game/parser helpers are reused.
                for origin in occupied:
                    pending, group, liberties = [origin], set(), set()
                    while pending:
                        current = pending.pop()
                        if current in group:
                            continue
                        group.add(current)
                        x, y = current
                        for neighbor in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                            if not all(0 <= value < size for value in neighbor):
                                continue
                            if neighbor not in occupied:
                                liberties.add(neighbor)
                            elif occupied[neighbor] == occupied[origin] and neighbor not in group:
                                pending.append(neighbor)
                    self.assertTrue(liberties, f"capture/suicide in {path}: {origin}")
            recorded = first["training_starts"]["files"][path.name]
            self.assertEqual(recorded["sha256"], _hash(path))
            self.assertEqual(recorded["size"], path.stat().st_size)
            self.assertNotIn(b"\r", path.read_bytes())
        self.assertEqual(observed, {(size, count) for size in (7, 8, 9, 13, 19) for count in (2, 4, 6)})
        self.assertEqual(dict(boards), {7: 6, 8: 6, 9: 9, 13: 6, 19: 3})
        self.assertEqual(len(positions), 30)
        for size, units in ((7, 2), (8, 2), (9, 3), (13, 2), (19, 1)):
            for stones in (2, 4, 6):
                self.assertEqual(depths[size, stones], units)
        pool = first["training_starts"]["sampling"]
        self.assertEqual(pool["pool_counts_by_board"], {"7": 6, "8": 6, "9": 9, "13": 6, "19": 3})
        self.assertEqual(pool["positions_per_sgf"], 1)
        self.assertTrue(pool["exact_board_weights"])
        self.assertEqual(pool["position_count"], 30)
        for size, wanted in ((7, .2), (8, .2), (9, .3), (13, .2), (19, .1)):
            self.assertAlmostEqual(pool["actual_pool_board_probabilities"][str(size)], wanted)
            self.assertAlmostEqual(pool["nominal_overall_board_probabilities"][str(size)], wanted)
        self.output = self.root / "second-experiment"
        second = self.prepare()
        self.assertEqual(_hashes(Path(second["training_starts"]["directory"])), first_hashes)
        self.assertIn("training", first["training_starts"]["domain"])
        self.assertIn("independently sampled", first["randomness_note"])

    def test_arbitrary_float_weights_have_bounded_explicit_approximate_pool_counts(self) -> None:
        self.config["board_weights"] = [.123456789, .23456789, .3456789, .111111, .185185421]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        manifest_path = self.source / "mixed_manifest.json"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifest["identity"] = MixedBoardRunner(self.config_path).manifest_identity()
        manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

        manifest = self.prepare()

        pool = manifest["training_starts"].get("sampling")
        self.assertIsNotNone(pool, "the manifest must disclose actual integer sampling weights")
        self.assertEqual(pool["pool_counts_by_board"], {"7": 36, "8": 69, "9": 105, "13": 33, "19": 57})
        self.assertEqual(pool["position_count"], 300)
        self.assertLessEqual(pool["position_count"], pool["maximum_position_count"])
        self.assertFalse(pool["exact_board_weights"])
        self.assertAlmostEqual(pool["max_abs_pool_probability_error"], .004814579)
        self.assertIn("approx", pool["limitation"].lower())
        self.assertAlmostEqual(pool["nominal_overall_board_probabilities"]["7"], .1231111101)

    def test_frozen_verification_never_needs_current_source_files(self) -> None:
        manifest = self.prepare()
        self.checkpoint.unlink()
        self.config_path.unlink()
        (self.source / "mixed_manifest.json").unlink()
        for path in self.source.glob("selfplay/**/tdata/*.npz"):
            path.unlink()
        module = importlib.import_module("weiqi.katago_mixed_experiments")
        verify = getattr(module, "verify_prepared_experiment", None)
        self.assertIsNotNone(verify, "snapshot-independent preparation verification is required")
        self.assertEqual(verify(self.output), manifest)

    def test_frozen_verification_rejects_extra_arm_replay_and_missing_inventory_entries(self) -> None:
        manifest = self.prepare()
        module = importlib.import_module("weiqi.katago_mixed_experiments")
        verify = getattr(module, "verify_prepared_experiment", None)
        self.assertIsNotNone(verify, "strict preparation preflight is required before training")
        extra = Path(manifest["arms"]["sgf_starts"]["run_root"]) / "selfplay/new/tdata/extra.npz"
        np.savez_compressed(extra, globalInputNC=np.zeros((1, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "replay|inventory"):
            verify(self.output)
        extra.unlink()
        del manifest["prepared_files"]["common/train/gogogo/checkpoint.ckpt"]
        Path(manifest["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inventory"):
            verify(self.output)

    @unittest.skipUnless(shutil.which("katago"), "optional native SGF traversal check needs KataGo on PATH")
    def test_upstream_sgf_traversal_exposes_exactly_one_nonempty_uniform_position_per_file(self) -> None:
        manifest = self.prepare()
        sampled = self.root / "native-samples"
        completed = subprocess.run([
            shutil.which("katago"), "samplesgfs", "-sgfdir", manifest["training_starts"]["directory"],
            "-outdir", str(sampled), "-sample-prob", "1", "-turn-weight-lambda", "0",
            "-num-threads", "1", "-for-testing", "-flip-if-pass",
        ], capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        positions = [json.loads(line) for path in sampled.glob("*.startposes.txt")
                     for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(positions), 30)
        self.assertEqual(dict(Counter(position["xSize"] for position in positions)),
                         {7: 6, 8: 6, 9: 9, 13: 6, 19: 3})
        self.assertEqual(len({(position["board"], position["nextPla"]) for position in positions}), 30)
        for position in positions:
            self.assertEqual(position["nextPla"], "B")
            self.assertEqual(position["moveLocs"], [])
            self.assertEqual(position["movePlas"], [])
            self.assertEqual(position["weight"], 1.0)
            stones = position["board"].count("X") + position["board"].count("O")
            self.assertIn(stones, (2, 4, 6))
            self.assertEqual(position["initialTurnNumber"], stones)

    def test_source_manifest_mismatch_is_rejected_without_mutating_source(self) -> None:
        path = self.source / "mixed_manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["identity"]["board_weights"] = [1, 1, 1, 1, 1]
        path.write_text(json.dumps(document), encoding="utf-8")
        before = _hashes(self.source)
        with self.assertRaisesRegex(ValueError, "manifest|清单|incompatible"):
            self.prepare()
        self.assertEqual(_hashes(self.source), before)
        self.assertFalse(self.output.exists())

    def test_repeat_rejects_checkpoint_mutation_even_if_its_inventory_entry_is_removed(self) -> None:
        manifest = self.prepare()
        checkpoint = Path(manifest["snapshot"]["run_root"]) / "train/gogogo/checkpoint.ckpt"
        checkpoint.write_bytes(b"mutated")
        path = Path(manifest["manifest_path"])
        del manifest["prepared_files"][checkpoint.relative_to(self.output.resolve()).as_posix()]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        before = _hashes(self.output)
        with self.assertRaisesRegex(ValueError, "inventory|manifest|artifact|清单|mutat"):
            self.prepare()
        self.assertEqual(_hashes(self.output), before)

    def test_repeat_rejects_training_start_manifest_mutation(self) -> None:
        manifest = self.prepare()
        manifest["training_starts"]["seed"] += 1
        Path(manifest["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
        before = _hashes(self.output)
        with self.assertRaisesRegex(ValueError, "manifest|start|清单|不兼容"):
            self.prepare()
        self.assertEqual(_hashes(self.output), before)

    def test_repeat_rejects_extra_replay_that_would_change_initial_logical_rows(self) -> None:
        manifest = self.prepare()
        extra = Path(manifest["arms"]["control"]["run_root"]) / "selfplay/new/tdata/extra.npz"
        np.savez_compressed(extra, globalInputNC=np.zeros((2, 1), dtype=np.float32))
        before = _hashes(self.output)
        with self.assertRaisesRegex(ValueError, "replay|inventory|artifact|changed"):
            self.prepare()
        self.assertEqual(_hashes(self.output), before)

    def test_valid_legacy_full_checkpoint_can_seed_both_arms_without_resaving(self) -> None:
        torch.save(self.state, self.checkpoint, _use_new_zipfile_serialization=False)
        os.utime(self.checkpoint, ns=(200_000_000_000, 200_000_000_000))
        original = self.checkpoint.read_bytes()

        try:
            manifest = self.prepare()
        except ValueError as error:
            self.fail(f"A loadable full checkpoint must not be rejected for its serialization format: {error}")

        for arm in manifest["arms"].values():
            copied = Path(arm["run_root"]) / "train/gogogo/checkpoint.ckpt"
            self.assertEqual(copied.read_bytes(), original)

    def test_source_change_during_copy_leaves_unidentified_partial_output_that_cannot_be_adopted(self) -> None:
        copy2 = shutil.copy2
        changed = False

        def copy_then_touch(source, destination):
            nonlocal changed
            result = copy2(source, destination)
            if Path(source) == self.checkpoint.resolve() and not changed:
                changed = True
                os.utime(self.checkpoint, ns=(201_000_000_000, 201_000_000_000))
            return result

        # Fault injection at real file I/O models a writer racing the copy;
        # checkpoint loading, file copies, NPZ counting and ledgers stay real.
        with patch("weiqi.katago_mixed_experiments.shutil.copy2", side_effect=copy_then_touch):
            with self.assertRaisesRegex(ValueError, "Source changed|pause"):
                self.prepare()
        self.assertTrue(changed)
        self.assertFalse((self.output / "experiment_manifest.json").exists())
        before = _hashes(self.output)
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.prepare()
        self.assertEqual(_hashes(self.output), before)

    def test_invalid_wall_budget_never_creates_output(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "wall_seconds|budget"):
                    self.prepare(wall_seconds=value)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
