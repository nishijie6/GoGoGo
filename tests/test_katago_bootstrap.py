import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from weiqi.katago_bootstrap import BootstrapManager, assess_bootstrap_probe


BOARDS = (7, 8, 9, 13, 19)


def write_probe(root, *, moves=40, pass_probability=.1, games_per_board=20):
    folder = root / "selfplay/candidate"
    (folder / "sgfs").mkdir(parents=True)
    (folder / "tdata").mkdir()
    games = []
    for size in BOARDS:
        sequence = "".join(f";{'B' if i % 2 == 0 else 'W'}[{chr(97+i%size)}{chr(97+i//size)}]"
                           for i in range(moves))
        games.extend([f"(;GM[1]SZ[{size}]RE[W+6.5]{sequence})"] * games_per_board)
    (folder / "sgfs/games.sgfs").write_text("\n".join(games))
    masks = np.zeros((5, 22, 19, 19), dtype=np.uint8)
    for index, size in enumerate(BOARDS):
        masks[index, 0, :size, :size] = 1
    policy = np.zeros((5, 2, 362), dtype=np.float32)
    policy[:, :, 0] = (1-pass_probability) * 100
    policy[:, :, -1] = pass_probability * 100
    target = np.zeros((5, 64), dtype=np.float32)
    target[:, 25:27] = 1
    np.savez_compressed(folder / "tdata/data.npz",
        binaryInputNCHWPacked=np.packbits(masks.reshape(5, 22, 361), axis=2),
        policyTargetsNCMove=policy, globalTargetsNC=target)


class BootstrapAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_healthy_probe_is_ready_even_with_all_white_wins_and_records_hashes(self):
        write_probe(self.root)
        report = assess_bootstrap_probe(self.root, BOARDS)
        self.assertTrue(report["passed"])
        self.assertEqual(report["finished_games"], 100)
        self.assertEqual(len(report["source_hashes"]), 2)
        self.assertEqual(report["thresholds"]["policy_pass_probability_max"], .2)
        for row in report["by_board"].values():
            self.assertAlmostEqual(row["policy_pass_probability"], .1)

    def test_short_games_cannot_pass(self):
        write_probe(self.root, moves=2)
        self.assertFalse(assess_bootstrap_probe(self.root, BOARDS)["passed"])

    def test_exact_policy_distribution_pass_bias_cannot_pass(self):
        # The SGF contains no chosen passes; policy targets alone expose the bias.
        write_probe(self.root, pass_probability=.8)
        self.assertFalse(assess_bootstrap_probe(self.root, BOARDS)["passed"])

    def test_incomplete_damaged_pending_and_no_result_probes_cannot_pass(self):
        for kind in ("incomplete", "damaged", "pending", "no_result"):
            with self.subTest(kind=kind):
                root = self.root / kind
                write_probe(root, games_per_board=19 if kind == "incomplete" else 20)
                sgf = root / "selfplay/candidate/sgfs/games.sgfs"
                if kind == "damaged":
                    with sgf.open("a") as stream:
                        stream.write("\n(;GM[1]SZ[broken]RE[W+6.5];B[aa])")
                elif kind == "pending":
                    with sgf.open("a") as stream:
                        stream.write("\n(;GM[1]SZ[19];B[aa]")
                elif kind == "no_result":
                    sgf.write_text(sgf.read_text().replace("RE[W+6.5]", "RE[Void]", 1))
                self.assertFalse(assess_bootstrap_probe(root, BOARDS)["passed"])

    def test_invalid_npz_stops_assessment(self):
        write_probe(self.root)
        (self.root / "selfplay/candidate/tdata/data.npz").write_bytes(b"invalid")
        with self.assertRaises(ValueError):
            assess_bootstrap_probe(self.root, BOARDS)

    def test_one_board_with_fewer_than_four_finished_games_cannot_pass(self):
        write_probe(self.root)
        path = self.root / "selfplay/candidate/sgfs/games.sgfs"
        games = path.read_text().splitlines()
        # Keep 100 games overall, but only three games on 19x19.
        path.write_text("\n".join(games[:83] + games[:17]))
        report = assess_bootstrap_probe(self.root, BOARDS)
        self.assertEqual(report["finished_games"], 100)
        self.assertFalse(report["passed"])
        self.assertIn("insufficient_finished_games", report["by_board"]["19"]["reasons"])

    def test_policy_mass_uses_row_and_policy_weights_and_normalization(self):
        write_probe(self.root)
        path = self.root / "selfplay/candidate/tdata/data.npz"
        with np.load(path) as data:
            arrays = {key: np.repeat(data[key], 2, axis=0) for key in data.files}
        arrays["policyTargetsNCMove"][1::2, 0, :] = 0
        arrays["policyTargetsNCMove"][1::2, 0, -1] = 500
        arrays["globalTargetsNC"][1::2, 25] = 0
        np.savez_compressed(path, **arrays)
        self.assertTrue(assess_bootstrap_probe(self.root, BOARDS)["passed"])


class BootstrapManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runner = SimpleNamespace(run_root=self.root, models_dir=self.root / "models",
            config_path=self.root / "config.json", project_root=self.root,
            hardware_profile=None, mixed=SimpleNamespace(board_sizes=BOARDS))
        self.manager = BootstrapManager(self.runner)

    def candidate(self, samples):
        path = self.runner.models_dir / f"gogogo-s{samples}-d100/model.bin.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"trained-{samples}".encode())
        return path

    def test_fresh_state_actor_selection_and_legacy_resume(self):
        self.assertIsNone(self.manager.read_state())
        self.assertEqual(self.manager.models_dir(), self.runner.models_dir)
        state = self.manager.start()
        self.assertEqual((state["phase"], state["last_probe_samples"]), ("warming", 0))
        self.assertEqual(self.manager.models_dir(), self.root / "bootstrap_random_models")
        self.assertEqual(list(self.manager.models_dir().iterdir()), [])
        self.assertEqual(BootstrapManager(self.runner).read_state(), state)
        self.assertEqual(self.manager.start(), state)

    def test_corrupt_recorded_state_fails_closed(self):
        for state in ("not json", "{}", '{"schema_version":1,"phase":"ready","last_probe_samples":0}'):
            (self.root / "bootstrap_state.json").write_text(state)
            with self.assertRaises(ValueError):
                self.manager.models_dir()

    def test_warming_actor_directory_rejects_any_unexpected_contents(self):
        self.manager.start()
        directory = self.manager.models_dir()
        (directory / "model.bin.gz").write_bytes(b"unexpected actor")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.manager.models_dir()

    def test_ready_state_without_complete_valid_assessment_fails_closed(self):
        state = self.manager.start()
        state.update(phase="ready", last_probe_samples=4096, last_assessment={"passed": True})
        (self.root / "bootstrap_state.json").write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            self.manager.read_state()

    def probe_factory(self, *, fail=None, short=False):
        owner = self

        class Probe:
            def __init__(self, config_path, *, project_root, run_root_override, hardware_profile):
                owner.assertEqual(config_path, owner.runner.config_path)
                owner.assertEqual(project_root, owner.runner.project_root)
                owner.assertIs(hardware_profile, owner.runner.hardware_profile)
                self.run_root = run_root_override
                self.models_dir = self.run_root / "models"

            def ensure_manifest(self):
                owner.assertFalse(self.models_dir.exists())
                self.run_root.mkdir(parents=True)
                (self.run_root / "mixed_manifest.json").write_text("{}")

            def selfplay(self, *, games):
                owner.assertEqual(games, 100)
                state = owner.manager.read_state()
                owner.assertIsNotNone(state["pending_probe"])
                owner.assertGreaterEqual(state["last_probe_samples"], 4096)
                frozen = list(self.models_dir.glob("*/model.bin.gz"))
                owner.assertEqual(len(frozen), 1)
                owner.assertEqual(frozen[0].read_bytes(), owner.candidate_path.read_bytes())
                owner.assertFalse((owner.root / "selfplay").exists())
                if fail:
                    raise fail
                write_probe(self.run_root, moves=2 if short else 40)

        return Probe

    def test_probe_isolated_and_ready_actor_uses_trained_models(self):
        self.manager.start()
        self.candidate_path = self.candidate(4096)
        with patch("weiqi.katago_mixed.MixedBoardRunner", self.probe_factory()):
            state = self.manager.after_cycle()
        self.assertEqual(state["phase"], "ready")
        self.assertIsNone(state["pending_probe"])
        self.assertEqual(self.manager.models_dir(), self.runner.models_dir)
        self.assertEqual(BootstrapManager(self.runner).start(), state)
        self.assertFalse((self.root / "selfplay").exists())

    def test_quality_failure_rate_limits_by_new_learner_samples(self):
        self.manager.start()
        self.candidate_path = self.candidate(4095)
        with patch("weiqi.katago_mixed.MixedBoardRunner", side_effect=AssertionError("early probe")):
            self.manager.after_cycle()
        self.candidate_path = self.candidate(4096)
        with patch("weiqi.katago_mixed.MixedBoardRunner", self.probe_factory(short=True)):
            state = self.manager.after_cycle()
        self.assertEqual(state["phase"], "warming")
        self.assertEqual(state["last_probe_samples"], 4096)
        self.candidate_path = self.candidate(8191)
        with patch("weiqi.katago_mixed.MixedBoardRunner", side_effect=AssertionError("lucky repeat")):
            self.manager.after_cycle()
        self.candidate_path = self.candidate(8192)
        with patch("weiqi.katago_mixed.MixedBoardRunner", self.probe_factory()):
            self.assertEqual(self.manager.after_cycle()["phase"], "ready")

    def test_interrupt_and_runtime_error_persist_pending_without_promotion_or_repeat(self):
        for index, error in enumerate((KeyboardInterrupt(), RuntimeError("engine failed"))):
            with self.subTest(error=type(error).__name__):
                root = self.root / str(index)
                root.mkdir()
                self.runner.run_root = root
                self.manager = BootstrapManager(self.runner)
                self.manager.start()
                self.candidate_path = self.candidate(4096)
                with patch("weiqi.katago_mixed.MixedBoardRunner", self.probe_factory(fail=error)):
                    with self.assertRaises(type(error)):
                        self.manager.after_cycle()
                state = self.manager.read_state()
                self.assertEqual(state["phase"], "warming")
                self.assertEqual(state["last_probe_samples"], 4096)
                self.assertIsNotNone(state["pending_probe"])
                self.assertEqual(BootstrapManager(self.runner).start(), state)
                with patch("weiqi.katago_mixed.MixedBoardRunner", side_effect=AssertionError("repeat")):
                    self.assertEqual(self.manager.after_cycle()["phase"], "warming")


if __name__ == "__main__":
    unittest.main()
