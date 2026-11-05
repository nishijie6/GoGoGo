"""Frozen mixed-board evaluations preserve model identity and paired evidence."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from weiqi.engine import GoGame
from weiqi.katago_hardware import HardwareProfile
from weiqi.rl_curriculum_openings import ensure_evaluation_opening_suite
from weiqi.rl_match import build_match_config


PAIR = ("(;GM[1]SZ[{size}]KM[7.5]RU[koPOSITIONALscoreAREAtaxNONEsui0]"
        "PB[candidate]PW[baseline]RE[B+1.5])\n"
        "(;GM[1]SZ[{size}]KM[7.5]RU[koPOSITIONALscoreAREAtaxNONEsui0]"
        "PB[baseline]PW[candidate]RE[W+1.5])\n")


def config_values(path: Path) -> dict[str, str]:
    return {key.strip(): value.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line for key, value in [line.split("=", 1)]}


class EngineBoundary:
    """Only the external process is replaced; configs, suites and state are real."""

    def __init__(self, root: Path):
        self.run_root = root / "training"
        self.project_root = root
        self.mixed = SimpleNamespace(board_sizes=(7, 8, 9, 13, 19))
        self.config = SimpleNamespace(game=SimpleNamespace(
            komi=7.5, scoring_rule="area", ko_rule="positional_superko", allow_suicide=False))
        self.hardware_profile = HardwareProfile("mps", "mps", "float32", "Apple M4", 12, 4, False)
        self.katago_executable = root / "engine" / "katago"
        self.katago_executable.parent.mkdir()
        self.katago_executable.write_bytes(b"fake engine version A")
        self.calls = []
        self.fail_at = None
        self.result_text = None

    def _run(self, command, *, cwd=None, log_name=None):
        command = list(map(str, command))
        self.calls.append(command)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("engine interrupted")
        config = Path(command[command.index("-config") + 1])
        sgfs = Path(command[command.index("-sgf-output-dir") + 1])
        values = config_values(config)
        size = int(values["bSizes"])
        (sgfs / "pair.sgfs").write_text(
            self.result_text if self.result_text is not None else PAIR.format(size=size), encoding="utf-8")
        if log_name is not None:
            Path(log_name).write_text("mocked engine boundary\n", encoding="utf-8")
        return "mocked engine boundary\n"


class MixedEvaluationTests(unittest.TestCase):
    def test_parallel_pair_count_is_frozen_and_completed_pairs_are_reused(self):
        self.evaluate(games_per_board=8, parallel_pairs=4)
        self.assertEqual(len(self.runner.calls), 4)
        self.assertEqual(self.manifest()["protocol"]["parallel_pairs"], 4)
        progress = json.loads((self.output / "boards/9x9/progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["parallel_pairs"], 4)
        self.runner.calls.clear()
        self.evaluate(games_per_board=8, parallel_pairs=4)
        self.assertEqual(self.runner.calls, [])
        with self.assertRaises(ValueError):
            self.evaluate(games_per_board=8, parallel_pairs=2)
        self.assertEqual(self.runner.calls, [])

    def test_invalid_parallel_pair_count_fails_before_creating_output(self):
        for value in (0, -1, True, 1.5, 17):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(parallel_pairs=value)
        self.assertFalse(self.output.exists())

    def test_long_game_limit_is_explicit_and_bound_to_resume(self):
        self.evaluate(boards=[7], games_per_board=2, max_moves_factor=6.0)
        cfg = self.output / "boards/7x7/pairs/opening-000/match.cfg"
        self.assertEqual(config_values(cfg)["maxMovesPerGame"], "294")
        with self.assertRaises(ValueError):
            self.evaluate(boards=[7], games_per_board=2, max_moves_factor=5.0)

    def test_missing_terminal_result_is_not_misreported_as_wrong_opponent(self):
        self.runner.result_text = PAIR.format(size=7).replace("RE[B+1.5]", "")
        with self.assertRaisesRegex(RuntimeError, "终局结果"):
            self.evaluate(boards=[7], games_per_board=2)

    def test_invalid_move_limits_fail_before_creating_output(self):
        for value in (0, -1, True, float('nan'), float('inf'), 1e308):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.evaluate(max_moves_factor=value)
        self.assertFalse(self.output.exists())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runner = EngineBoundary(self.root)
        self.output = self.root / "evaluation"
        self.baseline = self.model("gogogo-s0-d0", b"stable baseline")
        self.candidate = self.model("gogogo-s200-d100", b"trained candidate")

    def model(self, name, content):
        path = self.runner.run_root / "models" / name / "model.bin.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def evaluate(self, **changes):
        self.assertIsNotNone(importlib.util.find_spec("weiqi.katago_mixed_eval"),
                             "frozen mixed evaluation implementation is missing")
        function = importlib.import_module("weiqi.katago_mixed_eval").evaluate_mixed
        return function(self.runner, self.output,
                        **{"candidate": self.candidate, "baseline": self.baseline,
                           "boards": [9], "games_per_board": 4, "visits": 16, **changes})

    def manifest(self):
        return json.loads((self.output / "evaluation_manifest.json").read_text(encoding="utf-8"))

    def test_all_five_sizes_use_frozen_models_singleton_pairs_and_metal(self):
        result = self.evaluate(boards=None)
        self.assertEqual(set(result["results_by_board"]), {"7", "8", "9", "13", "19"})
        self.assertNotIn("win_rate", result)
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(len(self.runner.calls), 10)
        manifest = self.manifest()
        self.assertEqual(manifest["candidate"]["source_path"], str(self.candidate))
        self.assertEqual(manifest["baseline"]["source_path"], str(self.baseline))
        self.assertEqual(manifest["candidate"]["sha256"], hashlib.sha256(b"trained candidate").hexdigest())
        suites = set()
        for command in self.runner.calls:
            self.assertEqual(command[1], "match")
            cfg = Path(command[command.index("-config") + 1])
            values = config_values(cfg)
            self.assertEqual(values["numGamesTotal"], "2")
            self.assertEqual(values["numGameThreads"], "2")
            self.assertEqual(values["nnMaxBatchSize"], "2")
            self.assertEqual(values["maxVisits"], "16")
            self.assertEqual(values["metalDeviceToUseThread0"], "0")
            self.assertFalse(any(key.startswith("cuda") for key in values))
            self.assertEqual(values["komiMean"], "7.5")
            self.assertEqual(values["koRules"], "POSITIONAL")
            self.assertEqual(values["scoringRules"], "AREA")
            self.assertEqual(values["allowResignation"], "false")
            self.assertEqual(values["rootNoiseEnabled"], "false")
            self.assertEqual(values["chosenMoveTemperature"], "0.0")
            self.assertEqual(values["hintPosesProb"], "1.0")
            for role, key in (("candidate", "nnModelFile0"), ("baseline", "nnModelFile1")):
                frozen = self.output / "frozen" / f"{role}.bin.gz"
                self.assertEqual(Path(values[key]), frozen)
                self.assertTrue(frozen.is_file())
            opening = Path(values["hintPosesDir"])
            self.assertEqual(len(list(opening.iterdir())), 1)
            sample = json.loads((opening / "opening.startposes.txt").read_text(encoding="utf-8"))
            self.assertEqual(sample["xSize"], int(values["bSizes"]))
            suites.add(opening.parent)
            self.assertTrue((cfg.parent / "match.log").exists())
        self.assertEqual(len(suites), 5)
        for size, row in result["results_by_board"].items():
            self.assertEqual(row["games"], 4)
            self.assertEqual(row["candidate_black_games"], 2)
            self.assertEqual(row["candidate_white_games"], 2)
            self.assertIn("descriptive", row["wilson_interpretation"])
            self.assertEqual(row["pair_statistics"]["opening_pairs"], 2)
            self.assertEqual(row["pair_statistics"]["scores"], [1.0, 1.0])
            self.assertLess(row["pair_statistics"]["hoeffding_95_lower"], .5)
            self.assertEqual(len(row["pair_artifacts"]), 2)
            self.assertTrue((self.output / "boards" / f"{size}x{size}" / "progress.json").is_file())

    def test_auto_selection_uses_samples_not_directory_name_or_mtime(self):
        self.model("gogogo-s90-d9000", b"later lexical name")
        self.model("unknown-model", b"not a trained sample name")
        result = self.evaluate(candidate=None)
        self.assertEqual(result["candidate"]["source_path"], str(self.candidate))
        self.assertEqual(result["candidate"]["trained_samples"], 200)

    def test_default_pilot_plays_twelve_games_per_board_and_keeps_wide_pair_intervals(self):
        module = importlib.import_module("weiqi.katago_mixed_eval")
        result = module.evaluate_mixed(self.runner, self.output, baseline=self.baseline)
        self.assertEqual(len(self.runner.calls), 30)
        for row in result["results_by_board"].values():
            self.assertEqual(row["games"], 12)
            self.assertEqual(row["pair_statistics"]["opening_pairs"], 6)
            self.assertLess(row["pair_statistics"]["hoeffding_95_lower"], .5)
        self.assertEqual(config_values(Path(self.runner.calls[0][3]))["maxVisits"], "32")

    def test_saturated_pilot_scores_do_not_report_a_finite_elo(self):
        candidate_sweep = PAIR.format(size=9)
        baseline_sweep = candidate_sweep.replace("RE[B+1.5]", "RE[W+2.5]").replace("RE[W+1.5]", "RE[B+2.5]")
        for score, text in ((1.0, candidate_sweep), (0.0, baseline_sweep)):
            with self.subTest(score=score):
                self.output = self.root / f"saturated-{score}"
                self.runner.result_text = text
                row = self.evaluate()["results_by_board"]["9"]
                self.assertEqual(row["win_rate"], score)
                self.assertIsNone(row["elo"])
                self.assertIn("unestimable", row.get("elo_interpretation", ""))

    def test_non_saturated_pilot_elo_is_only_an_exploratory_relative_estimate(self):
        self.runner.result_text = PAIR.format(size=9).replace("RE[B+1.5]", "RE[W+1.5]")
        row = self.evaluate()["results_by_board"]["9"]
        self.assertEqual(row["win_rate"], 0.5)
        self.assertEqual(row["elo"], 0.0)
        self.assertIn("exploratory", row.get("elo_interpretation", ""))

    def test_both_models_and_manifest_are_frozen_before_the_first_match(self):
        execute = self.runner._run

        def change_training_exports(command, **kwargs):
            self.assertTrue((self.output / "evaluation_manifest.json").is_file())
            self.assertEqual((self.output / "frozen/candidate.bin.gz").read_bytes(), b"trained candidate")
            self.assertEqual((self.output / "frozen/baseline.bin.gz").read_bytes(), b"stable baseline")
            self.candidate.write_bytes(b"a newer training export")
            self.baseline.write_bytes(b"changed external baseline")
            return execute(command, **kwargs)

        self.runner._run = change_training_exports
        result = self.evaluate(boards=[7, 19])
        self.assertEqual(result["candidate"]["sha256"], hashlib.sha256(b"trained candidate").hexdigest())
        self.assertEqual(result["baseline"]["sha256"], hashlib.sha256(b"stable baseline").hexdigest())

    def test_resume_binds_auto_selection_and_survives_pruned_sources(self):
        first = self.evaluate(candidate=None)
        before = (self.output / "evaluation_manifest.json").read_bytes()
        self.model("gogogo-s99999-d5000", b"newly exported model")
        self.candidate.unlink()
        self.baseline.unlink()
        self.runner.calls.clear()
        second = self.evaluate(candidate=None)
        self.assertEqual(second["candidate"], first["candidate"])
        self.assertEqual(self.runner.calls, [])
        self.assertEqual((self.output / "evaluation_manifest.json").read_bytes(), before)
        progress = json.loads((self.output / "boards/9x9/progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["reused_pairs"], 2)

    def test_interrupted_evaluation_reuses_finished_pairs_after_source_pruning(self):
        self.runner.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.evaluate(candidate=None)
        self.assertFalse((self.output / "summary.json").exists())
        first_sgf = self.output / "boards/9x9/pairs/opening-000/sgfs/pair.sgfs"
        first_bytes = first_sgf.read_bytes()
        self.model("gogogo-s300-d110", b"new candidate")
        self.candidate.unlink()
        self.baseline.unlink()
        self.runner.calls.clear()
        self.runner.fail_at = None
        result = self.evaluate(candidate=None)
        self.assertEqual(len(self.runner.calls), 1)
        self.assertIn("retry-", self.runner.calls[0][self.runner.calls[0].index("-config") + 1])
        self.assertEqual(result["candidate"]["trained_samples"], 200)
        self.assertEqual(first_sgf.read_bytes(), first_bytes)
        self.assertEqual(result["results_by_board"]["9"]["games"], 4)

    def test_completed_pair_is_sealed_before_starting_the_next_pair(self):
        self.runner.fail_at = 2
        execute = self.runner._run
        seals_seen_before_next_pair = []

        def inspect_pair_evidence(command, **kwargs):
            if self.runner.calls:
                evidence = self.output / "boards/9x9/pairs/opening-000/pair-evidence.json"
                seals_seen_before_next_pair.append(evidence.is_file())
            return execute(command, **kwargs)

        self.runner._run = inspect_pair_evidence
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.evaluate()
        self.assertEqual(seals_seen_before_next_pair, [True])
        pair_root = self.output / "boards/9x9/pairs/opening-000"
        evidence = json.loads((pair_root / "pair-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(evidence["pair_artifact"]["sgfs"][0]["sha256"],
                         hashlib.sha256((pair_root / "sgfs/pair.sgfs").read_bytes()).hexdigest())

    def test_interrupted_board_cannot_reuse_a_modified_completed_pair(self):
        self.runner.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.evaluate()
        self.runner.fail_at = None
        self.runner.calls.clear()
        first_sgf = self.output / "boards/9x9/pairs/opening-000/sgfs/pair.sgfs"
        first_sgf.write_text(first_sgf.read_text(encoding="utf-8").replace("RE[B+1.5]", "RE[W+1.5]"),
                             encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA|校验|改变|不一致"):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])
        self.assertFalse((self.output / "summary.json").exists())

    def test_complete_pair_without_its_evidence_is_not_adopted_on_resume(self):
        self.runner.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.evaluate()
        (self.output / "boards/9x9/pairs/opening-000/pair-evidence.json").unlink(missing_ok=True)
        self.runner.fail_at = None
        self.runner.calls.clear()
        with self.assertRaisesRegex(ValueError, "证据|evidence|校验"):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])

    def test_wrong_board_opponent_rules_or_komi_cannot_be_sealed(self):
        valid = PAIR.format(size=9)
        cases = (
            valid.replace("SZ[9]", "SZ[19]"),
            valid.replace("baseline", "another-opponent"),
            valid.replace("SZ[9]", ""),
            valid.replace("KM[7.5]", "KM[6.5]"),
            valid.replace("KM[7.5]", ""),
            valid.replace("koPOSITIONAL", "koSIMPLE"),
            valid.replace("RU[koPOSITIONALscoreAREAtaxNONEsui0]", ""),
            valid.replace("PW[baseline]", "PW[baseline]PW[another-opponent]"),
        )
        for index, text in enumerate(cases):
            with self.subTest(index=index):
                self.output = self.root / f"wrong-protocol-{index}"
                self.runner.result_text = text
                with self.assertRaises(RuntimeError):
                    self.evaluate()
                self.assertFalse((self.output / "summary.json").exists())
                self.assertFalse(any(self.output.rglob("pair-evidence.json")))

    def test_same_path_engine_replacement_cannot_resume_or_rewrite_outputs(self):
        self.evaluate()
        manifest = (self.output / "evaluation_manifest.json").read_bytes()
        summary = (self.output / "summary.json").read_bytes()
        self.runner.calls.clear()
        self.runner.katago_executable.write_bytes(b"fake engine version B")
        with self.assertRaisesRegex(ValueError, "不一致|引擎|engine|SHA"):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])
        self.assertEqual((self.output / "evaluation_manifest.json").read_bytes(), manifest)
        self.assertEqual((self.output / "summary.json").read_bytes(), summary)

    def test_legacy_manifest_without_engine_identity_is_rejected_without_mutation(self):
        self.evaluate()
        manifest = self.manifest()
        manifest["protocol"]["version"] = 1
        manifest["protocol"].pop("katago_sha256", None)
        path = self.output / "evaluation_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (self.output / "evaluation_manifest.sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")
        before = {item: item.read_bytes() for item in self.output.rglob("*") if item.is_file()}
        self.runner.calls.clear()
        with self.assertRaisesRegex(ValueError, "不一致|协议|引擎|版本"):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])
        self.assertEqual({item: item.read_bytes() for item in before}, before)

    def test_changed_arguments_cannot_reuse_or_overwrite_an_evaluation(self):
        self.evaluate()
        before = (self.output / "evaluation_manifest.json").read_bytes()
        self.runner.calls.clear()
        other = self.model("gogogo-s300-d120", b"other candidate")
        for changes in ({"games_per_board": 6}, {"visits": 32}, {"boards": [13]},
                        {"candidate": other}, {"baseline": other}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.evaluate(**changes)
        self.assertEqual(self.runner.calls, [])
        self.assertEqual((self.output / "evaluation_manifest.json").read_bytes(), before)

    def test_frozen_content_and_manifest_tampering_are_rejected(self):
        self.evaluate()
        self.runner.calls.clear()
        frozen = self.output / "frozen/candidate.bin.gz"
        frozen.write_bytes(b"changed frozen bytes")
        with self.assertRaisesRegex(ValueError, "hash|哈希|SHA|改变|校验"):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])
        frozen.write_bytes(b"trained candidate")
        manifest = self.output / "evaluation_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])

    def test_changed_opening_or_saved_config_is_not_silently_regenerated(self):
        self.evaluate()
        manifest = self.manifest()
        self.runner.calls.clear()
        sample = next(self.output.glob("evaluation_openings/9x9/v*/opening-000/opening.startposes.txt"))
        original = sample.read_bytes()
        sample.write_bytes(b"{}\n")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(sample.read_bytes(), b"{}\n")
        sample.write_bytes(original)
        cfg = self.output / "boards/9x9/pairs/opening-000/match.cfg"
        cfg.write_text(cfg.read_text(encoding="utf-8") + "maxVisits = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])
        self.assertEqual(self.manifest(), manifest)

    def test_completed_sgf_evidence_cannot_be_changed_on_resume(self):
        self.evaluate()
        self.runner.calls.clear()
        sgf = self.output / "boards/9x9/pairs/opening-000/sgfs/pair.sgfs"
        sgf.write_text(sgf.read_text(encoding="utf-8").replace("RE[B+1.5]", "RE[W+1.5]"),
                       encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA|校验|改变|不一致"):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])

    def test_changed_supplied_export_is_rejected_but_frozen_snapshot_is_preserved(self):
        self.evaluate()
        self.runner.calls.clear()
        self.baseline.write_bytes(b"different baseline at same source path")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual((self.output / "frozen/baseline.bin.gz").read_bytes(), b"stable baseline")
        self.assertEqual(self.runner.calls, [])

    def test_existing_unidentified_output_is_not_adopted(self):
        self.output.mkdir()
        marker = self.output / "old-result.json"
        marker.write_text("preserve me", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
        self.assertEqual(self.runner.calls, [])

    def test_invalid_sizes_counts_rules_and_komi_fail_before_engine_execution(self):
        for changes in ({"boards": []}, {"boards": [7, 7]}, {"boards": [True]},
                        {"boards": [6]}, {"games_per_board": 3}, {"games_per_board": True},
                        {"visits": 0}, {"visits": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.evaluate(**changes)
        self.runner.config.game.komi = float("nan")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.runner.config.game.komi = 6.5
        self.runner.config.game.ko_rule = "simple"
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])
        self.assertFalse(self.output.exists())

    def test_hardware_and_komi_changes_cannot_resume_an_existing_protocol(self):
        self.evaluate()
        self.runner.calls.clear()
        self.runner.config.game.komi = 6.5
        with self.assertRaises(ValueError):
            self.evaluate()
        self.runner.config.game.komi = 7.5
        self.runner.hardware_profile = replace(self.runner.hardware_profile, tier="cpu", device="cpu")
        with self.assertRaises(ValueError):
            self.evaluate()
        self.assertEqual(self.runner.calls, [])

    def test_no_result_and_truncated_sgfs_never_publish_a_score(self):
        bad = (PAIR.format(size=9).replace("RE[B+1.5]", "RE[Void]"),
               PAIR.format(size=9) + "(;PB[candidate]")
        for index, text in enumerate(bad):
            with self.subTest(index=index):
                self.output = self.root / f"invalid-{index}"
                self.runner.result_text = text
                with self.assertRaises(RuntimeError):
                    self.evaluate()
                self.assertFalse((self.output / "summary.json").exists())
                self.assertFalse((self.output / "boards/9x9/match-result.json").exists())
                progress = json.loads((self.output / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["status"], "failed")


class SmallBoardEvaluationConfigTests(unittest.TestCase):
    def test_katago_small_board_support_does_not_expand_native_gui_board_sizes(self):
        self.assertEqual(GoGame.SUPPORTED_SIZES, (9, 13, 19))
        with tempfile.TemporaryDirectory() as temporary:
            for size in (7, 8):
                with self.subTest(size=size):
                    roots = ensure_evaluation_opening_suite(Path(temporary), board_size=size, opening_count=6)
                    self.assertEqual(len(roots), 6)
                    text = build_match_config(candidate_model=Path("c.bin.gz"), baseline_model=Path("b.bin.gz"),
                                              board_size=size, visits=16, games=2, komi=7.5)
                    self.assertIn(f"bSizes = {size}\n", text)
                    self.assertIn("komiMean = 7.5\n", text)


if __name__ == "__main__":
    unittest.main()
