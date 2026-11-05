"""Per-size self-play health uses completed SGFs and independent sample windows."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.katago_mixed_health import build_mixed_health
from weiqi.sgf import parse_sgf_game


def game(size, moves=8, result="B+1", *, extra=""):
    coordinates = [chr(97 + x) + chr(97 + y) for y in range(size) for x in range(size)]
    sequence = "".join(f";{'B' if index % 2 == 0 else 'W'}[{coordinates[index]}]"
                       for index in range(moves))
    return f"(;GM[1]SZ[{size}]KM[6.5]RE[{result}]{extra}{sequence})"


class MixedHealthTests(unittest.TestCase):
    def test_unified_health_reports_extreme_scores_as_observation(self):
        self.write("seed/sgfs/games.sgfs", "\n".join(
            game(9, 40, "B+999" if i % 2 else "W+999") for i in range(32)))
        report = self.health(sizes=(9,), minimum_games=32, standard="unified_v1")
        row = report["by_board"]["9"]
        self.assertEqual(row["status"], "ok")
        self.assertIn("extreme_margin_rate", row["warnings"])
        self.assertEqual(row["health_reasons"], [])
        self.assertEqual(report["training_standard"]["name"], "unified_v1")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, relative, content):
        path = self.root / "selfplay" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def health(self, sizes=(7, 8, 19), **kwargs):
        return build_mixed_health(self.root, sizes, **kwargs)

    def test_sizes_have_independent_results_and_warmup_counts(self):
        self.write("seed/sgfs/mixed.sgfs", "\n".join([
            game(7, result="B+1"), game(8, result="W+2"),
            game(19, 40, "B+20"), game(7, result="W+1"),
            game(19, 40, "W+20"), game(7, result="0"),
        ]))

        report = self.health(window_games=8, minimum_games=2)

        seven, eight, nineteen = (report["by_board"][str(size)] for size in (7, 8, 19))
        self.assertEqual(seven["cumulative_valid_games"], 3)
        self.assertEqual(seven["result_counts"], {"black": 1, "white": 1, "draw": 1, "no_result": 0})
        self.assertEqual(seven["black_score_rate"], 0.5)
        self.assertEqual(seven["status"], "ok")
        self.assertEqual(eight["window_games"], 1)
        self.assertFalse(eight["estimable"])
        self.assertEqual(eight["status"], "warming_up")
        self.assertEqual(nineteen["average_moves"], 40)
        self.assertEqual(nineteen["extreme_margin_rate"], 0)
        self.assertEqual(report["inventory"]["valid_games"], 6)
        self.assertEqual(json.loads((self.root / "health_by_board.json").read_text()), report)

    def test_default_minimum_is_reachable_without_filling_the_global_window(self):
        for size, moves in ((7, 8), (8, 8), (19, 40)):
            self.write(f"seed/sgfs/{size}.sgfs", "\n".join(
                game(size, moves, "B+1" if index % 2 else "W+1") for index in range(64)))

        report = self.health()

        for size in (7, 8, 19):
            with self.subTest(size=size):
                board = report["by_board"][str(size)]
                self.assertEqual(board["window_games"], 64)
                self.assertTrue(board["estimable"])
                self.assertFalse(board["window_full"])
                self.assertEqual(board["status"], "ok")
        self.assertEqual(report["status"], "ok")

    def test_degenerate_window_has_actionable_original_sgf_and_explicit_thresholds(self):
        comment = r"C[not a tree: (;SZ[19\]RE[B+999\];B[\];W[\]) escaped \] bracket]"
        original = ("(;GM[1]SZ[8]KM[6.5]PB[black model]PW[white model]RE[W+63.5]"
                    + comment + ";B[];W[];B[aa];W[])")
        self.write("seed/sgfs/double-pass.sgfs", "\n".join([
            original, original, original.replace("RE[W+63.5]", "RE[B+63.5]"),
            original.replace("RE[W+63.5]", "RE[B+63.5]"),
        ]))

        report = self.health(window_games=4, minimum_games=4)

        board = report["by_board"]["8"]
        self.assertEqual(board["status"], "needs_review")
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(board["early_double_pass_rate"], 1)
        self.assertEqual(board["short_game_rate"], 1)
        self.assertEqual(board["extreme_margin_rate"], 1)
        self.assertEqual(board["short_game_move_cutoff"], 7)
        self.assertEqual(set(board["warnings"]), {
            "early_double_pass_rate", "short_game_rate", "extreme_margin_rate"})
        self.assertEqual(report["thresholds"]["extreme_margin_area_fraction"], 0.75)
        self.assertLessEqual(len(board["examples"]), 3)
        example = board["examples"][0]
        self.assertEqual(example["komi"], "6.5")
        self.assertEqual(example["pass_moves"], 3)
        self.assertEqual(example["players"], {"black": "black model", "white": "white model"})
        self.assertEqual(example["original_sgf"], original.replace("RE[W+63.5]", "RE[B+63.5]"))
        self.assertEqual(board["by_source"]["seed"]["window_games"], 4)

    def test_recent_window_replaces_historical_degeneration_in_mtime_order(self):
        old = self.write("old/sgfs/z.sgfs", "\n".join(["(;SZ[8]RE[W+1];B[];W[])"] * 4))
        newest = self.write("new/sgfs/a.sgfs", "\n".join(
            game(8, result="B+1" if index % 2 else "W+1") for index in range(4)))
        os.utime(old, ns=(1_000_000_000, 1_000_000_000))
        os.utime(newest, ns=(2_000_000_000, 2_000_000_000))

        board = self.health((8,), window_games=4, minimum_games=4)["by_board"]["8"]

        self.assertEqual(board["cumulative_valid_games"], 8)
        self.assertEqual(board["window_games"], 4)
        self.assertTrue(board["window_full"])
        self.assertEqual(board["early_double_pass_rate"], 0)
        self.assertEqual(board["average_moves"], 8)
        self.assertEqual(board["status"], "ok")
        self.assertEqual(board["warnings"], [])
        self.assertEqual(board["examples"], [])
        self.assertEqual(set(board["by_source"]), {"new"})

    def test_short_game_cutoffs_scale_by_board_area_and_use_strictly_under(self):
        self.write("seed/sgfs/mixed.sgfs", "\n".join([
            game(7, 4, "B+1"), game(7, 5, "W+1"),
            game(8, 6, "B+1"), game(8, 7, "W+1"),
            game(19, 36, "B+1"), game(19, 37, "W+1"),
        ]))

        report = self.health(window_games=2, minimum_games=2)

        for size, cutoff in ((7, 5), (8, 7), (19, 37)):
            with self.subTest(size=size):
                board = report["by_board"][str(size)]
                self.assertEqual(board["short_game_move_cutoff"], cutoff)
                self.assertEqual(board["short_game_rate"], 0.5)
                self.assertEqual(board["early_double_pass_rate"], 0)

    def test_unknown_results_are_valid_and_excluded_from_black_score_denominator(self):
        self.write("seed/sgfs/results.sgfs", "\n".join(
            game(7, result=result) for result in ("B+R", "W+R", "0", "Void")))

        report = self.health((7,), window_games=4, minimum_games=4)

        board = report["by_board"]["7"]
        self.assertEqual(board["result_counts"], {"black": 1, "white": 1, "draw": 1, "no_result": 1})
        self.assertEqual(board["black_score_rate"], 0.5)
        self.assertEqual(board["no_result_rate"], 0.25)
        self.assertEqual(board["extreme_margin_rate"], 0)
        self.assertIn("no_result_rate", board["warnings"])
        self.assertEqual(report["inventory"]["damaged_games"], 0)

    def test_incomplete_live_tree_is_pending_and_can_become_a_valid_game(self):
        damaged = "(;SZ[8]RE[0]BROKEN)"
        prefix = game(7) + "\n" + damaged + "\n(;SZ[19]KM[6.5]RE[B+1];B[aa"
        path = self.write("seed/sgfs/live.sgfs", prefix)

        first = self.health(window_games=4, minimum_games=2)
        second = self.health(window_games=4, minimum_games=2)

        self.assertEqual(first["inventory"]["damaged_games"], 1)
        self.assertEqual(first["inventory"]["pending_games"], 1)
        self.assertEqual(second["inventory"]["pending_games"], 1)
        self.assertEqual(first["by_board"]["19"]["cumulative_valid_games"], 0)
        path.write_text(prefix + "];W[bb])", encoding="utf-8")
        completed = self.health(window_games=4, minimum_games=2)
        self.assertEqual(completed["inventory"]["damaged_games"], 1)
        self.assertEqual(completed["inventory"]["pending_games"], 0)
        self.assertEqual(completed["inventory"]["valid_games"], 2)
        self.assertEqual(completed["by_board"]["19"]["cumulative_valid_games"], 1)

    def test_immutable_files_reuse_real_parsed_records_and_preserve_source_bytes(self):
        path = self.write("seed/sgfs/cache.sgfs", game(7, result="B+1") + "\n" + game(7, result="W+1"))
        before = path.read_bytes(), path.stat().st_mtime_ns

        with patch("weiqi.katago_mixed_health.parse_sgf_game", wraps=parse_sgf_game) as parser:
            first = self.health(window_games=2, minimum_games=2)
            second = self.health(window_games=2, minimum_games=2)

        self.assertEqual(parser.call_count, 2)
        self.assertEqual(first["inventory"]["parsed_files"], 1)
        self.assertEqual(second["inventory"]["parsed_files"], 0)
        self.assertEqual(second["inventory"]["cached_files"], 1)
        self.assertEqual(second["by_board"]["7"]["black_score_rate"], 0.5)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        cache = json.loads((self.root / ".mixed_health_inventory.json").read_text())
        self.assertIn("selfplay/seed/sgfs/cache.sgfs", cache["files"])

    def test_cache_invalidates_for_same_size_mtime_changes_and_same_mtime_appends(self):
        path = self.write("seed/sgfs/cache.sgfs", game(7, result="B+1") + "\n" + game(7, result="W+1"))
        self.health(window_games=4, minimum_games=2)
        original = path.stat()
        path.write_text(path.read_text().replace("SZ[7]", "SZ[8]", 1), encoding="utf-8")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))

        changed = self.health(window_games=4, minimum_games=2)

        self.assertEqual(path.stat().st_size, original.st_size)
        self.assertEqual(changed["inventory"]["parsed_files"], 1)
        self.assertEqual(changed["by_board"]["7"]["cumulative_valid_games"], 1)
        self.assertEqual(changed["by_board"]["8"]["cumulative_valid_games"], 1)
        changed_stat = path.stat()
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + game(19, 40))
        os.utime(path, ns=(changed_stat.st_atime_ns, changed_stat.st_mtime_ns))
        appended = self.health(window_games=4, minimum_games=2)
        self.assertEqual(appended["inventory"]["parsed_files"], 1)
        self.assertEqual(appended["by_board"]["19"]["cumulative_valid_games"], 1)
        path.unlink()
        removed = self.health(window_games=4, minimum_games=2)
        self.assertEqual(removed["inventory"]["valid_games"], 0)
        self.assertEqual(json.loads((self.root / ".mixed_health_inventory.json").read_text())["files"], {})

    def test_live_writer_change_during_example_read_keeps_the_snapshot_reportable(self):
        path = self.write("seed/sgfs/live.sgfs", "\n".join(["(;SZ[8]RE[W+1];B[];W[])"] * 2))
        read_text = Path.read_text
        reads = 0

        def read_with_live_replacement(current, *args, **kwargs):
            nonlocal reads
            if current == path:
                reads += 1
                if reads == 2:
                    # A real writer replaces the source after the example's stat.
                    current.write_text("(BROKEN)", encoding="utf-8")
            return read_text(current, *args, **kwargs)

        with patch.object(Path, "read_text", new=read_with_live_replacement):
            try:
                report = self.health((8,), window_games=2, minimum_games=2)
            except ValueError as error:
                self.fail(f"a source changing during the snapshot must not break health reporting: {error}")

        board = report["by_board"]["8"]
        self.assertEqual(board["window_games"], 2)
        self.assertEqual(board["early_double_pass_rate"], 1)
        self.assertTrue(all(example["source_changed_since_scan"] for example in board["examples"]))
        self.assertTrue(all("original_sgf" not in example for example in board["examples"]))

    def test_other_board_sizes_are_not_malformed_and_single_sgf_files_are_included(self):
        self.write("seed/sgfs/other.sgf", game(5))
        self.write("seed/sgfs/mixed.sgfs", game(7))

        report = self.health(window_games=4, minimum_games=2)

        self.assertEqual(report["inventory"]["valid_games"], 2)
        self.assertEqual(report["inventory"]["damaged_games"], 0)
        self.assertEqual(report["inventory"]["unconfigured_board_games"], {"5": 1})
        self.assertEqual(report["by_board"]["7"]["cumulative_valid_games"], 1)

    def test_invalid_sampling_settings_fail_before_writing_a_report(self):
        cases = [((7, 7), {}), ((), {}), ((True,), {}), ((7,), {"window_games": 0}),
                 ((7,), {"minimum_games": 0}), ((7,), {"window_games": 2, "minimum_games": 3})]
        for sizes, options in cases:
            with self.subTest(sizes=sizes, options=options):
                with self.assertRaises(ValueError):
                    self.health(sizes, **options)
        self.assertFalse((self.root / "health_by_board.json").exists())


if __name__ == "__main__":
    unittest.main()
