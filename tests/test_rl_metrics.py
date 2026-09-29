"""Tests for persistent KataGo reinforcement-learning telemetry."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weiqi.rl_metrics import (
    RLMetricStore,
    parse_model_name,
    parse_selfplay_output,
    parse_sgfs,
    render_dashboard,
    rolling_health_metrics,
)


class RLMetricParsingTests(unittest.TestCase):
    def test_model_name_contains_samples_and_data_rows(self) -> None:
        self.assertEqual(parse_model_name("gogogo-s1083392-d171432"), (1083392, 171432))
        self.assertIsNone(parse_model_name("model.bin.gz"))

    def test_selfplay_summary_uses_final_values(self) -> None:
        output = """
Final games finished: 4
Final data rows: 100
Final games finished: 128
Final moves played: 7052
Final data rows: 1921
Final NN rows: 392422
Final NN batches: 31576
Final NN avg batch size: 12.4279
Total selfplay runtime (seconds): 31.938
"""
        self.assertEqual(
            parse_selfplay_output(output),
            {
                "selfplay_games": 128,
                "selfplay_moves": 7052,
                "selfplay_data_rows": 1921,
                "selfplay_nn_rows": 392422,
                "selfplay_nn_batches": 31576,
                "selfplay_avg_batch_size": 12.4279,
                "selfplay_seconds": 31.938,
            },
        )

    def test_sgf_results_moves_and_degenerate_games(self) -> None:
        collection = "\n".join(
            (
                r"(;FF[4]GM[1]SZ[9]RE[B+31.5]C[escaped \] ;B[]];B[];W[])",
                r"(;FF[4]GM[1]SZ[9]RE[W+R];B[aa];W[bb])",
                r"(;FF[4]GM[1]SZ[9]RE[0];B[cc])",
                r"(;FF[4]GM[1]SZ[9];B[dd];W[])",
                r"(;FF[4]GM[1]SZ[9]RE[W+2.5];B[ee];W[ff]",
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "games.sgfs"
            path.write_text(collection, encoding="utf-8")
            metrics = parse_sgfs(path)

        self.assertEqual(metrics["sgf_games"], 4)
        self.assertEqual(metrics["sgf_entries"], 5)
        self.assertEqual(metrics["black_wins"], 1)
        self.assertEqual(metrics["white_wins"], 1)
        self.assertEqual(metrics["draws"], 1)
        self.assertEqual(metrics["no_result_games"], 1)
        self.assertEqual(metrics["damaged_games"], 1)
        self.assertEqual(metrics["opening_double_pass_games"], 1)
        self.assertEqual(metrics["extreme_result_games"], 1)
        self.assertEqual(metrics["last_game_result"], "无结果")
        self.assertAlmostEqual(metrics["average_score_margin"], 31.5)
        self.assertAlmostEqual(metrics["average_moves"], 1.75)
        self.assertAlmostEqual(metrics["black_win_rate"], 0.5)
        self.assertAlmostEqual(metrics["invalid_game_rate"], 0.4)

    def test_rolling_health_requires_ten_cycles_and_1280_games(self) -> None:
        records = [
            {
                "sgf_games": 128,
                "sgf_entries": 128,
                "black_wins": 64,
                "white_wins": 64,
                "draws": 0,
                "no_result_games": 0,
                "damaged_games": 0,
                "opening_double_pass_games": 1,
                "extreme_result_games": 6,
            }
            for _ in range(10)
        ]
        incomplete = rolling_health_metrics(records[:9])
        self.assertFalse(incomplete["health_window_complete"])
        self.assertFalse(incomplete["health_passed"])

        health = rolling_health_metrics(records)
        self.assertTrue(health["health_window_complete"])
        self.assertTrue(health["health_passed"])
        self.assertEqual(health["health_window_games"], 1280)
        self.assertAlmostEqual(health["health_black_win_rate"], 0.5)
        self.assertAlmostEqual(health["health_opening_double_pass_rate"], 10 / 1280)
        self.assertAlmostEqual(health["health_extreme_result_rate"], 60 / 1280)

        unhealthy = [dict(record, extreme_result_games=7) for record in records]
        self.assertFalse(rolling_health_metrics(unhealthy)["health_passed"])


class RLMetricStoreTests(unittest.TestCase):
    def test_sync_keeps_history_after_old_models_are_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            first = "gogogo-s1024-d128"
            first_dir = run_root / "models" / first
            first_dir.mkdir(parents=True)
            (first_dir / "model.bin.gz").write_bytes(b"first")
            store = RLMetricStore(run_root)
            self.assertEqual([record["model"] for record in store.sync()], [first])

            (first_dir / "model.bin.gz").unlink()
            second = "gogogo-s2048-d256"
            second_dir = run_root / "models" / second
            second_dir.mkdir(parents=True)
            (second_dir / "model.bin.gz").write_bytes(b"second")
            old_sgf = run_root / "selfplay" / first / "sgfs" / "games.sgfs"
            old_sgf.parent.mkdir(parents=True)
            old_sgf.write_text(
                "(;FF[4]GM[1]SZ[9]RE[B+0.5];B[aa];W[bb])\n",
                encoding="utf-8",
            )

            records = store.sync()
            self.assertEqual(
                [record["model"] for record in records],
                [first, second],
            )
            self.assertEqual(records[0]["black_wins"], 1)

    def test_sync_backfills_and_writes_all_dashboard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            model = "gogogo-s4096-d2048"
            model_dir = run_root / "models" / model
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin.gz").write_bytes(b"model")
            checkpoint = run_root / "torchmodels_toexport" / model / "model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            sgf = run_root / "selfplay" / model / "sgfs" / "games.sgfs"
            sgf.parent.mkdir(parents=True)
            sgf.write_text(
                "(;FF[4]GM[1]SZ[9]RE[B+6.5];B[aa];W[bb])\n"
                "(;FF[4]GM[1]SZ[9]RE[W+1.5];B[cc];W[dd])\n",
                encoding="utf-8",
            )
            store = RLMetricStore(run_root)

            with patch(
                "weiqi.rl_metrics.checkpoint_metrics",
                return_value={
                    "trained_samples": 4096,
                    "data_rows": 2048,
                    "loss": 42.25,
                    "policy_loss": 3.5,
                    "value_loss": 0.8,
                    "score_loss": 0.6,
                    "policy_accuracy": 0.12,
                },
            ):
                records = store.sync(
                    live_model=model,
                    live_metrics={"selfplay_games": 128, "cycle_seconds": 61.5},
                )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["cycle"], 1)
            self.assertEqual(records[0]["source"], "completed-cycle")
            self.assertEqual(records[0]["selfplay_games"], 128)
            self.assertEqual(records[0]["black_wins"], 1)
            self.assertEqual(records[0]["white_wins"], 1)
            self.assertAlmostEqual(records[0]["average_score_margin"], 4.0)
            self.assertTrue(store.dashboard_path.is_file())
            self.assertTrue(store.csv_path.is_file())
            self.assertTrue(store.jsonl_path.is_file())
            self.assertTrue(store.latest_path.is_file())

            latest = json.loads(store.latest_path.read_text(encoding="utf-8"))
            self.assertEqual(latest["model"], model)
            with store.csv_path.open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[0]["trained_samples"], "4096")
            dashboard = store.dashboard_path.read_text(encoding="utf-8")
            self.assertIn("KataGo 9×9 强化学习监控", dashboard)
            self.assertIn("history.csv", dashboard)
            self.assertIn('role="img"', dashboard)
            self.assertIn("末局结果", dashboard)
            self.assertIn("自对弈健康趋势", dashboard)

    def test_render_dashboard_handles_no_models(self) -> None:
        dashboard = render_dashboard([], "2026-08-30T23:00:00+08:00")
        self.assertIn("尚无已接纳模型", dashboard)
        self.assertIn('content="15"', dashboard)


if __name__ == "__main__":
    unittest.main()
