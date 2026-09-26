"""Independent evaluation signals and data integrity for weak 9x9 models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import tempfile
import unittest
from pathlib import Path

from weiqi.engine import BLACK, WHITE, GoGame
from weiqi.rl.eval_positions import (
    atomic_json, canonical_hash, generate_suite, read_suite, replay_position,
)
from weiqi.rl.eval_stats import confirmed_improvement, paired_confidence, paired_sign_test

@dataclass
class MatchRecord:
    index: int
    seed: int
    candidate_color: int
    winner: int | None
    reason: str


def result(index: int, seed: int, candidate_wins: bool,
           *, truncated: bool = False) -> MatchRecord:
    color = BLACK if index % 2 == 0 else WHITE
    winner = None if truncated else (color if candidate_wins else WHITE if color == BLACK else BLACK)
    return MatchRecord(index, seed, color, winner,
                       "length_limit" if truncated else "two_passes")


class PairedEvaluationTests(unittest.TestCase):
    def test_color_swapped_openings_are_the_statistical_unit(self):
        games = [result(0, 7, True), result(1, 7, False),
                 result(2, 8, True), result(3, 8, True)]
        stats = paired_confidence(games)
        self.assertEqual(stats["complete_pairs"], 2)
        self.assertEqual(stats["mean_score"], 0.75)
        self.assertLessEqual(stats["lower"], 0.75)
        self.assertGreaterEqual(stats["upper"], 0.75)

    def test_truncation_cannot_count_as_a_winning_pair_or_promotion(self):
        games = [result(0, 7, True), result(1, 7, True),
                 result(2, 8, True, truncated=True), result(3, 8, True)]
        stats = paired_confidence(games)
        self.assertEqual((stats["complete_pairs"], stats["truncated_pairs"]), (1, 1))
        self.assertFalse(confirmed_improvement({"truncated": 1, "score_rate": 0.75,
                                                "paired": stats}, threshold=0.55))
        with self.assertRaisesRegex(ValueError, "colors are not paired"):
            paired_confidence([result(0, 7, True), result(1, 8, True)])

    def test_perfect_short_screen_is_not_a_formal_promotion(self):
        stats = paired_confidence([result(i, i // 2, True) for i in range(4)])
        self.assertEqual(stats["mean_score"], 1.0)
        self.assertFalse(confirmed_improvement({"truncated": 0, "score_rate": 1.0,
                                                "paired": stats}, threshold=0.55))

    def test_strong_complete_match_can_pass_a_paired_confirmation(self):
        stats = paired_confidence([result(i, i // 2, True) for i in range(20)])
        self.assertGreater(stats["lower"], 0.5)
        self.assertTrue(confirmed_improvement({"truncated": 0, "score_rate": 1.0,
                                               "paired": stats}, threshold=0.55))

    def test_paired_sign_gate_uses_complete_opening_pairs_without_claiming_hoeffding_lower_bound(self):
        games = [result(i, i // 2, i < 18) for i in range(20)]
        conservative = paired_confidence(games)
        sign = paired_sign_test(games)
        self.assertLess(conservative["lower"], 0.5)
        self.assertEqual((sign["wins"], sign["losses"], sign["ties"]), (9, 1, 0))
        self.assertAlmostEqual(sign["p_value"], 11 / 1024)
        summary = {"games": 20, "wins": 18, "truncated": 0, "score_rate": 0.9,
                   "paired": conservative, "paired_sign": sign}
        self.assertTrue(confirmed_improvement(summary, threshold=0.55,
                                              method="paired_sign"))
        games[0] = result(0, 0, True, truncated=True)
        summary.update(truncated=1, paired_sign=paired_sign_test(games))
        self.assertFalse(confirmed_improvement(summary, threshold=0.55,
                                               method="paired_sign"))

    def test_paired_sign_gate_rejects_ambiguous_small_samples(self):
        games = [result(i, i // 2, i < 16) for i in range(20)]
        sign = paired_sign_test(games)
        self.assertEqual((sign["wins"], sign["losses"]), (8, 2))
        self.assertGreater(sign["p_value"], 0.05)
        self.assertFalse(confirmed_improvement(
            {"games": 20, "wins": 16, "truncated": 0, "score_rate": 0.8,
             "paired": paired_confidence(games), "paired_sign": sign},
            threshold=0.55, method="paired_sign"))

    def test_default_40_game_confirmation_can_accept_a_clear_paired_advantage(self):
        games = [result(i, i // 2, i < 30) for i in range(40)]
        sign = paired_sign_test(games)
        self.assertEqual((sign["wins"], sign["losses"], sign["ties"]), (15, 5, 0))
        self.assertLess(sign["p_value"], 0.05)
        conservative = paired_confidence(games)
        self.assertLess(conservative["lower"], 0.5)
        self.assertTrue(confirmed_improvement({
            "truncated": 0, "score_rate": 0.75,
            "paired": conservative, "paired_sign": sign,
        }, threshold=0.55, method="paired_sign"))


class FixedPositionTests(unittest.TestCase):
    def test_suite_is_reproducible_legal_and_balanced_by_color(self):
        first = generate_suite(count=8, seed=99)
        again = generate_suite(count=8, seed=99)
        self.assertEqual(canonical_hash(first), canonical_hash(again))
        self.assertEqual({record["to_play"] for record in first["positions"]}, {BLACK, WHITE})
        self.assertEqual({record["phase"] for record in first["positions"]},
                         {"opening", "middle", "endgame"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "positions.json"
            atomic_json(first, path)
            loaded = read_suite(path, size=9, komi=6.5)
        self.assertEqual(loaded, first)
        for record in first["positions"]:
            game = replay_position(record, size=9, komi=6.5)
            self.assertFalse(game.game_over)
            self.assertTrue(list(game.legal_moves()))

    def test_board_or_move_history_tampering_is_rejected(self):
        record = generate_suite(count=4, seed=99)["positions"][0]
        changed = {**record, "board": [row[:] for row in record["board"]]}
        changed["board"][0][0] = 3
        with self.assertRaisesRegex(ValueError, "mismatched board"):
            replay_position(changed, size=9, komi=6.5)
        changed = {**record, "moves": [move[:] for move in record["moves"]]}
        changed["moves"][0][0] = WHITE
        with self.assertRaisesRegex(ValueError, "player history"):
            replay_position(changed, size=9, komi=6.5)


class KataGoLabelTests(unittest.TestCase):
    def test_labels_use_exact_move_info_and_black_score_perspective(self):
        from benchmark_teacher import _label_position
        game = GoGame(9, record_undo=False)
        game.play(0, 0)
        policy = [-1.0] + [1 / 81] * 81
        response = {"rootInfo": {"currentPlayer": "W", "visits": 64,
                                 "scoreLead": 999.0},
                    "policy": policy,
                    "moveInfos": [{"move": "B9", "order": 0, "scoreLead": 2.0}]}
        label = _label_position(game, response)
        self.assertEqual(label["score_lead_black_by_action"], {"1": 2.0})
        self.assertNotIn(999.0, label["score_lead_black_by_action"].values())
        self.assertEqual(label["top_action"], 1)
        response["policy"][0] = 0.01
        with self.assertRaisesRegex(ValueError, "illegal local action"):
            _label_position(game, response)

    def test_analysis_query_opt_in_keeps_history_and_requests_policy(self):
        from unittest.mock import patch
        from weiqi.katago import KataGoEngine, KataGoSettings
        with tempfile.TemporaryDirectory() as temporary:
            exe, model = Path(temporary) / "katago.exe", Path(temporary) / "model.bin.gz"
            exe.write_bytes(b"test")
            model.write_bytes(b"test")
            engine = KataGoEngine(KataGoSettings(executable=str(exe), model=str(model)))
            with patch.object(engine, "_send_analysis_query", return_value={"rootInfo": {}}) as send:
                engine.analyze_position(GoGame(9), max_visits=7, include_policy=True,
                                        preserve_history=True, include_ownership=False)
            query = send.call_args.args[1]
            self.assertEqual(query["maxVisits"], 7)
            self.assertEqual(query["boardXSize"], 9)
            self.assertTrue(query["includePolicy"])
            self.assertFalse(query["overrideSettings"]["ignorePreRootHistory"])
            engine.close()


class EvaluationOverrideTests(unittest.TestCase):
    def test_resumed_policy_changes_only_opponents_and_confirmation(self):
        import json
        from train import with_training_policy
        from weiqi.rl_config import resolve_rl_training_config

        previous = resolve_rl_training_config(overrides={
            "self_play": {"champion_fraction": 0.5, "milestone_fraction": 0.0},
            "evaluation": {"games": 20, "promotion_test": "paired_hoeffding",
                           "confirmation_max_game_length_factor": 2.5},
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.json"
            path.write_text(json.dumps({"self_play": {"champion_fraction": 0.25,
                                                      "milestone_fraction": 0.25},
                                        "evaluation": {"games": 40,
                                                       "promotion_test": "paired_sign",
                                                       "confirmation_max_game_length_factor": 4.0}}),
                            encoding="utf-8")
            adjusted = with_training_policy(previous, path)
            self.assertEqual((adjusted.self_play.champion_fraction,
                              adjusted.self_play.milestone_fraction), (0.25, 0.25))
            self.assertEqual((adjusted.evaluation.games,
                              adjusted.evaluation.promotion_test), (40, "paired_sign"))
            self.assertEqual(adjusted.optimizer, previous.optimizer)
            self.assertEqual(adjusted.network, previous.network)
            self.assertEqual(adjusted.game, previous.game)
            path.write_text(json.dumps({"network": {"channels": 128}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported policy section"):
                with_training_policy(previous, path)
            path.write_text(json.dumps({"self_play": {"workers": 99}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported policy field"):
                with_training_policy(previous, path)

    def test_resumed_run_can_adjust_match_budget_without_changing_model(self):
        import json
        from train import with_evaluation_overrides
        from weiqi.rl_config import resolve_rl_training_config

        previous = resolve_rl_training_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            path.write_text(json.dumps({"games": 100, "full_every_iterations": 5}),
                            encoding="utf-8")
            adjusted = with_evaluation_overrides(previous, path)
            self.assertEqual(adjusted.evaluation.games, 100)
            self.assertEqual(adjusted.evaluation.full_every_iterations, 5)
            self.assertEqual(adjusted.network, previous.network)
            self.assertEqual(adjusted.optimizer, previous.optimizer)
            self.assertEqual(adjusted.self_play, previous.self_play)
            path.write_text(json.dumps({"network": {"channels": 128}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "未知字段"):
                with_evaluation_overrides(previous, path)
            path.write_text(json.dumps({"games": 3}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "偶数"):
                with_evaluation_overrides(previous, path)

    def test_trend_warns_when_teacher_labels_change_under_the_same_model(self):
        import json
        from report_evaluation import render
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number, label_digest in ((1, "first-label-set"), (2, "second-label-set")):
                path = root / "iterations" / f"{number:06d}" / "summary.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "iteration": number, "training_steps": number, "replay_samples": 32,
                    "position_quality": {
                        "status": "measured", "suite_sha256": "same-suite",
                        "teacher_labels_sha256": label_digest,
                        "teacher_model_sha256": "same-model", "teacher_engine_sha256": "same-engine",
                        "teacher_config_sha256": "same-config", "teacher_visits": 64,
                        "candidate_visits": 8, "groups": {"all": {
                            "mean_teacher_log_loss": 2.0, "teacher_top_move_rate": 0.2,
                            "score_delta_coverage": 2, "positions": 8}},
                    }, "evaluation": {"status": "scheduled"}, "promoted": False,
                }), encoding="utf-8")
            report = render(root)
        self.assertIn("评测局面、KataGo 配置或候选搜索预算记录不一致", report)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"),
                     "Optional training dependencies are absent")
class FrozenOpponentTests(unittest.TestCase):
    def test_frozen_anchor_rejects_replacement_and_excludes_candidate_itself(self):
        import torch
        from weiqi.rl.eval_pool import save_anchor, select_anchors
        from weiqi.rl.network import PolicyValueNet
        from weiqi.rl.storage import cpu_state, fingerprint
        from weiqi.rl_config import resolve_rl_training_config
        config = resolve_rl_training_config(overrides={"network": {"channels": 8,
                                                                     "residual_blocks": 1}})
        model = PolicyValueNet(config)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = save_anchor(directory, model, config, kind="accepted", iteration=0)
            checksum = fingerprint(cpu_state(model))
            self.assertEqual(select_anchors(directory, config, candidate_sha256=checksum), [])
            with torch.no_grad():
                next(model.parameters()).add_(0.25)
            self.assertEqual(len(select_anchors(
                directory, config, candidate_sha256=fingerprint(cpu_state(model)))), 1)
            with self.assertRaisesRegex(ValueError, "changed"):
                save_anchor(directory, model, config, kind="accepted", iteration=0)
            self.assertTrue(path.is_file())
