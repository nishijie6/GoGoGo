import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from weiqi.katago_mixed import MixedBoardRunner
from weiqi.katago_rl import KataGoRLRunnerError
from weiqi.rl_curriculum_openings import ensure_evaluation_opening_suite
from weiqi.training_monitor import MonitorStore


class PromotionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.profile = self.root / "mixed.json"
        self.profile.write_text(json.dumps({
            "schema_version": 1, "board_sizes": [9, 19], "board_weights": [1, 1],
            "shuffle_rows_per_file": 256, "promotion": {"every_cycles": 2},
            "training": {"schema_version": 1, "preset": "balanced", "overrides": {
                "game": {"board_size": 19},
                "network": {"channels": 128, "residual_blocks": 10},
                "runtime": {"output_directory": str(self.root / "run")},
            }},
        }))
        self.runner = self.new_runner()
        self.runner.ensure_manifest()
        self.baseline = self.export(100)
        self.record_cycle(1)

    def new_runner(self):
        return MixedBoardRunner(self.profile, project_root=self.root)

    def export(self, samples):
        path = self.runner.models_dir / f"gogogo-s{samples}-d100" / "model.bin.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(str(samples).encode())
        return path

    def record_cycle(self, cycle, samples=None):
        path = self.runner.metric_store.latest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"cycle": cycle}
        if samples is not None:
            record["trained_samples"] = samples
        path.write_text(json.dumps(record))

    def use_standard(self, **promotion):
        profile = json.loads(self.profile.read_text())
        profile["promotion"].update(standard="unified_v1", **promotion)
        profile["training"]["overrides"]["optimizer"] = {
            "batch_size": 8, "training_steps_per_iteration": 128,
        }
        self.profile.write_text(json.dumps(profile))
        try:
            self.runner = self.new_runner()
        except ValueError as error:
            self.fail(f"shared promotion standard must be supported: {error}")
        self.all_black = False
        self.runner._promotion_probe = self.probe

    def probe(self, model, root, board, games):
        sgfs = root / "selfplay" / "actor" / "sgfs"
        sgfs.mkdir(parents=True, exist_ok=True)
        moves = "".join(f";{'B' if n % 2 == 0 else 'W'}[aa]" for n in range(45))
        collapsed = model.parent.name == "gogogo-s200-d100" and self.all_black and board == 19
        (sgfs / f"{board}.sgfs").write_text("".join(
            f"(;SZ[{board}]RE[{'B' if collapsed or n % 2 == 0 else 'W'}+1.5]{moves})"
            for n in range(games)
        ))

    @staticmethod
    def match(score):
        def evaluate(runner, root, *, games_per_board, **kwargs):
            return {"status": "complete", "results_by_board": {
                str(board): {"games": games_per_board, "no_result_games": 0,
                             "damaged_games": 0, "pair_statistics": {
                                 "scores": [score] * (games_per_board // 2)}}
                for board in runner.mixed.board_sizes
            }}
        return evaluate

    def exercise(self, score, all_black=False):
        self.all_black = all_black
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(3)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(score)):
            self.runner.promotion.advance()
        return self.new_runner().promotion.read_state()

    def test_winner_becomes_champion_even_after_export_is_pruned_and_runner_restarts(self):
        state = self.exercise(1.0)
        (self.runner.models_dir / "gogogo-s200-d100" / "model.bin.gz").unlink()
        resumed = self.new_runner().promotion
        actor, = resumed.actor_models().glob("*/model.bin.gz")
        self.assertEqual(actor.read_bytes(), b"200")
        self.assertEqual(state["promotions"], 1)

    def test_all_black_on_one_board_vetoes_confirmation_winner(self):
        state = self.exercise(1.0, all_black=True)
        self.assertEqual(state["champion"], "gogogo-s100-d100")
        self.assertEqual(state["promotions"], 0)
        self.assertIn("19:black_score_rate", state["last_result"]["health_reasons"])

    def test_extreme_margins_are_recorded_without_vetoing_promotion(self):
        normal_probe = self.probe

        def extreme_probe(model, root, board, games):
            normal_probe(model, root, board, games)
            path = root / "selfplay/actor/sgfs" / f"{board}.sgfs"
            path.write_text(path.read_text().replace("+1.5]", "+999.5]"))

        self.probe = extreme_probe
        state = self.exercise(1.0)
        self.assertEqual(state["promotions"], 1)
        self.assertEqual(state["last_result"]["health_reasons"], [])
        health = json.loads(Path(state["last_result"]["health_report"]).read_text())
        for row in health["by_board"].values():
            self.assertEqual(row["extreme_margin_rate"], 1.0)
            self.assertIn("extreme_margin_rate", row["warnings"])

    def test_losing_candidate_cannot_enter_actor_directory(self):
        state = self.exercise(0.0)
        self.assertEqual(state["last_result"]["outcome"], "screen_rejected")
        actor, = self.new_runner().promotion.actor_models().glob("*/model.bin.gz")
        self.assertEqual(actor.read_bytes(), b"100")

    def test_configured_match_boards_can_promote_at_55_percent_without_confidence_gate(self):
        profile = json.loads(self.profile.read_text())
        profile["promotion"].update(evaluation_boards=[19], require_confidence=False, parallel_pairs=2)
        self.profile.write_text(json.dumps(profile))
        self.runner = self.new_runner()
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(3)

        def evaluate(runner, root, *, boards, games_per_board, parallel_pairs, **kwargs):
            self.assertEqual(tuple(boards), (19,))
            self.assertEqual(parallel_pairs, 2)
            pairs = [0.5] * 6 if games_per_board == 12 else [1.0] * 11 + [0.0] * 9
            return {"status": "complete", "results_by_board": {"19": {
                "games": games_per_board, "no_result_games": 0, "damaged_games": 0,
                "pair_statistics": {"scores": pairs},
            }}}

        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=evaluate):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        result = state["last_result"]
        self.assertEqual(state["promotions"], 1)
        self.assertEqual(result["confirmation_score"], .55)
        self.assertLess(result["confirmation_lower"], .5)
        self.assertEqual(result["evaluation_boards"], [19])
        self.assertFalse(result["require_confidence"])
        self.assertEqual(result["parallel_pairs"], 2)
        self.assertEqual(set(json.loads(Path(result["health_report"]).read_text())["by_board"]), {"9", "19"})

    def test_live_all_black_stops_before_shuffle_and_stays_stopped_after_restart(self):
        runner = self.runner
        self.all_black = False
        runner._promotion_probe = self.probe
        runner.promotion.advance()
        # The old collapse had unanimous 19x19 black wins despite games ending normally.
        sgfs = runner.selfplay_dir / self.baseline.parent.name / "sgfs"
        sgfs.mkdir(parents=True)
        moves = "".join(f";{'B' if n % 2 == 0 else 'W'}[aa]" for n in range(45))
        (sgfs / "collapse.sgfs").write_text(f"(;SZ[19]RE[B+1.5]{moves})" * 24)
        runner.selfplay_config.parent.mkdir(parents=True, exist_ok=True)
        runner.selfplay_config.write_text("bSizes = 19\n")
        with patch.object(runner, "_require_runtime"), patch.object(runner, "_run", return_value="") as run:
            with self.assertRaisesRegex(KataGoRLRunnerError, "19:black_score_rate"):
                runner.selfplay()
        command = run.call_args.args[0]
        pool = command[command.index("-models-dir") + 1]
        self.assertEqual([model.read_bytes() for model in pool.glob("*/model.bin.gz")], [b"100"])
        with self.assertRaisesRegex(KataGoRLRunnerError, "健康停机"):
            self.new_runner().promotion.actor_models()

    def test_interrupted_confirmation_keeps_champion_and_resumes_without_double_count(self):
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(3)
        with patch("weiqi.katago_promotion.evaluate_mixed",
                   side_effect=[self.match(1.0)(self.runner, None, games_per_board=12), KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.promotion.advance()
        profile = json.loads(self.profile.read_text())
        profile["promotion"].update(evaluation_boards=[19], require_confidence=False, parallel_pairs=4)
        self.profile.write_text(json.dumps(profile))
        resumed = self.new_runner()
        self.assertEqual(resumed.promotion.read_state()["champion"], "gogogo-s100-d100")
        resumed._promotion_probe = self.probe
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(1.0)):
            resumed.promotion.advance()
            resumed.promotion.advance()
        self.assertEqual(resumed.promotion.read_state()["promotions"], 1)
        result = resumed.promotion.read_state()["last_result"]
        self.assertEqual(result["evaluation_boards"], [9, 19])
        self.assertTrue(result["require_confidence"])
        self.assertEqual(result["parallel_pairs"], 1)

    def test_engine_error_records_failed_evaluation_and_leaves_champion_active(self):
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(3)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=RuntimeError("engine failed")):
            self.runner.promotion.advance()
        state = self.new_runner().promotion.read_state()
        self.assertEqual(state["last_result"]["outcome"], "evaluation_failed")
        self.assertEqual(state["next_cycle"], 5)
        self.assertEqual(state["champion"], "gogogo-s100-d100")
        self.assertEqual(state["promotions"], 0)

    def test_legacy_profile_cannot_bypass_persisted_promotion_state(self):
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        profile = json.loads(self.profile.read_text())
        profile.pop("promotion")
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(KataGoRLRunnerError, "晋级"):
            self.new_runner().selfplay()

    def test_smoke_cannot_feed_untested_exports_into_existing_run(self):
        with patch.object(self.runner, "_require_runtime"):
            with self.assertRaisesRegex(KataGoRLRunnerError, "独立目录"):
                self.runner.selfplay(smoke=True)

    def test_screen_and_confirmation_use_different_positions(self):
        screen = ensure_evaluation_opening_suite(self.root / "screen", board_size=9, opening_count=6)
        confirm = ensure_evaluation_opening_suite(self.root / "confirm", board_size=9, opening_count=20,
                                                  opening_offset=6)
        def moves(paths):
            return {tuple(json.loads((path / "opening.startposes.txt").read_text())["moveLocs"])
                    for path in paths}
        self.assertFalse(moves(screen) & moves(confirm))

    def test_shared_standard_uses_samples_instead_of_hardware_dependent_cycles(self):
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        self.assertEqual(self.runner.promotion.read_state()["next_samples"], 51300)
        self.export(51300)
        self.record_cycle(300, 51299)
        self.runner.promotion.advance()
        self.assertIsNone(self.runner.promotion.read_state()["last_result"])
        self.record_cycle(301, 51300)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(.75)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 1)
        self.assertEqual(state["next_samples"], 102500)

    def test_standard_migration_preserves_remaining_legacy_due_work_once(self):
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.use_standard()
        # A hardware batch change must not stretch work left on the old schedule.
        self.runner.config = replace(self.runner.config, optimizer=replace(
            self.runner.config.optimizer, batch_size=32))
        self.record_cycle(2, 2024)
        self.runner.promotion.advance()
        self.assertEqual(self.runner.promotion.read_state()["next_samples"], 3048)
        self.export(3048)
        self.record_cycle(200, 3047)
        self.runner.promotion.advance()
        self.assertIsNone(self.runner.promotion.read_state()["last_result"])
        self.record_cycle(201, 3048)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(.75)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 1)
        self.assertEqual(state["next_samples"], 54248)

    def test_shared_policy_records_low_confidence_without_vetoing_55_percent(self):
        self.use_standard(require_confidence=True, win_rate=.9)
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        self.export(51300)
        self.record_cycle(2, 51300)

        def evaluate(runner, root, *, boards, games_per_board, **kwargs):
            pairs = [1.0] * 11 + [0.0] * 9 if games_per_board == 40 else [.5] * 6
            return {"status": "complete", "results_by_board": {str(board): {
                "games": games_per_board, "no_result_games": 0, "damaged_games": 0,
                "pair_statistics": {"scores": pairs},
            } for board in boards}}

        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=evaluate):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 1)
        result = state["last_result"]
        self.assertEqual(result["confirmation_score"], .55)
        self.assertLess(result["confirmation_lower"], .5)
        self.assertFalse(result["require_confidence"])
        self.assertEqual(result["win_rate"], .55)

    def test_standard_pending_freezes_full_policy_and_match_protocol(self):
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        self.export(51300)
        self.record_cycle(2, 51300)
        with patch("weiqi.katago_promotion.evaluate_mixed",
                   side_effect=[self.match(.75)(self.runner, None, games_per_board=12), KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.promotion.advance()
        profile = json.loads(self.profile.read_text())
        profile["promotion"].pop("standard")
        profile["promotion"].update(screen_games_per_board=2, confirmation_games_per_board=2, visits=1)
        self.profile.write_text(json.dumps(profile))
        self.runner = self.new_runner()
        self.runner._promotion_probe = self.probe

        def valid_evaluate(runner, root, *, boards, games_per_board, visits, **kwargs):
            self.assertEqual(visits, 32)
            self.assertEqual(games_per_board, 12 if root.name == "screen" else 40)
            pairs = [1.0] * 11 + [0.0] * 9 if games_per_board == 40 else [.5] * 6
            return {"status": "complete", "results_by_board": {str(board): {
                "games": games_per_board, "no_result_games": 0, "damaged_games": 0,
                "pair_statistics": {"scores": pairs},
            } for board in boards}}

        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=valid_evaluate):
            self.runner.promotion.advance()
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 1)
        self.assertEqual(state["last_result"]["training_standard"]["name"], "unified_v1")
        self.assertNotIn("training_standard", state)
        self.assertNotIn("next_samples", state)
        snapshot = MonitorStore(self.root).snapshot("run")
        self.assertIsNone(snapshot["training_standard"])
        self.assertIsNone(snapshot["evaluation_schedule"])
        self.assertEqual(snapshot["championship"]["next_cycle"], 4)
        self.export(60000)
        self.record_cycle(3, 60000)
        self.runner.promotion.advance()
        self.assertEqual(self.runner.promotion.read_state()["last_result"]["candidate"],
                         "gogogo-s51300-d100")
        self.record_cycle(4, 60001)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(0.)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["last_result"]["candidate"], "gogogo-s60000-d100")
        self.assertNotIn("training_standard", state["last_result"])

    def test_shared_policy_rejects_a_regressing_board_despite_60_percent_overall(self):
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        self.export(51300)
        self.record_cycle(2, 51300)

        def evaluate(runner, root, *, boards, games_per_board, **kwargs):
            return {"status": "complete", "results_by_board": {str(board): {
                "games": games_per_board, "no_result_games": 0, "damaged_games": 0,
                "pair_statistics": {"scores": ([.5] * 6 if games_per_board == 12
                    else [1.] * (8 if board == 9 else 16) + [0.] * (12 if board == 9 else 4))},
            } for board in boards}}

        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=evaluate):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 0)
        self.assertEqual(state["last_result"]["confirmation_score"], .6)
        self.assertIn("board_score", state["last_result"]["promotion_reasons"])

    def test_selecting_standard_preserves_pending_legacy_custom_protocol_and_gate(self):
        profile = json.loads(self.profile.read_text())
        profile["promotion"].update(screen_games_per_board=2, confirmation_games_per_board=4, visits=7)
        self.profile.write_text(json.dumps(profile))
        self.runner = self.new_runner()
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(3)
        with patch("weiqi.katago_promotion.evaluate_mixed",
                   side_effect=[self.match(.75)(self.runner, None, games_per_board=2), KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.promotion.advance()
        self.use_standard()

        def evaluate(runner, root, *, games_per_board, visits, **kwargs):
            self.assertEqual(visits, 7)
            self.assertEqual(games_per_board, 2 if root.name == "screen" else 4)
            return self.match(.75)(runner, root, games_per_board=games_per_board)

        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=evaluate):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 0)
        self.assertEqual(state["last_result"]["outcome"], "confirmation_rejected")
        self.assertTrue(state["last_result"]["require_confidence"])
        self.assertNotIn("training_standard", state["last_result"])

    def test_shared_policy_preserves_black_collapse_veto(self):
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(2, 51300)
        self.all_black = True
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(1.0)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 0)
        self.assertEqual(state["last_result"]["outcome"], "health_rejected")
        self.assertIn("19:black_score_rate", state["last_result"]["health_reasons"])

    def test_shared_policy_observes_extreme_margins_without_veto(self):
        normal_probe = self.probe

        def extreme_probe(model, root, board, games):
            normal_probe(model, root, board, games)
            path = root / "selfplay/actor/sgfs" / f"{board}.sgfs"
            path.write_text(path.read_text().replace("+1.5]", "+999.5]"))

        self.probe = extreme_probe
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        self.export(51300)
        self.record_cycle(2, 51300)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(.75)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["promotions"], 1)
        health = json.loads(Path(state["last_result"]["health_report"]).read_text())
        self.assertEqual(health["training_standard"]["name"], "unified_v1")
        for row in health["by_board"].values():
            self.assertEqual(row["extreme_margin_rate"], 1.)
            self.assertEqual(row["health_reasons"], [])

    def test_shared_live_black_collapse_halts_and_survives_restart(self):
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        sgfs = self.runner.selfplay_dir / self.baseline.parent.name / "sgfs"
        sgfs.mkdir(parents=True)
        moves = "".join(f";{'B' if n % 2 == 0 else 'W'}[aa]" for n in range(45))
        (sgfs / "collapse.sgfs").write_text(f"(;SZ[19]RE[B+1.5]{moves})" * 24)
        with self.assertRaisesRegex(KataGoRLRunnerError, "19:black_score_rate"):
            self.runner.promotion.inspect_live_selfplay()
        with self.assertRaisesRegex(KataGoRLRunnerError, "健康停机"):
            self.new_runner().promotion.actor_models()

    def test_empty_standard_run_waits_for_a_trained_export_without_metrics(self):
        self.use_standard()
        self.baseline.unlink()
        self.runner.metric_store.latest_path.unlink()
        self.runner.promotion.advance()
        self.assertIsNone(self.runner.promotion.read_state())

    def test_selecting_standard_after_legacy_round_anchors_next_schedule_to_actual_samples(self):
        self.all_black = False
        self.runner._promotion_probe = self.probe
        self.runner.promotion.advance()
        self.export(200)
        self.record_cycle(3)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.runner.promotion.advance()
        self.use_standard()
        self.record_cycle(9, 9000)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(.75)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["training_standard"], "unified_v1")
        self.assertEqual(state["next_samples"], 60200)
        self.assertNotIn("training_standard", state["last_result"])
        snapshot = MonitorStore(self.root).snapshot("run")
        self.assertEqual(snapshot["training_standard"]["name"], "unified_v1")
        self.assertEqual(snapshot["evaluation_schedule"], {
            "trained_samples": 9000, "next_samples": 60200, "remaining_samples": 51200,
        })
        self.export(10000)
        self.record_cycle(10, 10000)
        self.runner.promotion.advance()
        self.assertEqual(self.runner.promotion.read_state()["last_result"]["candidate"],
                         "gogogo-s200-d100")

    def test_removing_standard_between_rounds_updates_schedule_and_monitor_immediately(self):
        self.use_standard()
        self.record_cycle(1, 100)
        self.runner.promotion.advance()
        profile = json.loads(self.profile.read_text())
        profile["promotion"].pop("standard")
        self.profile.write_text(json.dumps(profile))
        self.runner = self.new_runner()
        self.runner._promotion_probe = self.probe
        self.export(200)
        self.record_cycle(2, 200)
        self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertIsNone(state["last_result"])
        self.assertNotIn("training_standard", state)
        self.assertNotIn("next_samples", state)
        snapshot = MonitorStore(self.root).snapshot("run")
        self.assertIsNone(snapshot["training_standard"])
        self.assertIsNone(snapshot["evaluation_schedule"])
        self.assertEqual(snapshot["championship"]["next_cycle"], 3)
        self.record_cycle(3, 201)
        with patch("weiqi.katago_promotion.evaluate_mixed", side_effect=self.match(0.)):
            self.runner.promotion.advance()
        state = self.runner.promotion.read_state()
        self.assertEqual(state["last_result"]["candidate"], "gogogo-s200-d100")
        self.assertNotIn("training_standard", state["last_result"])

    def test_trained_export_without_sample_metrics_cannot_start_standard_evaluation(self):
        self.use_standard()
        self.runner.metric_store.latest_path.unlink()
        with self.assertRaisesRegex(KataGoRLRunnerError, "trained_samples"):
            self.runner.promotion.advance()

    def test_standard_bootstrap_warmup_does_not_require_sample_metrics(self):
        self.use_standard()
        self.runner.metric_store.latest_path.unlink()
        self.runner.bootstrap.start()
        self.runner.promotion.advance()
        self.assertIsNone(self.runner.promotion.read_state())
