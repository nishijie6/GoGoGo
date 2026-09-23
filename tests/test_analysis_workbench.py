"""Tests at the three-entry-point analysis-workbench interface."""

from __future__ import annotations

import threading
import unittest
from dataclasses import FrozenInstanceError
from typing import Any, Optional

from weiqi.analysis_workbench import (
    AnalysisProtocolError,
    AnalysisSpec,
    AnalysisWorkbench,
    Back,
    FollowCandidate,
    ForeignFormalGameError,
    IllegalVariationMove,
    PlayMove,
    SelectNode,
    VariationMove,
    WorkbenchStateError,
)
from weiqi.engine import BLACK, WHITE, GoGame


class ScriptedResponseAdapter:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def analyze_position(
        self,
        game: GoGame,
        max_visits: int,
        pv_length: int,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "board": game.board_hash(),
                "moves": tuple(game.moves),
                "max_visits": max_visits,
                "pv_length": pv_length,
                "include_ownership": include_ownership,
                "cancelled": cancel_event.is_set() if cancel_event else False,
            }
        )
        if not self.responses:
            raise AssertionError("没有更多 scripted response")
        return self.responses.pop(0)


def response(
    *,
    ownership: Optional[list[float]] = None,
    candidates: Optional[list[dict[str, Any]]] = None,
    winrate: float = 0.55,
    score_lead: float = 1.25,
    visits: int = 100,
    current_player: str = "B",
) -> dict[str, Any]:
    if candidates is None:
        candidates = [
            {
                "move": "D4",
                "order": 0,
                "pv": ["D4", "E4", "E5"],
                "winrate": 0.57,
                "scoreLead": 2.0,
                "visits": 60,
                "prior": 0.25,
            },
            {
                "move": "E5",
                "order": 1,
                "pv": ["E5", "D4"],
                "winrate": 0.54,
                "scoreLead": 0.5,
                "visits": 30,
                "prior": 0.18,
            },
        ]
    return {
        "rootInfo": {
            "winrate": winrate,
            "scoreLead": score_lead,
            "visits": visits,
            "currentPlayer": current_player,
        },
        "ownership": [0.0] * 81 if ownership is None else ownership,
        "moveInfos": candidates,
        "engineFingerprint": "scripted-katago",
    }


class AnalysisWorkbenchTests(unittest.TestCase):
    def test_analysis_spec_rejects_non_integer_and_boolean_like_values(self) -> None:
        invalid_kwargs = (
            {"max_visits": "100"},
            {"max_visits": 10.5},
            {"top_k": "5"},
            {"top_k": 2.5},
            {"pv_length": "12"},
            {"pv_length": 8.5},
            {"force_refresh": 1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AnalysisSpec(**kwargs)  # type: ignore[arg-type]

    def test_analyze_normalizes_complete_result_and_request_options(self) -> None:
        adapter = ScriptedResponseAdapter(response())
        workbench = AnalysisWorkbench(adapter)
        initial = workbench.sync_formal(GoGame(9))

        view = workbench.analyze(
            initial.selected_node_id,
            AnalysisSpec(max_visits=123, top_k=2, pv_length=9),
        )

        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(adapter.calls[0]["max_visits"], 123)
        self.assertEqual(adapter.calls[0]["pv_length"], 9)
        self.assertTrue(adapter.calls[0]["include_ownership"])
        self.assertFalse(adapter.calls[0]["cancelled"])
        analysis = view.selected_analysis
        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(analysis.black_winrate, 0.55)
        self.assertEqual(analysis.black_score_lead, 1.25)
        self.assertEqual(analysis.visits, 100)
        self.assertEqual(len(analysis.ownership), 81)
        self.assertEqual(len(analysis.candidates), 2)
        first = analysis.candidates[0]
        self.assertEqual(first.move.color, BLACK)
        self.assertEqual([move.color for move in first.pv], [BLACK, WHITE, BLACK])
        self.assertEqual(first.move, first.pv[0])
        with self.assertRaises(FrozenInstanceError):
            view.revision = 99  # type: ignore[misc]

    def test_formal_game_is_never_mutated_by_exploration(self) -> None:
        formal = GoGame(9)
        self.assertTrue(formal.play(4, 4).legal)
        before = (
            formal.board_hash(),
            formal.current_player,
            tuple(formal.moves),
            dict(formal.captures),
            formal.can_undo,
        )
        workbench = AnalysisWorkbench(ScriptedResponseAdapter())
        view = workbench.sync_formal(formal)

        view = workbench.explore(
            PlayMove(VariationMove("play", WHITE, (3, 3)))
        )

        self.assertEqual(view.position.move_number, 2)
        self.assertEqual(
            (
                formal.board_hash(),
                formal.current_player,
                tuple(formal.moves),
                dict(formal.captures),
                formal.can_undo,
            ),
            before,
        )

    def test_candidate_clicks_create_sibling_branches_and_back_keeps_them(self) -> None:
        workbench = AnalysisWorkbench(ScriptedResponseAdapter(response()))
        root = workbench.sync_formal(GoGame(9))
        analyzed = workbench.analyze(root.selected_node_id)
        assert analyzed.selected_analysis is not None
        first, second = analyzed.selected_analysis.candidates

        branch_a = workbench.explore(FollowCandidate(first.id))
        branch_a_id = branch_a.selected_node_id
        at_root = workbench.explore(Back())
        self.assertEqual(at_root.selected_node_id, root.selected_node_id)
        branch_b = workbench.explore(FollowCandidate(second.id))
        branch_b_id = branch_b.selected_node_id

        root_node = next(item for item in branch_b.tree if item.id == root.selected_node_id)
        self.assertEqual(len(root_node.child_ids), 2)
        self.assertIn(branch_a_id, root_node.child_ids)
        self.assertIn(branch_b_id, root_node.child_ids)
        selected_a = workbench.explore(SelectNode(branch_a_id))
        self.assertEqual(selected_a.selected_node_id, branch_a_id)

    def test_follow_candidate_can_expand_and_reuse_a_pv_chain(self) -> None:
        workbench = AnalysisWorkbench(ScriptedResponseAdapter(response()))
        root = workbench.sync_formal(GoGame(9))
        analyzed = workbench.analyze(root.selected_node_id)
        assert analyzed.selected_analysis is not None
        candidate = analyzed.selected_analysis.candidates[0]

        expanded = workbench.explore(FollowCandidate(candidate.id, pv_plies=3))
        self.assertEqual(expanded.position.move_number, 3)
        node_count = len(expanded.tree)
        workbench.explore(SelectNode(root.selected_node_id))
        reused = workbench.explore(FollowCandidate(candidate.id, pv_plies=3))
        self.assertEqual(len(reused.tree), node_count)
        self.assertEqual(reused.selected_node_id, expanded.selected_node_id)

    def test_pv_is_truncated_at_first_locally_illegal_move(self) -> None:
        bad_pv = [
            {
                "move": "A9",
                "order": 0,
                "pv": ["A9", "A9", "B9"],
                "winrate": 0.51,
                "scoreLead": 0.1,
                "visits": 40,
            }
        ]
        workbench = AnalysisWorkbench(
            ScriptedResponseAdapter(response(candidates=bad_pv))
        )
        root = workbench.sync_formal(GoGame(9))

        analyzed = workbench.analyze(root.selected_node_id)

        assert analyzed.selected_analysis is not None
        candidate = analyzed.selected_analysis.candidates[0]
        self.assertEqual(len(candidate.pv), 1)
        self.assertTrue(
            any("PV 在第 2 手截断" in item for item in analyzed.selected_analysis.warnings)
        )

    def test_all_locally_illegal_candidates_fail_without_partial_commit(self) -> None:
        formal = GoGame(9)
        self.assertTrue(formal.play(0, 0).legal)
        illegal = [
            {
                "move": "A9",
                "order": 0,
                "pv": ["A9"],
                "winrate": 0.5,
                "scoreLead": 0.0,
                "visits": 20,
            }
        ]
        workbench = AnalysisWorkbench(
            ScriptedResponseAdapter(response(candidates=illegal, current_player="W"))
        )
        view = workbench.sync_formal(formal)

        with self.assertRaises(AnalysisProtocolError):
            workbench.analyze(view.selected_node_id)

        current = workbench.explore(SelectNode(view.selected_node_id))
        self.assertIsNone(current.selected_analysis)
        self.assertEqual(len(current.history.formal), 2)
        self.assertTrue(
            all(point.source == "heuristic" for point in current.history.formal)
        )

    def test_invalid_ownership_reanalysis_keeps_last_good_analysis(self) -> None:
        adapter = ScriptedResponseAdapter(
            response(),
            response(ownership=[0.0] * 80),
        )
        workbench = AnalysisWorkbench(adapter)
        root = workbench.sync_formal(GoGame(9))
        good = workbench.analyze(root.selected_node_id)
        assert good.selected_analysis is not None

        with self.assertRaises(AnalysisProtocolError):
            workbench.analyze(
                root.selected_node_id,
                AnalysisSpec(force_refresh=True),
            )

        current = workbench.explore(SelectNode(root.selected_node_id))
        self.assertEqual(current.selected_analysis, good.selected_analysis)
        self.assertEqual(len(current.history.formal), 1)
        self.assertEqual(current.history.formal[0].source, "katago")

    def test_formal_history_records_analyzed_positions(self) -> None:
        adapter = ScriptedResponseAdapter(
            response(),
            response(current_player="W", winrate=0.48, score_lead=-0.8),
        )
        formal = GoGame(9)
        workbench = AnalysisWorkbench(adapter)
        root = workbench.sync_formal(formal)
        workbench.analyze(root.selected_node_id)
        self.assertTrue(formal.play(4, 4).legal)
        tip = workbench.sync_formal(formal)

        analyzed_tip = workbench.analyze(tip.selected_node_id)

        self.assertEqual(
            [point.ply for point in analyzed_tip.history.formal],
            [0, 1],
        )
        self.assertEqual(analyzed_tip.history.formal[-1].black_winrate, 0.48)
        self.assertEqual(
            [point.source for point in analyzed_tip.history.formal],
            ["katago", "katago"],
        )

    def test_formal_history_has_labeled_estimates_before_katago_backfill(self) -> None:
        formal = GoGame(9)
        self.assertTrue(formal.play(4, 4).legal)
        self.assertTrue(formal.play(3, 3).legal)
        workbench = AnalysisWorkbench(ScriptedResponseAdapter())

        view = workbench.sync_formal(formal)

        self.assertEqual([point.ply for point in view.history.formal], [0, 1, 2])
        self.assertEqual(
            [point.source for point in view.history.formal],
            ["heuristic", "heuristic", "heuristic"],
        )
        self.assertTrue(
            all(0.0 <= point.black_winrate <= 1.0 for point in view.history.formal)
        )

    def test_fast_history_estimate_does_not_claim_open_board_as_territory(self) -> None:
        formal = GoGame(19)
        self.assertTrue(formal.play(3, 3).legal)
        workbench = AnalysisWorkbench(ScriptedResponseAdapter())

        point = workbench.sync_formal(formal).history.formal[-1]

        self.assertEqual(point.source, "heuristic")
        self.assertLess(abs(point.black_score_lead), 2.0)
        self.assertLess(abs(point.black_winrate - 0.5), 0.1)

    def test_formal_undo_and_new_move_preserve_old_line_as_branch(self) -> None:
        formal = GoGame(9)
        workbench = AnalysisWorkbench(ScriptedResponseAdapter())
        root = workbench.sync_formal(formal)
        self.assertTrue(formal.play(3, 3).legal)
        old_tip = workbench.sync_formal(formal).formal_tip_id

        self.assertEqual(formal.undo(), 1)
        undone = workbench.sync_formal(formal)
        self.assertEqual(undone.formal_tip_id, root.formal_tip_id)
        self.assertTrue(formal.play(4, 4).legal)
        changed = workbench.sync_formal(formal)

        root_node = next(item for item in changed.tree if item.id == root.formal_tip_id)
        self.assertEqual(len(root_node.child_ids), 2)
        old_node = next(item for item in changed.tree if item.id == old_tip)
        self.assertFalse(old_node.is_formal)
        new_node = next(item for item in changed.tree if item.id == changed.formal_tip_id)
        self.assertTrue(new_node.is_formal)

    def test_manual_moves_use_current_player_and_local_legality(self) -> None:
        workbench = AnalysisWorkbench(ScriptedResponseAdapter())
        root = workbench.sync_formal(GoGame(9))
        with self.assertRaises(IllegalVariationMove):
            workbench.explore(PlayMove(VariationMove("play", WHITE, (0, 0))))

        first = workbench.explore(
            PlayMove(VariationMove("play", BLACK, (0, 0)))
        )
        with self.assertRaises(IllegalVariationMove):
            workbench.explore(PlayMove(VariationMove("play", WHITE, (0, 0))))
        restored = workbench.explore(Back())
        self.assertEqual(restored.selected_node_id, root.selected_node_id)
        self.assertNotEqual(first.selected_node_id, root.selected_node_id)

    def test_workbench_rejects_another_formal_game_object(self) -> None:
        workbench = AnalysisWorkbench(ScriptedResponseAdapter())
        workbench.sync_formal(GoGame(9))
        with self.assertRaises(ForeignFormalGameError):
            workbench.sync_formal(GoGame(9))

    def test_cancel_prevents_any_later_analysis_or_exploration(self) -> None:
        adapter = ScriptedResponseAdapter(response())
        workbench = AnalysisWorkbench(adapter)
        view = workbench.sync_formal(GoGame(9))

        workbench.cancel()

        with self.assertRaisesRegex(WorkbenchStateError, "关闭"):
            workbench.analyze(view.selected_node_id)
        with self.assertRaisesRegex(WorkbenchStateError, "关闭"):
            workbench.explore(Back())
        self.assertEqual(adapter.calls, [])


if __name__ == "__main__":
    unittest.main()
