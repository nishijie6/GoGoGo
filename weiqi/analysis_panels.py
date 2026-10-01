"""Candidate list, variation tree, history chart and button states of the AI analysis window."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from .analysis_style import (
    field,
    finite_float,
    format_lead,
    format_probability,
    point_from,
    sequence,
)
from .engine import Point


class AnalysisPanels:
    """Mixed into :class:`~weiqi.analysis_gui.AnalysisWorkbenchWindow`; relies on
    the widgets, variables and view state it creates."""

    def _analysis_candidates(self, analysis: object) -> tuple[object, ...]:
        return sequence(field(analysis, "candidates", "moves", default=()))

    def _candidate_point(self, candidate: object) -> Optional[Point]:
        return point_from(field(candidate, "move", "point"))

    def _candidate_pv(self, candidate: object) -> tuple[Optional[Point], ...]:
        result: list[Optional[Point]] = []
        for move in sequence(field(candidate, "pv", "principal_variation", default=())):
            kind = str(field(move, "kind", default="")).lower()
            if kind == "pass" or move is None:
                result.append(None)
            else:
                point = point_from(move)
                if point is not None:
                    result.append(point)
        return tuple(result)

    def _refresh_candidates(self, analysis: object) -> None:
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        self._candidate_by_item.clear()
        for index, candidate in enumerate(self._analysis_candidates(analysis), start=1):
            point = self._candidate_point(candidate)
            move_text = "虚手" if point is None else self._coordinate(point)
            evaluation = field(candidate, "evaluation", default=candidate)
            probability = field(
                evaluation,
                "black_win_probability",
                "black_winrate",
                "winrate",
                "win_rate",
            )
            lead = field(
                evaluation,
                "black_score_lead",
                "black_lead",
                "score_lead",
            )
            visits = field(evaluation, "visits", "analysis_visits", default="—")
            pv_text = " ".join(
                "虚手" if move is None else self._coordinate(move)
                for move in self._candidate_pv(candidate)
            )
            item = self.candidate_tree.insert(
                "",
                "end",
                values=(
                    f"{index}. {move_text}",
                    format_probability(probability),
                    format_lead(lead),
                    visits,
                    pv_text or "—",
                ),
            )
            self._candidate_by_item[item] = candidate

    def _tree_nodes(self, tree: object) -> tuple[object, ...]:
        if tree is None:
            return ()
        nodes = field(tree, "nodes")
        if isinstance(nodes, Mapping):
            return tuple(nodes.values())
        if nodes is not None:
            return sequence(nodes)
        return sequence(tree)

    def _refresh_branches(self, tree: object) -> None:
        self.branch_tree.delete(*self.branch_tree.get_children())
        self._node_by_item.clear()
        nodes = self._tree_nodes(tree)
        if not nodes:
            return
        remaining = list(nodes)
        inserted: dict[object, str] = {}
        selected_ref = field(
            self.view,
            "selected_node_id",
            "selected",
            "selected_ref",
        )
        self.branch_tree.tag_configure(
            "selected",
            background="#3e5f4a",
            foreground="#ffffff",
        )
        while remaining:
            progressed = False
            for node in remaining[:]:
                reference = field(node, "ref", "position", "node_id", "id", default=id(node))
                parent_reference = field(node, "parent", "parent_ref", "parent_id")
                parent_item = inserted.get(parent_reference, "")
                if parent_reference is not None and not parent_item:
                    continue
                incoming = field(
                    node,
                    "incoming_move",
                    "move",
                    "move_from_parent",
                )
                point = point_from(incoming)
                move_text = "根" if parent_reference is None else (
                    "虚手" if point is None else self._coordinate(point)
                )
                evaluation = field(node, "evaluation", "analysis")
                probability = field(
                    evaluation,
                    "black_win_probability",
                    "winrate",
                    "win_rate",
                )
                is_formal = bool(field(node, "is_formal", default=False))
                has_analysis = bool(field(node, "has_analysis", default=False))
                label = str(
                    field(
                        node,
                        "label",
                        "name",
                        default=(
                            f"正式 {int(field(node, 'ply', default=0))}"
                            if is_formal
                            else f"分支 {len(inserted) + 1}"
                        ),
                    )
                )
                item = self.branch_tree.insert(
                    parent_item,
                    "end",
                    text=label,
                    values=(
                        move_text,
                        format_probability(probability)
                        if evaluation is not None
                        else ("已分析" if has_analysis else "待分析"),
                    ),
                    open=True,
                    tags=("selected",) if reference == selected_ref else (),
                )
                inserted[reference] = item
                self._node_by_item[item] = node
                if selected_ref is not None and reference == selected_ref:
                    self.branch_tree.see(item)
                remaining.remove(node)
                progressed = True
            if not progressed:
                # Malformed/cyclic input should remain visible rather than
                # hanging the Tk event loop.
                for node in remaining:
                    item = self.branch_tree.insert(
                        "",
                        "end",
                        text="孤立节点",
                        values=("—", "—"),
                    )
                    self._node_by_item[item] = node
                break

    def _history_items(self) -> tuple[object, ...]:
        history = field(self.view, "history")
        points = field(history, "points", "items")
        if points is not None:
            return sequence(points)
        formal = sequence(field(history, "formal", default=()))
        active = sequence(field(history, "active", default=()))
        if formal or active:
            selected_id = field(self.view, "selected_node_id")
            selected_node = next(
                (
                    node
                    for node in self._tree_nodes(field(self.view, "tree"))
                    if field(node, "id", "node_id") == selected_id
                ),
                None,
            )
            if bool(field(selected_node, "is_formal", default=False)):
                return formal
            return active or formal
        return sequence(history)

    def _draw_history(self) -> None:
        if not hasattr(self, "history_canvas"):
            return
        canvas = self.history_canvas
        width = max(100, canvas.winfo_width())
        height = max(80, canvas.winfo_height())
        canvas.delete("all")
        self._history_hits.clear()
        items = self._history_items()
        if not items:
            canvas.create_text(
                width / 2,
                height / 2,
                text="分析过的历史局面会显示在这里",
                fill="#7f9083",
                font=("Microsoft YaHei UI", 9),
            )
            return

        left, right, top, bottom = 38.0, width - 14.0, 14.0, height - 25.0
        canvas.create_line(left, top, left, bottom, fill="#58665c")
        canvas.create_line(left, bottom, right, bottom, fill="#58665c")
        canvas.create_line(left, (top + bottom) / 2, right, (top + bottom) / 2, fill="#344238", dash=(3, 3))

        move_numbers = [
            int(field(item, "move_number", "ply", "index", default=index))
            for index, item in enumerate(items)
        ]
        minimum_move = min(move_numbers)
        maximum_move = max(move_numbers)
        span = max(1, maximum_move - minimum_move)
        win_points: list[tuple[float, float]] = []
        win_markers: list[tuple[float, float, str]] = []
        lead_points: list[tuple[float, float]] = []
        lead_values = [
            finite_float(
                field(
                    field(item, "evaluation", "analysis", default=item),
                    "black_score_lead",
                    "black_lead",
                    "score_lead",
                )
            )
            for item in items
        ]
        lead_scale = max(5.0, max((abs(value) for value in lead_values if value is not None), default=5.0))

        for item, move_number, lead_value in zip(items, move_numbers, lead_values):
            evaluation = field(item, "evaluation", "analysis", default=item)
            probability = finite_float(
                field(
                    evaluation,
                    "black_win_probability",
                    "black_winrate",
                    "winrate",
                    "win_rate",
                )
            )
            x = left + (move_number - minimum_move) / span * (right - left)
            if probability is not None:
                if probability > 1.0:
                    probability /= 100.0
                probability = max(0.0, min(1.0, probability))
                y = bottom - probability * (bottom - top)
                win_points.append((x, y))
                win_markers.append(
                    (x, y, str(field(item, "source", default="katago")))
                )
                self._history_hits.append((x, y, item))
            if lead_value is not None:
                normalized = max(-1.0, min(1.0, lead_value / lead_scale))
                y = (top + bottom) / 2 - normalized * (bottom - top) * 0.45
                lead_points.append((x, y))
                self._history_hits.append((x, y, item))

        if len(win_points) > 1:
            canvas.create_line(*[coordinate for point in win_points for coordinate in point], fill="#e5ad52", width=2, smooth=True)
        if len(lead_points) > 1:
            canvas.create_line(*[coordinate for point in lead_points for coordinate in point], fill="#6fa6d8", width=2, smooth=True)
        for x, y, source in win_markers:
            exact = source == "katago"
            canvas.create_oval(
                x - 3,
                y - 3,
                x + 3,
                y + 3,
                fill="#e5ad52" if exact else "#101713",
                outline="#e5ad52" if exact else "#a69678",
            )

        canvas.create_text(4, top, text="100%", anchor="w", fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(8, bottom, text="0%", anchor="w", fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(left, height - 10, text=str(minimum_move), fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(right, height - 10, text=str(maximum_move), fill="#9eaa9f", font=("Segoe UI", 8))
        canvas.create_text(
            right - 150,
            top + 7,
            text="胜率（实心=KataGo）",
            fill="#e5ad52",
            font=("Microsoft YaHei UI", 8),
        )
        canvas.create_text(right - 58, top + 7, text="目差", fill="#6fa6d8", font=("Microsoft YaHei UI", 8))

    def _update_action_states(self) -> None:
        if not hasattr(self, "analyze_button"):
            return
        unavailable = self._busy or self._closed
        position = field(self.view, "position")
        game_over = bool(field(position, "game_over", default=False))
        analysis = field(self.view, "selected_analysis", "analysis")
        has_candidates = bool(self._analysis_candidates(analysis))
        selected_id = field(self.view, "selected_node_id")
        selected_node = next(
            (
                node
                for node in self._tree_nodes(field(self.view, "tree"))
                if field(node, "id", "node_id") == selected_id
            ),
            None,
        )
        can_go_back = selected_node is not None and field(
            selected_node,
            "parent_id",
            "parent",
            "parent_ref",
        ) is not None
        self.analyze_button.configure(
            state="disabled" if unavailable or game_over else "normal"
        )
        self.expand_pv_button.configure(
            state="disabled" if unavailable or not has_candidates else "normal"
        )
        self.back_button.configure(
            state="disabled" if unavailable or not can_go_back else "normal"
        )
        self.pass_button.configure(
            state="disabled" if unavailable or game_over else "normal"
        )
