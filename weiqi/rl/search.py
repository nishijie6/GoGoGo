"""PUCT search. Values are always from the node's player-to-move perspective."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..rl_config import SearchTrainingConfig
from .state import Position


Evaluator = Callable[[np.ndarray], tuple[np.ndarray, float]]
BatchEvaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class Node:
    """A search state; statistics of its children are kept in parallel arrays.

    Child ``i`` is ``actions[i]``; its visit count and value sum (from the
    child's player-to-move perspective) are ``child_visits[i]`` and
    ``child_values[i]``. Child nodes are created only when first selected.
    """

    __slots__ = ("visits", "value_sum", "actions", "priors", "child_visits",
                 "child_values", "children", "position")

    def __init__(self, position: Position | None = None) -> None:
        self.visits = 0
        self.value_sum = 0.0
        self.actions: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.child_visits: np.ndarray | None = None
        self.child_values: np.ndarray | None = None
        self.children: list[Node | None] = []
        self.position = position

    @property
    def expanded(self) -> bool:
        return self.actions is not None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def select(self, c_puct: float, rng: np.random.Generator) -> int:
        """Index of the child with the highest PUCT score; ties broken at random."""
        visits = self.child_visits
        # A child's value has the opposite sign to its parent's value.
        q = np.divide(self.child_values, visits, out=np.zeros_like(visits), where=visits > 0)
        scale = c_puct * np.sqrt(max(1, self.visits))
        scores = -q + scale * self.priors / (1 + visits)
        tied = np.flatnonzero(scores == scores.max())
        return int(rng.choice(tied))


def expand(node: Node, position: Position, evaluate: Evaluator) -> float:
    logits, value = evaluate(position.features())
    return expand_with(node, position, logits, value)


def expand_with(node: Node, position: Position, logits, value) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != position.legal.shape or not np.isfinite(logits).all():
        raise ValueError("Evaluator returned invalid policy logits")
    if not np.isfinite(value) or not -1.00001 <= value <= 1.00001:
        raise ValueError("Evaluator returned invalid value")
    actions = np.flatnonzero(position.legal)
    weights = np.exp(logits[actions] - logits[actions].max())
    weights /= weights.sum()
    node.actions, node.priors = actions, weights
    node.child_visits = np.zeros(len(actions))
    node.child_values = np.zeros(len(actions))
    node.children = [None] * len(actions)
    return float(value)


def search(
    position: Position,
    evaluate: Evaluator,
    config: SearchTrainingConfig,
    rng: np.random.Generator,
    *,
    simulations: int | None = None,
    add_noise: bool = False,
    evaluate_batch: BatchEvaluator | None = None,
) -> tuple[np.ndarray, float]:
    if position.game.game_over:
        raise ValueError("Cannot search a finished game")
    count = config.simulations_per_move if simulations is None else simulations
    if count < 1:
        raise ValueError("Search needs at least one simulation")
    root = Node(position)
    expand(root, position, evaluate)
    if add_noise:
        noise = rng.dirichlet(np.full(len(root.actions), config.dirichlet_alpha))
        root.priors = (root.priors * (1 - config.dirichlet_epsilon)
                       + noise * config.dirichlet_epsilon)
    batch_size = config.leaf_batch_size if evaluate_batch is not None else 1
    done = 0
    while done < count:
        # Select up to batch_size leaves; virtual loss steers later picks elsewhere.
        pending: list[tuple[list[tuple[Node, int]], Position, Node]] = []
        pending_ids: set[int] = set()
        while done < count and len(pending) < batch_size:
            node, state = root, position
            path: list[tuple[Node, int]] = []  # (parent, child index) edges
            while node.expanded and not state.game.game_over:
                index = node.select(config.c_puct, rng)
                child = node.children[index]
                if child is None:
                    child = node.children[index] = Node(state.play(int(node.actions[index])))
                path.append((node, index))
                node, state = child, child.position
            if state.game.game_over:
                _backup(root, path, state.terminal_value(), 0.0)
                done += 1
                continue
            if id(node) in pending_ids:
                break  # the same unexpanded leaf again; flush the batch first
            if batch_size > 1:
                _backup_virtual(root, path)
            pending.append((path, state, node))
            pending_ids.add(id(node))
            done += 1
        if not pending:
            continue
        if batch_size == 1:
            path, state, node = pending[0]
            _backup(root, path, expand(node, state, evaluate), 0.0)
            continue
        logits, values = evaluate_batch(np.stack([state.features() for _, state, _ in pending]))
        for (path, state, node), row, value in zip(pending, logits, values):
            _backup(root, path, expand_with(node, state, row, float(value)), 1.0)
    policy = np.zeros(len(position.legal), dtype=np.float32)
    policy[root.actions] = root.child_visits
    policy /= policy.sum()
    return policy, root.value


def _visit(parent: Node, index: int, visits: int, value: float) -> None:
    """Add to a child's statistics, both in its parent's arrays and on the child."""
    parent.child_visits[index] += visits
    parent.child_values[index] += value
    child = parent.children[index]
    child.visits += visits
    child.value_sum += value


def _backup_virtual(root: Node, path: list[tuple[Node, int]]) -> None:
    """Count a pending visit as a loss for each node's parent."""
    root.visits += 1
    for parent, index in path:
        _visit(parent, index, 1, 1.0)


def _backup(root: Node, path: list[tuple[Node, int]], value: float, virtual: float) -> None:
    """Propagate value up the path, replacing any virtual loss that was applied.

    ``value`` is from the leaf's player-to-move perspective and flips sign at
    every edge toward the root.
    """
    for parent, index in reversed(path):
        if virtual:
            _visit(parent, index, 0, value - virtual)
        else:
            _visit(parent, index, 1, value)
        value = -value
    root.value_sum += value
    if not virtual:
        root.visits += 1  # a virtual visit already counted the root's


def choose_action(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature == 0:
        return int(rng.choice(np.flatnonzero(policy == policy.max())))
    positive = policy > 0
    log_weights = np.log(policy[positive].astype(np.float64)) / temperature
    weights = np.exp(log_weights - log_weights.max())
    return int(rng.choice(np.flatnonzero(positive), p=weights / weights.sum()))
