"""PUCT search. Values are always from the node's player-to-move perspective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..rl_config import SearchTrainingConfig
from .state import Position


Evaluator = Callable[[np.ndarray], tuple[np.ndarray, float]]


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    position: Position | None = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def expand(node: Node, position: Position, evaluate: Evaluator) -> float:
    logits, value = evaluate(position.features())
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != position.legal.shape or not np.isfinite(logits).all():
        raise ValueError("Evaluator returned invalid policy logits")
    if not np.isfinite(value) or not -1.00001 <= value <= 1.00001:
        raise ValueError("Evaluator returned invalid value")
    actions = np.flatnonzero(position.legal)
    weights = np.exp(logits[actions] - logits[actions].max())
    weights /= weights.sum()
    node.children = {int(a): Node(float(p)) for a, p in zip(actions, weights)}
    return float(value)


def search(
    position: Position,
    evaluate: Evaluator,
    config: SearchTrainingConfig,
    rng: np.random.Generator,
    *,
    simulations: int | None = None,
    add_noise: bool = False,
) -> tuple[np.ndarray, float]:
    if position.game.game_over:
        raise ValueError("Cannot search a finished game")
    count = config.simulations_per_move if simulations is None else simulations
    if count < 1:
        raise ValueError("Search needs at least one simulation")
    root = Node()
    expand(root, position, evaluate)
    if add_noise:
        noise = rng.dirichlet(np.full(len(root.children), config.dirichlet_alpha))
        for child, amount in zip(root.children.values(), noise):
            child.prior = (
                child.prior * (1 - config.dirichlet_epsilon)
                + float(amount) * config.dirichlet_epsilon
            )
    for _ in range(count):
        node, state = root, position
        path = [node]
        while node.children and not state.game.game_over:
            # A child's value has the opposite sign to its parent's value.
            scale = config.c_puct * np.sqrt(max(1, node.visits))
            scores = [
                -child.value + scale * child.prior / (1 + child.visits)
                for child in node.children.values()
            ]
            maximum = max(scores)
            tied = [a for a, score in zip(node.children, scores) if score == maximum]
            action = int(rng.choice(tied))
            node = node.children[action]
            if node.position is None:
                node.position = state.play(action)
            state = node.position
            path.append(node)
        value = state.terminal_value() if state.game.game_over else expand(node, state, evaluate)
        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value
    policy = np.zeros(len(position.legal), dtype=np.float32)
    for action, child in root.children.items():
        policy[action] = child.visits
    policy /= policy.sum()
    return policy, root.value


def choose_action(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature == 0:
        return int(rng.choice(np.flatnonzero(policy == policy.max())))
    positive = policy > 0
    log_weights = np.log(policy[positive].astype(np.float64)) / temperature
    weights = np.exp(log_weights - log_weights.max())
    return int(rng.choice(np.flatnonzero(positive), p=weights / weights.sum()))
