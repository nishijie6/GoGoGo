"""Replay data and atomic, weights-only-compatible training artifacts."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..fileio import atomic_open, atomic_write_json
from ..rl_config import RL_CONFIG_VERSION, resolve_rl_training_config
from .state import FEATURE_VERSION, INPUT_PLANES, augment_batch


CHECKPOINT_VERSION = 1


def atomic_torch_save(payload: dict, path: Path) -> None:
    with atomic_open(path, "wb") as stream:
        torch.save(payload, stream)


def atomic_json(payload: dict, path: Path) -> None:
    atomic_write_json(path, payload, sort_keys=False, allow_nan=False)


def cpu_state(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fingerprint(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_config(payload: dict):
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Checkpoint feature encoding does not match this trainer")
    raw = payload["config"]
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != RL_CONFIG_VERSION:
        raise ValueError("Checkpoint configuration version is unsupported")
    overrides = deepcopy({key: value for key, value in raw.items()
                          if key not in ("preset", "schema_version")})
    # Schema-1 checkpoints predate these policy fields. Preserve their actual
    # training/evaluation behavior rather than inheriting newer preset defaults.
    overrides.setdefault("self_play", {}).setdefault("milestone_fraction", 0.0)
    evaluation = overrides.setdefault("evaluation", {})
    evaluation.setdefault("promotion_test", "paired_hoeffding")
    evaluation.setdefault("confirmation_max_game_length_factor",
                          overrides["self_play"].get("max_game_length_factor", 2.5))
    overrides.setdefault("network", {}).setdefault("auxiliary_heads", False)
    overrides.setdefault("search", {}).setdefault("full_search_probability", 1.0)
    overrides["search"].setdefault("leaf_batch_size", 1)
    overrides.setdefault("optimizer", {}).setdefault("target_sample_reuse", None)
    return resolve_rl_training_config(raw["preset"], overrides)


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint")
    config = checkpoint_config(payload)
    return payload, config


class ReplayBuffer:
    """Training samples ``(features, policy, value, ownership, score, aux_weight)``.

    ``aux_weight`` is 0 when a sample has no trustworthy final board (resigned
    games, replays saved before auxiliary targets existed) so its ownership and
    score rows are masked out of the loss.
    """

    def __init__(self, capacity: int, size: int, use_symmetry: bool = True):
        self.capacity, self.size, self.use_symmetry = capacity, size, use_symmetry
        self.samples: list[tuple] = []

    def _normalize(self, sample) -> tuple:
        if len(sample) == 3:
            features, policy, value = sample
            return (features, policy, float(value),
                    np.zeros(self.size ** 2, dtype=np.float32), 0.0, 0.0)
        return tuple(sample)

    def extend(self, samples) -> None:
        self.samples.extend(self._normalize(sample) for sample in samples)
        self.samples = self.samples[-self.capacity:]

    def __len__(self):
        return len(self.samples)

    def sample(self, batch_size: int):
        """Draw a batch with replacement in-process; no worker pickling of the pool.

        Returns float32 tensors: features, policies, values, ownership, scores,
        and auxiliary-target weights.
        """
        indices = torch.randint(len(self.samples), (batch_size,)).tolist()
        rows = [self.samples[index] for index in indices]
        features = np.stack([row[0] for row in rows])
        policies = np.stack([row[1] for row in rows])
        ownership = np.stack([row[3] for row in rows])
        if self.use_symmetry:
            features, policies, ownership = augment_batch(
                features, policies, torch.randint(8, (batch_size,)).numpy(), ownership)
        columns = (features, policies, np.array([row[2] for row in rows], dtype=np.float32),
                   ownership, np.array([row[4] for row in rows], dtype=np.float32),
                   np.array([row[5] for row in rows], dtype=np.float32))
        return tuple(torch.from_numpy(np.ascontiguousarray(column)) for column in columns)

    def state(self) -> dict:
        area = self.size ** 2
        if not self.samples:
            return {"features": torch.empty((0, INPUT_PLANES, self.size, self.size)),
                    "policies": torch.empty((0, area + 1)), "values": torch.empty(0),
                    "ownership": torch.empty((0, area)), "scores": torch.empty(0),
                    "aux_weights": torch.empty(0)}
        return {
            "features": torch.from_numpy(np.stack([row[0] for row in self.samples])),
            "policies": torch.from_numpy(np.stack([row[1] for row in self.samples])),
            "values": torch.tensor([row[2] for row in self.samples], dtype=torch.float32),
            "ownership": torch.from_numpy(np.stack([row[3] for row in self.samples])),
            "scores": torch.tensor([row[4] for row in self.samples], dtype=torch.float32),
            "aux_weights": torch.tensor([row[5] for row in self.samples], dtype=torch.float32),
        }

    def restore(self, state: dict) -> None:
        features, policies, values = (state[key].numpy() for key in ("features", "policies", "values"))
        length, area = len(values), self.size ** 2
        if "ownership" in state:
            ownership, scores, weights = (state[key].numpy()
                                          for key in ("ownership", "scores", "aux_weights"))
        else:
            # Replays saved before auxiliary targets: keep them, but mask those heads.
            ownership = np.zeros((length, area), dtype=np.float32)
            scores = np.zeros(length, dtype=np.float32)
            weights = np.zeros(length, dtype=np.float32)
        arrays = (features, policies, values, ownership, scores, weights)
        if (features.shape != (length, INPUT_PLANES, self.size, self.size)
                or policies.shape != (length, area + 1) or values.shape != (length,)
                or ownership.shape != (length, area) or scores.shape != (length,)
                or weights.shape != (length,)):
            raise ValueError("Checkpoint replay shapes are invalid")
        if any(array.dtype != np.float32 for array in arrays):
            raise ValueError("Checkpoint replay must use float32")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Checkpoint replay contains non-finite values")
        if ((policies < 0).any() or not np.allclose(policies.sum(axis=1), 1, atol=1e-5)
                or (np.abs(values) > 1).any() or (np.abs(ownership) > 1).any()
                or not np.isin(weights, (0.0, 1.0)).all()):
            raise ValueError("Checkpoint replay targets are invalid")
        self.samples = list(zip(features, policies, values.tolist(), ownership,
                                scores.tolist(), weights.tolist()))[-self.capacity:]


def write_games(games, config, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    size = config.game.board_size
    for game in games:
        stem = f"game_{game.index:04d}"
        atomic_json(game.record(config), directory / (stem + ".json"))
        outcome = ""
        if game.reason != "length_limit":
            if game.winner is None:
                outcome = "RE[0]"
            else:
                color = "B" if game.winner == 1 else "W"
                margin = "R" if game.reason == "resign" else f"{abs(game.black_score - game.white_score):g}"
                outcome = f"RE[{color}+{margin}]"
        body = f"(;GM[1]FF[4]CA[UTF-8]SZ[{size}]KM[{config.game.komi:g}]RU[Chinese]{outcome}C[{game.reason}]"
        for color, action in game.moves:
            coordinate = "" if action == size ** 2 else chr(97 + action % size) + chr(97 + action // size)
            body += f";{'B' if color == 1 else 'W'}[{coordinate}]"
        (directory / (stem + ".sgf")).write_text(body + ")\n", encoding="utf-8")
