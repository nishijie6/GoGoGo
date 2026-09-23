"""Replay data and atomic, weights-only-compatible training artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from ..rl_config import RL_CONFIG_VERSION, resolve_rl_training_config
from .state import FEATURE_VERSION, INPUT_PLANES, augment


CHECKPOINT_VERSION = 1


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
    return resolve_rl_training_config(
        raw["preset"], {key: value for key, value in raw.items() if key not in ("preset", "schema_version")}
    )


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint")
    config = checkpoint_config(payload)
    return payload, config


class ReplayBuffer(Dataset):
    def __init__(self, capacity: int, size: int, use_symmetry: bool = True):
        self.capacity, self.size, self.use_symmetry = capacity, size, use_symmetry
        self.samples: list[tuple[np.ndarray, np.ndarray, float]] = []

    def extend(self, samples) -> None:
        self.samples.extend(samples)
        self.samples = self.samples[-self.capacity:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        features, policy, target = self.samples[index]
        if self.use_symmetry:
            features, policy = augment(features, policy, int(torch.randint(8, ()).item()))
        return torch.from_numpy(features), torch.from_numpy(policy), torch.tensor(target, dtype=torch.float32)

    def state(self) -> dict:
        if not self.samples:
            return {"features": torch.empty((0, INPUT_PLANES, self.size, self.size)),
                    "policies": torch.empty((0, self.size ** 2 + 1)), "values": torch.empty(0)}
        return {
            "features": torch.from_numpy(np.stack([row[0] for row in self.samples])),
            "policies": torch.from_numpy(np.stack([row[1] for row in self.samples])),
            "values": torch.tensor([row[2] for row in self.samples], dtype=torch.float32),
        }

    def restore(self, state: dict) -> None:
        features, policies, values = (state[key].numpy() for key in ("features", "policies", "values"))
        length = len(values)
        if (features.shape != (length, INPUT_PLANES, self.size, self.size)
                or policies.shape != (length, self.size ** 2 + 1) or values.shape != (length,)):
            raise ValueError("Checkpoint replay shapes are invalid")
        if any(array.dtype != np.float32 for array in (features, policies, values)):
            raise ValueError("Checkpoint replay must use float32")
        if not all(np.isfinite(array).all() for array in (features, policies, values)):
            raise ValueError("Checkpoint replay contains non-finite values")
        if (policies < 0).any() or not np.allclose(policies.sum(axis=1), 1, atol=1e-5) or (np.abs(values) > 1).any():
            raise ValueError("Checkpoint replay targets are invalid")
        self.samples = list(zip(features, policies, values.tolist()))[-self.capacity:]


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
