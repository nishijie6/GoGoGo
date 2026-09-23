"""Frozen model references and repeatable paired matches against a pool."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..engine import BLACK, WHITE
from ..rl_config import RLTrainingConfig
from .eval_stats import paired_confidence
from .network import PolicyValueNet, Runtime
from .selfplay import GameJob, evaluation_summary, run_games
from .storage import atomic_json, atomic_torch_save, cpu_state, fingerprint, load_checkpoint, write_games


def save_anchor(directory: Path, model, config: RLTrainingConfig, *, kind: str,
                iteration: int) -> Path:
    if kind not in ("accepted", "milestone") or iteration < 0:
        raise ValueError("Invalid frozen model role or iteration")
    path = directory / f"{kind}_{iteration:06d}.pt"
    state = cpu_state(model)
    checksum = fingerprint(state)
    if path.exists():
        saved, _ = load_checkpoint(path)
        if saved.get("kind") != "model" or fingerprint(saved["model"]) != checksum:
            raise ValueError(f"Frozen opponent changed: {path}")
        return path
    payload = {"checkpoint_version": 1, "feature_version": 1,
               "config": config.to_dict(), "kind": "model", "model_role": "anchor",
               "iteration": iteration, "best_iteration": iteration if kind == "accepted" else 0,
               "training_steps": 0, "anchor_kind": kind, "anchor_iteration": iteration,
               "model": state}
    atomic_torch_save(payload, path)
    return path


def select_anchors(directory: Path, config: RLTrainingConfig, *, candidate_sha256: str,
                   maximum: int = 4,
                   kinds: tuple[str, ...] = ("accepted", "milestone")) -> list[dict]:
    if maximum < 1:
        raise ValueError("Pool size must be positive")
    seen, anchors = {candidate_sha256}, []
    for path in sorted(directory.glob("*.pt")):
        payload, old_config = load_checkpoint(path)
        if payload.get("kind") != "model" or payload.get("model_role") != "anchor":
            raise ValueError(f"Unexpected file in frozen model pool: {path}")
        if payload.get("anchor_kind") not in ("accepted", "milestone"):
            raise ValueError(f"Unexpected frozen model kind: {path}")
        if old_config.game != config.game or old_config.network != config.network:
            raise ValueError(f"Frozen opponent has incompatible rules or architecture: {path}")
        if payload["anchor_kind"] not in kinds:
            continue
        checksum = fingerprint(payload["model"])
        if checksum in seen:
            continue
        seen.add(checksum)
        anchors.append({"name": path.stem, "path": path,
                        "iteration": payload["anchor_iteration"],
                        "kind": payload["anchor_kind"], "sha256": checksum})
    anchors.sort(key=lambda item: (item["iteration"], item["kind"], item["name"]))
    if len(anchors) > maximum:
        indices = sorted({round(index * (len(anchors) - 1) / (maximum - 1))
                          for index in range(maximum)}) if maximum > 1 else [0]
        anchors = [anchors[index] for index in indices]
    return anchors


def match_jobs(config: RLTrainingConfig, games: int, *, seed_offset: int = 0) -> list[GameJob]:
    if games < 2 or games % 2:
        raise ValueError("Paired matches need a positive even game count")
    return [GameJob(index, config.runtime.random_seed + seed_offset + index // 2,
                    BLACK if index % 2 == 0 else WHITE) for index in range(games)]


def evaluate_pool(model, runtime: Runtime, config: RLTrainingConfig, anchors: list[dict],
                  output: Path, *, games: int, simulations: int,
                  check=lambda: None, progress=lambda event: None) -> dict:
    if simulations < 1:
        raise ValueError("Pool MCTS budget must be positive")
    match_config = replace(config, evaluation=replace(config.evaluation, games=games,
                                                       simulations_per_move=simulations))
    candidate_sha = fingerprint(cpu_state(model))
    opponents = [{"name": "uniform_policy_mcts", "path": None, "sha256": None}, *anchors]
    rows = []
    for opponent in opponents:
        if opponent["path"] is None:
            baseline = lambda batch: (
                np.zeros((len(batch), config.action_size), dtype=np.float32),
                np.zeros(len(batch), dtype=np.float32))
        else:
            payload, old_config = load_checkpoint(opponent["path"])
            if old_config.game != config.game or old_config.network != config.network:
                raise ValueError("Frozen opponent is incompatible with the candidate")
            baseline_model = PolicyValueNet(old_config).to(runtime.device)
            baseline_model.load_state_dict(payload["model"])
            baseline = runtime.evaluator(baseline_model)
        # Each reference gets its own fixed opening seeds. Repeating the pool
        # against a later candidate will use precisely the same openings.
        jobs = match_jobs(config, games, seed_offset=(
            int(opponent["sha256"][:8], 16) if opponent["sha256"] else 0))
        results = run_games(match_config, jobs,
                            {0: baseline, 1: runtime.evaluator(model)},
                            training=False, check=check, progress=progress)
        write_games(results, match_config, output / opponent["name"])
        row = {"opponent": opponent["name"], "opponent_sha256": opponent["sha256"],
               **evaluation_summary(results),
               "paired": paired_confidence(results, confidence=config.evaluation.confidence_level)}
        rows.append(row)
        progress({"event": "pool_opponent_completed", **row})
    report = {"candidate_sha256": candidate_sha, "games_per_opponent": games,
              "simulations_per_move": simulations, "opponents": rows}
    atomic_json(report, output / "summary.json")
    return report
