"""Standalone strength checks of saved models, outside the training loop.

``evaluate`` plays color-swapped games against another checkpoint (or a
uniform-policy MCTS baseline); ``benchmark`` scores fixed KataGo-labelled
positions and plays the frozen opponent pool. Both write their own run
directory and never touch a training run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..engine import BLACK, WHITE
from .control import Progress, RunLock, TrainingControl
from .eval_pool import evaluate_pool, select_anchors
from .eval_quality import evaluate_positions
from .eval_stats import paired_confidence
from .network import PolicyValueNet, Runtime
from .selfplay import GameJob, evaluation_summary, run_games
from .storage import atomic_json, cpu_state, fingerprint, load_checkpoint, write_games


def evaluate(checkpoint: Path, opponent: Path | None, output: Path, games: int | None = None):
    payload, config = load_checkpoint(checkpoint)
    if games is not None:
        from ..rl_config import resolve_rl_training_config
        overrides = {key: value for key, value in config.to_dict().items()
                     if key not in ("schema_version", "preset")}
        overrides["evaluation"]["games"] = games
        config = resolve_rl_training_config(config.preset, overrides)
    with RunLock(output):
        if (output / "events.jsonl").exists():
            raise ValueError("Evaluation output already exists; choose a new --output")
        with Progress(output, operation="evaluate") as progress:
            control = TrainingControl(output, config.runtime.pause_while_game_is_active, progress)
            control()
            runtime = Runtime(config)
            candidate = PolicyValueNet(config).to(runtime.device)
            candidate.load_state_dict(payload["model"])
            if opponent is not None:
                other_payload, other_config = load_checkpoint(opponent)
                if other_config.game != config.game:
                    raise ValueError("Evaluation checkpoints use different board sizes or rules")
                other = PolicyValueNet(other_config).to(runtime.device)
                other.load_state_dict(other_payload["model"])
                baseline = runtime.evaluator(other)
            else:
                # Uniform priors and zero values, still using the same MCTS budget.
                baseline = lambda batch: (np.zeros((len(batch), config.action_size), dtype=np.float32),
                                          np.zeros(len(batch), dtype=np.float32))
            match_metadata = {
                "candidate_sha256": fingerprint(payload["model"]),
                "opponent_name": opponent.stem if opponent else "uniform_policy_mcts",
                "opponent_sha256": fingerprint(other_payload["model"]) if opponent else None,
            }
            jobs = [GameJob(index, config.runtime.random_seed + index // 2,
                            BLACK if index % 2 == 0 else WHITE,
                            opponent_name=match_metadata["opponent_name"],
                            opponent_sha256=match_metadata["opponent_sha256"])
                    for index in range(config.evaluation.games)]
            progress.phase = "evaluation"
            results = run_games(config, jobs, {0: baseline, 1: runtime.evaluator(candidate)},
                                training=False, check=control, progress=progress,
                                metadata=match_metadata)
            write_games(results, config, output / "games")
            summary = {**match_metadata, **evaluation_summary(results),
                       "paired": paired_confidence(results, confidence=config.evaluation.confidence_level),
                       "checkpoint": str(checkpoint),
                       "opponent": str(opponent) if opponent else "uniform_policy_mcts"}
            atomic_json(summary, output / "summary.json")
            progress({"event": "evaluation_completed", **summary})
            return summary


def benchmark(checkpoint: Path, output: Path, *, positions: Path | None = None,
              teacher: Path | None = None, pool: Path | None = None,
              opponents: list[Path] | None = None, games: int | None = None,
              simulations: int | None = None, position_simulations: int | None = None):
    """Independent frozen-position and fixed-opponent report for any export."""
    payload, config = load_checkpoint(checkpoint)
    with RunLock(output):
        if (output / "events.jsonl").exists():
            raise ValueError("Benchmark output already exists; choose a new --output")
        with Progress(output, operation="benchmark") as progress:
            control = TrainingControl(output, config.runtime.pause_while_game_is_active, progress)
            control()
            runtime = Runtime(config)
            model = PolicyValueNet(config).to(runtime.device)
            model.load_state_dict(payload["model"])
            candidate_sha = fingerprint(cpu_state(model))
            root = Path(__file__).resolve().parents[2]
            positions = positions or (root / config.evaluation.position_suite_path
                                      if config.evaluation.position_suite_path else None)
            teacher = teacher or (root / config.evaluation.teacher_labels_path
                                  if config.evaluation.teacher_labels_path else None)
            report = {"checkpoint": str(checkpoint), "candidate_sha256": candidate_sha}
            if positions is not None and teacher is not None:
                progress.phase = "position_diagnostics"
                quality = evaluate_positions(
                    model, runtime, config, positions, teacher,
                    simulations=(config.evaluation.position_simulations_per_move
                                 if position_simulations is None else position_simulations),
                    check=control, progress=progress,
                )
                atomic_json(quality, output / "position_quality.json")
                report["position_quality"] = {
                    "suite_sha256": quality["suite_sha256"], "candidate_visits": quality["candidate_visits"],
                    "teacher_visits": quality["teacher_visits"], "groups": quality["groups"],
                }
            else:
                report["position_quality"] = {"status": "not_configured"}
            anchors = select_anchors(pool or checkpoint.parent / "anchors", config,
                                     candidate_sha256=candidate_sha)
            for index, path in enumerate(opponents or [], 1):
                other, other_config = load_checkpoint(path)
                if other_config.game != config.game or other_config.network != config.network:
                    raise ValueError(f"Opponent uses incompatible rules or architecture: {path}")
                checksum = fingerprint(other["model"])
                if checksum == candidate_sha or any(anchor["sha256"] == checksum for anchor in anchors):
                    continue
                anchors.append({"name": f"manual_{index:02d}_{path.stem}",
                                "path": path, "sha256": checksum})
            progress.phase = "opponent_pool"
            report["opponent_pool"] = evaluate_pool(
                model, runtime, config, anchors, output / "pool",
                games=config.evaluation.pool_games if games is None else games,
                simulations=(config.evaluation.pool_simulations_per_move
                             if simulations is None else simulations),
                check=control, progress=progress,
            )
            atomic_json(report, output / "summary.json")
            progress({"event": "benchmark_completed", "output": str(output),
                      "candidate_sha256": candidate_sha,
                      "opponents": len(report["opponent_pool"]["opponents"])})
            return report
