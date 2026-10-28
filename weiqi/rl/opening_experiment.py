"""Isolated, fixed-budget A/B training with matched held-out opening pairs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from ..engine import BLACK, WHITE
from ..fileio import sha256_file
from ..rl_config import resolve_rl_training_config
from ..rl_native_handoff import apply_native_hardware
from . import assessment
from .control import Progress, RunLock
from .eval_pool import select_anchors
from .openings import JOSEKI_SUITE_VERSION, joseki_openings
from .runner import Trainer
from .storage import atomic_json, atomic_torch_save, fingerprint, load_checkpoint


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2 ** 32):
        raise ValueError("Experiment seeds must be integers between 0 and 2**32-1")


def seeded_resume(payload: dict, seed: int | None) -> dict:
    """Replace random streams while keeping initial learned training state.

    Merely changing config.random_seed is insufficient: Trainer restores the
    saved RNG fields after initialization. A local generator avoids changing
    the caller's global Torch state while preparing the paired resumes.
    """
    _validate_seed(seed)
    if seed is None:
        return payload
    return {**payload, "numpy_rng": np.random.default_rng(seed).bit_generator.state,
            "torch_rng": torch.Generator(device="cpu").manual_seed(seed).get_state(),
            "cuda_rng": []}


def make_evaluation_config(config, *, games: int, visits: int, seed: int | None = None,
                           length_factor: float | None = None,
                           leaf_batch_size: int | None = None):
    """Validate the actual standalone-match controls independently of training."""
    _validate_seed(seed)
    raw = config.to_dict()
    raw["self_play"]["joseki_fraction"] = 0
    raw["evaluation"].update(games=games, simulations_per_move=visits)
    if seed is not None:
        raw["runtime"]["random_seed"] = seed
    if length_factor is not None:
        raw["self_play"]["max_game_length_factor"] = length_factor
    if leaf_batch_size is not None:
        raw["search"]["leaf_batch_size"] = leaf_batch_size
    return resolve_rl_training_config(config.preset, {
        key: value for key, value in raw.items() if key not in ("schema_version", "preset")})


def arm_configs(config, output: Path, *, fraction: float,
                training_games: int | None = None, training_steps: int | None = None,
                training_visits: int | None = None, training_seed: int | None = None,
                training_leaf_batch_size: int | None = None,
                freeze_training_opponents: bool = False,
                checkpoint_retention: int | None = None):
    """Resolve identical budgets, changing only the opening fraction and output."""
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Treatment fraction must be greater than zero and at most one")
    _validate_seed(training_seed)
    if type(freeze_training_opponents) is not bool:
        raise ValueError("freeze_training_opponents must be a boolean")
    raw = config.to_dict()
    raw.pop("schema_version")
    raw.pop("preset")
    # Match gradient-update budgets even when opening lengths change the number
    # of collected samples. Each arm still uses the same checkpoint replay.
    raw["optimizer"]["target_sample_reuse"] = None
    if training_seed is not None:
        raw["runtime"]["random_seed"] = training_seed
    if freeze_training_opponents:
        # Promotion is disabled by the caller; suppress branch-specific new
        # milestones as well, so both arms keep the same frozen opponents.
        raw["evaluation"]["milestone_every_iterations"] = 2 ** 31 - 1
    for section, field, value in (
            ("self_play", "games_per_iteration", training_games),
            ("optimizer", "training_steps_per_iteration", training_steps),
            ("search", "simulations_per_move", training_visits),
            ("search", "leaf_batch_size", training_leaf_batch_size),
            ("runtime", "keep_checkpoints", checkpoint_retention)):
        if value is not None:
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
            raw[section][field] = value
    configs = {}
    for arm, opening_fraction in (("control", 0.0), ("joseki", fraction)):
        values = deepcopy(raw)
        values["self_play"]["joseki_fraction"] = opening_fraction
        values["runtime"]["output_directory"] = str(output / arm)
        configs[arm] = resolve_rl_training_config(config.preset, values)
    return configs


def matched_delta(control: list[dict], treatment: list[dict], *, planned_games: int) -> dict:
    """Exact one-sided sign test of treatment-minus-control opening pair scores.

    All four games in a matched unit share the same random two-stone opening.
    Truncation in either arm excludes that entire unit. The fixed-sample test is
    descriptive until the planned match completes, with at least twenty pairs.
    """
    if type(planned_games) is not int or planned_games < 2 or planned_games % 2:
        raise ValueError("Evaluation needs a positive even planned game count")
    groups = []
    for records in (control, treatment):
        by_index = {row["index"]: row for row in records}
        if (len(by_index) != len(records) or len(records) % 2
                or set(by_index) != set(range(len(records))) or len(records) > planned_games):
            raise ValueError("Evaluation indices must be unique, contiguous color-swapped pairs")
        groups.append(by_index)
    if set(groups[0]) != set(groups[1]):
        raise ValueError("Evaluation arms have different game indices")
    deltas, excluded = [], 0
    for index in range(0, len(control), 2):
        rows = [group[index + offset] for group in groups for offset in (0, 1)]
        if (len({row["seed"] for row in rows}) != 1
                or [row["candidate_color"] for row in rows] != [BLACK, WHITE, BLACK, WHITE]
                or any(row.get("winner") not in (None, BLACK, WHITE) for row in rows)):
            raise ValueError("Evaluation opening seeds or candidate colors do not match")
        openings = [row.get("moves", [])[:2] for row in rows]
        if any(opening != openings[0] for opening in openings):
            raise ValueError("Evaluation arms did not use identical opening moves")
        if any(row["reason"] == "length_limit" for row in rows):
            excluded += 1
            continue
        scores = [0.5 if row["winner"] is None else float(
            row["winner"] == row["candidate_color"]) for row in rows]
        deltas.append((scores[2] + scores[3] - scores[0] - scores[1]) / 2)
    positive, negative = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    decisive = positive + negative
    p_value = (sum(math.comb(decisive, wins) for wins in range(positive, decisive + 1))
               / 2 ** decisive) if decisive else 1.0
    mean = sum(deltas) / len(deltas) if deltas else None
    # Differences lie in [-1, 1], twice the range of a single pair score.
    radius = math.sqrt(2 * math.log(40) / len(deltas)) if deltas else None
    complete = len(control) == planned_games and excluded == 0
    return {"method": "matched_opening_pair_exact_one_sided_sign_test",
            "planned_games_per_arm": planned_games, "observed_games_per_arm": len(control),
            "complete_pairs": len(deltas), "excluded_truncated_pairs": excluded,
            "positive_pairs": positive, "negative_pairs": negative,
            "tied_pairs": len(deltas) - decisive, "mean_score_delta": mean,
            "one_sided_p_value": p_value, "alpha": .05,
            "hoeffding_95_lower": max(-1.0, mean - radius) if deltas else None,
            "hoeffding_95_upper": min(1.0, mean + radius) if deltas else None,
            "complete_planned_match": complete,
            "evidence_of_improvement": bool(complete and len(deltas) >= 20
                                              and mean - radius > 0 and p_value <= .05)}


def _copy_frozen(source: Path, target: Path) -> dict:
    checksum = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != checksum or sha256_file(source) != checksum:
        raise ValueError(f"Source changed while freezing: {source}")
    return {"path": str(source), "sha256": checksum, "bytes": target.stat().st_size}


def _freeze_context(source_run: Path, output: Path, payload: dict, config) -> list[dict]:
    """Keep all anchors and only the fallback history the trainer can consume."""
    context = output / "frozen_context"
    inventory = []
    for source in sorted((source_run / "anchors").glob("*.pt")):
        relative = Path("anchors") / source.name
        inventory.append({"relative_path": str(relative),
                          **_copy_frozen(source, context / relative)})
    candidate_sha = fingerprint(payload["model"])
    milestones = select_anchors(context / "anchors", config,
                                candidate_sha256=candidate_sha, kinds=("milestone",), recent=True)
    if config.self_play.milestone_fraction > 0 and not milestones:
        for source in sorted((source_run / "checkpoints").glob("iteration_*.pt"), reverse=True):
            previous, previous_config = load_checkpoint(source)
            if previous.get("kind") != "training" or previous["iteration"] >= payload["iteration"]:
                continue
            if previous_config.game != config.game or previous_config.network != config.network:
                raise ValueError(f"Historical checkpoint is incompatible: {source}")
            if fingerprint(previous["model"]) == candidate_sha:
                continue
            relative = Path("checkpoints") / source.name
            inventory.append({"relative_path": str(relative),
                              **_copy_frozen(source, context / relative)})
            break
    return inventory


class _TrainingWithoutEvaluation(Trainer):
    """Explicit experiment option: both arms skip diagnostics and promotion."""

    def _position_quality(self, directory):
        return {"status": "disabled_for_experiment"}

    def _opponent_pool(self, number, directory):
        return {"status": "disabled_for_experiment"}

    def _screening(self, directory):
        return {}, {"status": "disabled_for_experiment"}

    def _promotion(self, number, directory, match_metadata, screening):
        return {"status": "disabled_for_experiment"}, False


def _train_arm(config, output: Path, frozen: Path, context: Path, iterations: int,
               skip_training_evaluation: bool, update, training_seed: int | None = None):
    # Own the directory before installing historical files; train() would reject
    # that intentionally preseeded output as an occupied unrelated run.
    with RunLock(output):
        if any(path.name != "run.lock" for path in output.iterdir()):
            raise ValueError("Training arm requires a fresh output directory")
        for directory in ("anchors", "checkpoints"):
            if (context / directory).exists():
                shutil.copytree(context / directory, output / directory)
        resume, _ = load_checkpoint(frozen)
        resume = seeded_resume(resume, training_seed)
        with Progress(output, operation="opening_experiment_train") as progress:
            trainer_type = _TrainingWithoutEvaluation if skip_training_evaluation else Trainer
            trainer = trainer_type(config, output, resume, progress=progress)
            summaries, joseki_samples = [], 0
            for _ in range(iterations):
                summaries.append(trainer.run_iteration())
                game_directory = output / "iterations" / f"{trainer.iteration:06d}" / "selfplay"
                for path in game_directory.glob("game_*.json"):
                    game = json.loads(path.read_text(encoding="utf-8"))
                    if game.get("opening", {}).get("id"):
                        joseki_samples += game["samples"]
                update(trainer.iteration)
            return {"iterations": len(summaries), "final_iteration": trainer.iteration,
                    "final_training_steps": trainer.training_steps,
                    "training_steps": sum(row["training"]["steps"] for row in summaries),
                    "new_samples": sum(row["new_samples"] for row in summaries),
                    "selfplay_games": sum(row["selfplay_games"] for row in summaries),
                    "joseki_games": sum(row.get("selfplay_joseki_games", 0) for row in summaries),
                    "joseki_samples": joseki_samples,
                    "model_sha256": summaries[-1]["model_sha256"]}


def compare_training(resume: Path, output: Path, *, iterations: int = 3,
                     fraction: float = .25, games: int = 80, visits: int = 32,
                     hardware_overrides: Path | None = None,
                     training_games: int | None = None, training_steps: int | None = None,
                     training_visits: int | None = None,
                     skip_training_evaluation: bool = False,
                     training_seed: int | None = None, evaluation_seed: int | None = None,
                     training_leaf_batch_size: int | None = None,
                     evaluation_leaf_batch_size: int | None = None,
                     evaluation_length_factor: float | None = None,
                     freeze_training_opponents: bool = False,
                     checkpoint_retention: int | None = None) -> dict:
    """Freeze one full checkpoint, train two fresh branches, assess both vs source."""
    for name, value in (("iterations", iterations), ("visits", visits), ("games", games)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if games < 2 or games % 2:
        raise ValueError("Evaluation games must be a positive even number")
    if freeze_training_opponents and not skip_training_evaluation:
        raise ValueError("Freezing training opponents requires --skip-training-evaluation")
    resume, output = Path(resume).resolve(), Path(output).resolve()
    source_run = resume.parent.parent if resume.parent.name == "checkpoints" else resume.parent
    if output.is_relative_to(source_run):
        raise ValueError("Experiment output must be outside the source run")
    if output.exists():
        raise ValueError("Experiment requires a fresh output directory")
    source_initial_checksum = sha256_file(resume)
    payload, config = load_checkpoint(resume)
    if payload.get("kind") != "training":
        raise ValueError("Experiment resume requires a full training checkpoint")
    required = {"model", "best_model", "optimizer", "scaler", "replay", "numpy_rng",
                "torch_rng", "cuda_rng", "iteration", "training_steps", "best_iteration"}
    if required - payload.keys():
        raise ValueError("Full training checkpoint is missing resume state")
    config = apply_native_hardware(config, hardware_overrides)
    configs = arm_configs(config, output, fraction=fraction, training_games=training_games,
                          training_steps=training_steps, training_visits=training_visits,
                          training_seed=training_seed, training_leaf_batch_size=training_leaf_batch_size,
                          freeze_training_opponents=freeze_training_opponents,
                          checkpoint_retention=checkpoint_retention)
    # Reject invalid evaluation options before creating any run artifacts.
    make_evaluation_config(config, games=games, visits=visits, seed=evaluation_seed,
                           length_factor=evaluation_length_factor,
                           leaf_batch_size=evaluation_leaf_batch_size)
    openings = [{"name": opening.name, "actions": list(opening.actions)}
                for opening in joseki_openings(config.game.board_size)]
    if not openings:
        raise ValueError("No training joseki openings support this board size")
    args = {"resume": str(resume), "output": str(output), "iterations": iterations,
            "fraction": fraction, "games": games, "visits": visits,
            "hardware_overrides_sha256": sha256_file(hardware_overrides) if hardware_overrides else None,
            "training_games": training_games, "training_steps": training_steps,
            "training_visits": training_visits, "skip_training_evaluation": skip_training_evaluation,
            "training_seed": training_seed, "evaluation_seed": evaluation_seed,
            "training_leaf_batch_size": training_leaf_batch_size,
            "evaluation_leaf_batch_size": evaluation_leaf_batch_size,
            "evaluation_length_factor": evaluation_length_factor,
            "freeze_training_opponents": freeze_training_opponents,
            "checkpoint_retention": checkpoint_retention}
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "phase": "freezing", "arguments_sha256": _digest(args)}

    def update(**changes):
        status.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(status, output / "status.json")

    with RunLock(output):
        try:
            update()
            frozen = output / "frozen_source.pt"
            source = _copy_frozen(resume, frozen)
            if source["sha256"] != source_initial_checksum:
                raise ValueError("Source checkpoint changed after configuration was loaded")
            # Always deserialize the frozen bytes, even if latest.pt is replaced
            # by a running source trainer immediately after the copy.
            payload, _ = load_checkpoint(frozen)
            context = _freeze_context(source_run, output, payload, config)
            manifest = {"schema_version": 1, "arguments": args,
                        "arguments_sha256": _digest(args), "source": source,
                        "source_iteration": payload["iteration"],
                        "source_training_steps": payload["training_steps"],
                        "source_model_sha256": fingerprint(payload["model"]),
                        "arm_configs": {name: value.to_dict() for name, value in configs.items()},
                        "arm_config_sha256": {name: _digest(value.to_dict())
                                              for name, value in configs.items()},
                        "frozen_context": context,
                        "joseki_suite": {"version": JOSEKI_SUITE_VERSION,
                                         "sha256": _digest(openings), "openings": openings},
                        "training_evaluation": "disabled" if skip_training_evaluation else "source_policy",
                        "common_training_overrides": {
                            "optimizer.target_sample_reuse": None,
                            "freeze_training_opponents": freeze_training_opponents},
                        "evaluation_openings": "identical seeded random two-stone color-swapped pairs",
                        "history_policy": "all anchors; newest distinct prior checkpoint only when needed for milestone fallback"}
            atomic_json(manifest, output / "manifest.json")
            training, evaluation = {}, {}
            for arm, arm_config in configs.items():
                update(phase="training", arm=arm, iteration=payload["iteration"])
                training[arm] = _train_arm(
                    arm_config, output / arm, frozen, output / "frozen_context", iterations,
                    skip_training_evaluation, lambda number, name=arm: update(arm=name, iteration=number),
                    training_seed=training_seed)
                atomic_json(training, output / "training.json")
            for arm in configs:
                update(phase="evaluation", arm=arm)
                candidate, candidate_config = load_checkpoint(output / arm / "candidate.pt")
                evaluation_config = make_evaluation_config(
                    candidate_config, games=games, visits=visits, seed=evaluation_seed,
                    length_factor=evaluation_length_factor,
                    leaf_batch_size=evaluation_leaf_batch_size)
                candidate["config"] = evaluation_config.to_dict()
                evaluation_export = output / f"evaluation_{arm}.pt"
                atomic_torch_save(candidate, evaluation_export)
                evaluation[arm] = assessment.evaluate(
                    evaluation_export, frozen, output / f"evaluate_{arm}", games=games)
            records = {arm: [json.loads(path.read_text(encoding="utf-8")) for path in
                             sorted((output / f"evaluate_{arm}" / "games").glob("game_*.json"))]
                       for arm in configs}
            matched = matched_delta(records["control"], records["joseki"], planned_games=games)
            limitations = ["single_training_seed_pilot"]
            if not training["joseki"]["joseki_games"]:
                limitations.append("no_treatment_games")
            if not training["joseki"]["joseki_samples"]:
                limitations.append("no_treatment_samples")
            if any(not row["training_steps"] for row in training.values()):
                limitations.append("no_training_updates")
            if any(not row["new_samples"] for row in training.values()):
                limitations.append("no_new_training_samples")
            if not matched["complete_planned_match"]:
                limitations.append("incomplete_or_truncated_evaluation")
            if matched["complete_pairs"] < 20:
                limitations.append("small_evaluation_sample")
            supported = matched["evidence_of_improvement"] and len(limitations) == 1
            report = {"status": "complete", "source_sha256": source["sha256"],
                      "source_model_sha256": manifest["source_model_sha256"],
                      "training_evaluation": manifest["training_evaluation"],
                      "training": training, "evaluation": evaluation, "matched": matched,
                      "conclusion": "pilot_evidence_of_improvement" if supported else "inconclusive",
                      "limitations": limitations,
                      "interpretation": "A single training seed cannot establish a general training benefit; repeat across seeds."}
            atomic_json(report, output / "report.json")
            update(status="complete", phase="complete")
            return report
        except BaseException as error:
            update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
