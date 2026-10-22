"""Predeclared continuation-seed replications of the native joseki experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from ..engine import BLACK, WHITE
from ..fileio import sha256_file
from .control import RunLock
from .opening_experiment import _copy_frozen, _digest, _freeze_context, compare_training, matched_delta
from .openings import JOSEKI_SUITE_VERSION, joseki_openings
from .storage import atomic_json, fingerprint, load_checkpoint


DEFAULT_TRAINING_SEEDS = tuple(range(2026101701, 2026101706))


def pair_deltas(control, treatment, *, planned_games: int, evaluation_seed: int) -> dict:
    """Reconstruct valid four-game matched units from saved evaluator records."""
    summary = matched_delta(control, treatment, planned_games=planned_games)
    by_arm = [{row["index"]: row for row in records} for records in (control, treatment)]
    deltas = []
    for index in range(0, len(control), 2):
        rows = [arm[index + offset] for arm in by_arm for offset in (0, 1)]
        for row in rows:
            opening = row.get("moves", [])[:2]
            if (row["seed"] != evaluation_seed + index // 2
                    or row["reason"] not in ("two_passes", "length_limit")
                    or len(opening) != 2 or [move[0] for move in opening] != [BLACK, WHITE]
                    or opening[0][1] == opening[1][1]):
                raise ValueError("Evaluation records do not match the declared random opening jobs")
        if any(row["reason"] == "length_limit" for row in rows):
            continue
        scores = [.5 if row["winner"] is None else float(row["winner"] == row["candidate_color"])
                  for row in rows]
        deltas.append((scores[2] + scores[3] - scores[0] - scores[1]) / 2)
    return {"deltas": deltas, "complete_planned_match": summary["complete_planned_match"],
            "excluded_truncated_pairs": summary["excluded_truncated_pairs"]}


def aggregate_replicates(results, *, planned_seeds, bootstrap_draws=10000,
                         bootstrap_seed=90720261017) -> dict:
    """Weight training seeds equally; opening pairs stay inside seed clusters."""
    planned_seeds = list(planned_seeds)
    if not planned_seeds or len(set(planned_seeds)) != len(planned_seeds):
        raise ValueError("Study plan has empty or duplicate training seeds")
    seeds = [row["training_seed"] for row in results]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Study results contain duplicate training seeds")
    if set(seeds) - set(planned_seeds):
        raise ValueError("Study results contain undeclared training seeds")
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("Bootstrap seed must be a nonnegative integer")
    ordered = sorted(results, key=lambda row: row["training_seed"])
    clusters, means = [], []
    for row in ordered:
        values = np.asarray(row["pair_deltas"], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or (np.abs(values) > 1).any():
            raise ValueError("Matched pair deltas must be finite values in [-1, 1]")
        clusters.append(values)
        means.append(float(values.mean()) if len(values) else None)
    positive, negative = sum(value is not None and value > 0 for value in means), sum(
        value is not None and value < 0 for value in means)
    decisive = positive + negative
    p_value = (sum(math.comb(decisive, wins) for wins in range(positive, decisive + 1))
               / 2 ** decisive) if decisive else 1.0
    interval = {"method": "hierarchical_percentile_bootstrap", "approximate": True,
                "confidence": .95, "draws": bootstrap_draws, "random_seed": bootstrap_seed,
                "clusters": len(ordered), "lower": None, "upper": None,
                "limitation": "Few training-seed clusters give an approximate and unstable interval."}
    if clusters and all(len(cluster) for cluster in clusters):
        rng = np.random.default_rng(bootstrap_seed)
        selected = rng.integers(len(clusters), size=(bootstrap_draws, len(clusters)))
        draws = np.zeros(bootstrap_draws, dtype=np.float64)
        for position in range(len(clusters)):
            for index, cluster in enumerate(clusters):
                targets = np.flatnonzero(selected[:, position] == index)
                sample = rng.integers(len(cluster), size=(len(targets), len(cluster)))
                draws[targets] += cluster[sample].mean(axis=1)
        draws /= len(clusters)
        interval["lower"], interval["upper"] = map(float, np.quantile(draws, [.025, .975]))
    complete = set(seeds) == set(planned_seeds)
    ready = bool(ordered) and all(
        row["complete_planned_match"] and not row["excluded_truncated_pairs"]
        and len(row["pair_deltas"]) * 2 == row["planned_games"]
        and all(row["training"][arm]["training_steps"] > 0
                and row["training"][arm]["new_samples"] > 0 for arm in ("control", "joseki"))
        and row["training"]["control"]["joseki_samples"] == 0
        and row["training"]["control"]["joseki_games"] == 0
        and row["training"]["joseki"]["joseki_samples"] > 0
        and row["training"]["joseki"]["joseki_games"] > 0 for row in ordered)
    supported = (complete and len(planned_seeds) >= 5 and positive == len(planned_seeds)
                 and p_value <= .05 and ready and interval["lower"] is not None
                 and interval["lower"] > 0)
    return {"unit_of_replication": "training_seed", "planned_training_seeds": planned_seeds,
            "training_seeds": len(ordered), "positive_training_seeds": positive,
            "negative_training_seeds": negative,
            "tied_training_seeds": sum(value == 0 for value in means),
            "uninformative_training_seeds": sum(value is None for value in means),
            "per_seed_mean_score_delta": [
                {"training_seed": row["training_seed"], "mean_score_delta": mean,
                 "complete_pairs": len(row["pair_deltas"])} for row, mean in zip(ordered, means)],
            "mean_score_delta": sum(means) / len(means) if means and None not in means else None,
            "seed_sign_p_value": p_value, "seed_sign_alpha": .05,
            "hierarchical_bootstrap": interval, "all_planned_replicates_complete": complete,
            "all_replicates_eligible": ready,
            "conclusion": "consistent_empirical_improvement" if supported else "inconclusive",
            "limitations": ["conditional_on_one_source_checkpoint_and_frozen_opponents",
                            "limited_training_seed_power", "approximate_cluster_bootstrap"],
            "interpretation": "The sign test concerns direction consistency across continuation seeds. "
                              "It does not establish a population mean gain. An inconclusive gate "
                              "does not establish no effect."}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _replicate_result(path, *, training_seed, evaluation_seed, planned_games, source_sha256):
    report = _read(path / "report.json")
    manifest = _read(path / "manifest.json")
    arguments = manifest["arguments"]
    if (arguments.get("training_seed") != training_seed
            or arguments.get("evaluation_seed") != evaluation_seed):
        raise ValueError("Actual replicate seeds do not match the predeclared study seeds")
    if manifest["source"]["sha256"] != source_sha256:
        raise ValueError("Replicate manifest does not match the frozen source")
    if report.get("status") != "complete" or report["source_sha256"] != source_sha256:
        raise ValueError("Replicate report does not match the frozen source")
    records = {arm: [_read(record) for record in sorted(
        (path / f"evaluate_{arm}" / "games").glob("game_*.json"))] for arm in ("control", "joseki")}
    matched = pair_deltas(records["control"], records["joseki"],
                          planned_games=planned_games, evaluation_seed=evaluation_seed)
    mean = sum(matched["deltas"]) / len(matched["deltas"]) if matched["deltas"] else None
    if mean != report["matched"]["mean_score_delta"]:
        raise ValueError("Saved game records disagree with the replicate report")
    for arm in ("control", "joseki"):
        if report["evaluation"][arm]["games"] != len(records[arm]):
            raise ValueError("Saved game counts disagree with the replicate report")
    return {"training_seed": training_seed, "evaluation_seed": evaluation_seed,
            "attempt": str(path), "planned_games": planned_games,
            "pair_deltas": matched["deltas"],
            "complete_planned_match": matched["complete_planned_match"],
            "excluded_truncated_pairs": matched["excluded_truncated_pairs"],
            "training": report["training"],
            "evaluation_records_sha256": _digest(records),
            "report_sha256": sha256_file(path / "report.json"),
            "manifest_sha256": sha256_file(path / "manifest.json")}


def _verify_snapshot(output, manifest):
    source = output / "source"
    expected = {"latest.pt": manifest["source"]["sha256"], **{
        row["relative_path"]: row["sha256"] for row in manifest["frozen_context"]}}
    actual = {str(path.relative_to(source)) for path in source.rglob("*.pt")}
    if actual != set(expected) or any(sha256_file(source / name) != checksum
                                      for name, checksum in expected.items()):
        raise ValueError("Study frozen source or opponent snapshot changed")


def run_study(resume: Path, output: Path, *, training_seeds=DEFAULT_TRAINING_SEEDS,
              iterations=10, fraction=.25, games=200, visits=32,
              training_games=16, training_steps=100, training_visits=64,
              training_leaf_batch_size=8, evaluation_leaf_batch_size=4,
              evaluation_length_factor=6.0, evaluation_seed_base=810000001,
              checkpoint_retention=2, hardware_overrides=None,
              bootstrap_draws=10000, bootstrap_seed=90720261017,
              continue_study=False) -> dict:
    """Predeclare, freeze, execute, and durably aggregate independent seed pairs."""
    resume, output = Path(resume).resolve(), Path(output).resolve()
    seeds = list(training_seeds)
    for name, value in (("iterations", iterations), ("games", games), ("visits", visits),
                        ("training_games", training_games), ("training_steps", training_steps),
                        ("training_visits", training_visits), ("training_leaf_batch_size", training_leaf_batch_size),
                        ("evaluation_leaf_batch_size", evaluation_leaf_batch_size),
                        ("checkpoint_retention", checkpoint_retention), ("bootstrap_draws", bootstrap_draws)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if games % 2 or games < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Study needs an even evaluation count and distinct training seeds")
    if (any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)
            or type(evaluation_seed_base) is not int or evaluation_seed_base < 0
            or type(bootstrap_seed) is not int or bootstrap_seed < 0
            or isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1
            or isinstance(evaluation_length_factor, bool)
            or not math.isfinite(evaluation_length_factor) or evaluation_length_factor < 1):
        raise ValueError("Study seeds, fraction, or evaluation length factor are invalid")
    evaluation_seeds = [evaluation_seed_base + index * 10000 for index in range(len(seeds))]
    seed_sets = [set(range(seed, seed + games // 2)) for seed in evaluation_seeds]
    if (games // 2 > 10000 or any(values & set(range(2026, 2066)) for values in seed_sets)
            or evaluation_seeds[-1] + games // 2 > 2 ** 32):
        raise ValueError("Evaluation opening seed ranges overlap or reuse prior pilot openings")
    source_run = resume.parent.parent if resume.parent.name == "checkpoints" else resume.parent
    if output.is_relative_to(source_run):
        raise ValueError("Study output must be outside the source run")
    if output.exists() and not continue_study:
        raise ValueError("Study requires a fresh output directory, or --continue-study")
    if continue_study and not (output / "manifest.json").is_file():
        raise ValueError("Continuation requires an existing study manifest")
    prior = _read(output / "manifest.json") if continue_study else None
    source_checksum = prior["source"]["sha256"] if prior else sha256_file(resume)
    checkpoint = output / "source/latest.pt" if prior else resume
    payload, config = load_checkpoint(checkpoint)
    if payload.get("kind") != "training":
        raise ValueError("Study requires a full training checkpoint")
    suite = [{"name": opening.name, "actions": list(opening.actions)}
             for opening in joseki_openings(config.game.board_size)]
    plan = {"resume": str(resume), "output": str(output), "training_seeds": seeds,
            "evaluation_seeds": evaluation_seeds, "iterations": iterations,
            "fraction": fraction, "games": games, "visits": visits,
            "training_games": training_games, "training_steps": training_steps,
            "training_visits": training_visits, "training_leaf_batch_size": training_leaf_batch_size,
            "evaluation_leaf_batch_size": evaluation_leaf_batch_size,
            "evaluation_length_factor": evaluation_length_factor,
            "checkpoint_retention": checkpoint_retention,
            "skip_training_evaluation": True, "freeze_training_opponents": True,
            "hardware_overrides_sha256": sha256_file(hardware_overrides) if hardware_overrides else None,
            "source_sha256": source_checksum, "joseki_suite_version": JOSEKI_SUITE_VERSION,
            "joseki_suite_sha256": _digest(suite), "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "gates": {"minimum_training_seeds": 5, "all_seed_means_positive": True,
                      "seed_sign_alpha": .05, "approximate_bootstrap_lower_above": 0,
                      "complete_no_truncation": True, "actual_treatment_and_updates": True}}
    plan_hash = _digest(plan)
    if prior and (prior["plan_sha256"] != plan_hash or _digest(prior["plan"]) != plan_hash):
        raise ValueError("Continuation arguments do not match the predeclared study plan")
    if not prior:
        output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "phase": "freezing", "plan_sha256": plan_hash}

    def update(**changes):
        status.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(status, output / "status.json")

    with RunLock(output):
        try:
            update()
            if prior:
                manifest = prior
            else:
                snapshot = output / "source/latest.pt"
                source = _copy_frozen(resume, snapshot)
                if source["sha256"] != source_checksum:
                    raise ValueError("Source checkpoint changed before the study snapshot")
                context = _freeze_context(source_run, output, payload, config)
                for name in ("anchors", "checkpoints"):
                    directory = output / "frozen_context" / name
                    if directory.exists():
                        directory.rename(output / "source" / name)
                if (output / "frozen_context").exists():
                    (output / "frozen_context").rmdir()
                manifest = {"schema_version": 1, "plan": plan, "plan_sha256": plan_hash,
                            "source": source, "source_iteration": payload["iteration"],
                            "source_model_sha256": fingerprint(payload["model"]),
                            "frozen_context": context, "joseki_openings": suite}
                if hardware_overrides:
                    manifest["hardware"] = _copy_frozen(Path(hardware_overrides),
                                                       output / "hardware.json")
                atomic_json(manifest, output / "manifest.json")
            _verify_snapshot(output, manifest)
            results = []
            common = {key: plan[key] for key in (
                "iterations", "fraction", "games", "visits", "training_games", "training_steps",
                "training_visits", "training_leaf_batch_size", "evaluation_leaf_batch_size",
                "evaluation_length_factor", "checkpoint_retention", "skip_training_evaluation",
                "freeze_training_opponents")}
            if hardware_overrides:
                if sha256_file(output / "hardware.json") != plan["hardware_overrides_sha256"]:
                    raise ValueError("Frozen hardware overrides changed")
                common["hardware_overrides"] = output / "hardware.json"
            for seed, evaluation_seed in zip(seeds, evaluation_seeds):
                update(phase="replicate", training_seed=seed, completed_replicates=len(results))
                parent = output / "replicates" / f"seed_{seed}"
                completed = None
                attempts = sorted(parent.glob("attempt-*"))
                for attempt in attempts:
                    receipt_path = attempt / "study_replicate.json"
                    if not receipt_path.is_file():
                        continue
                    receipt = _read(receipt_path)
                    if (receipt.get("plan_sha256") != plan_hash or receipt.get("training_seed") != seed
                            or receipt.get("evaluation_seed") != evaluation_seed):
                        raise ValueError("Completed replicate receipt does not match study plan")
                    if (sha256_file(attempt / "report.json") != receipt["report_sha256"]
                            or sha256_file(attempt / "manifest.json") != receipt["manifest_sha256"]):
                        raise ValueError("Completed replicate artifacts changed")
                    completed = _replicate_result(attempt, training_seed=seed,
                        evaluation_seed=evaluation_seed, planned_games=games, source_sha256=source_checksum)
                    comparison = {key: value for key, value in completed.items()
                                  if key != "evaluation_records_sha256"
                                  or "evaluation_records_sha256" in receipt}
                    if _digest(comparison) != _digest({key: receipt[key] for key in comparison}):
                        raise ValueError("Completed replicate game records changed")
                    if "evaluation_records_sha256" not in receipt:
                        # Schema-1 receipts from an in-flight study already
                        # pin deltas, reports, and manifests. Certify the current
                        # full records only after those older checks agree.
                        atomic_json({"plan_sha256": plan_hash, **completed,
                                     "legacy_receipt_certification": {
                                         "method": "revalidated_existing_receipt_and_current_records",
                                         "certified_at": datetime.now(timezone.utc).isoformat()}},
                                    receipt_path)
                if completed is None:
                    number = max([int(path.name.split("-")[-1]) for path in attempts] + [0]) + 1
                    attempt = parent / f"attempt-{number:04d}"
                    _verify_snapshot(output, manifest)
                    compare_training(output / "source/latest.pt", attempt, training_seed=seed,
                                     evaluation_seed=evaluation_seed, **common)
                    completed = _replicate_result(attempt, training_seed=seed,
                        evaluation_seed=evaluation_seed, planned_games=games, source_sha256=source_checksum)
                    atomic_json({"plan_sha256": plan_hash, **completed}, attempt / "study_replicate.json")
                results.append(completed)
                atomic_json({"plan_sha256": plan_hash, "replicates": results}, output / "results.json")
                aggregation = aggregate_replicates(results, planned_seeds=seeds,
                    bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed)
                atomic_json({"status": "running", "plan_sha256": plan_hash,
                             "replicates": results, **aggregation}, output / "report.json")
            report = {"status": "complete", "plan_sha256": plan_hash,
                      "source_sha256": source_checksum, "replicates": results, **aggregation}
            atomic_json(report, output / "report.json")
            update(status="complete", phase="complete", completed_replicates=len(results))
            return report
        except BaseException as error:
            update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
