"""Finite serial pilots from a prepared mixed-board experiment.

Preparation is repeatable before execution. Execution is deliberately one-shot:
an interrupted pilot keeps all artifacts but cannot silently receive extra time.
Frozen evaluation directories can be resumed with mixed-evaluate separately.
"""
from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .fileio import (acquire_file_lock, release_file_lock, atomic_write_json,
                     local_timestamp, sha256_file)
from .katago_mixed import MixedBoardRunner
from .katago_mixed_eval import evaluate_mixed
from .katago_mixed_experiments import _checkpoint_counters, verify_prepared_experiment
from .katago_mixed_health import build_mixed_health
from .katago_rl import PROJECT_ROOT, KataGoRLRunnerError
from .process_control import managed_popen
from .replay_accounting import ReplayRowLedger


def run_budgeted_process(command: list[str], log_path: Path, *, wall_seconds: int) -> dict:
    """Interrupt at a fixed deadline, distinguishing shutdown from training time."""
    if type(wall_seconds) is not int or wall_seconds <= 0:
        raise ValueError("wall_seconds must be a positive integer")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log, managed_popen(
            command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT) as process:
        try:
            code = process.wait(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            stopped = time.monotonic()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                code = process.wait(timeout=45)
            except subprocess.TimeoutExpired as error:
                # The managed context removes the process tree on this error.
                raise KataGoRLRunnerError(f"Training did not stop after SIGINT; see {log_path}") from error
            if code not in (0, 130, -signal.SIGINT):
                raise KataGoRLRunnerError(f"Training exit {code} during budget shutdown; see {log_path}")
        else:
            raise KataGoRLRunnerError(f"Training exited before its fixed budget (exit {code}); see {log_path}")
    ended = time.monotonic()
    return {"pid": process.pid, "command": command, "returncode": code,
            "budget_seconds": wall_seconds, "training_seconds": stopped - started,
            "shutdown_seconds": ended - stopped, "stdout": str(log_path),
            "stop_reason": "fixed_wall_budget", "finished_at": local_timestamp()}


def summarize_budget(initial: dict, final: dict, records: list[dict]) -> dict:
    completed = [r for r in records if r.get("trained_samples", 0) > initial["global_step_samples"]]
    stages = ("selfplay_stage_seconds", "shuffle_seconds", "train_seconds", "export_seconds")
    return {
        "trained_samples_delta": final["global_step_samples"] - initial["global_step_samples"],
        "trainer_data_rows_delta": final["total_num_data_rows"] - initial["total_num_data_rows"],
        "completed_cycles": len(completed),
        "completed_stages_seconds": {key: sum(r.get(key, 0) or 0 for r in completed) for key in stages},
        "generated_rows_in_completed_cycles": sum(r.get("selfplay_data_rows", 0) or 0 for r in completed),
        "timing_note": "Stage sums include completed cycles only; partial work and startup are included in wall time.",
    }


def _export_checkpoint(runner: MixedBoardRunner) -> Path:
    checkpoint = runner.run_root / "train/gogogo/checkpoint.ckpt"
    counters = _checkpoint_counters(checkpoint)
    name = f"gogogo-s{counters['global_step_samples']}-d{counters['total_num_data_rows']}"
    target = runner.run_root / "torchmodels_toexport" / name / "model.ckpt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(checkpoint, target)
    runner.export_new_models()
    model = runner.models_dir / name / "model.bin.gz"
    if not model.is_file():
        raise KataGoRLRunnerError(f"No model exported for completed checkpoint: {checkpoint}")
    return model


def _validate_preparation(output: Path, manifest: dict) -> None:
    if manifest.get("kind") != "mixed_fixed_budget_ab" or manifest.get("schema_version") != 1:
        raise ValueError("Expected a compatible prepared mixed experiment")
    if set(manifest.get("arms", {})) != {"control", "sgf_starts"}:
        raise ValueError("Expected control and sgf_starts arms")
    for name, arm in manifest["arms"].items():
        if arm != {"config_path": str(output / "arms" / name / "config.json"),
                   "run_root": str(output / "arms" / name / "run")}:
            raise ValueError(f"Experiment arm path changed: {name}")
    if manifest.get("snapshot") != {"run_root": str(output / "common")}:
        raise ValueError("Experiment snapshot path changed")
    # The strict verifier checks the complete expected inventory, identities,
    # checkpoint hashes, all NPZs (including unlisted extras), and SGF pools.
    # It uses the frozen common snapshot, never a moving baseline checkpoint.
    if verify_prepared_experiment(output) != manifest:
        raise ValueError("Prepared manifest changed during verification")


def run_prepared_experiment(output_root: Path, *, hardware_profile,
                            games_per_board: int = 12, visits: int = 32) -> dict:
    """Run each prepared arm once for the same wall budget, then frozen matches.

    The source training lock must be available. This prevents this controller
    from contending with the baseline job on a single GPU. It never stops or
    restarts user processes itself.
    """
    if type(games_per_board) is not int or games_per_board < 2 or games_per_board % 2:
        raise ValueError("Evaluation games must be a positive even number, at least two")
    if type(visits) is not int or visits < 1:
        raise ValueError("Evaluation visits must be positive")
    if hardware_profile is None:
        raise ValueError("A detected hardware profile is required for a fixed-budget pilot")
    output = Path(output_root).resolve(strict=True)
    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    source_runner = MixedBoardRunner(Path(source["config"]["path"]),
                                    run_root_override=Path(source["run_root"]))
    with source_runner.lock():
        handle = acquire_file_lock(output / "controller.lock")
        try:
            return _run_prepared(output, manifest_path, manifest, hardware_profile,
                                 games_per_board, visits)
        finally:
            release_file_lock(handle)


def _run_prepared(output, manifest_path, manifest, profile, games, visits):
    state_path = output / "trial_result.json"
    if state_path.exists():
        raise ValueError("Pilot execution already started; preserve its budget and use a new experiment directory")
    _validate_preparation(output, manifest)
    wall_seconds = manifest["budget"]["wall_seconds"]
    if type(wall_seconds) is not int or wall_seconds < 1:
        raise ValueError("Invalid prepared wall budget")
    state = {"schema_version": 1, "status": "running", "started_at": local_timestamp(),
             "experiment_manifest_sha256": sha256_file(manifest_path),
             "hardware": profile.to_dict(),
             "evaluation_protocol": {"games_per_board": games, "visits": visits},
             "arms": {}, "automatic_promotion": False,
             "interpretation": "Single-run finite pilot; independent randomness and small samples do not establish causal improvement."}
    atomic_write_json(state_path, state)
    try:
        control = manifest["arms"]["control"]
        control_runner = MixedBoardRunner(Path(control["config_path"]), hardware_profile=profile)
        # Export a common pre-treatment SWA reference before either arm changes.
        with control_runner.lock():
            baseline_source = _export_checkpoint(control_runner)
        baseline = output / "common/baseline.bin.gz"
        shutil.copy2(baseline_source, baseline)
        state["baseline"] = {"file": str(baseline), "sha256": sha256_file(baseline)}
        atomic_write_json(state_path, state)
        candidates = {}
        for name in ("control", "sgf_starts"):
            arm = manifest["arms"][name]
            runner = MixedBoardRunner(Path(arm["config_path"]), hardware_profile=profile)
            checkpoint = runner.run_root / "train/gogogo/checkpoint.ckpt"
            with runner.lock():
                before = _checkpoint_counters(checkpoint)
                before_boards = runner.board_statistics()
                # Both arms receive their initial model outside the timed
                # interval, including the control used for the reference above.
                _export_checkpoint(runner)
            command = [sys.executable, str(PROJECT_ROOT / "train_rl.py"), "mixed",
                       "--mixed-config", arm["config_path"], "--hardware", profile.tier]
            state["active_arm"] = name
            atomic_write_json(state_path, state)
            timing = run_budgeted_process(command, runner.run_root / "training.stdout.log",
                                           wall_seconds=wall_seconds)
            with runner.lock():
                after = _checkpoint_counters(checkpoint)
                after_boards = runner.board_statistics()
                replay = ReplayRowLedger(runner.run_root).sync()
                records = runner.metric_store._read_existing().values()
                accounting = summarize_budget(before, after, list(records))
                if accounting["trained_samples_delta"] <= 0:
                    raise KataGoRLRunnerError(f"Arm {name} completed no training updates within its budget")
                candidate = _export_checkpoint(runner)
                frozen = output / "frozen" / f"{name}.bin.gz"
                frozen.parent.mkdir(exist_ok=True)
                shutil.copy2(candidate, frozen)
                candidates[name] = frozen
                state["arms"][name] = {"timing": timing, "initial_counters": before,
                    "final_counters": after, "accounting": accounting,
                    "retained_rows_by_board_before": before_boards["rows_by_board"],
                    "retained_rows_by_board_after": after_boards["rows_by_board"],
                    "games_by_board": after_boards["games_by_board"],
                    "generated_rows_delta": replay.logical_rows - manifest["initial_counters"]["logical_replay_rows"],
                    "candidate": {"source": str(candidate), "file": str(frozen), "sha256": sha256_file(frozen)},
                    "health": build_mixed_health(runner.run_root, runner.mixed.board_sizes)}
            atomic_write_json(state_path, state)
        state["active_arm"] = None
        for name, candidate in candidates.items():
            arm = manifest["arms"][name]
            runner = MixedBoardRunner(Path(arm["config_path"]), hardware_profile=profile)
            state["arms"][name]["evaluation"] = evaluate_mixed(
                runner, output / "evaluation" / name, candidate=candidate, baseline=baseline,
                games_per_board=games, visits=visits)
            atomic_write_json(state_path, state)
        state.update(status="complete", finished_at=local_timestamp(), decision="retain_baseline_pending_replication")
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=str(error), finished_at=local_timestamp())
        atomic_write_json(state_path, state)
        raise
    atomic_write_json(state_path, state)
    return state
