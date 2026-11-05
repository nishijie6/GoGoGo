"""Prepare independent one-factor resource candidates from a frozen SGF pilot.

Only immutable common inputs are read, so the pilot's active arms may continue.
No training, evaluation, process control or promotion is performed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from .fileio import atomic_write_json, local_timestamp, sha256_file
from .katago_mixed import MixedBoardRunner
from .katago_mixed_experiments import _checkpoint_counters, _copy_checked, _read_json, _record
from .katago_rl import PROJECT_ROOT
from .replay_accounting import ReplayRowLedger
from .rl_retention import npz_row_count


_CHECKPOINT = "train/gogogo/checkpoint.ckpt"
_FACTORS = {
    "control": None,
    "search64": ("training.overrides.search.simulations_per_move", 64),
    "cycle96": ("training.overrides.self_play.games_per_iteration", 96),
    "replay25000": ("training.overrides.optimizer.replay_buffer_capacity", 25000),
    "more19": ("board_weights", [2, 2, 3, 2, 2]),
}


def _core(record: dict) -> dict:
    return {key: record[key] for key in ("sha256", "size", "mtime_ns")}


def _frozen_inputs(base: Path) -> tuple[dict, dict, dict, dict, dict]:
    manifest_path = base / "experiment_manifest.json"
    manifest_record = _record(manifest_path)
    manifest = _read_json(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "mixed_fixed_budget_ab"
            or manifest.get("snapshot") != {"run_root": str(base / "common")}):
        raise ValueError("Expected an original prepared SGF experiment manifest")
    try:
        source, prepared = manifest["source"], manifest["prepared_files"]
        replay = source["replay"]["files"]
        expected = {"source_config.json": _core(source["config"]),
                    _CHECKPOINT: _core(source["checkpoint"]),
                    **{relative: _core(record) for relative, record in replay.items()}}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Original experiment manifest inventory is incomplete") from error
    common = base / "common"
    observed = {path.relative_to(common).as_posix() for path in common.glob("selfplay/**/tdata/*.npz")}
    if not replay or observed != set(replay):
        raise ValueError("Frozen replay snapshot inventory changed")
    for relative, record in expected.items():
        path = common / relative
        if (not path.resolve().is_relative_to(common)
                or prepared.get(f"common/{relative}") != record or _record(path) != record):
            raise ValueError(f"Frozen snapshot artifact/inventory changed: {relative}")
    raw = _read_json(common / "source_config.json")
    if raw.get("exploration") is not None:
        raise ValueError("Budget candidates do not support source exploration dependencies")
    runner = MixedBoardRunner(common / "source_config.json", run_root_override=common)
    if runner.manifest_identity() != source["mixed_manifest"]["identity"]:
        raise ValueError("Frozen source config differs from its original mixed manifest")
    counters = _checkpoint_counters(common / _CHECKPOINT)
    physical = sum(npz_row_count(common / relative) for relative in replay)
    old_rows = sum(npz_row_count(common / relative) for relative, record in replay.items()
                   if record["mtime_ns"] <= expected[_CHECKPOINT]["mtime_ns"])
    offset = max(0, max(counters["total_num_data_rows"], counters["train_bucket_level_at_row"]) - old_rows)
    counters.update(retained_replay_rows=physical, deleted_rows_offset=offset,
                    logical_replay_rows=offset + physical)
    if counters != manifest.get("initial_counters") or _record(manifest_path) != manifest_record:
        raise ValueError("Frozen snapshot counters or original manifest changed")
    budget = manifest.get("budget", {})
    if type(budget.get("wall_seconds")) is not int or budget["wall_seconds"] < 1 or budget.get("per_arm") is not True:
        raise ValueError("Original experiment needs a positive equal per-arm wall budget")
    return manifest, raw, expected, counters, manifest_record


def _set(config: dict, path: str, value) -> object:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    before = deepcopy(target.get(parts[-1]))
    target[parts[-1]] = value
    return before


def prepare_budget_candidates(base_experiment: Path, output_root: Path) -> dict:
    """Copy common full learner state into five isolated preparation-only runs.

    Every output must be fresh. All replay is retained initially, including for
    replay25000: its first retention pass, after training, applies the new cap.
    Commands require the same hardware and wall budget; cycle96 explicitly
    overrides the hardware profile's automatic games-per-iteration cap.
    """
    base = Path(base_experiment).resolve(strict=True)
    output_path = Path(output_root)
    output = output_path.resolve()
    if output_path.is_symlink() or output.exists():
        raise ValueError("Budget candidate output must be fresh and unidentified output cannot be adopted")
    manifest, raw, files, counters, manifest_record = _frozen_inputs(base)
    source_run = Path(manifest["source"]["run_root"]).resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (base, source_run)):
        raise ValueError("Budget candidate output must not overlap the source experiment or original run")
    base_runner = MixedBoardRunner(base / "common/source_config.json", run_root_override=base / "common")
    mixed, training = base_runner.mixed, base_runner.config
    if mixed.board_sizes != (7, 8, 9, 13, 19) or mixed.board_weights != (2, 2, 3, 2, 1):
        raise ValueError("more19 requires the default ordered five boards and weights [2,2,3,2,1]")
    if training.search.fast_simulations_per_move != 8:
        raise ValueError("search64 preserves cheap search at 8; source cheap search must already be 8")
    if (training.search.simulations_per_move == 64 or training.self_play.games_per_iteration == 96
            or training.optimizer.replay_buffer_capacity == 25000):
        raise ValueError("A candidate target already matches the source; each variant must change one factor")
    hardware = training.hardware.device if training.hardware.device in ("mps", "cpu") else "auto"
    output.mkdir(parents=True, exist_ok=False)
    profiles = {}
    for name, factor in _FACTORS.items():
        root, config_path = output / name / "run", output / name / "config.json"
        config = deepcopy(raw)
        mutations = []
        if factor is not None:
            path, value = factor
            mutations.append({"path": path, "before": _set(config, path, deepcopy(value)), "after": value})
        previous_output = _set(config, "training.overrides.runtime.output_directory", str(root))
        atomic_write_json(config_path, config, allow_nan=False)
        runner = MixedBoardRunner(config_path)
        runner.ensure_manifest()  # Identify the new run before introducing seeded data.
        for relative, expected in files.items():
            if relative != "source_config.json":
                _copy_checked(base / "common" / relative, root / relative, expected)
        ledger = ReplayRowLedger(root).sync()
        if (ledger.physical_rows, ledger.deleted_rows_offset, ledger.logical_rows) != (
                counters["retained_replay_rows"], counters["deleted_rows_offset"], counters["logical_replay_rows"]):
            raise ValueError(f"Candidate replay counters differ from frozen snapshot: {name}")
        normalized = deepcopy(config)
        _set(normalized, "training.overrides.runtime.output_directory", "<candidate-run>")
        identity = {"name": name, "config": normalized, "source_files": files, "budget": manifest["budget"]}
        profile_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        command = [sys.executable, str(PROJECT_ROOT / "train_rl.py"), "mixed", "--mixed-config",
                   str(config_path), "--hardware", hardware]
        if name == "cycle96":
            command.extend(("--games", "96"))
        profiles[name] = {
            "config_path": str(config_path), "run_root": str(root), "mutations": mutations,
            "output_directory_change": {"before": previous_output, "after": str(root)},
            "profile_identity": profile_hash, "config_sha256": sha256_file(config_path),
            "manifest_identity": runner.manifest_identity(),
            "initial_replay_policy": "preserve_full_window_then_apply_capacity",
            "note": "The first retention pass applies the candidate capacity; preparation preserves all initial rows.",
            "execution": {"api": "weiqi.katago_mixed_trials.run_budgeted_process", "command": command,
                          "log_path": str(root / "training.stdout.log"), "wall_seconds": manifest["budget"]["wall_seconds"],
                          "effective_overrides": {"games": 96} if name == "cycle96" else {}},
        }
    _after, _raw, after_files, after_counters, after_manifest = _frozen_inputs(base)
    if after_files != files or after_counters != counters or after_manifest != manifest_record:
        raise ValueError("Frozen source changed during candidate preparation")
    report = {"schema_version": 1, "kind": "mixed_budget_candidates", "created_at": local_timestamp(),
              "source": {"base_experiment": str(base), "manifest_sha256": manifest_record["sha256"],
                         "common_root": str(base / "common"), "files": files},
              "initial_counters": counters, "budget": manifest["budget"], "profiles": profiles,
              "automatic_promotion": False,
              "execution_note": "Preparation only. Export the common initial model outside each equal wall budget, "
                  "then run candidates serially via the recorded API on the same explicit hardware tier."}
    atomic_write_json(output / "budget_candidates_manifest.json", report, allow_nan=False)
    return report


__all__ = ["prepare_budget_candidates"]
