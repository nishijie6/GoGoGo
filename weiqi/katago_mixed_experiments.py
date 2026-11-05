"""Prepare isolated, reproducible mixed-board training A/B experiments.

Callers must stop or pause every source writer (including replay retention)
before calling :func:`prepare_mixed_experiment`. File inventories and hashes
detect observed changes; they are not a substitute for quiescing the source.
This module prepares artifacts only and never starts or stops training jobs.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from .fileio import atomic_write_json, atomic_write_text, local_timestamp, sha256_file
from .katago_mixed import MixedBoardRunner
from .replay_accounting import ReplayRowLedger
from .rl_config import resolve_rl_training_config
from .rl_retention import npz_row_count


EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_MANIFEST = "experiment_manifest.json"
TRAINING_START_DOMAIN = "gogogo/mixed/training-starts/v2"
TRAINING_START_SEED = 20261024
_START_STONE_COUNTS = (2, 4, 6)
_MAX_WEIGHT_UNITS = 100
_CHECKPOINT = Path("train/gogogo/checkpoint.ckpt")
_ARM_NAMES = ("control", "sgf_starts")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read compatible experiment manifest/config: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _record(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular, non-symlink artifact: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source/artifact changed during hashing; pause all writers: {path}")
    return {"sha256": digest, "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def _checkpoint_counters(path: Path) -> dict:
    import torch

    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError) as error:
            # Legacy torch.save files are loadable but cannot be memory mapped.
            if isinstance(error, RuntimeError) and "mmap" not in str(error):
                raise
            state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Cannot load full training checkpoint on CPU: {path}") from error
    if (not isinstance(state, dict)
            or not isinstance(state.get("model"), dict) or not state["model"]
            or not isinstance(state.get("optimizer"), dict) or not state["optimizer"]
            or not isinstance(state.get("train_state"), dict)):
        raise ValueError("A full checkpoint with model, optimizer and train_state is required")
    counters = {}
    for name in ("global_step_samples", "total_num_data_rows", "train_bucket_level_at_row"):
        value = state["train_state"].get(name, 0 if name == "train_bucket_level_at_row" else None)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid full checkpoint train_state counter: {name}")
        counters[name] = value
    return counters


def _require_unexplored_baseline(raw: dict) -> None:
    if raw.get("exploration") is not None:
        raise ValueError("This experiment version requires a source baseline without exploration")


def _source_inventory(source: Path, config_path: Path) -> tuple[dict, dict, dict]:
    config_record = _record(config_path)
    raw = _read_json(config_path)
    _require_unexplored_baseline(raw)
    runner = MixedBoardRunner(config_path, run_root_override=source)
    manifest_path = source / "mixed_manifest.json"
    manifest_record = _record(manifest_path)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("identity") != runner.manifest_identity():
        raise ValueError("Source mixed manifest is incompatible with the supplied config")
    checkpoint = source / _CHECKPOINT
    checkpoint_record = _record(checkpoint)
    counters = _checkpoint_counters(checkpoint)
    files = {}
    for path in sorted(source.glob("selfplay/**/tdata/*.npz")):
        try:
            path.resolve(strict=True).relative_to(source)
        except (OSError, ValueError) as error:
            raise ValueError(f"Replay path leaves source run: {path}") from error
        record = _record(path)
        try:
            rows = npz_row_count(path)
        except Exception as error:
            raise ValueError(f"Invalid retained replay NPZ: {path}") from error
        if rows <= 0:
            raise ValueError(f"Retained replay NPZ has no rows: {path}")
        files[path.relative_to(source).as_posix()] = {**record, "rows": rows}
    if not files:
        raise ValueError("A retained selfplay replay window is required for the experiment")
    physical = sum(item["rows"] for item in files.values())
    accounted = sum(item["rows"] for item in files.values()
                    if item["mtime_ns"] <= checkpoint_record["mtime_ns"])
    watermark = max(counters["total_num_data_rows"], counters["train_bucket_level_at_row"])
    offset = max(0, watermark - accounted)
    counters.update(retained_replay_rows=physical, deleted_rows_offset=offset,
                    logical_replay_rows=offset + physical)
    inventory = {
        "run_root": str(source),
        "config": {"path": str(config_path), **config_record},
        "mixed_manifest": {"identity": manifest["identity"], **manifest_record},
        "checkpoint": {"relative_path": _CHECKPOINT.as_posix(), **checkpoint_record},
        "replay": {"files": files, "physical_rows": physical},
    }
    return raw, inventory, counters


def _weight_units(weights: list[float]) -> tuple[list[int], list[Fraction]]:
    fractions = [Fraction(str(weight)) for weight in weights]
    total = sum(fractions)
    probabilities = [weight / total for weight in fractions]
    denominator = math.lcm(*(weight.denominator for weight in fractions))
    ratios = [int(weight * denominator) for weight in fractions]
    divisor = math.gcd(*ratios)
    units = [ratio // divisor for ratio in ratios]
    if sum(units) > _MAX_WEIGHT_UNITS:
        # Keep every requested size/depth represented, then allocate remaining
        # units to the largest deficit. Fractions avoid overflow for finite but
        # extremely different input floats and make ties deterministic.
        units = [1] * len(weights)
        targets = [probability * _MAX_WEIGHT_UNITS for probability in probabilities]
        for _ in range(_MAX_WEIGHT_UNITS - len(units)):
            index = max(range(len(units)), key=lambda i: (targets[i] - units[i], -i))
            units[index] += 1
    return units, probabilities


def _training_start_plan(boards: list[int], weights: list[float], komi: float) -> tuple[dict[str, str], dict]:
    units, requested = _weight_units(weights)
    starts = {}
    for size, count_per_depth in zip(boards, units):
        points = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)
                  if (x + y) % 2 == 0]
        for stones in _START_STONE_COUNTS:
            seed = hashlib.sha256(
                f"{TRAINING_START_DOMAIN}:{TRAINING_START_SEED}:{size}:{stones}".encode()).digest()
            rng = random.Random(seed)
            seen = set()
            for index in range(count_per_depth):
                for _ in range(10000):
                    chosen = rng.sample(points, stones)
                    black, white = tuple(sorted(chosen[::2])), tuple(sorted(chosen[1::2]))
                    if (black, white) not in seen:
                        seen.add((black, white))
                        break
                else:
                    raise ValueError(f"Cannot generate enough unique legal training starts for {size}")
                # Nonadjacent interior points have four liberties each. Balanced
                # setup stones are reachable by alternating noncapturing moves.
                placements = "".join(color + "".join(f"[{chr(97+x)}{chr(97+y)}]" for x, y in group)
                                     for color, group in (("AB", black), ("AW", white)))
                starts[f"board-{size}-stones-{stones}-position-{index:03d}.sgf"] = (
                    f"(;GM[1]FF[4]CA[UTF-8]AP[GoGoGo:training-starts-v2]SZ[{size}]"
                    f"KM[{komi:g}]RU[Chinese]PL[B]{placements}"
                    f"C[{TRAINING_START_DOMAIN} seed={TRAINING_START_SEED}])\n")
    total_units = sum(units)
    actual = [Fraction(count, total_units) for count in units]
    sampling = {
        "position_count": len(starts), "maximum_position_count": len(_START_STONE_COUNTS) * _MAX_WEIGHT_UNITS,
        "positions_per_sgf": 1, "format": "single root AB/AW setup with PL[B], no moves or variations",
        "pool_counts_by_board": {str(size): count * len(_START_STONE_COUNTS) for size, count in zip(boards, units)},
        "pool_counts_by_board_and_stones": {str(size): {str(stones): count for stones in _START_STONE_COUNTS}
                                            for size, count in zip(boards, units)},
        "requested_board_probabilities": {str(size): float(p) for size, p in zip(boards, requested)},
        "actual_pool_board_probabilities": {str(size): float(p) for size, p in zip(boards, actual)},
        "nominal_overall_board_probabilities": {
            str(size): float(Fraction(9, 10) * target + Fraction(1, 10) * pool)
            for size, target, pool in zip(boards, requested, actual)},
        "exact_board_weights": requested == actual,
        "max_abs_pool_probability_error": float(max(abs(target - pool) for target, pool in zip(requested, actual))),
        "upstream_contract": ("Sgf::iterAllUniquePositions applies root placements then emits one position; "
                              "global uniqueness, startPosesLoadProb=1 and startPosesTurnWeightLambda=0 give uniform pool weights."),
        "limitation": ("Bounded integer counts approximate arbitrary floating board weights; one unit per size/depth is retained. "
                       "The nominal 90% ordinary/10% SGF mixture is not a finite-game or replay-row proportion guarantee. "
                       "Setup roots carry stones without move history."),
    }
    return starts, sampling


def _arm_config(raw: dict, run_root: Path, starts_dir: Path, name: str) -> dict:
    config = deepcopy(raw)
    config.pop("exploration", None)
    runtime = config["training"]["overrides"].setdefault("runtime", {})
    runtime["output_directory"] = str(run_root)
    if name == "sgf_starts":
        config["exploration"] = {
            "start_positions_dir": str(starts_dir), "start_probability": 0.1,
            "early_fork_probability": 0.0, "fork_probability": 0.0, "policy_init": False,
        }
    return config


def _layout(output: Path) -> tuple[dict, Path, Path]:
    arms = {name: {"config_path": str(output / "arms" / name / "config.json"),
                   "run_root": str(output / "arms" / name / "run")}
            for name in _ARM_NAMES}
    return arms, output / "common", output / "training_starts"


def _copy_checked(source: Path, destination: Path, expected: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _record(destination) != {key: expected[key] for key in ("sha256", "size", "mtime_ns")}:
        raise ValueError(f"Snapshot copy changed bytes or modification time: {source}")


def _start_manifest(starts_dir: Path, starts: dict[str, str], sampling: dict) -> dict:
    return {
        "directory": str(starts_dir), "domain": TRAINING_START_DOMAIN,
        "seed": TRAINING_START_SEED,
        "sampling": sampling,
        "files": {name: {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "size": len(text.encode("utf-8"))}
                  for name, text in starts.items()},
    }


def _expected_paths(output: Path, source: dict, arms: dict,
                    common: Path, starts_dir: Path, starts: dict[str, str]) -> set[str]:
    paths = {common / "source_config.json"}
    paths.update(starts_dir / name for name in starts)
    for root in (common, *(Path(arm["run_root"]) for arm in arms.values())):
        paths.add(root / _CHECKPOINT)
        paths.update(root / relative for relative in source["replay"]["files"])
    for arm in arms.values():
        root = Path(arm["run_root"])
        paths.update((Path(arm["config_path"]), root / "mixed_manifest.json", root / "replay_rows.json"))
    return {path.relative_to(output).as_posix() for path in paths}


def _verify_existing(output: Path, raw: dict, source: dict, counters: dict,
                     wall_seconds: int, starts: dict[str, str], sampling: dict) -> dict:
    manifest = _read_json(output / EXPERIMENT_MANIFEST)
    arms, common, starts_dir = _layout(output)
    expected = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "kind": "mixed_fixed_budget_ab",
        "source": source, "initial_counters": counters,
        "budget": {"wall_seconds": wall_seconds, "per_arm": True},
        "arms": arms, "snapshot": {"run_root": str(common)},
        "training_starts": _start_manifest(starts_dir, starts, sampling),
        "manifest_path": str(output / EXPERIMENT_MANIFEST),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing experiment manifest is incompatible with source/config/budget")
    records = manifest.get("prepared_files")
    if (not isinstance(records, dict)
            or set(records) != _expected_paths(output, source, arms, common, starts_dir, starts)):
        raise ValueError("Existing experiment manifest has an incompatible prepared artifact inventory")
    for relative, record in records.items():
        path = output / relative
        if not path.resolve().is_relative_to(output) or _record(path) != record:
            raise ValueError(f"Prepared experiment artifact was mutated: {path}")
    for root in (common, *(Path(arm["run_root"]) for arm in arms.values())):
        replay_paths = {path.relative_to(root).as_posix()
                        for path in root.glob("selfplay/**/tdata/*.npz")}
        if replay_paths != set(source["replay"]["files"]):
            raise ValueError(f"Prepared replay inventory changed: {root}")
        copies = {_CHECKPOINT.as_posix(): source["checkpoint"], **source["replay"]["files"]}
        for relative, source_record in copies.items():
            if _record(root / relative) != {key: source_record[key]
                                           for key in ("sha256", "size", "mtime_ns")}:
                raise ValueError(f"Prepared experiment artifact differs from source snapshot: {root / relative}")
    for name, arm in arms.items():
        if _read_json(Path(arm["config_path"])) != _arm_config(raw, Path(arm["run_root"]), starts_dir, name):
            raise ValueError(f"Incompatible arm config: {name}")
        MixedBoardRunner(Path(arm["config_path"])).ensure_manifest()
    for name, text in starts.items():
        if (starts_dir / name).read_text(encoding="utf-8") != text:
            raise ValueError(f"Training start artifact was mutated: {name}")
    return manifest


def verify_prepared_experiment(output_root: Path) -> dict:
    """Verify an unmodified preparation using only its frozen local artifacts.

    No current source-run file or original config is read. Call before any arm
    is initialized or trained; subsequent state changes deliberately fail this
    preflight rather than granting a second execution budget.
    """
    output = Path(output_root).resolve(strict=True)
    manifest = _read_json(output / EXPERIMENT_MANIFEST)
    try:
        source, counters, budget = manifest["source"], manifest["initial_counters"], manifest["budget"]["wall_seconds"]
        config_path = output / "common/source_config.json"
        if _record(config_path) != {key: source["config"][key] for key in ("sha256", "size", "mtime_ns")}:
            raise ValueError("Frozen source config differs from preparation metadata")
        raw = _read_json(config_path)
        _require_unexplored_baseline(raw)
        training = raw["training"]
        config = resolve_rl_training_config(preset=training["preset"], overrides=training["overrides"])
        starts, sampling = _training_start_plan(raw["board_sizes"], raw["board_weights"], config.game.komi)
        actual_counters = _checkpoint_counters(output / "common" / _CHECKPOINT)
        if any(counters.get(key) != value for key, value in actual_counters.items()):
            raise ValueError("Frozen checkpoint counters differ from preparation metadata")
    except (KeyError, TypeError) as error:
        raise ValueError("Incompatible prepared experiment manifest/config") from error
    if type(budget) is not int or budget <= 0:
        raise ValueError("Invalid prepared wall_seconds budget")
    return _verify_existing(output, raw, source, counters, budget, starts, sampling)


def prepare_mixed_experiment(source_run: Path, config_path: Path, output_root: Path,
                             *, wall_seconds: int = 180) -> dict:
    """Freeze a full learner checkpoint and retained replay into two new arms.

    The caller must quiesce all source writers first. Checkpoint and NPZ mtimes
    are copied unchanged because ``ReplayRowLedger`` needs their ordering to
    recover rows pruned before this snapshot. Neither source manifests nor
    source ledgers are adopted. Repeating the same request verifies prepared
    artifacts and returns the existing manifest; it never overwrites results.
    This version requires an original baseline without exploration controls.

    The budget is a per-arm execution plan. This function performs no runtime,
    GPU, subprocess or process-control operation.
    """
    if type(wall_seconds) is not int or wall_seconds <= 0:
        raise ValueError("wall_seconds budget must be a positive integer")
    source = Path(source_run).resolve(strict=True)
    config_path = Path(config_path).resolve(strict=True)
    output_root = Path(output_root)
    if output_root.is_symlink():
        raise ValueError("Experiment output cannot be a symlink")
    output = output_root.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Experiment output must be isolated from the source run")
    raw, source_info, counters = _source_inventory(source, config_path)
    runner = MixedBoardRunner(config_path, run_root_override=source)
    starts, sampling = _training_start_plan(raw["board_sizes"], raw["board_weights"], runner.config.game.komi)
    if output.exists():
        return _verify_existing(output, raw, source_info, counters, wall_seconds, starts, sampling)

    # Exclusive creation prevents adopting old or partial experiments. On any
    # later error keep the partial directory for inspection, never reuse it.
    output.mkdir(parents=True, exist_ok=False)
    arms, common, starts_dir = _layout(output)
    source_config = common / "source_config.json"
    _copy_checked(config_path, source_config, source_info["config"])
    _copy_checked(source / _CHECKPOINT, common / _CHECKPOINT, source_info["checkpoint"])
    for relative, record in source_info["replay"]["files"].items():
        _copy_checked(source / relative, common / relative, record)
    starts_dir.mkdir()
    for name, text in starts.items():
        path = starts_dir / name
        atomic_write_text(path, text)

    for name, arm in arms.items():
        root, path = Path(arm["run_root"]), Path(arm["config_path"])
        atomic_write_json(path, _arm_config(raw, root, starts_dir, name))
        # Create a manifest for this arm before introducing its seeded data.
        MixedBoardRunner(path).ensure_manifest()
        _copy_checked(common / _CHECKPOINT, root / _CHECKPOINT, source_info["checkpoint"])
        for relative, record in source_info["replay"]["files"].items():
            _copy_checked(common / relative, root / relative, record)
        ledger = ReplayRowLedger(root).sync()
        if (ledger.physical_rows != counters["retained_replay_rows"]
                or ledger.deleted_rows_offset != counters["deleted_rows_offset"]
                or ledger.logical_rows != counters["logical_replay_rows"]):
            raise ValueError(f"Initial replay ledger counters differ for arm: {name}")

    # Hash again after every copy to detect a writer or retention race. Source
    # data is read only; a mismatch leaves this partial experiment untouched.
    _raw_after, source_after, counters_after = _source_inventory(source, config_path)
    if source_after != source_info or counters_after != counters:
        raise ValueError("Source changed during preparation; pause all writers and use a fresh output")
    prepared = {path.relative_to(output).as_posix(): _record(path)
                for path in sorted(output.rglob("*")) if path.is_file()}
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION, "kind": "mixed_fixed_budget_ab",
        "created_at": local_timestamp(), "manifest_path": str(output / EXPERIMENT_MANIFEST),
        "budget": {"wall_seconds": wall_seconds, "per_arm": True},
        "source": source_info, "initial_counters": counters,
        "snapshot": {"run_root": str(common)}, "arms": arms,
        "training_starts": _start_manifest(starts_dir, starts, sampling),
        "randomness_note": "Upstream random sequences are independently sampled; exact common RNG is not guaranteed.",
        "source_consistency": "Caller must pause all source writers and retention before preparation.",
        "prepared_files": prepared,
    }
    atomic_write_json(output / EXPERIMENT_MANIFEST, manifest)
    return manifest


__all__ = ["prepare_mixed_experiment", "verify_prepared_experiment"]
