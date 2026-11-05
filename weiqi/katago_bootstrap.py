"""Keep cold learners out of the actor role until an isolated data-quality probe."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from .fileio import atomic_write_json, local_timestamp
from .katago_mixed_health import build_mixed_health


_THRESHOLDS = {"probe_games": 100, "minimum_games_per_board": 4,
               "probe_interval_samples": 4096, "short_game_rate_max": .25,
               "short_game_area_fraction": .10, "short_game_min_moves": 4,
               "policy_pass_probability_max": .2}
_MODEL_NAME = re.compile(r"gogogo-s([0-9]+)-d([0-9]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_assessment(assessment: dict, board_sizes) -> None:
    if (not isinstance(assessment, dict) or type(assessment.get("passed")) is not bool
            or assessment.get("schema_version") != 1 or assessment.get("thresholds") != _THRESHOLDS
            or type(assessment.get("finished_games")) is not int or assessment["finished_games"] < 0
            or not isinstance(assessment.get("source_hashes"), dict)
            or not isinstance(assessment.get("reasons"), list)
            or any(not isinstance(reason, str) for reason in assessment["reasons"])
            or not isinstance(assessment.get("by_board"), dict)
            or set(assessment["by_board"]) != {str(size) for size in board_sizes}):
        raise ValueError("invalid recorded bootstrap assessment")
    if any(not isinstance(name, str) or not isinstance(digest, str)
           or not re.fullmatch(r"[0-9a-f]{64}", digest)
           for name, digest in assessment["source_hashes"].items()):
        raise ValueError("invalid bootstrap source hashes")
    for row in assessment["by_board"].values():
        if (not isinstance(row, dict) or type(row.get("finished_games")) is not int
                or row["finished_games"] < 0 or type(row.get("no_result_games")) is not int
                or row["no_result_games"] < 0):
            raise ValueError("invalid bootstrap board assessment")
        for key in ("short_game_rate", "policy_pass_probability"):
            number = row.get(key)
            if number is not None and (type(number) not in (int, float) or not math.isfinite(number)
                                       or not 0 <= number <= 1):
                raise ValueError("invalid bootstrap rate")
        if assessment["passed"] and (
                row["finished_games"] < _THRESHOLDS["minimum_games_per_board"]
                or row["no_result_games"] != 0 or row.get("short_game_rate") is None
                or row["short_game_rate"] > _THRESHOLDS["short_game_rate_max"]
                or row.get("policy_pass_probability") is None
                or row["policy_pass_probability"] > _THRESHOLDS["policy_pass_probability_max"]):
            raise ValueError("recorded successful bootstrap assessment fails its thresholds")
    if assessment["passed"] and (assessment["reasons"] or not assessment["source_hashes"]
                                 or assessment["finished_games"] < _THRESHOLDS["probe_games"]):
        raise ValueError("recorded successful bootstrap assessment is incomplete")


def _policy_pass_statistics(root: Path, board_sizes) -> dict:
    """Use the actual visit distributions, weighted by sample and policy weights."""
    import numpy as np

    totals = {size: [0., 0., 0] for size in board_sizes}
    for path in sorted((root / "selfplay").glob("**/tdata/*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                packed = data["binaryInputNCHWPacked"]
                policies = data["policyTargetsNCMove"]
                target = data["globalTargetsNC"]
                if (packed.ndim != 3 or packed.dtype != np.uint8 or packed.shape[1] != 22
                        or policies.ndim != 3 or policies.shape[1] != 2
                        or target.ndim != 2 or target.shape[1] < 27
                        or not packed.shape[0] == policies.shape[0] == target.shape[0]):
                    raise ValueError("incompatible training tensor shapes")
                area = policies.shape[2] - 1
                side = int(area ** .5)
                if side * side != area or packed.shape[2] != (area + 7) // 8:
                    raise ValueError("incompatible policy and board dimensions")
                policy = policies[:, 0, :].astype(np.float64)
                weights = target[:, 25].astype(np.float64) * target[:, 26].astype(np.float64)
                if (not np.isfinite(policy).all() or (policy < 0).any()
                        or not np.isfinite(target).all() or (target[:, 25:27] < 0).any()
                        or not np.isfinite(weights).all()):
                    raise ValueError("nonfinite or negative training targets")
                masses = policy.sum(axis=1)
                if not np.isfinite(masses).all() or (masses <= 0).any():
                    raise ValueError("empty policy visit distribution")
                probabilities = policy[:, -1] / masses
                masks = np.unpackbits(packed[:, 0, :], axis=1)[:, :area].reshape(-1, side, side)
                for index, mask in enumerate(masks):
                    size = int(int(mask.sum()) ** .5)
                    expected = np.zeros((side, side), dtype=np.uint8)
                    expected[:size, :size] = 1
                    if size not in totals or not np.array_equal(mask, expected):
                        raise ValueError("unknown or malformed board mask")
                    totals[size][0] += float(weights[index] * probabilities[index])
                    totals[size][1] += float(weights[index])
                    totals[size][2] += 1
        except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
            raise ValueError(f"Invalid bootstrap probe NPZ {path}: {error}") from error
    return {str(size): {"policy_pass_probability": weighted / mass if mass > 0 else None,
                        "policy_weight": mass, "policy_rows": rows}
            for size, (weighted, mass, rows) in totals.items()}


def assess_bootstrap_probe(probe_root: Path, board_sizes) -> dict:
    """Assess all recorded probe games; winning color is deliberately not a gate."""
    probe_root = Path(probe_root)
    board_sizes = tuple(board_sizes)
    sources = sorted({*probe_root.glob("selfplay/**/*.sgf"),
                      *probe_root.glob("selfplay/**/*.sgfs"),
                      *probe_root.glob("selfplay/**/tdata/*.npz")})
    hashes = {str(path.relative_to(probe_root)): _sha256(path) for path in sources}
    # A probe is finite (100 games), so this window includes every game.
    health = build_mixed_health(probe_root, board_sizes, window_games=100000,
                                minimum_games=_THRESHOLDS["minimum_games_per_board"])
    policy = _policy_pass_statistics(probe_root, board_sizes)
    inventory = health["inventory"]
    reasons = []
    if inventory["valid_games"] < _THRESHOLDS["probe_games"]:
        reasons.append("insufficient_finished_games")
    for key in ("damaged_games", "pending_games", "read_errors", "unconfigured_board_games"):
        if inventory[key]:
            reasons.append(key)
    boards = {}
    for size in board_sizes:
        key = str(size)
        game = health["by_board"][key]
        row = {"finished_games": game["window_games"],
               "short_game_move_cutoff": game["short_game_move_cutoff"],
               "short_game_rate": game["short_game_rate"],
               "no_result_games": game["result_counts"]["no_result"], **policy[key]}
        failed = []
        if row["finished_games"] < _THRESHOLDS["minimum_games_per_board"]:
            failed.append("insufficient_finished_games")
        if row["no_result_games"]:
            failed.append("no_result_games")
        if row["short_game_rate"] is None or row["short_game_rate"] > _THRESHOLDS["short_game_rate_max"]:
            failed.append("short_games")
        if (row["policy_pass_probability"] is None
                or row["policy_pass_probability"] > _THRESHOLDS["policy_pass_probability_max"]):
            failed.append("policy_pass_probability")
        row["reasons"] = failed
        boards[key] = row
        reasons.extend(f"board_{size}:{reason}" for reason in failed)
    if any(_sha256(path) != hashes[str(path.relative_to(probe_root))] for path in sources):
        raise ValueError("Bootstrap probe sources changed during assessment")
    report = {"schema_version": 1, "assessed_at": local_timestamp(), "passed": not reasons,
              "finished_games": inventory["valid_games"], "thresholds": dict(_THRESHOLDS),
              "by_board": boards, "reasons": reasons, "inventory": inventory,
              "source_hashes": hashes}
    atomic_write_json(probe_root / "bootstrap_assessment.json", report, allow_nan=False)
    return report


class BootstrapManager:
    def __init__(self, runner):
        self.runner = runner
        self.path = runner.run_root / "bootstrap_state.json"

    def read_state(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(state, dict) or type(state.get("schema_version")) is not int
                    or state["schema_version"] != 1 or state.get("phase") not in ("warming", "ready")
                    or type(state.get("last_probe_samples")) is not int or state["last_probe_samples"] < 0
                    or "pending_probe" not in state or "last_assessment" not in state):
                raise ValueError("invalid bootstrap state fields")
            pending = state["pending_probe"]
            if pending is not None and (
                    not isinstance(pending, dict) or type(pending.get("samples")) is not int
                    or pending["samples"] != state["last_probe_samples"]
                    or not isinstance(pending.get("model"), str) or not _MODEL_NAME.fullmatch(pending["model"])
                    or not isinstance(pending.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", pending["sha256"])
                    or not isinstance(pending.get("path"), str)):
                raise ValueError("invalid pending bootstrap probe")
            assessment = state["last_assessment"]
            if assessment is not None:
                _validate_assessment(assessment, self.runner.mixed.board_sizes)
            if state["phase"] == "ready" and (pending is not None or not assessment or not assessment["passed"]):
                raise ValueError("bootstrap ready state lacks a passed assessment")
            return state
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Invalid bootstrap state {self.path}: {error}") from error

    def start(self) -> dict:
        state = self.read_state()
        if state is not None:
            return state
        state = {"schema_version": 1, "phase": "warming", "last_probe_samples": 0,
                 "pending_probe": None, "last_assessment": None}
        atomic_write_json(self.path, state, allow_nan=False)
        return state

    def models_dir(self) -> Path:
        state = self.read_state()
        if state is None or state["phase"] == "ready":
            return self.runner.models_dir
        directory = self.runner.run_root / "bootstrap_random_models"
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError(f"Native bootstrap actor directory must be empty: {directory}")
        return directory

    def after_cycle(self) -> dict | None:
        state = self.read_state()
        if state is None or state["phase"] == "ready":
            return state
        candidates = []
        for path in self.runner.models_dir.glob("*/model.bin.gz"):
            match = _MODEL_NAME.fullmatch(path.parent.name)
            if match and path.is_file() and int(match[1]) > 0:
                candidates.append((int(match[1]), int(match[2]), path))
        if not candidates:
            return state
        samples, _rows, model = max(candidates)
        if samples < state["last_probe_samples"] + _THRESHOLDS["probe_interval_samples"]:
            return state
        probe_root = self.runner.run_root / "bootstrap_probes" / f"s{samples}"
        state.update(last_probe_samples=samples, pending_probe={"samples": samples,
            "model": model.parent.name, "sha256": _sha256(model), "path": str(probe_root)})
        state.pop("last_probe_error", None)
        atomic_write_json(self.path, state, allow_nan=False)
        try:
            if probe_root.exists():
                raise ValueError(f"Bootstrap probe directory already exists: {probe_root}")
            # Import lazily: MixedBoardRunner owns this manager itself.
            from .katago_mixed import MixedBoardRunner
            probe = MixedBoardRunner(self.runner.config_path, project_root=self.runner.project_root,
                run_root_override=probe_root, hardware_profile=self.runner.hardware_profile)
            probe.ensure_manifest()
            target = probe.models_dir / model.parent.name / "model.bin.gz"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, target)
            if _sha256(target) != state["pending_probe"]["sha256"]:
                raise ValueError("Bootstrap candidate changed while being frozen")
            probe.selfplay(games=_THRESHOLDS["probe_games"])
            assessment = assess_bootstrap_probe(probe_root, self.runner.mixed.board_sizes)
            assessment["candidate"] = dict(state["pending_probe"])
            state.update(last_assessment=assessment, pending_probe=None)
            if assessment["passed"]:
                state["phase"] = "ready"
            atomic_write_json(self.path, state, allow_nan=False)
            return state
        except (Exception, KeyboardInterrupt) as error:
            state["last_probe_error"] = {"type": type(error).__name__, "message": str(error)}
            atomic_write_json(self.path, state, allow_nan=False)
            raise
