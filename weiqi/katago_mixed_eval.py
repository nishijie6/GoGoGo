"""Frozen, resumable fixed-opening evaluations of one mixed-board network.

Results stay separate by board size. The two colors of an opening form one
experimental unit; game-level Wilson intervals are descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Sequence
import uuid

from .fileio import (atomic_open, atomic_write_json, atomic_write_text,
                     local_timestamp, sha256_file)
from .rl_curriculum_openings import ensure_evaluation_opening_suite
from .rl_match import build_match_config, parse_match_game, summarize_match_sgfs
from .rl_match_jobs import _complete_pair, run_opening_pairs
from .rl_metrics import parse_model_name
from .sgf import MalformedSGF, _read_value, parse_sgf_game, split_sgf_collection


_SUPPORTED_BOARDS = (7, 8, 9, 13, 19)
_MANIFEST = "evaluation_manifest.json"
_MANIFEST_HASH = "evaluation_manifest.sha256"
_PAIR_EVIDENCE = "pair-evidence.json"
_SGF_RULES = "koPOSITIONALscoreAREAtaxNONEsui0"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_hash(path: Path, expected: str) -> None:
    try:
        valid = path.is_file() and sha256_file(path) == expected
    except OSError:
        valid = False
    if not valid:
        raise ValueError(f"评测文件 SHA256 校验失败，不能复用：{path}")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"无法读取评测清单：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"评测清单格式无效：{path}")
    return value


def _protocol(runner, output_dir: Path, boards, games: int, visits: int) -> dict:
    game = runner.config.game
    if (game.scoring_rule != "area" or game.ko_rule != "positional_superko"
            or game.allow_suicide):
        raise ValueError("混训评测要求 area、positional_superko 和禁止自杀规则")
    if type(game.komi) not in (float, int) or not math.isfinite(game.komi):
        raise ValueError("混训评测贴目必须是有限数值")
    executable = Path(runner.katago_executable).resolve()
    try:
        executable_hash = sha256_file(executable)
    except OSError as error:
        raise ValueError(f"无法校验评测引擎：{executable}") from error
    profile = runner.hardware_profile
    return {
        "version": 2,
        "run_root": str(Path(runner.run_root).resolve()),
        "output_directory": str(output_dir),
        "board_sizes": list(boards),
        "games_per_board": games,
        "visits": visits,
        "rules": {"komi": game.komi, "scoring": "AREA", "ko": "POSITIONAL",
                  "allow_suicide": False, "tax": "NONE", "has_buttons": False},
        "root_noise": False,
        "temperature": 0.0,
        "resignation": False,
        "parallel_pairs": 1,
        "game_threads": 2,
        "inference_batch_size": 2,
        "katago_executable": str(executable),
        "katago_sha256": executable_hash,
        "inference_overrides": profile.inference_overrides() if profile else None,
        "backend": profile.backend if profile else None,
    }


def _select_candidate(run_root: Path) -> Path:
    candidates = []
    for path in (run_root / "models").glob("*/model.bin.gz"):
        parsed = parse_model_name(path.parent.name)
        if parsed is not None and path.is_file():
            candidates.append((parsed, path.parent.name, path))
    if not candidates:
        raise ValueError("混训目录没有带训练样本数的可评测 model.bin.gz")
    return max(candidates)[-1]


def _snapshot(source: Path, destination: Path, role: str, selected_by: str) -> dict:
    source = source.resolve()
    if not source.is_file():
        raise ValueError(f"评测模型不存在：{source}")
    stat = source.stat()
    original_hash = sha256_file(source)
    with source.open("rb") as reader, atomic_open(destination, "wb") as writer:
        shutil.copyfileobj(reader, writer)
    frozen_hash = sha256_file(destination)
    if frozen_hash != original_hash or sha256_file(source) != original_hash:
        raise ValueError(f"冻结期间模型发生变化：{source}")
    parsed = parse_model_name(source.parent.name)
    return {
        "source_path": str(source),
        "source_mtime_ns": stat.st_mtime_ns,
        "model_name": source.parent.name,
        "selected_by": selected_by,
        "trained_samples": parsed[0] if parsed is not None else None,
        "data_rows": parsed[1] if parsed is not None else None,
        "frozen_file": f"frozen/{role}.bin.gz",
        "size_bytes": destination.stat().st_size,
        "sha256": frozen_hash,
    }


def _config(runner, root: Path, board: int, opening: Path, protocol: dict) -> str:
    return build_match_config(
        candidate_model=root / "frozen/candidate.bin.gz",
        baseline_model=root / "frozen/baseline.bin.gz",
        board_size=board,
        visits=protocol["visits"],
        games=2,
        game_threads=2,
        inference_batch_size=2,
        gpu_index=(runner.hardware_profile.gpu_index or 0) if runner.hardware_profile else 0,
        opening_directory=opening,
        hardware_profile=runner.hardware_profile,
        komi=protocol["rules"]["komi"],
        max_moves_factor=protocol.get("max_moves_factor", 2.5),
    )


def _initialize(runner, root: Path, candidate: Path | None, baseline: Path,
                protocol: dict) -> dict:
    """Publish the complete frozen identity atomically before starting matches."""
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("评测输出目录已有内容但没有匹配清单，不能覆盖或采用")
    candidate_path = candidate if candidate is not None else _select_candidate(Path(runner.run_root))
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(f".{root.name}.prepare-{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    try:
        manifest = {
            "schema_version": 1,
            "created_at": local_timestamp(),
            "protocol": protocol,
            "candidate": _snapshot(candidate_path, staging / "frozen/candidate.bin.gz", "candidate",
                                   "explicit" if candidate is not None else "highest_trained_samples"),
            "baseline": _snapshot(baseline, staging / "frozen/baseline.bin.gz", "baseline", "explicit"),
            "opening_suites": {},
        }
        for board in protocol["board_sizes"]:
            openings = ensure_evaluation_opening_suite(
                staging, board_size=board, opening_count=protocol["games_per_board"] // 2,
                opening_offset=protocol.get("opening_offset", 0))
            suite_manifest = openings[0].parent / "manifest.json"
            manifest["opening_suites"][str(board)] = {
                "manifest_file": str(suite_manifest.relative_to(staging)),
                "manifest_sha256": sha256_file(suite_manifest),
                "opening_directories": [str(path.relative_to(staging)) for path in openings],
                "opening_sha256": [sha256_file(path / "opening.startposes.txt") for path in openings],
                "config_sha256": [_text_hash(_config(runner, root, board,
                    root / path.relative_to(staging), protocol)) for path in openings],
            }
        atomic_write_json(staging / _MANIFEST, manifest, allow_nan=False)
        atomic_write_text(staging / _MANIFEST_HASH, sha256_file(staging / _MANIFEST) + "\n")
        # os.rename refuses a concurrently published nonempty evaluation directory.
        os.rename(staging, root)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_manifest(root: Path, protocol: dict, candidate: Path | None, baseline: Path) -> dict:
    try:
        saved_hash = (root / _MANIFEST_HASH).read_text(encoding="ascii").strip()
    except OSError as error:
        raise ValueError("评测清单缺少 SHA256 校验，不能复用") from error
    _verify_hash(root / _MANIFEST, saved_hash)
    manifest = _read_json(root / _MANIFEST)
    saved_protocol = manifest.get("protocol", {})
    if not isinstance(saved_protocol, dict) or saved_protocol.get("version") != 2:
        raise ValueError("旧评测协议未固定引擎 SHA256 和逐对局证据，不能继续；请使用新输出目录")
    if manifest.get("schema_version") != 1 or manifest.get("protocol") != protocol:
        raise ValueError("评测参数、规则、硬件或输出目录与冻结清单不一致")
    for role, requested in (("candidate", candidate), ("baseline", baseline)):
        metadata = manifest.get(role, {})
        if metadata.get("frozen_file") != f"frozen/{role}.bin.gz":
            raise ValueError("评测冻结模型清单无效")
        _verify_hash(root / metadata["frozen_file"], metadata.get("sha256", ""))
        if requested is not None:
            if str(requested.resolve()) != metadata.get("source_path"):
                raise ValueError(f"评测 {role} 模型参数与冻结清单不一致")
            # Deleted exports are expected under retention. A supplied existing
            # export must still name the same model contents as its frozen copy.
            if requested.exists():
                _verify_hash(requested, metadata["sha256"])
    return manifest


def _root_properties(tree: str) -> dict[str, list[str]]:
    """Read only the root node, honoring escaped brackets and comment values."""
    offset = 1  # split_sgf_collection returns trees beginning with '('.
    while tree[offset].isspace():
        offset += 1
    if tree[offset] != ";":
        raise MalformedSGF("SGF root node is missing")
    offset += 1
    properties = {}
    while offset < len(tree):
        while tree[offset].isspace():
            offset += 1
        if tree[offset] in ";()":
            break
        start = offset
        while tree[offset].isalpha():
            offset += 1
        key = tree[start:offset]
        if not key or key != key.upper():
            raise MalformedSGF("invalid root property identifier")
        while tree[offset].isspace():
            offset += 1
        values = []
        while tree[offset] == "[":
            value, offset = _read_value(tree, offset)
            values.append(value)
            while tree[offset].isspace():
                offset += 1
        if not values:
            raise MalformedSGF("root property value is missing")
        properties.setdefault(key, []).extend(values)
    return properties


def _validate_pair_sgfs(paths: Sequence[Path], board: int, komi: float) -> None:
    """Check actual SGF roots before any pair is sealed or reused."""
    try:
        for path in paths:
            trees, truncated = split_sgf_collection(path.read_text(encoding="utf-8"))
            if truncated:
                raise MalformedSGF("truncated match SGF")
            for tree in trees:
                parsed = parse_sgf_game(tree)
                properties = _root_properties(tree)
                if not properties.get("RE"):
                    raise RuntimeError("评测棋谱缺少终局结果（可能达到手数上限），不能计分")
                for key in ("SZ", "PB", "PW", "RE", "KM", "RU"):
                    if len(properties.get(key, [])) != 1:
                        raise MalformedSGF(f"root {key} must appear exactly once")
                names = (properties["PB"][0], properties["PW"][0])
                if (parsed["board_size"] != board
                        or names not in (("candidate", "baseline"), ("baseline", "candidate"))
                        or properties["RU"][0] != _SGF_RULES
                        or not math.isclose(float(properties["KM"][0]), komi, abs_tol=1e-9)):
                    raise MalformedSGF("match board, opponents, rules, or komi differ")
                # The shared summarizer must read the same root values, rather
                # than accidentally finding property-shaped text in a comment.
                match = parse_match_game(tree)
                if (match.black, match.white, match.result) != (*names, properties["RE"][0]):
                    raise MalformedSGF("ambiguous match properties")
        if not _complete_pair(paths):
            raise MalformedSGF("match must contain two complete color-swapped games")
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError("评测 SGF 与固定棋盘、对手、规则或贴目不一致，不能计分") from error


def _pair_record(root: Path, attempt: Path, index: int, board: int, komi: float) -> dict:
    config = attempt / "match.cfg"
    paths = sorted((attempt / "sgfs").glob("*.sgfs"))
    _validate_pair_sgfs(paths, board, komi)
    summary = summarize_match_sgfs(paths, candidate_name="candidate", requested_games=2)
    return {
        "opening_index": index,
        "score": summary.win_rate,
        "config_file": str(config.relative_to(root)),
        "config_sha256": sha256_file(config),
        "sgfs": [{"file": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in paths],
    }


def _pair_evidence(root: Path, attempt: Path, index: int, board: int, protocol: dict) -> dict:
    return {
        "schema_version": 1,
        "evaluation_manifest_sha256": sha256_file(root / _MANIFEST),
        "board_size": board,
        "pair_artifact": _pair_record(root, attempt, index, board, protocol["rules"]["komi"]),
    }


def _verify_pair_evidence(root: Path, attempt: Path, index: int, board: int, protocol: dict) -> dict:
    saved = _read_json(attempt / _PAIR_EVIDENCE)
    try:
        expected = _pair_evidence(root, attempt, index, board, protocol)
    except RuntimeError as error:
        raise ValueError(f"已完成评测对局的 SGF 证据校验失败：{attempt}") from error
    if saved != expected:
        raise ValueError(f"已完成评测对局的不可变证据校验不一致：{attempt}")
    return saved["pair_artifact"]


def _preflight(runner, root: Path, manifest: dict) -> dict[int, tuple[Path, ...]]:
    """Check all boards before any process, without repairing immutable inputs."""
    protocol = manifest["protocol"]
    suites = manifest.get("opening_suites", {})
    if set(suites) != set(map(str, protocol["board_sizes"])):
        raise ValueError("评测开局清单的棋盘集合不一致")
    by_board = {}
    count = protocol["games_per_board"] // 2
    for board in protocol["board_sizes"]:
        suite = suites[str(board)]
        suite_path = root / suite["manifest_file"]
        _verify_hash(suite_path, suite["manifest_sha256"])
        directories = suite["opening_directories"]
        if len(directories) != count or len(suite["opening_sha256"]) != count:
            raise ValueError("评测开局清单的开局数不一致")
        openings = tuple(root / path for path in directories)
        expected_names = {"manifest.json", *(path.name for path in openings)}
        if {path.name for path in suite_path.parent.iterdir()} != expected_names:
            raise ValueError("固定评测开局目录出现未知内容")
        for index, opening in enumerate(openings):
            _verify_hash(opening / "opening.startposes.txt", suite["opening_sha256"][index])
            if {path.name for path in opening.iterdir()} != {"opening.startposes.txt"}:
                raise ValueError("固定评测开局必须是单局面目录")
            text = _config(runner, root, board, opening, protocol)
            if _text_hash(text) != suite["config_sha256"][index]:
                raise ValueError("评测引擎配置与冻结清单不一致")
            pair_root = root / "boards" / f"{board}x{board}" / "pairs" / f"opening-{index:03d}"
            for attempt in [pair_root, *sorted(pair_root.glob("retry-*"))]:
                cfg = attempt / "match.cfg"
                if cfg.exists() and cfg.read_text(encoding="utf-8") != text:
                    raise ValueError(f"已有评测配置发生变化，不能复用：{cfg}")
                if (attempt / _PAIR_EVIDENCE).exists():
                    _verify_pair_evidence(root, attempt, index, board, protocol)
                elif _complete_pair(sorted((attempt / "sgfs").glob("*.sgfs"))):
                    raise ValueError(f"已完成评测对局缺少不可变证据，不能复用：{attempt}")
        result_path = root / "boards" / f"{board}x{board}" / "result.json"
        if result_path.exists():
            saved = _read_json(result_path)
            records = saved.get("pair_artifacts", [])
            if saved.get("board_size") != board or len(records) != count:
                raise ValueError("已有评测对局证据清单不一致")
            for record in records:
                _verify_hash(root / record["config_file"], record["config_sha256"])
                for sgf in record["sgfs"]:
                    _verify_hash(root / sgf["file"], sgf["sha256"])
        by_board[board] = openings
    return by_board


def _pair_artifacts(root: Path, output_dir: Path, openings, config_builder,
                    board: int, protocol: dict) -> list[dict]:
    records = []
    for index, opening in enumerate(openings):
        pair_root = output_dir / "pairs" / f"opening-{index:03d}"
        for attempt in [pair_root, *sorted(pair_root.glob("retry-*"))]:
            config = attempt / "match.cfg"
            paths = sorted((attempt / "sgfs").glob("*.sgfs"))
            if (config.is_file() and config.read_text(encoding="utf-8") == config_builder(opening)
                    and _complete_pair(paths)):
                records.append(_verify_pair_evidence(root, attempt, index, board, protocol))
                break
        else:
            raise RuntimeError(f"评测开局 {index} 缺少完整黑白对局")
    return records


def _pair_statistics(records: list[dict]) -> dict:
    scores = [record["score"] for record in records]
    mean = sum(scores) / len(scores)
    # Each score lies in [0,1]. Unlike a binomial game interval, this bounded
    # mean interval allows the two games in a pair to be arbitrarily correlated.
    radius = math.sqrt(math.log(2 / .05) / (2 * len(scores)))
    return {
        "opening_pairs": len(scores),
        "scores": scores,
        "mean_score": mean,
        "hoeffding_95_lower": max(0.0, mean - radius),
        "hoeffding_95_upper": min(1.0, mean + radius),
        "interpretation": ("Exploratory bounded-mean Hoeffding interval assuming independent opening pairs; "
                           "fixed pilot starts do not establish general playing strength or a promotion gate."),
    }


def evaluate_mixed(
    runner,
    output_dir: Path,
    *,
    candidate: Path | None = None,
    baseline: Path,
    boards: Sequence[int] | None = None,
    games_per_board: int = 12,
    visits: int = 32,
    max_moves_factor: float = 2.5,
    opening_offset: int = 0,
    parallel_pairs: int = 1,
) -> dict:
    """Evaluate frozen candidate/baseline snapshots, resuming matching pairs only.

    Omitting candidate selects the largest trained-sample export once. On resume
    the manifest's frozen snapshot remains authoritative even if exports were
    pruned or a newer model was published. Protocol 1 outputs are preserved but
    cannot resume because their original engine identity cannot be established.
    """
    selected = tuple(runner.mixed.board_sizes if boards is None else boards)
    if (not selected or any(type(size) is not int or size not in _SUPPORTED_BOARDS for size in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("评测棋盘必须是不重复的 7、8、9、13、19 路尺寸")
    if type(games_per_board) is not int or games_per_board < 2 or games_per_board % 2:
        raise ValueError("每种棋盘的评测局数必须是正偶数")
    if type(visits) is not int or visits < 1:
        raise ValueError("评测 visits 必须是正整数")
    if type(parallel_pairs) is not int or not 1 <= parallel_pairs <= 16:
        raise ValueError("评测并发换色开局组数必须是 1 到 16 的整数")
    if (type(max_moves_factor) not in (int, float) or not math.isfinite(max_moves_factor)
            or max_moves_factor <= 0 or not math.isfinite(max(selected) ** 2 * max_moves_factor)):
        raise ValueError("评测手数倍数必须是有限正数")
    root = Path(output_dir).resolve()
    candidate = Path(candidate).resolve() if candidate is not None else None
    baseline = Path(baseline).resolve()
    protocol = _protocol(runner, root, selected, games_per_board, visits)
    protocol["parallel_pairs"] = parallel_pairs
    # Retain byte-compatible identities for existing default-limit protocols.
    # Any explicit nondefault cap becomes part of the frozen comparison.
    if max_moves_factor != 2.5:
        protocol["max_moves_factor"] = max_moves_factor
    if type(opening_offset) is not int or opening_offset < 0:
        raise ValueError("评测开局偏移必须是非负整数")
    if opening_offset:
        protocol["opening_offset"] = opening_offset
    manifest = (_load_manifest(root, protocol, candidate, baseline)
                if (root / _MANIFEST).exists()
                else _initialize(runner, root, candidate, baseline, protocol))
    by_board = _preflight(runner, root, manifest)
    results = {}
    current = None

    def progress(status, error=None):
        atomic_write_json(root / "progress.json", {
            "status": status, "completed_boards": list(results), "current_board": current,
            "total_boards": len(selected), "updated_at": local_timestamp(), "error": error,
        }, allow_nan=False)

    progress("running")
    try:
        for board, openings in by_board.items():
            current = board
            progress("running")
            board_root = root / "boards" / f"{board}x{board}"

            def config_builder(opening):
                return _config(runner, root, board, opening, protocol)

            def execute(config, sgfs, log):
                _verify_hash(Path(protocol["katago_executable"]), protocol["katago_sha256"])
                runner._run([runner.katago_executable, "match", "-config", config,
                             "-sgf-output-dir", sgfs], log_name=str(log.resolve()))
                _verify_hash(Path(protocol["katago_executable"]), protocol["katago_sha256"])
                index = int(config.relative_to(board_root / "pairs").parts[0].removeprefix("opening-"))
                _verify_hash(config, _text_hash(config_builder(openings[index])))
                evidence_path = config.parent / _PAIR_EVIDENCE
                evidence = _pair_evidence(root, config.parent, index, board, protocol)
                if evidence_path.exists():
                    if _read_json(evidence_path) != evidence:
                        raise ValueError("已有评测对局的不可变证据不一致，不能覆盖")
                else:
                    # Seal this pair before the scheduler reuses its worker.
                    atomic_write_json(evidence_path, evidence, allow_nan=False)
                return 0

            match = run_opening_pairs(openings, board_root, parallel_pairs=parallel_pairs,
                                      config_builder=config_builder, execute=execute)
            records = _pair_artifacts(root, board_root, openings, config_builder, board, protocol)
            saturated = match.win_rate in (0.0, 1.0)
            row = {
                "board_size": board,
                **match.to_dict(),
                "elo": None if saturated else match.elo,
                "elo_interpretation": (
                    "unestimable at a saturated 0% or 100% pilot score; no finite Elo estimate"
                    if saturated else
                    "exploratory relative estimate from a small fixed-opening pilot; not a stable Elo rating"
                ),
                "wilson_interpretation": ("descriptive per-game Wilson interval; the two games from "
                                          "each opening are correlated and are not independent trials"),
                "pair_statistics": _pair_statistics(records),
                "opening_suite": manifest["opening_suites"][str(board)],
                "pair_artifacts": records,
            }
            atomic_write_json(board_root / "result.json", row, allow_nan=False)
            results[str(board)] = row
            progress("running")
        result = {
            "schema_version": 1,
            "status": "complete",
            "evaluation_manifest": str(root / _MANIFEST),
            "evaluation_manifest_sha256": sha256_file(root / _MANIFEST),
            "candidate": manifest["candidate"],
            "baseline": manifest["baseline"],
            "results_by_board": results,
            "automatic_promotion": False,
            "interpretation": ("Separate fixed-opening pilot results for each board size; "
                               "no aggregate win rate or automatic promotion decision."),
        }
        atomic_write_json(root / "summary.json", result, allow_nan=False)
        current = None
        progress("complete")
        return result
    except Exception as error:
        progress("failed", str(error))
        raise


__all__ = ["evaluate_mixed"]
