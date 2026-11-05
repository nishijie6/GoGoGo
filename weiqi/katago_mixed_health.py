"""Read-only self-play diagnostics for each board size in a mixed run.

These statistics describe recorded games, not playing strength. Building a
report only reads SGFs and writes its own JSON reports; it never controls a
training process or imports a neural-network runtime.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from .fileio import atomic_open, atomic_write_json, local_timestamp
from .sgf import MalformedSGF, _read_value, classify_result, parse_sgf_game, split_sgf_collection
from .training_standard import get_training_standard


_THRESHOLDS = {
    "early_double_pass_rate_max": 0.05,
    "short_game_area_fraction": 0.10,
    "short_game_min_moves": 4,
    "short_game_rate_max": 0.25,
    "black_score_rate_min": 0.10,
    "black_score_rate_max": 0.90,
    "extreme_margin_area_fraction": 0.75,
    "extreme_margin_rate_max": 0.10,
    "no_result_rate_max": 0.05,
}


class _Game(NamedTuple):
    tree_index: int
    board_size: int
    moves: int
    result: str
    outcome: str
    margin: Optional[float]
    early_double_pass: bool
    pass_moves: int


class _Observed(NamedTuple):
    path: str
    source: str
    signature: tuple[int, int]
    game: _Game


def _scan_file(path: Path) -> dict:
    trees, pending = split_sgf_collection(path.read_text(encoding="utf-8-sig", errors="replace"))
    records = []
    damaged = 0
    damaged_examples = []
    for index, tree in enumerate(trees):
        try:
            parsed = parse_sgf_game(tree)
        except MalformedSGF as error:
            damaged += 1
            if len(damaged_examples) < 3:
                damaged_examples.append({"tree_index": index, "error": str(error)})
            continue
        moves = list(parsed["moves"])
        result = str(parsed["result"])
        outcome, margin = classify_result(result)
        double_pass = (len(moves) >= 2 and moves[0][1] == moves[1][1] == ""
                       and moves[0][0] != moves[1][0])
        records.append(_Game(index, int(parsed["board_size"]), len(moves), result,
                             outcome, margin, double_pass, sum(point == "" for _, point in moves)))
    return {"records": records, "damaged_games": damaged, "pending_games": pending,
            "damaged_examples": damaged_examples}


def _load_cache(path: Path) -> dict:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (not isinstance(cached, dict) or cached.get("schema_version") != 1
            or cached.get("record_fields") != list(_Game._fields)
            or not isinstance(cached.get("files"), dict)):
        return {}
    return cached["files"]


def _cached_entry_matches(entry: object, signature: list[int]) -> bool:
    if (not isinstance(entry, dict) or entry.get("signature") != signature
            or not isinstance(entry.get("records"), list)
            or type(entry.get("damaged_games")) is not int or entry["damaged_games"] < 0
            or type(entry.get("pending_games")) is not int or entry["pending_games"] not in (0, 1)
            or not isinstance(entry.get("damaged_examples"), list)):
        return False
    for row in entry["records"]:
        if not isinstance(row, list) or len(row) != len(_Game._fields):
            return False
        index, size, moves, result, outcome, margin, double_pass, passes = row
        if (type(index) is not int or index < 0 or type(size) is not int or size <= 0
                or type(moves) is not int or moves < 0 or not isinstance(result, str)
                or outcome not in ("black", "white", "draw", "no_result")
                or (margin is not None and (type(margin) not in (int, float) or not math.isfinite(margin)))
                or type(double_pass) is not bool or type(passes) is not int or not 0 <= passes <= moves):
            return False
    return all(isinstance(example, dict) and type(example.get("tree_index")) is int
               and isinstance(example.get("error"), str) for example in entry["damaged_examples"])


def _summarize(records: Sequence[_Game], *, board_size: int, minimum_games: int,
               standard: str | None = None) -> dict:
    count = len(records)
    results = Counter(record.outcome for record in records)
    decided = results["black"] + results["white"] + results["draw"]
    cutoff = max(_THRESHOLDS["short_game_min_moves"],
                 math.ceil(board_size ** 2 * _THRESHOLDS["short_game_area_fraction"]))
    early_passes = sum(record.early_double_pass for record in records)
    short_games = sum(record.moves < cutoff for record in records)
    extreme_games = sum(record.margin is not None
                        and record.margin >= board_size ** 2 * _THRESHOLDS["extreme_margin_area_fraction"]
                        for record in records)
    moves = sum(record.moves for record in records)
    passes = sum(record.pass_moves for record in records)
    summary = {
        "window_games": count,
        "estimable": count >= minimum_games,
        "average_moves": moves / count if count else None,
        "early_double_pass_games": early_passes,
        "early_double_pass_rate": early_passes / count if count else None,
        "short_game_move_cutoff": cutoff,
        "short_games": short_games,
        "short_game_rate": short_games / count if count else None,
        "result_counts": {outcome: results[outcome] for outcome in ("black", "white", "draw", "no_result")},
        "black_score_rate": (results["black"] + 0.5 * results["draw"]) / decided if decided else None,
        "extreme_margin_games": extreme_games,
        "extreme_margin_rate": extreme_games / count if count else None,
        "no_result_rate": results["no_result"] / count if count else None,
        "pass_moves": passes,
        "pass_move_rate": passes / moves if moves else None,
    }
    warnings = []
    if summary["estimable"]:
        for metric in ("early_double_pass_rate", "short_game_rate", "extreme_margin_rate", "no_result_rate"):
            if summary[metric] > _THRESHOLDS[f"{metric}_max"]:
                warnings.append(metric)
        score_rate = summary["black_score_rate"]
        if score_rate is not None and not _THRESHOLDS["black_score_rate_min"] <= score_rate <= _THRESHOLDS["black_score_rate_max"]:
            warnings.append("black_score_rate")
    summary["warnings"] = warnings
    summary["status"] = ("needs_review" if warnings else "ok") if summary["estimable"] else "warming_up"
    policy = get_training_standard(standard)
    if policy:
        reasons = policy.health_reasons(games=count, minimum_games=minimum_games,
            black_score_rate=summary["black_score_rate"],
            early_double_pass_rate=summary["early_double_pass_rate"],
            short_game_rate=summary["short_game_rate"], invalid_rate=summary["no_result_rate"],
            pass_move_rate=summary["pass_move_rate"])
        summary["health_reasons"] = list(reasons)
        summary["status"] = ("needs_review" if reasons else "ok") if summary["estimable"] else "warming_up"
    return summary


def _root_metadata(tree: str) -> dict[str, str]:
    """Read only root metadata, skipping SGF values with the strict escape reader."""
    metadata = {}
    index = tree.index(";") + 1
    while index < len(tree):
        if tree[index] in ";()":
            break
        if not tree[index].isalpha():
            index += 1
            continue
        start = index
        while index < len(tree) and tree[index].isalpha():
            index += 1
        identifier = tree[start:index].upper()
        while index < len(tree) and tree[index].isspace():
            index += 1
        while index < len(tree) and tree[index] == "[":
            value, index = _read_value(tree, index)
            if identifier in ("KM", "PB", "PW"):
                metadata.setdefault(identifier, value)
            while index < len(tree) and tree[index].isspace():
                index += 1
    return metadata


def _examples(run_root: Path, window: Sequence[_Observed], warnings: Sequence[str]) -> list[dict]:
    if not warnings:
        return []
    examples = []
    trees_by_path = {}
    for observed in reversed(window):
        record = observed.game
        reasons = []
        if "early_double_pass_rate" in warnings and record.early_double_pass:
            reasons.append("early_double_pass_rate")
        cutoff = max(4, math.ceil(record.board_size ** 2 * _THRESHOLDS["short_game_area_fraction"]))
        if "short_game_rate" in warnings and record.moves < cutoff:
            reasons.append("short_game_rate")
        if ("extreme_margin_rate" in warnings and record.margin is not None
                and record.margin >= record.board_size ** 2 * _THRESHOLDS["extreme_margin_area_fraction"]):
            reasons.append("extreme_margin_rate")
        if "no_result_rate" in warnings and record.outcome == "no_result":
            reasons.append("no_result_rate")
        if "black_score_rate" in warnings and record.outcome != "no_result":
            reasons.append("black_score_rate")
        if not reasons:
            continue
        example = {"path": observed.path, "source": observed.source, "tree_index": record.tree_index,
                   "board_size": record.board_size, "moves": record.moves, "result": record.result,
                   "pass_moves": record.pass_moves, "early_double_pass": record.early_double_pass,
                   "reason_codes": reasons}
        if observed.path not in trees_by_path:
            path = run_root / observed.path
            try:
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != observed.signature:
                    trees_by_path[observed.path] = []
                else:
                    content = path.read_text(encoding="utf-8-sig", errors="replace")
                    after = path.stat()
                    trees_by_path[observed.path] = (split_sgf_collection(content)[0]
                        if (after.st_size, after.st_mtime_ns) == observed.signature else [])
            except OSError:
                trees_by_path[observed.path] = []
        trees = trees_by_path[observed.path]
        if record.tree_index < len(trees):
            tree = trees[record.tree_index]
            metadata = _root_metadata(tree)
            example.update(original_sgf=tree, komi=metadata.get("KM"),
                           players={"black": metadata.get("PB"), "white": metadata.get("PW")})
        else:
            example["source_changed_since_scan"] = True
        examples.append(example)
        if len(examples) == 3:
            break
    return examples


def build_mixed_health(run_root: Path, board_sizes: Sequence[int], *,
                       window_games: int = 256, minimum_games: int = 64,
                       standard: str | None = None) -> dict:
    """Summarize trailing completed valid games independently for each board.

    Cumulative counts cover all currently present self-play SGFs. Windows use
    file modification time, relative path, then tree order as a deterministic
    approximation to game completion order. The minimum is independent of the
    requested window capacity, and must fit within it.
    """
    policy = get_training_standard(standard)
    board_sizes = tuple(board_sizes)
    if (not board_sizes or any(type(size) is not int or size <= 0 for size in board_sizes)
            or len(set(board_sizes)) != len(board_sizes)):
        raise ValueError("board_sizes must contain distinct positive integers")
    if (type(window_games) is not int or type(minimum_games) is not int
            or not 1 <= minimum_games <= window_games):
        raise ValueError("require 1 <= minimum_games <= window_games")
    run_root = Path(run_root)
    cache_path = run_root / ".mixed_health_inventory.json"
    old = _load_cache(cache_path)
    inventory = {"files": 0, "parsed_files": 0, "cached_files": 0, "valid_games": 0,
                 "damaged_games": 0, "pending_games": 0, "damaged_examples": [], "read_errors": []}
    files = {}
    candidates = []
    selfplay = run_root / "selfplay"
    for pattern in ("**/*.sgfs", "**/*.sgf"):
        for path in selfplay.glob(pattern):
            key = path.relative_to(run_root).as_posix()
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            except OSError as error:
                inventory["read_errors"].append({"path": key, "error": str(error)})
                continue
            candidates.append((stat.st_mtime_ns, key, path, [stat.st_size, stat.st_mtime_ns]))
    totals = Counter()
    windows = {size: deque(maxlen=window_games) for size in board_sizes}
    for _mtime, key, path, signature in sorted(candidates):
        cached = old.get(key)
        if _cached_entry_matches(cached, signature):
            entry = cached
            inventory["cached_files"] += 1
        else:
            try:
                entry = _scan_file(path)
            except FileNotFoundError:
                continue
            except OSError as error:
                inventory["read_errors"].append({"path": key, "error": str(error)})
                continue
            entry["signature"] = signature
            inventory["parsed_files"] += 1
        files[key] = entry
        inventory["files"] += 1
        inventory["damaged_games"] += entry["damaged_games"]
        inventory["pending_games"] += entry["pending_games"]
        for example in entry["damaged_examples"]:
            if len(inventory["damaged_examples"]) < 3:
                inventory["damaged_examples"].append({"path": key, **example})
        source = path.relative_to(selfplay).parts[0]
        for row in entry["records"]:
            record = _Game(*row)
            totals[record.board_size] += 1
            if record.board_size in windows:
                windows[record.board_size].append(_Observed(key, source, tuple(signature), record))
    boards = {}
    for size, window in windows.items():
        board = _summarize([item.game for item in window], board_size=size, minimum_games=minimum_games,
                           standard=standard)
        by_source = defaultdict(list)
        for item in window:
            by_source[item.source].append(item.game)
        board.update(cumulative_valid_games=totals[size], window_full=len(window) == window_games,
                     examples=_examples(run_root, window, board["warnings"]),
                     by_source={source: _summarize(records, board_size=size, minimum_games=minimum_games, standard=standard)
                                for source, records in by_source.items()})
        boards[str(size)] = board
    inventory["valid_games"] = sum(totals.values())
    inventory["unconfigured_board_games"] = {str(size): count for size, count in sorted(totals.items())
                                               if size not in windows}
    warnings = []
    if inventory["damaged_games"]:
        warnings.append("malformed_complete_games")
    if inventory["read_errors"]:
        warnings.append("unreadable_sources")
    status = ("needs_review" if warnings or any(board["status"] == "needs_review" for board in boards.values())
              else "ok" if all(board["estimable"] for board in boards.values()) else "warming_up")
    report = {
        "schema_version": 1,
        "updated_at": local_timestamp(),
        "status": status,
        "warnings": warnings,
        "by_board": boards,
        "inventory": inventory,
        "thresholds": {**_THRESHOLDS, "window_games": window_games, "minimum_games": minimum_games},
        "window_order": "file_mtime_ns, relative_path, tree_index",
        "note": ("Heuristic self-play review flags, not a playing-strength estimate or a training stop. "
                 "Rates use completed valid games in each board's trailing window; short means strictly "
                 "fewer moves than max(4, ceil(board area * 0.10)). Black score excludes no-results. "
                 "Cumulative counts cover currently present self-play SGFs. Pending trailing trees are "
                 "separate from malformed complete games. Source/model breakdowns cover the same window."),
    }
    if policy:
        report["training_standard"] = policy.to_dict()
    with atomic_open(cache_path) as stream:
        json.dump({"schema_version": 1, "record_fields": list(_Game._fields), "files": files},
                  stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    atomic_write_json(run_root / "health_by_board.json", report, allow_nan=False)
    return report
