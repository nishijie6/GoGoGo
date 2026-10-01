"""KataGo training metrics: SGF health, self-play logs, checkpoint losses.

The persistent per-run history lives in ``rl_metric_store`` and its HTML page
in ``rl_metrics_dashboard``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from .sgf import MalformedSGF, classify_result, parse_sgf_game, split_sgf_collection


MODEL_NAME_PATTERN = re.compile(r"^gogogo-s(?P<samples>\d+)-d(?P<data_rows>\d+)$")
SELFPLAY_PATTERNS = {
    "selfplay_games": re.compile(r"Final games finished:\s*(\d+)"),
    "selfplay_moves": re.compile(r"Final moves played:\s*(\d+)"),
    "selfplay_data_rows": re.compile(r"Final data rows:\s*(\d+)"),
    "selfplay_nn_rows": re.compile(r"Final NN rows:\s*(\d+)"),
    "selfplay_nn_batches": re.compile(r"Final NN batches:\s*(\d+)"),
    "selfplay_avg_batch_size": re.compile(
        r"Final NN avg batch size:\s*([0-9]+(?:\.[0-9]+)?)"
    ),
    "selfplay_seconds": re.compile(
        r"Total selfplay runtime \(seconds\):\s*([0-9]+(?:\.[0-9]+)?)"
    ),
}

# A margin spanning at least a quarter of all intersections is a useful,
# board-size-independent signal for the degenerate "one side owns the board"
# games seen early in reinforcement learning.  Keep the value public so the
# curriculum quality gate and its tests use exactly the same definition.
EXTREME_MARGIN_BOARD_FRACTION = 0.25
HEALTH_WINDOW_CYCLES = 10
HEALTH_WINDOW_GAMES = 1280
HEALTH_DOUBLE_PASS_MAX = 0.01
HEALTH_EXTREME_RESULT_MAX = 0.05
HEALTH_BLACK_WIN_MIN = 0.45
HEALTH_BLACK_WIN_MAX = 0.55
HEALTH_INVALID_MAX = 0.005

CSV_FIELDS = (
    "cycle",
    "timestamp",
    "model",
    "trained_samples",
    "sample_delta",
    "data_rows",
    "data_rows_delta",
    "accepted_models",
    "loss",
    "policy_loss",
    "value_loss",
    "score_loss",
    "policy_accuracy",
    "gradient_norm",
    "selfplay_games",
    "selfplay_moves",
    "selfplay_data_rows",
    "selfplay_nn_rows",
    "selfplay_nn_batches",
    "selfplay_avg_batch_size",
    "selfplay_seconds",
    "board_size",
    "sgf_games",
    "sgf_entries",
    "last_game_result",
    "black_wins",
    "white_wins",
    "draws",
    "no_result_games",
    "damaged_games",
    "average_score_margin",
    "average_moves",
    "opening_double_pass_games",
    "opening_double_pass_rate",
    "extreme_result_games",
    "extreme_result_rate",
    "black_win_rate",
    "invalid_game_rate",
    "health_window_cycles",
    "health_window_games",
    "health_black_wins",
    "health_white_wins",
    "health_draws",
    "health_no_result_games",
    "health_damaged_games",
    "health_opening_double_pass_games",
    "health_extreme_result_games",
    "health_black_win_rate",
    "health_opening_double_pass_rate",
    "health_extreme_result_rate",
    "health_invalid_game_rate",
    "health_window_complete",
    "health_passed",
    "shuffle_seconds",
    "train_seconds",
    "export_seconds",
    "cycle_seconds",
    "model_bytes",
    "source",
)


def parse_sgfs(paths: Path | Iterable[Path]) -> dict[str, object]:
    """Summarize KataGo self-play SGFs using only the Python standard library.

    A malformed tree is counted separately and never contributes to result,
    move-count, or degeneration rates.  Unknown/missing ``RE`` values are valid
    SGFs with no result, which lets callers distinguish engine no-results from
    damaged storage.
    """

    if isinstance(paths, Path):
        path_list: Sequence[Path] = (paths,)
    else:
        path_list = tuple(Path(path) for path in paths)
    black_wins = white_wins = draws = no_results = damaged = 0
    double_passes = extreme_results = 0
    score_margins: list[float] = []
    move_counts: list[int] = []
    last_result: Optional[str] = None
    board_size: Optional[int] = None
    parsed_games = 0
    total_bytes = 0
    for path in path_list:
        try:
            total_bytes += path.stat().st_size
            content = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            damaged += 1
            continue
        games, split_damage = split_sgf_collection(content)
        damaged += split_damage
        for game in games:
            try:
                parsed = parse_sgf_game(game)
            except MalformedSGF:
                damaged += 1
                continue
            parsed_games += 1
            current_size = int(parsed["board_size"])
            if board_size is None:
                board_size = current_size
            elif board_size != current_size:
                damaged += 1
                parsed_games -= 1
                continue
            result = str(parsed["result"])
            last_result = result or "无结果"
            outcome, margin = classify_result(result)
            if outcome == "black":
                black_wins += 1
            elif outcome == "white":
                white_wins += 1
            elif outcome == "draw":
                draws += 1
            else:
                no_results += 1
            if margin is not None and outcome in {"black", "white"}:
                score_margins.append(margin)
                if margin >= current_size * current_size * EXTREME_MARGIN_BOARD_FRACTION:
                    extreme_results += 1
            moves = list(parsed["moves"])
            move_counts.append(len(moves))
            if (
                len(moves) >= 2
                and moves[0][1] == ""
                and moves[1][1] == ""
                and moves[0][0] != moves[1][0]
            ):
                double_passes += 1

    entries = parsed_games + damaged
    decided = black_wins + white_wins + draws
    result: dict[str, object] = {
        "sgf_files": len(path_list),
        "sgf_file_bytes": total_bytes,
        "sgf_games": parsed_games,
        "sgf_entries": entries,
        "last_game_result": last_result,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "no_result_games": no_results,
        "damaged_games": damaged,
        "opening_double_pass_games": double_passes,
        "extreme_result_games": extreme_results,
        "opening_double_pass_rate": double_passes / parsed_games if parsed_games else 0.0,
        "extreme_result_rate": extreme_results / parsed_games if parsed_games else 0.0,
        "black_win_rate": (
            (black_wins + 0.5 * draws) / decided if decided else None
        ),
        "invalid_game_rate": (
            (no_results + damaged) / entries if entries else 0.0
        ),
        "average_score_margin": (
            sum(score_margins) / len(score_margins) if score_margins else None
        ),
        "average_moves": sum(move_counts) / len(move_counts) if move_counts else None,
    }
    if board_size is not None:
        result["board_size"] = board_size
    return result


def rolling_health_metrics(
    records: Iterable[Mapping[str, object]],
    *,
    max_cycles: int = HEALTH_WINDOW_CYCLES,
    required_games: int = HEALTH_WINDOW_GAMES,
) -> dict[str, object]:
    """Aggregate and evaluate the most recent self-play quality window."""

    eligible = [record for record in records if "sgf_entries" in record]
    window = eligible[-max_cycles:]
    count_fields = {
        "health_black_wins": "black_wins",
        "health_white_wins": "white_wins",
        "health_draws": "draws",
        "health_no_result_games": "no_result_games",
        "health_damaged_games": "damaged_games",
        "health_opening_double_pass_games": "opening_double_pass_games",
        "health_extreme_result_games": "extreme_result_games",
    }
    totals = {
        output: sum(int(record.get(source, 0) or 0) for record in window)
        for output, source in count_fields.items()
    }
    parsed_games = sum(int(record.get("sgf_games", 0) or 0) for record in window)
    entries = sum(int(record.get("sgf_entries", 0) or 0) for record in window)
    decided = totals["health_black_wins"] + totals["health_white_wins"] + totals["health_draws"]
    black_rate = (
        (totals["health_black_wins"] + 0.5 * totals["health_draws"]) / decided
        if decided
        else None
    )
    double_pass_rate = totals["health_opening_double_pass_games"] / parsed_games if parsed_games else 0.0
    extreme_rate = totals["health_extreme_result_games"] / parsed_games if parsed_games else 0.0
    invalid_rate = (
        (totals["health_no_result_games"] + totals["health_damaged_games"]) / entries
        if entries
        else 0.0
    )
    complete = len(window) >= max_cycles and entries >= required_games
    passed = bool(
        complete
        and black_rate is not None
        and HEALTH_BLACK_WIN_MIN <= black_rate <= HEALTH_BLACK_WIN_MAX
        and double_pass_rate <= HEALTH_DOUBLE_PASS_MAX
        and extreme_rate <= HEALTH_EXTREME_RESULT_MAX
        and invalid_rate <= HEALTH_INVALID_MAX
    )
    return {
        "health_window_cycles": len(window),
        "health_window_games": entries,
        **totals,
        "health_black_win_rate": black_rate,
        "health_opening_double_pass_rate": double_pass_rate,
        "health_extreme_result_rate": extreme_rate,
        "health_invalid_game_rate": invalid_rate,
        "health_window_complete": complete,
        "health_passed": passed,
    }


def parse_model_name(name: str) -> Optional[tuple[int, int]]:
    """Return ``(trained samples, data rows)`` encoded in an export name."""

    match = MODEL_NAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group("samples")), int(match.group("data_rows"))


def parse_selfplay_output(output: str) -> dict[str, int | float]:
    """Extract the final counters emitted by KataGo's self-play command."""

    result: dict[str, int | float] = {}
    float_fields = {"selfplay_avg_batch_size", "selfplay_seconds"}
    for field, pattern in SELFPLAY_PATTERNS.items():
        matches = pattern.findall(output)
        if not matches:
            continue
        raw = matches[-1]
        result[field] = float(raw) if field in float_fields else int(raw)
    return result


def _safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_average(
    sums: Mapping[str, object], weights: Mapping[str, object], key: str
) -> Optional[float]:
    numerator = _safe_float(sums.get(key))
    denominator = _safe_float(weights.get(key))
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def checkpoint_metrics(checkpoint: Path) -> dict[str, int | float]:
    """Read lightweight training state and EMA losses from a local checkpoint."""

    if not checkpoint.is_file():
        return {}
    # Import lazily so status/dashboard code still imports cleanly before the
    # optional reinforcement-learning dependencies are installed.
    import torch

    load_options = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        state = torch.load(checkpoint, mmap=True, **load_options)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        state = torch.load(checkpoint, **load_options)

    train_state = state.get("train_state", {})
    running = state.get("running_metrics", {})
    sums = running.get("sums", {})
    weights = running.get("weights", {})
    result: dict[str, int | float] = {}
    state_fields = {
        "trained_samples": "global_step_samples",
        "data_rows": "total_num_data_rows",
    }
    for output_name, state_name in state_fields.items():
        value = train_state.get(state_name)
        if value is not None:
            result[output_name] = int(value)
    metric_fields = {
        "loss": "loss_sum",
        "policy_loss": "p0loss_sum",
        "value_loss": "vloss_sum",
        "score_loss": "sloss_sum",
        "policy_accuracy": "pacc1_sum",
        "gradient_norm": "gnorm_batch",
    }
    for output_name, metric_name in metric_fields.items():
        value = _metric_average(sums, weights, metric_name)
        if value is not None:
            result[output_name] = value
    return result


__all__ = [
    "CSV_FIELDS",
    "EXTREME_MARGIN_BOARD_FRACTION",
    "HEALTH_BLACK_WIN_MAX",
    "HEALTH_BLACK_WIN_MIN",
    "HEALTH_DOUBLE_PASS_MAX",
    "HEALTH_EXTREME_RESULT_MAX",
    "HEALTH_INVALID_MAX",
    "HEALTH_WINDOW_CYCLES",
    "HEALTH_WINDOW_GAMES",
    "checkpoint_metrics",
    "parse_model_name",
    "parse_selfplay_output",
    "parse_sgfs",
    "rolling_health_metrics",
]
