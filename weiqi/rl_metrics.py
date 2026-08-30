"""Persistent metrics and an offline dashboard for local KataGo training."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


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


class _MalformedSGF(ValueError):
    """Internal marker for a game tree that cannot be trusted."""


def _split_sgf_collection(content: str) -> tuple[list[str], int]:
    """Split an SGF collection without being fooled by comments or escapes.

    KataGo's ``.sgfs`` files normally contain one complete tree per line, but
    parsing balanced game trees makes this also work for wrapped SGFs and SGF
    collections.  The second return value counts an unterminated trailing tree.
    """

    games: list[str] = []
    start: Optional[int] = None
    depth = 0
    in_value = False
    escaped = False
    for index, character in enumerate(content):
        if in_value:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "]":
                in_value = False
            continue
        if character == "[" and depth > 0:
            in_value = True
        elif character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                games.append(content[start : index + 1])
                start = None
    return games, int(depth != 0 or in_value)


def _read_sgf_value(game: str, offset: int) -> tuple[str, int]:
    if offset >= len(game) or game[offset] != "[":
        raise _MalformedSGF("SGF property is missing its value")
    offset += 1
    value: list[str] = []
    while offset < len(game):
        character = game[offset]
        if character == "]":
            return "".join(value), offset + 1
        if character == "\\":
            offset += 1
            if offset >= len(game):
                raise _MalformedSGF("unterminated SGF escape")
            escaped = game[offset]
            # SGF line continuation removes the escaped line break.
            if escaped == "\r" and offset + 1 < len(game) and game[offset + 1] == "\n":
                offset += 1
            elif escaped not in "\r\n":
                value.append(escaped)
        else:
            value.append(character)
        offset += 1
    raise _MalformedSGF("unterminated SGF property value")


def _parse_sgf_game(game: str) -> dict[str, object]:
    """Parse the root result and main-line moves of one SGF game tree."""

    if "\ufffd" in game:
        raise _MalformedSGF("invalid UTF-8 replacement character")
    depth = 0
    node_index = -1
    root: dict[str, list[str]] = {}
    moves: list[tuple[str, str]] = []
    index = 0
    while index < len(game):
        character = game[index]
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth -= 1
            if depth < 0:
                raise _MalformedSGF("unbalanced SGF game tree")
            index += 1
            continue
        if character == ";":
            if depth == 1:
                node_index += 1
            index += 1
            continue
        if not character.isalpha():
            index += 1
            continue

        property_start = index
        while index < len(game) and game[index].isalpha():
            index += 1
        identifier = game[property_start:index].upper()
        while index < len(game) and game[index].isspace():
            index += 1
        values: list[str] = []
        while index < len(game) and game[index] == "[":
            value, index = _read_sgf_value(game, index)
            values.append(value)
            while index < len(game) and game[index].isspace():
                index += 1
        if not values:
            # Alphabetic text is only legal as a property identifier followed
            # by one or more values.  Treat it as corruption, rather than
            # accidentally mining moves embedded in free-form text.
            raise _MalformedSGF(f"property {identifier!r} has no value")
        if depth == 1 and node_index == 0:
            root.setdefault(identifier, []).extend(values)
        if depth == 1 and node_index >= 1 and identifier in {"B", "W"}:
            moves.append((identifier, values[0]))
    if depth != 0 or node_index < 0:
        raise _MalformedSGF("incomplete SGF game tree")

    size_value = (root.get("SZ") or [""])[0]
    try:
        dimensions = [int(part) for part in size_value.split(":")]
    except ValueError as error:
        raise _MalformedSGF("invalid or missing board size") from error
    if not dimensions or dimensions[0] <= 0 or any(part != dimensions[0] for part in dimensions):
        raise _MalformedSGF("only square positive boards are supported")
    result = (root.get("RE") or [""])[0].strip()
    return {
        "board_size": dimensions[0],
        "result": result,
        "moves": moves,
    }


def _classify_result(result: str) -> tuple[str, Optional[float]]:
    normalized = result.strip()
    folded = normalized.casefold()
    if folded in {"0", "draw", "jigo"}:
        return "draw", 0.0
    match = re.fullmatch(r"([BWbw])\+(.+)", normalized)
    if match is None:
        return "no_result", None
    winner = "black" if match.group(1).upper() == "B" else "white"
    margin = _safe_float(match.group(2))
    return winner, abs(margin) if margin is not None else None


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
        games, split_damage = _split_sgf_collection(content)
        damaged += split_damage
        for game in games:
            try:
                parsed = _parse_sgf_game(game)
            except _MalformedSGF:
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
            outcome, margin = _classify_result(result)
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


def _timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        output.write(content)
    temporary.replace(path)


def _format_number(value: object, digits: int = 0) -> str:
    number = _safe_float(value)
    if number is None:
        return "—"
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def _format_percentage(value: object, digits: int = 1) -> str:
    number = _safe_float(value)
    if number is None:
        return "—"
    return f"{number * 100:.{digits}f}%"


def _compact_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}k"
    if absolute >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _line_chart(
    records: list[dict[str, object]],
    *,
    title: str,
    description: str,
    series: Iterable[tuple[str, str, str]],
    zero_baseline: bool = False,
) -> str:
    width, height = 920, 315
    left, right, top, bottom = 72, 22, 26, 44
    plot_width = width - left - right
    plot_height = height - top - bottom
    lines = list(series)
    plotted: list[tuple[str, str, str, list[tuple[float, float]]]] = []
    for key, label, color in lines:
        points: list[tuple[float, float]] = []
        for record in records:
            x = _safe_float(record.get("cycle"))
            y = _safe_float(record.get(key))
            if x is not None and y is not None:
                points.append((x, y))
        if points:
            plotted.append((key, label, color, points))
    if not plotted:
        return (
            f'<section class="panel"><h2>{html.escape(title)}</h2>'
            '<div class="empty">完成下一轮后会在这里绘制曲线。</div></section>'
        )

    x_values = [point[0] for _, _, _, points in plotted for point in points]
    y_values = [point[1] for _, _, _, points in plotted for point in points]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    data_min, data_max = min(y_values), max(y_values)
    y_min = 0.0 if zero_baseline else data_min
    if data_min == data_max:
        padding = max(abs(data_min) * 0.08, 1.0)
    else:
        padding = (data_max - data_min) * 0.08
    if not zero_baseline:
        y_min -= padding
    y_max = data_max + padding
    if y_min == y_max:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    svg: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(description)}">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(description)}</desc>",
    ]
    for index in range(5):
        fraction = index / 4
        y = top + fraction * plot_height
        value = y_max - fraction * (y_max - y_min)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{width - right}" y2="{y:.2f}" />'
        )
        svg.append(
            f'<text class="axis-label" x="{left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end">{html.escape(_compact_number(value))}</text>'
        )
    tick_count = min(6, max(2, int(x_max - x_min + 1)))
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        x = left + fraction * plot_width
        value = round(x_min + fraction * (x_max - x_min))
        svg.append(
            f'<line class="tick" x1="{x:.2f}" y1="{top + plot_height}" '
            f'x2="{x:.2f}" y2="{top + plot_height + 5}" />'
        )
        svg.append(
            f'<text class="axis-label" x="{x:.2f}" y="{height - 16}" '
            f'text-anchor="middle">{value}</text>'
        )
    svg.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_height}" />'
    )
    svg.append(
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
        f'x2="{width - right}" y2="{top + plot_height}" />'
    )
    for _, label, color, points in plotted:
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        svg.append(
            f'<polyline class="series" style="stroke:{color}" points="{coordinates}" />'
        )
        last_x, last_y = points[-1]
        svg.append(
            f'<circle cx="{sx(last_x):.2f}" cy="{sy(last_y):.2f}" r="4" '
            f'fill="{color}"><title>{html.escape(label)}: '
            f'{html.escape(_format_number(last_y, 4))}</title></circle>'
        )
    svg.append("</svg>")
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>'
        for _, label, color, _ in plotted
    )
    return (
        '<section class="panel chart-panel">'
        f'<div class="panel-heading"><div><h2>{html.escape(title)}</h2>'
        f'<p>{html.escape(description)}</p></div><div class="legend">{legend}</div></div>'
        + "".join(svg)
        + "</section>"
    )


def render_dashboard(
    records: list[dict[str, object]],
    generated_at: str,
    board_size: Optional[int] = None,
) -> str:
    """Render a self-contained dashboard that works without a web server."""

    latest = records[-1] if records else {}
    latest_game = next(
        (record for record in reversed(records) if "sgf_games" in record),
        {},
    )
    inferred_size = next(
        (
            int(record["board_size"])
            for record in reversed(records)
            if _safe_float(record.get("board_size")) is not None
        ),
        None,
    )
    # Nine was the original dashboard's fixed size, so keeping it as the API
    # fallback preserves old callers while RLMetricStore supplies 13/19 below.
    resolved_size = board_size or inferred_size or 9
    dashboard_title = f"KataGo {resolved_size}×{resolved_size} 强化学习监控"
    latest_timestamp = str(latest.get("timestamp", ""))
    sample_chart = _line_chart(
        records,
        title="累计训练规模",
        description="每个已接纳网络对应的累计训练样本与自对弈数据行。",
        series=(
            ("trained_samples", "训练样本", "#5b8def"),
            ("data_rows", "数据行", "#22b8a7"),
        ),
        zero_baseline=True,
    )
    loss_chart = _line_chart(
        records,
        title="总训练损失（EMA）",
        description="KataGo 检查点中的官方总损失指数移动平均；纵轴按当前范围缩放。",
        series=(("loss", "总损失", "#f59f3a"),),
    )
    component_chart = _line_chart(
        records,
        title="优化目标分量（EMA）",
        description="策略、胜负价值与目数预测损失；同一纵轴，越低通常代表拟合更好。",
        series=(
            ("policy_loss", "策略损失", "#a277ff"),
            ("value_loss", "价值损失", "#ff6b87"),
            ("score_loss", "目数损失", "#22b8a7"),
        ),
    )
    throughput_chart = _line_chart(
        records,
        title="每轮数据产出",
        description="相邻已接纳模型之间新增的自对弈训练行。",
        series=(("data_rows_delta", "新增数据行", "#5b8def"),),
        zero_baseline=True,
    )
    health_chart = _line_chart(
        records,
        title="自对弈健康趋势",
        description="单轮黑方得分率、开局双方立即停一手率与极端目差率（0–1 比例）。",
        series=(
            ("black_win_rate", "黑方得分率", "#5b8def"),
            ("opening_double_pass_rate", "开局双停率", "#ff6b87"),
            ("extreme_result_rate", "极端结果率", "#f59f3a"),
            ("invalid_game_rate", "无效棋谱率", "#a277ff"),
        ),
        zero_baseline=True,
    )

    recent_rows = []
    for record in reversed(records[-10:]):
        recent_rows.append(
            "<tr>"
            f'<td>{html.escape(str(record.get("cycle", "")))}</td>'
            f'<td class="mono">{html.escape(str(record.get("model", "")))}</td>'
            f'<td>{_format_number(record.get("trained_samples"))}</td>'
            f'<td>{_format_number(record.get("data_rows"))}</td>'
            f'<td>{_format_number(record.get("data_rows_delta"))}</td>'
            f'<td>{_format_number(record.get("loss"), 3)}</td>'
            f'<td>{_format_number(record.get("policy_loss"), 3)}</td>'
            f'<td>{_format_number(record.get("value_loss"), 3)}</td>'
            f'<td>{html.escape(str(record.get("last_game_result", "—")))}</td>'
            f'<td>{_format_number(record.get("black_wins"))}/{_format_number(record.get("white_wins"))}</td>'
            f'<td>{_format_number(record.get("average_score_margin"), 1)}</td>'
            f'<td>{_format_number(record.get("average_moves"), 1)}</td>'
            f'<td>{_format_percentage(record.get("opening_double_pass_rate"))}</td>'
            f'<td>{_format_percentage(record.get("extreme_result_rate"))}</td>'
            f'<td>{_format_number(record.get("cycle_seconds"), 1)}</td>'
            "</tr>"
        )
    table_body = "".join(recent_rows) or (
        '<tr><td colspan="15" class="empty">尚无已接纳模型。</td></tr>'
    )
    cards = (
        ("已接纳模型", _format_number(latest.get("accepted_models"))),
        ("累计训练样本", _format_number(latest.get("trained_samples"))),
        ("自对弈数据行", _format_number(latest.get("data_rows"))),
        ("总损失 EMA", _format_number(latest.get("loss"), 3)),
        ("策略命中率", (
            f'{float(latest["policy_accuracy"]) * 100:.1f}%'
            if _safe_float(latest.get("policy_accuracy")) is not None
            else "—"
        )),
        ("最近一轮", f'{_format_number(latest.get("cycle_seconds"), 1)} 秒'),
        ("末局结果", str(latest_game.get("last_game_result", "—"))),
        (
            "黑胜 / 白胜",
            f'{_format_number(latest_game.get("black_wins"))} / {_format_number(latest_game.get("white_wins"))}',
        ),
        ("平均目差", _format_number(latest_game.get("average_score_margin"), 1)),
        ("平均手数", _format_number(latest_game.get("average_moves"), 1)),
        ("开局双方双停", _format_percentage(latest_game.get("opening_double_pass_rate"))),
        ("极端棋局", _format_percentage(latest_game.get("extreme_result_rate"))),
        ("10 轮黑方得分率", _format_percentage(latest.get("health_black_win_rate"))),
        (
            "健康窗口",
            "通过" if latest.get("health_passed") else (
                "未达标" if latest.get("health_window_complete") else "收集中"
            ),
        ),
    )
    card_html = "".join(
        f'<article class="stat"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></article>'
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="refresh" content="15">
  <title>{html.escape(dashboard_title)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f3f5fa; --panel:#fff; --text:#18202f;
      --muted:#687386; --border:#dfe4ee; --grid:#e8ebf2; --axis:#8590a2;
      --shadow:0 14px 35px rgba(31,43,67,.08); }}
    @media (prefers-color-scheme:dark) {{ :root {{ --bg:#10141d; --panel:#171d29;
      --text:#edf1f8; --muted:#9ca8bb; --border:#2b3445; --grid:#293142;
      --axis:#77849a; --shadow:0 16px 38px rgba(0,0,0,.24); }} }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text);
      font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif; }}
    main {{ width:min(1480px,calc(100% - 32px)); margin:0 auto; padding:28px 0 44px }}
    header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:20px;
      margin-bottom:20px }} h1 {{ margin:0 0 7px; font-size:clamp(24px,3vw,38px); letter-spacing:-.03em }}
    header p,.panel-heading p {{ margin:0; color:var(--muted) }} .status {{ text-align:right }}
    .badge {{ display:inline-flex; align-items:center; gap:7px; padding:7px 11px;
      border:1px solid var(--border); border-radius:999px; background:var(--panel); font-weight:650 }}
    .dot {{ width:9px; height:9px; border-radius:50%; background:#22b8a7; box-shadow:0 0 0 4px rgba(34,184,167,.13) }}
    .status small {{ display:block; color:var(--muted); margin-top:7px }}
    .stats {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin-bottom:16px }}
    .stat,.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:16px; box-shadow:var(--shadow) }}
    .stat {{ padding:17px 18px }} .stat span {{ display:block; color:var(--muted); font-size:13px; margin-bottom:8px }}
    .stat strong {{ font-size:clamp(19px,2vw,28px); letter-spacing:-.03em }}
    .charts {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px }}
    .panel {{ padding:19px; min-width:0 }} .panel-heading {{ display:flex; justify-content:space-between; gap:15px; align-items:flex-start }}
    h2 {{ margin:0 0 5px; font-size:17px }} .panel-heading p {{ font-size:13px; line-height:1.5 }}
    svg {{ display:block; width:100%; height:auto; margin-top:8px; overflow:visible }}
    .grid {{ stroke:var(--grid); stroke-width:1 }} .axis,.tick {{ stroke:var(--axis); stroke-width:1 }}
    .axis-label {{ fill:var(--muted); font-size:12px }} .series {{ fill:none; stroke-width:2.6; stroke-linecap:round; stroke-linejoin:round }}
    .legend {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:10px; color:var(--muted); font-size:12px }}
    .legend span {{ white-space:nowrap }} .legend i {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px }}
    .table-panel {{ margin-top:16px; overflow:auto }} table {{ width:100%; border-collapse:collapse; font-size:13px }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:right; white-space:nowrap }}
    th {{ color:var(--muted); font-weight:600 }} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left }}
    tr:last-child td {{ border-bottom:0 }} .mono {{ font-family:"Cascadia Code",Consolas,monospace; font-size:12px }}
    .note {{ margin-top:16px; color:var(--muted); font-size:13px; line-height:1.65 }}
    .note a {{ color:#5b8def }} .empty {{ min-height:210px; display:grid; place-items:center; color:var(--muted) }}
    @media(max-width:1050px) {{ .stats {{ grid-template-columns:repeat(3,1fr) }} .charts {{ grid-template-columns:1fr }} }}
    @media(max-width:650px) {{ main {{ width:min(100% - 20px,1480px); padding-top:18px }} header {{ align-items:flex-start; flex-direction:column }}
      .status {{ text-align:left }} .stats {{ grid-template-columns:repeat(2,1fr) }} .panel {{ padding:13px }} .panel-heading {{ flex-direction:column }} }}
  </style>
</head>
<body>
<main>
  <header><div><h1>{html.escape(dashboard_title)}</h1><p>自对弈 → 洗牌 → BF16 训练 → 模型接纳</p></div>
    <div class="status"><div class="badge"><span class="dot"></span><span id="health">检查训练状态…</span></div>
    <small>生成于 {html.escape(generated_at)} · 页面每 15 秒刷新</small></div></header>
  <section class="stats">{card_html}</section>
  <section class="charts">{sample_chart}{loss_chart}{component_chart}{throughput_chart}{health_chart}</section>
  <section class="panel table-panel"><div class="panel-heading"><div><h2>最近 10 个模型</h2><p>完整原始数据可下载为 CSV 或 JSONL。</p></div>
    <div class="legend"><a href="metrics/history.csv">history.csv</a><a href="metrics/history.jsonl">history.jsonl</a></div></div>
    <table><thead><tr><th>轮次</th><th>模型</th><th>训练样本</th><th>数据行</th><th>新增行</th><th>总损失</th><th>策略损失</th><th>价值损失</th><th>末局</th><th>黑/白胜</th><th>平均目差</th><th>平均手数</th><th>双停率</th><th>极端率</th><th>耗时/秒</th></tr></thead>
    <tbody>{table_body}</tbody></table></section>
  <p class="note">损失来自 KataGo 官方检查点的 <code>running_metrics</code> 指数移动平均。极端棋局定义为终局目差达到棋盘交叉点总数的 {EXTREME_MARGIN_BOARD_FRACTION:.0%}；健康窗口严格使用最近 {HEALTH_WINDOW_CYCLES} 轮、至少 {HEALTH_WINDOW_GAMES} 盘。损失下降不等同于棋力或 Elo，可靠棋力结论仍需独立、固定条件的对局评测。图表和数据文件会在每轮模型接纳后原子更新。</p>
</main>
<script>
  const latest = Date.parse({json.dumps(latest_timestamp, ensure_ascii=False)});
  const node = document.getElementById('health');
  function updateHealth() {{
    const age = Date.now() - latest;
    node.textContent = Number.isFinite(age) && age < 5 * 60 * 1000 ? '持续训练活跃' : '等待新一轮数据';
  }}
  updateHealth(); setInterval(updateHealth, 15000);
</script>
</body>
</html>
"""


class RLMetricStore:
    """Synchronize accepted checkpoints into durable metrics and charts."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.models_dir = self.run_root / "models"
        self.exports_dir = self.run_root / "torchmodels_toexport"
        self.selfplay_dir = self.run_root / "selfplay"
        self.metrics_dir = self.run_root / "metrics"
        self.jsonl_path = self.metrics_dir / "history.jsonl"
        self.csv_path = self.metrics_dir / "history.csv"
        self.latest_path = self.metrics_dir / "latest.json"
        self.dashboard_path = self.run_root / "dashboard.html"
        size_match = re.fullmatch(r"(?P<size>\d+)x(?P=size)", self.run_root.name)
        self.board_size = int(size_match.group("size")) if size_match else None

    def paths(self) -> dict[str, str]:
        return {
            "metrics_jsonl": str(self.jsonl_path),
            "metrics_csv": str(self.csv_path),
            "metrics_latest": str(self.latest_path),
            "dashboard": str(self.dashboard_path),
        }

    def _read_existing(self) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        if not self.jsonl_path.is_file():
            return records
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = record.get("model")
            if isinstance(model, str):
                records[model] = record
        return records

    def _model_record(self, model_dir: Path) -> dict[str, object]:
        parsed = parse_model_name(model_dir.name)
        if parsed is None:
            raise ValueError(f"无法解析 KataGo 模型名：{model_dir.name}")
        trained_samples, data_rows = parsed
        model_file = model_dir / "model.bin.gz"
        record: dict[str, object] = {
            "schema_version": 1,
            "timestamp": _timestamp(model_dir),
            "model": model_dir.name,
            "trained_samples": trained_samples,
            "data_rows": data_rows,
            "model_bytes": model_file.stat().st_size if model_file.is_file() else None,
            "source": "checkpoint-bootstrap",
        }
        checkpoint = self.exports_dir / model_dir.name / "model.ckpt"
        try:
            record.update(checkpoint_metrics(checkpoint))
        except (ImportError, OSError, RuntimeError, ValueError):
            # The name itself still provides valid monotonic progress if an old
            # checkpoint has been pruned or is temporarily unavailable.
            pass
        return record

    def sync(
        self,
        *,
        live_model: Optional[str] = None,
        live_metrics: Optional[Mapping[str, object]] = None,
    ) -> list[dict[str, object]]:
        """Import new accepted models, then atomically refresh every artifact."""

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_existing()
        model_dirs = {
            path.name: path
            for path in self.models_dir.iterdir()
            if path.is_dir()
            and ".tmp-" not in path.name
            and (path / "model.bin.gz").is_file()
            and parse_model_name(path.name) is not None
        } if self.models_dir.is_dir() else {}
        # Model/checkpoint retention is intentionally much shorter than metric
        # retention.  Build from the union so pruning a model never truncates
        # the durable JSONL/CSV history on the next refresh.
        model_names = sorted(
            {
                name
                for name in (*existing.keys(), *model_dirs.keys())
                if parse_model_name(name) is not None
            },
            key=lambda name: parse_model_name(name) or (0, 0),
        )
        records: list[dict[str, object]] = []
        for model_name in model_names:
            model_dir = model_dirs.get(model_name)
            record = existing.get(model_name)
            if record is None:
                if model_dir is None:  # Defensive; union guarantees this cannot happen.
                    continue
                record = self._model_record(model_dir)
            else:
                record = dict(record)
            if model_name == live_model and live_metrics:
                record.update(live_metrics)
                record["source"] = "completed-cycle"
            sgf_paths = sorted(
                (self.selfplay_dir / model_name / "sgfs").glob("*.sgfs")
            )
            if sgf_paths:
                sgf_bytes = sum(path.stat().st_size for path in sgf_paths)
                if record.get("sgf_file_bytes") != sgf_bytes:
                    record.update(parse_sgfs(sgf_paths))
            records.append(record)

        previous_samples = 0
        previous_rows = 0
        previous_time: Optional[datetime] = None
        for index, record in enumerate(records, start=1):
            record["schema_version"] = 2
            record["cycle"] = index
            record["accepted_models"] = index
            samples = int(record.get("trained_samples", 0))
            rows = int(record.get("data_rows", 0))
            record["sample_delta"] = samples - previous_samples
            record["data_rows_delta"] = rows - previous_rows
            current_time: Optional[datetime]
            try:
                current_time = datetime.fromisoformat(str(record["timestamp"]))
            except (KeyError, TypeError, ValueError):
                current_time = None
            if "cycle_seconds" not in record and previous_time and current_time:
                record["cycle_seconds"] = max(
                    0.0, (current_time - previous_time).total_seconds()
                )
            previous_samples = samples
            previous_rows = rows
            previous_time = current_time or previous_time

        health_records: list[dict[str, object]] = []
        for record in records:
            if "sgf_entries" in record:
                health_records.append(record)
            record.update(rolling_health_metrics(health_records))

        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        jsonl = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        _atomic_write_text(self.jsonl_path, jsonl)
        csv_temporary = self.csv_path.with_name(
            f".{self.csv_path.name}.{os.getpid()}.tmp"
        )
        with csv_temporary.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        csv_temporary.replace(self.csv_path)
        latest = records[-1] if records else {}
        _atomic_write_text(
            self.latest_path,
            json.dumps(latest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write_text(
            self.dashboard_path,
            render_dashboard(records, generated_at, board_size=self.board_size),
        )
        return records


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
    "RLMetricStore",
    "checkpoint_metrics",
    "parse_model_name",
    "parse_selfplay_output",
    "parse_sgfs",
    "render_dashboard",
    "rolling_health_metrics",
]
