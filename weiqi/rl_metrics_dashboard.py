"""Offline HTML dashboard for one KataGo training run's metric history."""

from __future__ import annotations

import html
import json
from typing import Iterable, Optional

from .rl_metrics import (
    EXTREME_MARGIN_BOARD_FRACTION,
    HEALTH_WINDOW_CYCLES,
    HEALTH_WINDOW_GAMES,
    _safe_float,
)


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


__all__ = ["render_dashboard"]
