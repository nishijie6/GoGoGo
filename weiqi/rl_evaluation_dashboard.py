"""Render opponent-specific evaluation results without combining baselines."""

from __future__ import annotations

import html
from collections import defaultdict
from typing import Mapping


def evaluation_rows(stage):
    rows = []
    for record in stage.evaluations:
        opponents = record.get("opponents")
        if opponents is None and isinstance(record.get("match"), Mapping):
            # Legacy matches remain attached to their original fixed baseline.
            opponents = [{"model": record.get("baseline_model", "未知旧基准"),
                          "roles": ["fixed"], "match": record["match"]}]
        for item in opponents or []:
            rows.append({**item, "timestamp": record.get("timestamp", ""),
                         "samples": record.get("stage_samples", 0),
                         "candidate": record.get("candidate_model", ""),
                         "promoted": record.get("champion_promoted", False)})
    return rows


def _rate(value):
    return f"{float(value):.1%}" if value is not None else "—"


def _chart(rows, label, *, connect=True):
    if not rows:
        return '<p class="pool-empty">待评测：尚无此类对战成绩。</p>'
    left, top, width, height = 54, 20, 620, 180
    samples = [int(row["samples"]) for row in rows]
    lower, upper = min(samples), max(samples)
    x = lambda sample: left + (sample - lower) / max(1, upper - lower) * width
    y = lambda rate: top + (1 - float(rate)) * height
    svg = [f'<svg viewBox="0 0 720 260" role="img" aria-label="{html.escape(label)}">']
    for rate in (0, 0.5, 0.6, 1):
        svg.append(f'<line x1="{left}" x2="{left+width}" y1="{y(rate)}" y2="{y(rate)}" stroke="#718096" opacity=".25"/>')
        svg.append(f'<text x="46" y="{y(rate)+4}" text-anchor="end" fill="currentColor">{rate:.0%}</text>')
    if connect:
        points = ' '.join(f'{x(int(row["samples"]))},{y(row["match"]["win_rate"])}' for row in rows)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#38bdf8" stroke-width="2"/>')
    for row in rows:
        match = row["match"]
        px = x(int(row["samples"]))
        tooltip = html.escape(f'{row["candidate"]} vs {row["model"]}：{_rate(match.get("win_rate"))}，{match.get("games", 0)} 局')
        svg.append(f'<line x1="{px}" x2="{px}" y1="{y(match.get("wilson_lower", match["win_rate"]))}" y2="{y(match.get("wilson_upper", match["win_rate"]))}" stroke="#38bdf8" opacity=".55"/>')
        svg.append(f'<circle cx="{px}" cy="{y(match["win_rate"])}" r="4" fill="#38bdf8"><title>{tooltip}</title></circle>')
    svg.extend([
        f'<text x="54" y="223" fill="currentColor">{lower:,}</text>',
        f'<text x="674" y="223" text-anchor="end" fill="currentColor">{upper:,}</text>',
        '<text x="360" y="247" text-anchor="middle" fill="currentColor">阶段训练样本数</text></svg>',
    ])
    return ''.join(svg)


def render_opponent_pools(stage):
    rows = evaluation_rows(stage)
    champion_rows = [row for row in rows if "champion" in row["roles"]]
    fixed_rows = defaultdict(list)
    for row in rows:
        if "fixed" in row["roles"]:
            fixed_rows[row["model"]].append(row)
    escape = lambda value: html.escape(str(value))
    fixed_names = stage.fixed_baseline_models or [stage.baseline_model]
    groups = []
    for name in fixed_names:
        group = fixed_rows.get(name, [])
        groups.append(f'<div class="pool-model"><h3>{escape(name)}</h3>{_chart(group, f"对固定基准 {name} 的胜率")}</div>')
    table_rows = []
    for row in reversed(rows[-32:]):
        match = row["match"]
        roles = ' / '.join('滚动冠军' if role == 'champion' else '固定历史' for role in row['roles'])
        # An all-win/lose sample has an unbounded point estimate, not a finite Elo.
        rate = float(match["win_rate"])
        elo = '饱和，无法估计' if rate in (0, 1) else f'{float(match.get("elo", 0)):+.0f}'
        result = '冠军晋升' if row['promoted'] and 'champion' in row['roles'] else '已完成'
        table_rows.append(f'<tr><td>{escape(row["timestamp"])}</td><td>{roles}</td>'
                          f'<td>{escape(row["candidate"])}</td><td>{escape(row["model"])}</td>'
                          f'<td>{_rate(rate)} / {match.get("games", 0)} 局</td>'
                          f'<td>{_rate(match.get("wilson_lower"))}–{_rate(match.get("wilson_upper"))}</td>'
                          f'<td>{elo}</td><td>{result}</td></tr>')
    status = '经挑战确认' if stage.champion_established else '暂定冠军，等待挑战确认'
    return f'''<style>
    .pool-columns{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}}
    .pool-columns>article{{min-width:0}}.pool-columns p{{line-height:1.7;margin:8px 0}}
    .pool-model{{margin-top:14px}}.pool-model h3{{font-size:13px;overflow-wrap:anywhere}}
    .pool-empty{{padding:12px 0;color:#94a3b8}}.pool-table{{overflow:auto}}
    .pool-columns svg text{{font-size:12px}}.pool-name{{overflow-wrap:anywhere}}
    @media(max-width:800px){{.pool-columns{{grid-template-columns:1fr}}}}
    </style><section class="pool-columns">
    <article class="panel"><h2>滚动冠军 · 近期进步</h2>
    <p class="pool-name">{escape(stage.champion_model or '待初始化')}<br>{status}</p>
    <p>挑战结果达到胜率门槛且 Wilson 下界超过门槛才更换冠军；点的对手可能不同，不能连接成绝对棋力曲线。</p>
    {_chart(champion_rows, '冠军挑战胜率与 Wilson 区间', connect=False)}</article>
    <article class="panel"><h2>固定历史基准 · 长期进步</h2>
    <p>各基准独立统计。课程升级使用首个固定基准：<span class="pool-name">{escape(fixed_names[0])}</span>。</p>
    {''.join(groups)}</article></section>
    <article class="panel" style="margin-top:16px"><h2>最近对战成绩</h2>
    <p>每项标明真实对手；Elo 为相对此对手的差值，不能跨对手直接比较。旧评测保留原基准身份。</p>
    <div class="pool-table"><table><thead><tr><th>时间</th><th>类别</th><th>候选</th><th>对手</th><th>胜率 / 局数</th><th>Wilson 95% 区间</th><th>相对 Elo</th><th>结果</th></tr></thead>
    <tbody>{''.join(table_rows) or '<tr><td colspan="8">尚无评测</td></tr>'}</tbody></table></div></article>'''
