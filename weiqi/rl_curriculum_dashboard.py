"""Self-contained HTML overview of the 9/13/19 curriculum."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Mapping, Optional

from .fileio import atomic_write_text, local_timestamp
from .rl_curriculum import CurriculumConfig, CurriculumState


def _finite_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percent(value: object) -> str:
    number = _finite_float(value)
    return f"{number * 100:.1f}%" if number is not None else "—"


def _count_pair(left: object, right: object) -> str:
    left_number = _finite_float(left)
    right_number = _finite_float(right)
    if left_number is None or right_number is None:
        return "—"
    return f"{int(left_number):,} / {int(right_number):,}"


def render_curriculum_dashboard(
    state: CurriculumState,
    config: CurriculumConfig,
    *,
    generated_at: Optional[str] = None,
    latest_health: Optional[Mapping[str, object]] = None,
    last_training_at: Optional[str] = None,
) -> str:
    """Render the global, self-contained 9/13/19 curriculum dashboard."""

    from .rl_evaluation_dashboard import render_opponent_pools

    active = state.active
    stage_definition = config.stages[state.active_stage_index]
    minimum = stage_definition.min_stage_samples
    remaining = max(0, minimum - active.samples) if minimum is not None else None
    evaluations = active.evaluations
    latest_evaluation = evaluations[-1] if evaluations else {}
    health = latest_health if latest_health is not None else latest_evaluation.get("health", {})
    if not isinstance(health, Mapping):
        health = {}
    disk = state.disk
    disk_free = _finite_float(disk.get("free_gib"))
    disk_status_key = str(disk.get("status", "waiting"))
    disk_status = {
        "healthy": "磁盘空间正常",
        "cleanup": "磁盘偏低，正在清理",
        "paused": "磁盘不足，已暂停",
        "waiting": "等待磁盘检查",
    }.get(disk_status_key, disk_status_key)
    phase_labels = {
        "training": "训练阶段（保存状态）",
        "evaluating": "固定评测",
        "transition_ready": "等待迁移",
        "migrating": "安全迁移",
        "paused_disk": "磁盘不足，已安全暂停",
    }
    generated = generated_at or local_timestamp()
    cards = (
        ("当前棋盘", f"{active.board_size}×{active.board_size}"),
        ("课程状态", phase_labels.get(state.phase, state.phase)),
        ("阶段样本", f"{active.samples:,}"),
        ("距离最低门槛", "无限持续" if remaining is None else f"{remaining:,}"),
        ("下次两类评测", f"{active.next_evaluation_sample:,}"),
        ("连续达标", f"{active.consecutive_passes}/{config.required_consecutive_passes}"),
        ("黑方得分率", _percent(health.get("black_win_rate"))),
        (
            "黑胜 / 白胜",
            _count_pair(health.get("black_wins"), health.get("white_wins")),
        ),
        ("开局双停率", _percent(health.get("immediate_double_pass_rate"))),
        ("极端棋局率（观察）", _percent(health.get("extreme_result_rate"))),
        ("无效棋谱率", _percent(health.get("invalid_rate"))),
        ("磁盘剩余", f"{disk_free:.1f} GiB" if disk_free is not None else "—"),
    )
    cards_html = "".join(
        f"<article class=\"stat\"><span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong></article>"
        for label, value in cards
    )
    stage_rows = []
    for index, stage in enumerate(config.stages):
        progress = state.stages.get(stage.key)
        if index < state.active_stage_index:
            status = "已完成"
        elif index == state.active_stage_index:
            status = phase_labels.get(state.phase, state.phase)
        else:
            status = "等待"
        stage_rows.append(
            "<tr>"
            f"<td>{stage.board_size}×{stage.board_size}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{progress.samples:,}</td>" if progress else
            "<tr>"
            f"<td>{stage.board_size}×{stage.board_size}</td>"
            f"<td>{html.escape(status)}</td>"
            "<td>0</td>"
        )
        stage_rows[-1] += (
            f"<td>{'无限' if stage.min_stage_samples is None else f'{stage.min_stage_samples:,}'}</td>"
            f"<td>{html.escape(progress.latest_model if progress and progress.latest_model else '—')}</td>"
            "</tr>"
        )
    warning = state.warnings[-1].get("message") if state.warnings else "无"
    disk_class = " alarm" if state.phase == "paused_disk" else ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15"><title>KataGo 自动课程训练</title>
<style>
:root{{--bg:#f3f6fb;--panel:#fff;--text:#172033;--muted:#6a7589;--border:#dfe5ef;--blue:#4c7df0;--green:#17a589;--red:#df5367;--grid:#e5eaf3}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f141e;--panel:#171e2b;--text:#edf2fa;--muted:#9ba8bb;--border:#2a3548;--grid:#293447}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif}}
main{{width:min(1440px,calc(100% - 28px));margin:auto;padding:26px 0 42px}}header{{display:flex;justify-content:space-between;gap:18px;align-items:flex-end;margin-bottom:18px}}
h1{{margin:0 0 5px;font-size:clamp(25px,3vw,38px)}}p{{margin:0;color:var(--muted)}}.badge{{padding:8px 12px;border-radius:999px;background:var(--panel);border:1px solid var(--border)}}.badge.alarm{{color:var(--red);font-weight:700}}
.stats{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:11px}}.stat,.panel{{background:var(--panel);border:1px solid var(--border);border-radius:15px}}
.stat{{padding:15px}}.stat span{{display:block;color:var(--muted);font-size:12px;margin-bottom:7px}}.stat strong{{font-size:clamp(17px,2vw,25px)}}.grid2{{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;margin-top:14px}}
a{{color:var(--blue)}}.panel{{padding:18px;overflow:auto}}h2{{font-size:17px;margin:0 0 12px}}svg{{display:block;width:100%;height:auto}}.grid{{stroke:var(--grid);stroke-width:1}}.threshold{{stroke:var(--red);stroke-dasharray:5 4}}.axis-label{{fill:var(--muted);font-size:11px}}.series{{fill:none;stroke:var(--blue);stroke-width:2.8}}circle{{fill:var(--blue)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{color:var(--muted)}}.empty{{height:220px;display:grid;place-items:center;color:var(--muted)}}.empty-row{{text-align:center!important;color:var(--muted)}}.foot{{margin-top:14px;font-size:13px;line-height:1.6}}
@media(max-width:1000px){{.stats{{grid-template-columns:repeat(3,1fr)}}.grid2{{grid-template-columns:1fr}}}}@media(max-width:600px){{header{{align-items:flex-start;flex-direction:column}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main>
<header><div><h1>KataGo 9×9 → 13×13 → 19×19</h1><p>固定评测门槛 · 滚动冠军 + 固定历史基准 · SWA 权重迁移</p></div><div class="badge{disk_class}">{html.escape(disk_status)}</div></header>
<section class="stats">{cards_html}</section>
<p class="foot">最近训练记录：{html.escape(last_training_at or '尚未读取')}。页面刷新不代表训练正在运行。</p>
<nav class="foot"><a href="../9x9/dashboard.html">9×9 损失与吞吐</a> · <a href="../13x13/dashboard.html">13×13 训练曲线（阶段启动后）</a> · <a href="../19x19/dashboard.html">19×19 训练曲线（阶段启动后）</a></nav>
{render_opponent_pools(active)}
<article class="panel" style="margin-top:16px"><h2>课程阶段</h2><table><thead><tr><th>棋盘</th><th>状态</th><th>样本</th><th>最低门槛</th><th>最新模型</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table></article>
<p class="foot">最近警告：{html.escape(str(warning))}<br>生成于 {html.escape(generated)}；每 15 秒自动刷新。评测使用固定 6.5 贴目、面积计分、位置全局同形、关闭根噪声与温度且不认输。</p>
</main></body></html>"""


def write_curriculum_dashboard(
    state: CurriculumState,
    config: CurriculumConfig,
    path: Path,
) -> Path:
    """Atomically refresh the global dashboard."""

    atomic_write_text(path, render_curriculum_dashboard(state, config))
    return path


__all__ = ["render_curriculum_dashboard", "write_curriculum_dashboard"]
