"""Reassess the last two completed 9x9 evaluations after removing the score-margin gate.

Run only with the curriculum process stopped. The original state is backed up,
and each rewritten evaluation retains its previous decision for auditability.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from weiqi.rl_curriculum import (  # noqa: E402
    GlobalCurriculumLock,
    MatchSummary,
    compute_health_window,
    evaluate_quality_gate,
)
from weiqi.rl_curriculum_runtime import CurriculumRuntime  # noqa: E402


def reassess(runtime: CurriculumRuntime, *, apply: bool) -> dict[str, object]:
    with GlobalCurriculumLock(runtime.state_root):
        state = runtime.store.load()
        if state is None or state.active_stage_index != 0:
            raise RuntimeError("当前状态不是待晋级的 9×9 阶段")
        if state.phase not in {"training", "evaluating"}:
            raise RuntimeError(f"当前状态不适合重新核算：{state.phase}")
        evaluations = state.active.evaluations[-runtime.config.required_consecutive_passes :]
        if len(evaluations) != runtime.config.required_consecutive_passes:
            raise RuntimeError("完整历史评测不足两轮")

        stage = runtime.config.stages[0]
        decisions = []
        for record in evaluations:
            saved_health = record.get("health")
            saved_match = record.get("match")
            if not isinstance(saved_health, dict) or not isinstance(saved_match, dict):
                raise RuntimeError("历史评测缺少健康或对局汇总")
            if saved_health.get("complete") is not True or saved_health.get("cycles", 0) < runtime.config.health_window_cycles:
                raise RuntimeError("历史健康窗口不完整")
            health = compute_health_window(
                [{
                    "games": saved_health["games"],
                    "black_wins": saved_health["black_wins"],
                    "white_wins": saved_health["white_wins"],
                    "draws": saved_health["draws"],
                    "immediate_double_pass_games": saved_health["immediate_double_pass_games"],
                    "extreme_games": saved_health["extreme_games"],
                    "invalid_games": saved_health["invalid_games"],
                }],
                window_cycles=1,
                games_per_cycle=runtime.config.health_window_cycles * 128,
                thresholds=runtime.config.health,
            )
            match = MatchSummary(**saved_match)
            gate = evaluate_quality_gate(
                match, health, runtime.config,
                stage_samples=int(record["stage_samples"]),
                minimum_samples=stage.min_stage_samples,
            )
            if not gate.passed:
                raise RuntimeError(f"历史评测仍未达标：{record.get('candidate_model')} {gate.reasons}")
            if saved_match.get("damaged_games") or saved_match.get("no_result_games"):
                raise RuntimeError("历史对局存在损坏或无结果棋谱")
            decisions.append({
                "candidate_model": record["candidate_model"],
                "stage_samples": record["stage_samples"],
                "original_passed": record["passed"],
                "original_reasons": record["reasons"],
                "new_passed": gate.passed,
            })

        last = evaluations[-1]
        seed = str(state.active.latest_model)
        seed_samples = int(state.active.samples)
        runner = runtime._stage_runner(state)
        checkpoint = runner.run_root / "torchmodels_toexport" / seed / "model.ckpt"
        if not checkpoint.is_file():
            raise RuntimeError(f"找不到已通过评测的迁移检查点：{checkpoint}")
        summary: dict[str, object] = {
            "decisions": decisions,
            "migration_seed": seed,
            "migration_samples": seed_samples,
            "last_completed_evaluation_model": last["candidate_model"],
            "seed_evaluation_complete": seed == last["candidate_model"],
            "apply": apply,
        }
        if not apply:
            return summary

        original = copy.deepcopy(state.to_dict())
        backup = runtime.state_root / "backups" / f"before-extreme-gate-{datetime.now():%Y%m%d-%H%M%S}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for record in evaluations:
            record["policy_reassessment"] = {
                "previous_passed": record["passed"],
                "previous_reasons": copy.deepcopy(record["reasons"]),
                "policy": "extreme_result_observation_only",
            }
            record["passed"] = True
            record["reasons"] = []
            record["health"]["passed"] = True
            record["health"]["reasons"] = []
        state.active.consecutive_passes = runtime.config.required_consecutive_passes
        state.phase = "transition_ready"
        state.protected_models = list(dict.fromkeys(state.protected_models + [seed]))
        state.warnings.append({
            "timestamp": datetime.now().astimezone().isoformat(),
            "kind": "quality_gate_reassessment",
            "message": f"移除大分差晋级门槛后，已完成的最近两轮评测达标；使用当前 {seed} 检查点迁移，其本轮评测未完成",
        })
        runtime.store.save(state)
        runtime._render_dashboard(state)
        summary["backup"] = str(backup)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="写入重新核算的状态；默认只读预检")
    args = parser.parse_args()
    runtime = CurriculumRuntime(PROJECT_ROOT / "config/rl_curriculum.rtx5070ti.json")
    print(json.dumps(reassess(runtime, apply=args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
