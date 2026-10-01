"""Curriculum evaluation: fixed-opening KataGo matches against the opponent pools."""

from __future__ import annotations

import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .fileio import atomic_write_json, atomic_write_text, local_timestamp, sha256_file
from .katago import katago_subprocess_environment
from .rl_curriculum import CurriculumState, CurriculumStateError
from .rl_curriculum_openings import ensure_evaluation_opening_suite
from .rl_match import build_match_config, summarize_match_sgfs


class EvaluationSteps:
    """Evaluation half of :class:`~weiqi.rl_curriculum_runtime.CurriculumRuntime`.

    Mixed into the runtime; uses its ``config``, ``store``, ``controller``,
    ``state_root``, ``project_root`` and its ``_stage_runner``, ``_model_path``,
    ``_ensure_opponent_pools``, ``_health`` and ``_run_external`` methods.
    """

    def _evaluation_directory(self, state: CurriculumState) -> Path:
        active = state.active
        token = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = re.sub(r"[^A-Za-z0-9_.-]", "_", active.latest_model or "unknown")
        return (
            self.state_root
            / "evaluations"
            / f"{active.board_size}x{active.board_size}"
            / f"s{active.samples}-{candidate}-{token}-{uuid.uuid4().hex[:8]}"
        )

    def _match_opponent(self, state, runner, candidate_name, baseline_name, roles, output_dir):
        """Play the same color-swapped opening suite against one pool opponent."""
        candidate = self._model_path(runner, candidate_name)
        baseline = self._model_path(runner, baseline_name)
        output_dir.mkdir(parents=True, exist_ok=False)
        opening_count = self.config.evaluation_games // 2
        openings = ensure_evaluation_opening_suite(
            self.state_root,
            board_size=state.active.board_size,
            opening_count=opening_count,
        )
        suite_manifest = openings[0].parent / "manifest.json"
        atomic_write_json(
            output_dir / "evaluation_manifest.json",
            {
                "protocol_version": 2,
                "roles": roles,
                "candidate_model": candidate_name,
                "candidate_path": str(candidate),
                "candidate_sha256": sha256_file(candidate),
                "baseline_model": baseline_name,
                "baseline_path": str(baseline),
                "baseline_sha256": sha256_file(baseline),
                "board_size": state.active.board_size,
                "visits": runner.config.evaluation.simulations_per_move,
                "komi": 6.5,
                "games": self.config.evaluation_games,
                "opening_pairs": opening_count,
                "opening_suite": str(suite_manifest),
                "opening_suite_sha256": sha256_file(suite_manifest),
                "root_noise": False,
                "temperature": 0.0,
                "resignation": False,
                "created_at": local_timestamp(),
            },
        )
        environment = katago_subprocess_environment()
        environment["PYTHONUTF8"] = "1"
        for index, opening in enumerate(openings):
            pair_root = output_dir / "pairs" / f"opening-{index:03d}"
            sgf_dir = pair_root / "sgfs"
            sgf_dir.mkdir(parents=True, exist_ok=False)
            config_path = pair_root / "match.cfg"
            config_text = build_match_config(
                candidate_model=candidate,
                baseline_model=baseline,
                board_size=state.active.board_size,
                visits=runner.config.evaluation.simulations_per_move,
                games=2,
                game_threads=2,
                inference_batch_size=runner.config.self_play.inference_batch_size,
                gpu_index=0,
                opening_directory=opening,
            )
            atomic_write_text(config_path, config_text)
            execution = self._run_external(
                [
                    runner.katago_executable,
                    "match",
                    "-config",
                    config_path,
                    "-sgf-output-dir",
                    sgf_dir,
                ],
                cwd=self.project_root,
                env=environment,
                log_path=pair_root / "match.log",
            )
            if execution.returncode:
                raise RuntimeError(
                    f"固定开局 {index:03d} 的 KataGo match 退出代码 {execution.returncode}"
                )
            pair_summary = summarize_match_sgfs(
                sorted(sgf_dir.glob("*.sgfs")),
                candidate_name="candidate",
                requested_games=2,
            )
            if (
                pair_summary.candidate_black_games != 1
                or pair_summary.candidate_white_games != 1
            ):
                raise RuntimeError(
                    f"固定开局 {index:03d} 未完成严格黑白互换"
                )
        summary = summarize_match_sgfs(
            sorted(output_dir.rglob("*.sgfs")),
            candidate_name="candidate",
            requested_games=self.config.evaluation_games,
        )
        atomic_write_json(output_dir / "match-result.json", summary.to_dict())
        return summary

    def _evaluate(self, state: CurriculumState) -> None:
        runner = self._stage_runner(state)
        committed = False
        try:
            self._ensure_opponent_pools(state)
            candidate_name = state.active.latest_model
            opponents = self.controller.evaluation_opponents(state)
            if not opponents:
                raise CurriculumStateError("尚无不同于候选模型的评测对手，请先继续训练")
            output_dir = self._evaluation_directory(state)
            results = {}
            for index, (name, roles) in enumerate(opponents.items()):
                print(f"评测 {candidate_name} vs {name}（{' / '.join(roles)}）", flush=True)
                results[name] = self._match_opponent(
                    state, runner, candidate_name, name, roles,
                    output_dir / f"opponent-{index:02d}-{name}",
                )
            baseline_name = state.active.baseline_model
            if baseline_name not in results:
                raise CurriculumStateError("课程固定基准不能与候选模型相同")
            summary = results[baseline_name]
            health = self._health(runner)
            gate = self.controller.record_evaluation(
                state, candidate_model=str(candidate_name), match=summary,
                health=health, opponent_results=results,
            )
            committed = True
            atomic_write_json(output_dir / "summary.json", state.active.evaluations[-1])
            if state.active.evaluations[-1].get("champion_promoted"):
                print(f"滚动冠军已更新：{state.active.champion_model}", flush=True)
            if gate.passed:
                print(f"课程门槛通过：固定基准胜率 {summary.win_rate:.1%}", flush=True)
            else:
                state.warnings.append({
                    "timestamp": local_timestamp(), "kind": "quality_gate",
                    "message": "；".join(gate.reasons) or "课程门槛未通过",
                })
                self.store.save(state)
                print("课程门槛未达标，继续当前棋盘训练。", flush=True)
        except Exception as error:
            if committed:
                state.warnings.append({
                    "timestamp": local_timestamp(), "kind": "evaluation_artifact",
                    "message": f"评测状态已提交，但辅助文件或输出失败：{error}",
                })
                self.store.save(state)
                print(f"评测状态已提交；辅助文件记录失败：{error}", file=sys.stderr, flush=True)
            else:
                self.controller.record_evaluation_error(state, str(error))
                print(f"评测失败但检查点保持完好：{error}", file=sys.stderr, flush=True)
