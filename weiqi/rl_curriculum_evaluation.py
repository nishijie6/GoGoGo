"""Curriculum evaluation: fixed-opening KataGo matches against the opponent pools."""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from .fileio import atomic_write_json, local_timestamp, sha256_file
from .rl_curriculum import CurriculumState, CurriculumStateError
from .rl_curriculum_openings import ensure_evaluation_opening_suite
from .rl_match import build_match_config
from .rl_match_jobs import run_opening_pairs


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

    def _match_manifest(self, state, runner, candidate_name, baseline_name, roles):
        candidate = self._model_path(runner, candidate_name)
        baseline = self._model_path(runner, baseline_name)
        opening_count = self.config.evaluation_games // 2
        openings = ensure_evaluation_opening_suite(
            self.state_root,
            board_size=state.active.board_size,
            opening_count=opening_count,
        )
        suite_manifest = openings[0].parent / "manifest.json"
        return openings, {
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
        }

    def _parallel_pairs(self) -> int:
        requested = self.config.evaluation_parallel_pairs
        profile = self.hardware_profile
        if profile is None:
            return requested
        if profile.tier != "high":
            return 1
        # Independent engines each load two nets and inference workspaces.
        budget = min(self.config.max_gpu_memory_gib, max(0, profile.memory_gib - 2))
        per_engine_gib = 1.0 if self.config.evaluation_inference_batch_size <= 2 else 2.5
        return min(requested, max(1, int(budget // per_engine_gib)))

    @staticmethod
    def _manifest_matches(path, expected):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(saved, dict) and all(saved.get(key) == value for key, value in expected.items())

    def _pending_evaluation_directory(self, state, runner, candidate_name, opponents):
        parent = self.state_root / "evaluations" / f"{state.active.board_size}x{state.active.board_size}"
        prefix = f"s{state.active.samples}-{candidate_name}-"
        expected = {
            f"opponent-{index:02d}-{name}": self._match_manifest(
                state, runner, candidate_name, name, roles
            )[1]
            for index, (name, roles) in enumerate(opponents.items())
        }
        for path in sorted(parent.glob(prefix + "*"), reverse=True):
            if not path.is_dir() or (path / "summary.json").exists():
                continue
            directories = list(path.glob("opponent-*"))
            if directories and all(
                child.name in expected and self._manifest_matches(
                    child / "evaluation_manifest.json", expected[child.name]
                )
                for child in directories
            ):
                print(f"继续未完成评测：{path.name}", flush=True)
                return path
        return None

    def _match_opponent(self, state, runner, candidate_name, baseline_name, roles, output_dir):
        """Run bounded parallel pairs while preserving the fixed evaluation protocol."""
        openings, manifest = self._match_manifest(state, runner, candidate_name, baseline_name, roles)
        manifest_path = output_dir / "evaluation_manifest.json"
        if manifest_path.exists() and not self._manifest_matches(manifest_path, manifest):
            raise CurriculumStateError("评测模型、规则或开局清单已改变，不能复用原评测")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            atomic_write_json(manifest_path, {**manifest, "created_at": local_timestamp()})
        environment = runner._environment()
        atomic_write_json(output_dir / "execution.json", {
            "parallel_pairs": self._parallel_pairs(),
            "inference_batch_size": self.config.evaluation_inference_batch_size,
            "updated_at": local_timestamp(),
        })

        def config_builder(opening):
            return build_match_config(
                candidate_model=Path(manifest["candidate_path"]),
                baseline_model=Path(manifest["baseline_path"]),
                board_size=state.active.board_size,
                visits=runner.config.evaluation.simulations_per_move,
                games=2,
                game_threads=2,
                inference_batch_size=runner.config.self_play.inference_batch_size,
                gpu_index=(self.hardware_profile.gpu_index if self.hardware_profile else 0),
                opening_directory=opening,
                hardware_profile=self.hardware_profile,
            )

        def execute(config_path, sgf_dir, log_path):
            # Resource allocation only: keep the protocol config byte-identical
            # so previously completed opening/color pairs remain reusable.
            execution = self._run_external(
                [
                    runner.katago_executable,
                    "match",
                    "-config",
                    config_path,
                    "-override-config",
                    f"nnMaxBatchSize={self.config.evaluation_inference_batch_size}",
                    "-sgf-output-dir",
                    sgf_dir,
                ],
                cwd=self.project_root,
                env=environment,
                log_path=log_path,
            )
            return execution.returncode

        return run_opening_pairs(
            openings, output_dir, parallel_pairs=self._parallel_pairs(),
            config_builder=config_builder, execute=execute,
        )

    def _needs_champion_evaluation(self, state):
        active = state.active
        if not active.latest_model or not active.champion_model or active.latest_model == active.champion_model:
            return False
        return not any(record.get("candidate_model") == active.latest_model
                       and any("champion" in item.get("roles", [])
                               and item.get("match", {}).get("games") == self.config.evaluation_games
                               for item in (record.get("opponents") or []))
                       for record in active.evaluations)

    def _defer_disabled_champion_evaluation(self, state) -> bool:
        """Apply a changed per-cycle schedule without cancelling a due gate.

        Cached opening pairs remain on disk. Deferral is an audit event, never
        a match result or a champion promotion.
        """
        active = state.active
        if (self.config.champion_evaluation_every_cycle or state.phase != "evaluating"
                or not active.pending_champion_evaluation
                or active.samples >= active.next_evaluation_sample):
            return False
        state.warnings.append({
            "timestamp": local_timestamp(), "kind": "evaluation_deferred",
            "candidate_model": active.latest_model,
            "message": "已切换到按样本间隔评测；本轮冠军评测延期，已完成开局配对保留，冠军不变",
            "next_evaluation_sample": active.next_evaluation_sample,
        })
        active.pending_champion_evaluation = False
        state.phase = "training"
        self.store.save(state)
        return True

    def _evaluate_champion(self, state: CurriculumState) -> None:
        runner = self._stage_runner(state)
        self._ensure_opponent_pools(state)
        candidate_name, champion = state.active.latest_model, state.active.champion_model
        if not champion or candidate_name == champion:
            raise CurriculumStateError("冠军评测需要不同于候选的冠军模型")
        state.phase = "evaluating"
        state.active.pending_champion_evaluation = True
        self.store.save(state)
        opponents = {champion: ["champion"]}
        output_dir = self._pending_evaluation_directory(state, runner, candidate_name, opponents) or self._evaluation_directory(state)
        print(f"每轮冠军评测：{candidate_name} vs {champion} · {self.config.evaluation_games} 局", flush=True)
        # A failed or interrupted match stays pending. Restart reuses completed
        # opening pairs before allowing the next model to train.
        match = self._match_opponent(state, runner, candidate_name, champion, ["champion"],
                                     output_dir / f"opponent-00-{champion}")
        records = self._read_metric_records(runner)
        self.controller.record_champion_evaluation(state, match=match,
                                                   training_cycle=records[-1].get("cycle") if records else None)
        try:
            atomic_write_json(output_dir / "summary.json", state.active.evaluations[-1])
        except OSError as error:
            print(f"冠军评测已提交，辅助文件写入失败：{error}", file=sys.stderr, flush=True)
        print(f"冠军评测完成：胜率 {match.win_rate:.1%} · 冠军 {state.active.champion_model}", flush=True)

    def _evaluate(self, state: CurriculumState) -> None:
        runner = self._stage_runner(state)
        committed = False
        try:
            self._ensure_opponent_pools(state)
            candidate_name = state.active.latest_model
            opponents = self.controller.evaluation_opponents(state)
            if not opponents:
                raise CurriculumStateError("尚无不同于候选模型的评测对手，请先继续训练")
            output_dir = self._pending_evaluation_directory(
                state, runner, candidate_name, opponents
            ) or self._evaluation_directory(state)
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
                training_cycle=next((row.get("cycle") for row in reversed(self._read_metric_records(runner))
                                     if row.get("cycle") is not None), None),
            )
            committed = True
            atomic_write_json(output_dir / "summary.json", state.active.evaluations[-1])
            if state.active.evaluations[-1].get("champion_promoted"):
                print(f"滚动冠军已更新：{state.active.champion_model}", flush=True)
            if gate.passed:
                print("课程门槛通过：全部固定历史基准、样本数与健康检查均达标。", flush=True)
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
