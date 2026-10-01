"""Long-running, crash-resumable execution layer for KataGo curriculum RL.

``rl_curriculum`` contains the deliberately side-effect-light state machine and
validation primitives.  This module wires those primitives to the real KataGo
runner, subprocesses and the offline curriculum dashboard, and runs the main
loop.  :class:`CurriculumRuntime` combines three step mixins kept in their own
modules: evaluation matches (``rl_curriculum_evaluation``), batch autotuning and
stage migration (``rl_curriculum_transition``), and disk cleanup
(``rl_curriculum_cleanup``).

The runtime never mixes stage data: every runner is created from its stage's
own training profile and therefore its own ``output_directory``.  A transition
is prepared under a sibling temporary directory and is made visible only after
model-load, low-visit self-play, autotune, and one training-batch smoke checks
have all succeeded.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from .fileio import atomic_write_text, local_timestamp
from .katago_rl import PROJECT_ROOT, KataGoRLRunner
from .process_control import stream_command
from .rl_curriculum import (
    CurriculumController,
    CurriculumStage,
    CurriculumState,
    CurriculumStateError,
    CurriculumStateStore,
    GlobalCurriculumLock,
    compute_health_window,
    load_curriculum_config,
    validate_training_profiles,
)
from .rl_curriculum_cleanup import CleanupSteps
from .rl_curriculum_dashboard import render_curriculum_dashboard
from .rl_curriculum_evaluation import EvaluationSteps
from .rl_curriculum_transition import TransitionSteps
from .rl_metrics import parse_model_name
from .rl_retention import disk_space_status


DISK_POLL_SECONDS = 60
MIGRATION_RETRY_SECONDS = 60


@dataclass(frozen=True)
class CommandExecution:
    """Normalized result from an injected or real subprocess executor."""

    returncode: int
    stdout: str


def _model_sort_key(name: str) -> tuple[int, int, str]:
    parsed = parse_model_name(name)
    if parsed is None:
        return (-1, -1, name)
    return parsed[0], parsed[1], name


class CurriculumRuntime(EvaluationSteps, TransitionSteps, CleanupSteps):
    """Execute the persistent 9x9 -> 13x13 -> 19x19 curriculum.

    ``runner_factory`` and ``command_runner`` are injectable so recovery and
    failure paths can be verified without starting KataGo or allocating a GPU.
    A custom command runner receives ``(command, cwd=..., env=..., log_path=...)``
    and may return ``CommandExecution``, ``subprocess.CompletedProcess``, a
    ``(returncode, stdout)`` tuple, or just a stdout string.
    """

    def __init__(
        self,
        config_path: Path,
        project_root: Path = PROJECT_ROOT,
        *,
        runner_factory: Callable[..., KataGoRLRunner] = KataGoRLRunner,
        command_runner: Optional[Callable[..., object]] = None,
        sleep: Callable[[float], None] = time.sleep,
        disk_probe: Callable[..., object] = disk_space_status,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = (
            config_path if config_path.is_absolute() else self.project_root / config_path
        ).resolve()
        self.config = load_curriculum_config(self.config_path)
        validate_training_profiles(self.config, self.project_root)
        self.state_root = self.config.state_root(self.project_root)
        self.store = CurriculumStateStore(self.state_root)
        self.controller = CurriculumController(self.config, self.store)
        self.runner_factory = runner_factory
        self.command_runner = command_runner
        self.sleep = sleep
        self.disk_probe = disk_probe
        self.dashboard_path = self.state_root / "dashboard.html"

    def _make_runner(
        self, stage: CurriculumStage, *, run_root_override: Optional[Path] = None
    ) -> KataGoRLRunner:
        return self.runner_factory(
            config_path=stage.config_path(self.project_root),
            project_root=self.project_root,
            run_root_override=run_root_override,
        )

    def _stage_runner(self, state: CurriculumState) -> KataGoRLRunner:
        return self._make_runner(self.config.stages[state.active_stage_index])

    @staticmethod
    def _accepted_models(runner: KataGoRLRunner) -> list[str]:
        if not runner.models_dir.is_dir():
            return []
        return sorted(
            (
                path.name
                for path in runner.models_dir.iterdir()
                if path.is_dir()
                and (path / "model.bin.gz").is_file()
                and parse_model_name(path.name) is not None
            ),
            key=_model_sort_key,
        )

    @staticmethod
    def _model_path(runner: KataGoRLRunner, name: Optional[str]) -> Path:
        if not name:
            raise CurriculumStateError("课程状态缺少模型名")
        path = (runner.models_dir / name / "model.bin.gz").resolve()
        try:
            path.relative_to(runner.models_dir.resolve())
        except ValueError as error:
            raise CurriculumStateError(f"非法模型路径：{name}") from error
        if not path.is_file():
            raise CurriculumStateError(f"找不到已接纳模型：{path}")
        return path

    def _adopt_existing_9x9(self) -> CurriculumState:
        existing = self.store.load()
        if existing is not None:
            return existing
        runner = self._make_runner(self.config.stages[0])
        models = self._accepted_models(runner)
        if not models:
            raise CurriculumStateError(
                f"无法接管 9x9：{runner.models_dir} 中没有已接纳模型"
            )
        entry, latest = models[0], models[-1]
        parsed = parse_model_name(latest)
        if parsed is None:
            raise CurriculumStateError(f"无法从模型名读取训练样本：{latest}")
        # The controller saves state atomically.  Retention is intentionally
        # invoked only after this call, so the oldest evaluation baseline and
        # current checkpoint are protected before the first large cleanup.
        return self.controller.adopt_existing_9x9(
            trained_samples=parsed[0], latest_model=latest, entry_model=entry
        )

    def _ensure_opponent_pools(self, state: CurriculumState) -> None:
        """Initialize each pool once, and protect its models before cleanup."""
        candidate_state = copy.deepcopy(state)
        active = candidate_state.active
        definition = self.config.stages[state.active_stage_index]
        runner = self._stage_runner(state)
        fixed = active.fixed_baseline_models or list(definition.fixed_baseline_models) or [active.baseline_model]
        champion = active.champion_model or definition.initial_champion_model or active.latest_model
        for name in [*fixed, champion]:
            self._model_path(runner, name)
        changed = not active.fixed_baseline_models or not active.champion_model
        if changed:
            if active.baseline_model != fixed[0]:
                active.consecutive_passes = 0
                if candidate_state.phase == "transition_ready":
                    candidate_state.phase = "training"
            active.fixed_baseline_models = list(fixed)
            active.baseline_model = fixed[0]
            active.champion_model = champion
        protected = list(dict.fromkeys(state.protected_models + list(fixed) + [champion]))
        if changed or protected != state.protected_models:
            candidate_state.protected_models = protected
            self.store.save(candidate_state)
            state.__dict__.update(candidate_state.__dict__)

    def refresh_dashboard(self) -> Path:
        """Prepare the two opponent pools and refresh the page without training."""
        with GlobalCurriculumLock(self.state_root):
            state = self._adopt_existing_9x9()
            with self._stage_runner(state).lock():
                self._ensure_opponent_pools(state)
                self._render_dashboard(state)
        print(str(self.dashboard_path), flush=True)
        return self.dashboard_path

    def evaluate_now(self) -> dict[str, object]:
        """Run one bounded evaluation suite, without starting self-play training."""
        with GlobalCurriculumLock(self.state_root):
            state = self._adopt_existing_9x9()
            with self._stage_runner(state).lock():
                if state.phase not in {"training", "evaluating"}:
                    raise CurriculumStateError("只可在训练或评测阶段执行独立评测")
                if any(
                    row.get("candidate_model") == state.active.latest_model and row.get("opponents")
                    for row in state.active.evaluations
                ):
                    self._render_dashboard(state)
                    print("当前模型已有两类评测成绩，保留原结果；新模型导出后再评测。", flush=True)
                    return self._status_dict(state)
                self._evaluate(state)
                self._render_dashboard(state)
                if state.active.evaluations and state.active.evaluations[-1].get("error"):
                    raise CurriculumStateError(str(state.active.evaluations[-1]["error"]))
            return self._status_dict(state)

    @staticmethod
    def _read_metric_records(runner: KataGoRLRunner) -> list[dict[str, object]]:
        path = runner.metric_store.jsonl_path
        if not path.is_file():
            return []
        records: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def _health(self, runner: KataGoRLRunner):
        normalized: list[dict[str, object]] = []
        for record in self._read_metric_records(runner):
            # A newly exported model can have training counters before it has
            # produced any SGF. Counting its selfplay_games fallback would add
            # 128 result-less "ghost games" and dilute every health rate.
            if "sgf_entries" not in record:
                continue
            normalized.append(
                {
                    "games": record.get("sgf_entries", 0),
                    "black_wins": record.get("black_wins", 0),
                    "white_wins": record.get("white_wins", 0),
                    "draws": record.get("draws", 0),
                    "immediate_double_pass_games": record.get(
                        "opening_double_pass_games",
                        record.get("immediate_double_pass_games", 0),
                    ),
                    "extreme_games": record.get(
                        "extreme_result_games", record.get("extreme_games", 0)
                    ),
                    "invalid_games": record.get("no_result_games", 0),
                    "damaged_games": record.get("damaged_games", 0),
                }
            )
        return compute_health_window(
            normalized,
            window_cycles=self.config.health_window_cycles,
            games_per_cycle=128,
            thresholds=self.config.health,
        )

    def _refresh_progress(
        self, state: CurriculumState, runner: KataGoRLRunner
    ) -> bool:
        models = self._accepted_models(runner)
        if not models:
            raise CurriculumStateError(f"{state.active.board_size}x{state.active.board_size} 没有模型")
        latest = models[-1]
        parsed = parse_model_name(latest)
        if parsed is None:
            raise CurriculumStateError(f"无法解析模型名：{latest}")
        return self.controller.mark_cycle(
            state, trained_samples=parsed[0], latest_model=latest
        )

    def _normalize_command_result(self, result: object) -> CommandExecution:
        if isinstance(result, CommandExecution):
            return result
        if isinstance(result, subprocess.CompletedProcess):
            return CommandExecution(int(result.returncode), str(result.stdout or ""))
        if isinstance(result, tuple) and len(result) == 2:
            return CommandExecution(int(result[0]), str(result[1]))
        if isinstance(result, str):
            return CommandExecution(0, result)
        raise TypeError("command_runner 返回值必须包含 returncode/stdout")

    def _run_external(
        self,
        command: Sequence[object],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> CommandExecution:
        args = [str(item) for item in command]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.command_runner is not None:
            result = self.command_runner(
                args, cwd=cwd, env=dict(env), log_path=log_path
            )
            execution = self._normalize_command_result(result)
            atomic_write_text(log_path, execution.stdout)
            return execution

        with log_path.open("a", encoding="utf-8", newline="") as log:
            log.write("$ " + subprocess.list2cmdline(args) + "\n")
            log.flush()
            returncode, output = stream_command(args, cwd=str(cwd), env=dict(env), log=log)
        return CommandExecution(returncode, output)

    def _cycle(self, state: CurriculumState, *, smoke: bool) -> None:
        runner = self._stage_runner(state)
        batch = state.active.autotune_batch
        try:
            with runner.lock():
                runner.cycle(
                    smoke=smoke,
                    training_batch_size=batch,
                )
                due = self._refresh_progress(state, runner)
                if due:
                    self._evaluate(state)
                if state.phase in {"transition_ready", "migrating"}:
                    self._migrate(state)
        except Exception as error:
            state.warnings.append(
                {"timestamp": local_timestamp(), "kind": "training", "message": str(error)}
            )
            self.store.save(state)
            raise

    def _render_dashboard(self, state: CurriculumState) -> None:
        runner = self._stage_runner(state)
        health = self._health(runner).to_dict()
        records = self._read_metric_records(runner)
        last_training = next((str(row["timestamp"]) for row in reversed(records)
                              if row.get("timestamp") and row.get("source") == "completed-cycle"), None)
        document = render_curriculum_dashboard(
            state, self.config, latest_health=health, last_training_at=last_training,
        )
        atomic_write_text(self.dashboard_path, document)

    def run(self, iterations: int = 0, smoke: bool = False) -> dict[str, object]:
        """Run continuously, or for ``iterations`` completed training cycles."""

        if iterations < 0:
            raise ValueError("iterations 不能小于 0")
        self.state_root.mkdir(parents=True, exist_ok=True)
        with GlobalCurriculumLock(self.state_root):
            state = self._adopt_existing_9x9()
            completed = 0
            while iterations == 0 or completed < iterations:
                state = self.store.load() or state
                self._ensure_opponent_pools(state)
                disk_state = self._update_disk(state)
                self._apply_retention(state)
                self._render_dashboard(state)
                if disk_state == "paused":
                    print(
                        f"磁盘仅剩 {state.disk.get('free_gib', 0):.1f} GiB，课程已安全暂停。",
                        file=sys.stderr,
                        flush=True,
                    )
                    if iterations > 0:
                        break
                    self.sleep(DISK_POLL_SECONDS)
                    continue
                if state.phase == "evaluating":
                    with self._stage_runner(state).lock():
                        self._evaluate(state)
                if state.phase in {"transition_ready", "migrating"}:
                    with self._stage_runner(state).lock():
                        self._migrate(state)
                if state.phase in {"transition_ready", "migrating"}:
                    self._render_dashboard(state)
                    if iterations > 0:
                        break
                    self.sleep(MIGRATION_RETRY_SECONDS)
                    continue
                if state.phase != "training":
                    self._render_dashboard(state)
                    continue
                print(
                    f"\n===== 课程循环 {completed + 1} · {state.active.board_size}x{state.active.board_size} =====",
                    flush=True,
                )
                self._cycle(state, smoke=smoke)
                completed += 1
                state = self.store.load() or state
                self._apply_retention(state)
                self._render_dashboard(state)
            return self._status_dict(self.store.load() or state)

    def _status_dict(self, state: CurriculumState) -> dict[str, object]:
        return {
            "config": str(self.config_path),
            "state": str(self.store.path),
            "dashboard": str(self.dashboard_path),
            "active_board_size": state.active.board_size,
            "phase": state.phase,
            "samples": state.active.samples,
            "latest_model": state.active.latest_model,
            "champion_model": state.active.champion_model,
            "fixed_baseline_models": state.active.fixed_baseline_models,
            "next_evaluation_sample": state.active.next_evaluation_sample,
            "consecutive_passes": state.active.consecutive_passes,
            "autotune_batch": state.active.autotune_batch,
            "disk": dict(state.disk),
            "warnings": list(state.warnings[-10:]),
            "stages": {
                name: asdict(progress) for name, progress in state.stages.items()
            },
            "updated_at": state.updated_at,
        }

    def status(self) -> dict[str, object]:
        """Print persisted curriculum status without acquiring the training lock."""

        state = self.store.load()
        if state is None:
            result: dict[str, object] = {
                "config": str(self.config_path),
                "state": str(self.store.path),
                "dashboard": str(self.dashboard_path),
                "initialized": False,
            }
        else:
            result = {"initialized": True, **self._status_dict(state)}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result


__all__ = [
    "CommandExecution",
    "CurriculumRuntime",
]
