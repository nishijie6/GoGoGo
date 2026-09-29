"""Long-running, crash-resumable execution layer for KataGo curriculum RL.

``rl_curriculum`` contains the deliberately side-effect-light state machine and
validation primitives.  This module wires those primitives to the real KataGo
runner, subprocesses, filesystem retention, GPU batch benchmarking, and the
offline curriculum dashboard.

The runtime never mixes stage data: every runner is created from its stage's
own training profile and therefore its own ``output_directory``.  A transition
is prepared under a sibling temporary directory and is made visible only after
model-load, low-visit self-play, autotune, and one training-batch smoke checks
have all succeeded.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .katago import katago_subprocess_environment
from .katago_rl import PROJECT_ROOT, KataGoRLRunner, KataGoRLRunnerError
from .process_control import managed_popen
from .replay_accounting import ReplayAccountingError, ReplayRowLedger
from .rl_curriculum import (
    BenchmarkMeasurement,
    CurriculumController,
    CurriculumMigrationError,
    CurriculumStage,
    CurriculumState,
    CurriculumStateError,
    CurriculumStateStore,
    GlobalCurriculumLock,
    apply_retention_plan,
    build_benchmark_command,
    build_match_config,
    build_retention_plan,
    choose_autotune_batch,
    compute_health_window,
    disk_space_status,
    load_curriculum_config,
    prepare_stage_migration,
    summarize_match_sgfs,
    validate_training_profiles,
    verify_checkpoint_loads,
    render_curriculum_dashboard,
)
from .rl_metrics import parse_model_name


THROUGHPUT_RE = re.compile(
    r"Throughput:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+samples/s", re.IGNORECASE
)
PEAK_MEMORY_RE = re.compile(
    r"Peak GPU memory(?:\s*\([^)]*\))?:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB",
    re.IGNORECASE,
)
STALE_TEMP_SECONDS = 6 * 60 * 60
DISK_POLL_SECONDS = 60
MIGRATION_RETRY_SECONDS = 60
EVALUATION_OPENING_SUITE_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CommandExecution:
    """Normalized result from an injected or real subprocess executor."""

    returncode: int
    stdout: str


def parse_benchmark_output(
    output: str, *, batch_size: int, returncode: int = 0
) -> BenchmarkMeasurement:
    """Parse the stable summary emitted by ``benchmark_fresh_model.py``."""

    folded = output.casefold()
    oom = "out of memory" in folded or re.search(r"\boom\b", folded) is not None
    throughput_match = THROUGHPUT_RE.search(output)
    peak_match = PEAK_MEMORY_RE.search(output)
    throughput = (
        float(throughput_match.group(1).replace(",", ""))
        if throughput_match
        else 0.0
    )
    peak = float(peak_match.group(1)) if peak_match else math.inf
    success = returncode == 0 and not oom and throughput > 0 and math.isfinite(peak)
    error: Optional[str] = None
    if not success:
        if oom:
            error = "CUDA out of memory"
        elif returncode:
            error = f"benchmark 退出代码 {returncode}"
        else:
            error = "benchmark 输出缺少吞吐或峰值显存"
    return BenchmarkMeasurement(
        batch_size=batch_size,
        success=success,
        throughput=throughput,
        peak_gpu_gib=peak,
        oom=oom,
        error=error,
    )


def prepare_benchmark_npz(source: Path, target: Path, required_rows: int) -> Path:
    """Tile a target-board NPZ so every autotune candidate gets a full batch."""

    if required_rows < 1:
        raise ValueError("required_rows 必须为正数")
    import numpy as np

    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as data:
        if not data.files:
            raise ValueError(f"训练 NPZ 为空：{source}")
        row_candidates = [
            int(data[key].shape[0])
            for key in data.files
            if data[key].ndim > 0 and data[key].shape[0] > 0
        ]
        if not row_candidates:
            raise ValueError(f"训练 NPZ 没有样本维度：{source}")
        source_rows = max(set(row_candidates), key=row_candidates.count)
        indices = np.arange(required_rows, dtype=np.int64) % source_rows
        arrays: dict[str, object] = {}
        for key in data.files:
            value = data[key]
            arrays[key] = value[indices] if value.ndim and value.shape[0] == source_rows else value
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    return target


def _model_sort_key(name: str) -> tuple[int, int, str]:
    parsed = parse_model_name(name)
    if parsed is None:
        return (-1, -1, name)
    return parsed[0], parsed[1], name


def _directory_size(path: Path) -> int:
    if path.is_file():
        with contextlib.suppress(OSError):
            return path.stat().st_size
        return 0
    total = 0
    with contextlib.suppress(OSError):
        for child in path.rglob("*"):
            if child.is_file():
                with contextlib.suppress(OSError):
                    total += child.stat().st_size
    return total


def _assert_exact_descendant(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root = root.resolve()
    if resolved == root:
        raise ValueError("拒绝删除训练根目录")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"拒绝删除训练目录外路径：{resolved}") from error


def _gtp_coordinate(x: int, y: int, board_size: int) -> str:
    columns = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{columns[x]}{board_size - y}"


def _fixed_opening_sample(board_size: int, index: int) -> str:
    """Generate one legal, deterministic early-game PositionSample."""

    candidates = [
        (x, y)
        for y in range(1, board_size - 1)
        for x in range(1, board_size - 1)
        if (x + y) % 2 == 0
    ]
    candidates.sort(
        key=lambda point: hashlib.sha256(
            (
                f"gogogo-eval-v{EVALUATION_OPENING_SUITE_VERSION}:"
                f"{board_size}:{index}:{point[0]}:{point[1]}"
            ).encode("ascii")
        ).digest()
    )
    move_count = 4 + 2 * (index % 4)
    selected = candidates[:move_count]
    if len(selected) != move_count:
        raise CurriculumStateError(f"无法为 {board_size}x{board_size} 生成固定评测开局")
    sample = {
        "board": ("." * board_size + "/") * board_size,
        "hintLoc": "null",
        "initialTurnNumber": 0,
        "metadata": (
            f"gogogo-evaluation-v{EVALUATION_OPENING_SUITE_VERSION}-"
            f"{board_size}x{board_size}-{index:03d}"
        ),
        "moveLocs": [_gtp_coordinate(x, y, board_size) for x, y in selected],
        "movePlas": ["B" if move % 2 == 0 else "W" for move in range(move_count)],
        "nextPla": "B",
        "trainingWeight": 1.0,
        "weight": 1.0,
        "xSize": board_size,
        "ySize": board_size,
    }
    return json.dumps(sample, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def ensure_evaluation_opening_suite(
    state_root: Path, *, board_size: int, opening_count: int
) -> tuple[Path, ...]:
    """Create or verify immutable singleton opening directories for evaluation."""

    if board_size not in {9, 13, 19} or opening_count < 1:
        raise ValueError("固定评测开局参数无效")
    samples = [_fixed_opening_sample(board_size, index) for index in range(opening_count)]
    hashes = [hashlib.sha256(sample.encode("utf-8")).hexdigest() for sample in samples]
    if len(set(hashes)) != opening_count:
        raise CurriculumStateError("固定评测开局生成器产生了重复局面")
    suite_root = (
        state_root.resolve()
        / "evaluation_openings"
        / f"{board_size}x{board_size}"
        / f"v{EVALUATION_OPENING_SUITE_VERSION}"
    )
    manifest = {
        "schema_version": 1,
        "suite_version": EVALUATION_OPENING_SUITE_VERSION,
        "board_size": board_size,
        "opening_count": opening_count,
        "format": "KataGo PositionSample JSONL; one singleton directory per color-swapped pair",
        "sha256": hashes,
    }

    def is_valid() -> bool:
        try:
            stored = json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))
            files_match = all(
                (
                    suite_root
                    / f"opening-{index:03d}"
                    / "opening.startposes.txt"
                ).read_text(encoding="utf-8")
                == samples[index]
                for index in range(opening_count)
            )
        except (OSError, json.JSONDecodeError):
            return False
        return stored == manifest and files_match

    if suite_root.is_dir() and is_valid():
        return tuple(suite_root / f"opening-{index:03d}" for index in range(opening_count))

    suite_root.parent.mkdir(parents=True, exist_ok=True)
    if suite_root.exists():
        quarantine = suite_root.with_name(
            f".{suite_root.name}.invalid-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        suite_root.replace(quarantine)
    temporary = suite_root.with_name(f".{suite_root.name}.build-{uuid.uuid4().hex}.tmp")
    try:
        for index, sample in enumerate(samples):
            opening = temporary / f"opening-{index:03d}"
            opening.mkdir(parents=True, exist_ok=False)
            (opening / "opening.startposes.txt").write_text(sample, encoding="utf-8")
        _atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, suite_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(suite_root / f"opening-{index:03d}" for index in range(opening_count))


class CurriculumRuntime:
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
            _atomic_write_text(log_path, execution.stdout)
            return execution

        output: list[str] = []
        with log_path.open("a", encoding="utf-8", newline="") as log:
            log.write("$ " + subprocess.list2cmdline(args) + "\n")
            log.flush()
            with managed_popen(
                args,
                cwd=str(cwd),
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    output.append(line)
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return CommandExecution(process.wait(), "".join(output))

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
        _atomic_write_json(
            output_dir / "evaluation_manifest.json",
            {
                "protocol_version": 2,
                "roles": roles,
                "candidate_model": candidate_name,
                "candidate_path": str(candidate),
                "candidate_sha256": _sha256_file(candidate),
                "baseline_model": baseline_name,
                "baseline_path": str(baseline),
                "baseline_sha256": _sha256_file(baseline),
                "board_size": state.active.board_size,
                "visits": runner.config.evaluation.simulations_per_move,
                "komi": 6.5,
                "games": self.config.evaluation_games,
                "opening_pairs": opening_count,
                "opening_suite": str(suite_manifest),
                "opening_suite_sha256": _sha256_file(suite_manifest),
                "root_noise": False,
                "temperature": 0.0,
                "resignation": False,
                "created_at": _now(),
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
            _atomic_write_text(config_path, config_text)
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
        _atomic_write_json(output_dir / "match-result.json", summary.to_dict())
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
            _atomic_write_json(output_dir / "summary.json", state.active.evaluations[-1])
            if state.active.evaluations[-1].get("champion_promoted"):
                print(f"滚动冠军已更新：{state.active.champion_model}", flush=True)
            if gate.passed:
                print(f"课程门槛通过：固定基准胜率 {summary.win_rate:.1%}", flush=True)
            else:
                state.warnings.append({
                    "timestamp": _now(), "kind": "quality_gate",
                    "message": "；".join(gate.reasons) or "课程门槛未通过",
                })
                self.store.save(state)
                print("课程门槛未达标，继续当前棋盘训练。", flush=True)
        except Exception as error:
            if committed:
                state.warnings.append({
                    "timestamp": _now(), "kind": "evaluation_artifact",
                    "message": f"评测状态已提交，但辅助文件或输出失败：{error}",
                })
                self.store.save(state)
                print(f"评测状态已提交；辅助文件记录失败：{error}", file=sys.stderr, flush=True)
            else:
                self.controller.record_evaluation_error(state, str(error))
                print(f"评测失败但检查点保持完好：{error}", file=sys.stderr, flush=True)

    def _benchmark_measurement(
        self,
        runner: KataGoRLRunner,
        data_file: Path,
        batch_size: int,
        log_root: Path,
    ) -> BenchmarkMeasurement:
        command = build_benchmark_command(
            python_executable=Path(sys.executable),
            benchmark_script=runner.python_source / "benchmark_fresh_model.py",
            data_file=data_file,
            board_size=runner.config.game.board_size,
            batch_size=batch_size,
            gpu_index=0,
            model_kind=runner.model_kind,
        )
        environment = runner._environment()
        execution = self._run_external(
            command,
            cwd=runner.python_source,
            env=environment,
            log_path=log_root / f"batch-{batch_size}.log",
        )
        return parse_benchmark_output(
            execution.stdout, batch_size=batch_size, returncode=execution.returncode
        )

    def _autotune(
        self,
        stage: CurriculumStage,
        runner: KataGoRLRunner,
        source_npz: Path,
    ) -> tuple[int, Optional[str]]:
        if not stage.autotune_batches:
            return runner.config.optimizer.batch_size, None
        tune_root = runner.run_root / "autotune"
        data_file = prepare_benchmark_npz(
            source_npz,
            tune_root / "benchmark.npz",
            max(stage.autotune_batches),
        )
        persist_path = runner.run_root / "autotune.json"
        measurements: dict[int, BenchmarkMeasurement] = {}
        for batch in stage.autotune_batches:
            try:
                measurements[batch] = self._benchmark_measurement(
                    runner, data_file, batch, tune_root
                )
            except Exception as error:
                message = str(error)
                measurements[batch] = BenchmarkMeasurement(
                    batch_size=batch,
                    success=False,
                    oom=(
                        "out of memory" in message.casefold()
                        or re.search(r"\boom\b", message, re.IGNORECASE) is not None
                    ),
                    error=message,
                )
        try:
            result = choose_autotune_batch(
                stage.autotune_batches,
                lambda batch: measurements[batch],
                max_gpu_memory_gib=self.config.max_gpu_memory_gib,
                persist_path=None,
            )
            _atomic_write_json(
                persist_path,
                {
                    "selected_batch_size": result.selected_batch_size,
                    "measurements": [
                        {
                            **asdict(measurement),
                            "peak_gpu_gib": (
                                measurement.peak_gpu_gib
                                if math.isfinite(measurement.peak_gpu_gib)
                                else None
                            ),
                        }
                        for measurement in result.measurements
                    ],
                    "completed_at": _now(),
                },
            )
            return result.selected_batch_size, None
        except Exception as error:
            warning = f"自动批次测试无合格结果，拒绝迁移：{error}"
            _atomic_write_json(
                persist_path,
                {
                    "selected_batch_size": None,
                    "failed": True,
                    "warning": warning,
                    "measurements": [
                        {
                            **asdict(measurements[batch]),
                            "peak_gpu_gib": (
                                measurements[batch].peak_gpu_gib
                                if math.isfinite(measurements[batch].peak_gpu_gib)
                                else None
                            ),
                        }
                        for batch in stage.autotune_batches
                    ],
                    "completed_at": _now(),
                },
            )
            raise CurriculumMigrationError(warning) from error
        finally:
            with contextlib.suppress(FileNotFoundError):
                data_file.unlink()

    def _training_smoke(
        self,
        stage: CurriculumStage,
        seed_checkpoint: Path,
        source_npz: Path,
        batch_size: int,
        parent: Path,
    ) -> None:
        smoke_root = parent / f".training-smoke-{uuid.uuid4().hex}.tmp"
        try:
            checkpoint = smoke_root / "train" / "gogogo" / "checkpoint.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed_checkpoint, checkpoint)
            data_file = smoke_root / "selfplay" / "seed" / "tdata" / "smoke.npz"
            prepare_benchmark_npz(source_npz, data_file, max(2, batch_size * 2))
            smoke_runner = self._make_runner(stage, run_root_override=smoke_root)
            smoke_runner.shuffle(min_rows=1, smoke=True)
            smoke_runner.train(smoke=True, batch_size=batch_size)
            if not checkpoint.is_file():
                raise CurriculumMigrationError("单批训练冒烟未生成检查点")
        finally:
            if smoke_root.exists():
                shutil.rmtree(smoke_root, ignore_errors=True)

    def _migration_checks(
        self,
        stage: CurriculumStage,
        temporary_root: Path,
        seed_checkpoint: Path,
    ) -> tuple[int, Optional[str]]:
        runner = self._make_runner(stage, run_root_override=temporary_root)
        exported = runner.export_new_models()
        seed_model = temporary_root / "models" / "gogogo-s0-d0" / "model.bin.gz"
        if not seed_model.is_file() and not exported:
            raise CurriculumMigrationError("迁移种子未能导出为 KataGo 模型")
        runner.selfplay(games=4, visits=8, smoke=True)
        npz_files = sorted(
            (temporary_root / "selfplay").glob("**/tdata/*.npz"),
            key=lambda path: path.stat().st_mtime,
        )
        if not npz_files:
            raise CurriculumMigrationError("4 局低 visits 冒烟未生成目标棋盘 NPZ")
        selected, warning = self._autotune(stage, runner, npz_files[-1])
        self._training_smoke(
            stage, seed_checkpoint, npz_files[-1], selected, temporary_root
        )
        # Smoke games validate the target but must never enter its replay pool.
        for folder in (
            temporary_root / "selfplay",
            temporary_root / "shuffleddata",
            temporary_root / "shufflescratch",
            temporary_root / "autotune",
        ):
            if folder.exists():
                shutil.rmtree(folder)
        # The four validation games are intentionally discarded and must not
        # become the new stage's deleted-row offset.  Its real ledger starts at
        # zero after the temporary tree is published.
        with contextlib.suppress(FileNotFoundError):
            (temporary_root / "replay_rows.json").unlink()
        return selected, warning

    @staticmethod
    def _read_selected_batch(path: Path) -> Optional[int]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            selected = int(value["selected_batch_size"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return selected if selected > 0 else None

    def _record_migration_retry(self, state: CurriculumState, message: str) -> None:
        state.phase = "migrating"
        state.warnings.append(
            {"timestamp": _now(), "kind": "migration_retry", "message": message}
        )
        self.store.save(state)
        print(f"迁移暂时无法完成，将保留身份后重试：{message}", file=sys.stderr, flush=True)

    def _migrate(self, state: CurriculumState) -> None:
        if state.active_stage_index >= len(self.config.stages) - 1:
            state.phase = "training"
            self.store.save(state)
            return
        current_runner = self._stage_runner(state)
        source_model = state.active.latest_model
        if not source_model:
            self.controller.migration_failed(state, "迁移前缺少最新模型")
            return
        if source_model not in state.protected_models:
            state.protected_models.append(source_model)
            self.store.save(state)
        target_stage = self.config.stages[state.active_stage_index + 1]
        target_runner = self._make_runner(target_stage)
        target_root = target_runner.run_root
        source_checkpoint = (
            current_runner.run_root
            / "torchmodels_toexport"
            / source_model
            / "model.ckpt"
        )
        if not source_checkpoint.is_file():
            self.controller.migration_failed(
                state, f"找不到最新接纳模型的 SWA 检查点：{source_checkpoint}"
            )
            return

        try:
            source_sha256 = _sha256_file(source_checkpoint)
        except OSError as error:
            if state.phase == "migrating" and state.migration:
                self._record_migration_retry(state, str(error))
            else:
                state.warnings.append(
                    {
                        "timestamp": _now(),
                        "kind": "migration_retry",
                        "message": f"暂时无法读取迁移源检查点：{error}",
                    }
                )
                self.store.save(state)
                print(
                    f"迁移源暂时不可读，将保持待迁移状态后重试：{error}",
                    file=sys.stderr,
                    flush=True,
                )
            return

        identity = {
            "source_model": source_model,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_sha256": source_sha256,
            "target_board_size": target_stage.board_size,
            "seed_model": "gogogo-s0-d0",
        }
        if state.phase == "transition_ready":
            state.migration = {
                "id": uuid.uuid4().hex,
                **identity,
                "started_at": _now(),
            }
            self.store.save(state)
            self.controller.begin_migration(state)
        elif state.phase == "migrating":
            if not state.migration:
                # Backward-compatible recovery is safe only before a target
                # directory has become visible.  A visible directory without
                # an identity manifest could belong to another transition.
                if target_root.exists():
                    self.controller.migration_failed(
                        state,
                        f"目标目录缺少迁移身份，拒绝触碰现有目录：{target_root}",
                    )
                    return
                state.migration = {
                    "id": uuid.uuid4().hex,
                    **identity,
                    "started_at": _now(),
                }
                self.store.save(state)
            elif (
                not isinstance(state.migration.get("id"), str)
                or any(state.migration.get(key) != value for key, value in identity.items())
            ):
                suffix = ""
                if target_root.exists():
                    suffix = f"；未触碰现有目标目录 {target_root}"
                self.controller.migration_failed(
                    state, f"迁移源检查点或目标阶段已变化，拒绝继续旧迁移{suffix}"
                )
                return
        else:
            raise CurriculumStateError(f"当前 phase={state.phase}，不能迁移")

        selected_batch: Optional[int] = None
        warning: Optional[str] = None
        target_preexisting = target_root.exists()
        target_identity_confirmed = False
        target_published = False
        migration_committed = False
        source_stage_index = state.active_stage_index
        try:
            target_lock = (
                target_runner.lock() if target_preexisting else contextlib.nullcontext()
            )
            with target_lock:
                if target_preexisting:
                    if not target_root.exists():
                        raise CurriculumMigrationError("目标阶段目录在加锁期间消失")
                # Recovery for a crash after the atomic directory rename but
                # before complete_migration() persisted the new active stage.
                    checkpoint = target_root / "train" / "gogogo" / "checkpoint.ckpt"
                    model = target_root / "models" / "gogogo-s0-d0" / "model.bin.gz"
                    selected_batch = self._read_selected_batch(
                        target_root / "autotune.json"
                    )
                    manifest_path = target_root / "migration_manifest.json"
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (FileNotFoundError, json.JSONDecodeError):
                        manifest = None
                    manifest_matches = isinstance(manifest, dict) and all(
                        manifest.get(key) == value
                        for key, value in state.migration.items()
                    )
                    if isinstance(manifest, dict):
                        manifest_matches = manifest_matches and (
                            manifest.get("selected_batch_size") == selected_batch
                        )
                    if (
                        not checkpoint.is_file()
                        or not model.is_file()
                        or selected_batch is None
                        or not manifest_matches
                        or manifest.get("seed_checkpoint_sha256")
                        != _sha256_file(checkpoint)
                    ):
                        raise CurriculumMigrationError(
                            f"已存在目标目录不完整或不属于当前迁移；拒绝触碰 {target_root}"
                        )
                    target_identity_confirmed = True
                    verify_checkpoint_loads(
                        checkpoint,
                        boards=(9, 13, 19),
                        python_source=target_runner.python_source,
                    )
                else:
                    if target_root.exists():
                        raise CurriculumMigrationError(
                            f"目标目录在迁移准备期间被其他进程创建：{target_root}"
                        )
                    check_result: dict[str, object] = {}

                    def verify(checkpoint: Path, _board: int) -> None:
                        verify_checkpoint_loads(
                            checkpoint,
                            boards=(9, 13, 19),
                            python_source=target_runner.python_source,
                        )

                    def smoke(root: Path, checkpoint: Path, _board: int) -> None:
                        batch, tune_warning = self._migration_checks(
                            target_stage, root, checkpoint
                        )
                        _atomic_write_json(
                            root / "migration_manifest.json",
                            {
                                **state.migration,
                                "selected_batch_size": batch,
                                "seed_checkpoint_sha256": _sha256_file(checkpoint),
                                "completed_at": _now(),
                            },
                        )
                        check_result["batch"] = batch
                        check_result["warning"] = tune_warning

                    prepare_stage_migration(
                        source_checkpoint=source_checkpoint,
                        target_stage_root=target_root,
                        target_board_size=target_stage.board_size,
                        load_verifier=verify,
                        smoke_verifier=smoke,
                        seed_name="gogogo-s0-d0",
                    )
                    # From this point the verified tree and its identity
                    # manifest are atomically visible.  A later state-save
                    # failure must retain this migration identity so the next
                    # run can adopt that exact tree instead of orphaning it.
                    target_published = True
                    selected_batch = int(check_result["batch"])
                    warning_value = check_result.get("warning")
                    warning = str(warning_value) if warning_value else None
                self.controller.complete_migration(
                    state,
                    seed_model="gogogo-s0-d0",
                    autotune_batch=selected_batch,
                )
                migration_committed = True
            if warning:
                state.warnings.append(
                    {"timestamp": _now(), "kind": "autotune", "message": warning}
                )
                self.store.save(state)
            print(
                f"课程已切换到 {target_stage.board_size}x{target_stage.board_size}，训练批次 {selected_batch}",
                flush=True,
            )
        except Exception as error:
            # If complete_migration's atomic save returned an error after the
            # replacement, trust the durable state instead of the mutated
            # in-memory object.
            try:
                persisted = self.store.load()
                if persisted is not None:
                    state.__dict__.update(persisted.__dict__)
                    migration_committed = (
                        persisted.active_stage_index > source_stage_index
                    )
            except (CurriculumStateError, OSError) as reload_error:
                if target_published or target_preexisting:
                    # The target is visible but we cannot tell whether the
                    # atomic state replacement committed.  Any write here
                    # could regress a successfully committed migration or
                    # persist a half-mutated object, so fail closed and let a
                    # fresh process read the durable state.
                    raise CurriculumStateError(
                        "迁移目标已发布，但无法确认课程状态提交结果；拒绝继续写状态"
                    ) from reload_error
            if migration_committed:
                state.warnings.append(
                    {
                        "timestamp": _now(),
                        "kind": "migration_artifact",
                        "message": f"迁移状态已提交，但后续记录失败：{error}",
                    }
                )
                self.store.save(state)
                print(
                    f"迁移已提交；后续记录失败：{error}", file=sys.stderr, flush=True
                )
            elif target_published or (
                target_preexisting
                and (
                    isinstance(error, (KataGoRLRunnerError, OSError))
                    or target_identity_confirmed
                )
            ):
                self._record_migration_retry(state, str(error))
            else:
                self.controller.migration_failed(state, str(error))
                print(f"迁移失败，继续旧棋盘：{error}", file=sys.stderr, flush=True)

    @staticmethod
    def _stale_temp_paths(run_root: Path, now: float) -> list[Path]:
        candidates: list[Path] = []
        for parent in (
            run_root / "models",
            run_root / "torchmodels_toexport",
            run_root / "shuffleddata",
            run_root / "shufflescratch",
        ):
            if not parent.is_dir():
                continue
            for path in parent.iterdir():
                if not path.is_dir():
                    continue
                is_temp = path.name.endswith(".tmp") or ".tmp-" in path.name
                with contextlib.suppress(OSError):
                    if is_temp and now - path.stat().st_mtime >= STALE_TEMP_SECONDS:
                        candidates.append(path)
        return candidates

    def _retention_for_root(
        self, run_root: Path, protected: Iterable[str]
    ) -> dict[str, int]:
        if not run_root.is_dir():
            return {"files": 0, "directories": 0, "bytes": 0}
        ledger = ReplayRowLedger(run_root)
        try:
            # This durable inventory is the write-ahead record for any NPZ
            # deletion performed below.
            ledger.sync()
        except ReplayAccountingError as error:
            raise CurriculumStateError(
                f"无法安全同步 {run_root.name} 回放行账本：{error}"
            ) from error
        plan = build_retention_plan(
            run_root,
            protected_models=protected,
            keep_models=self.config.keep_models,
            keep_shuffles=self.config.keep_shuffles,
            replay_window_rows=self.config.replay_window_rows,
        )
        longterm = run_root / "train" / "gogogo" / "longterm_checkpoints"
        checkpoints = sorted(
            (path for path in longterm.glob("*.ckpt") if path.is_file()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        delete_checkpoints = checkpoints[:-self.config.keep_models]
        stale = self._stale_temp_paths(run_root, time.time())
        directory_targets = list(plan.delete_directories) + stale
        file_targets = list(plan.delete_npz) + delete_checkpoints
        unique_dirs = list(dict.fromkeys(directory_targets))
        unique_files = list(dict.fromkeys(file_targets))
        for path in (*unique_dirs, *unique_files):
            _assert_exact_descendant(path, run_root)
        deleted_bytes = sum(_directory_size(path) for path in (*unique_dirs, *unique_files))
        # Apply the core plan first (which repeats exact-target validation),
        # then the runtime-only longterm/tmp rules.
        apply_retention_plan(plan, run_root)
        try:
            # If interruption happens before this write, the next pre-cleanup
            # sync observes the same missing inventory entries exactly once.
            ledger.sync()
        except ReplayAccountingError as error:
            raise CurriculumStateError(
                f"无法提交 {run_root.name} 回放清理账本：{error}"
            ) from error
        for path in delete_checkpoints:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        for path in stale:
            if path.is_dir():
                shutil.rmtree(path)
        return {
            "files": len(unique_files),
            "directories": len(unique_dirs),
            "bytes": deleted_bytes,
        }

    @staticmethod
    def _cleanup_stale_migrations(target_root: Path) -> dict[str, int]:
        """Remove only old, exact sibling migration-temp directories."""

        parent = target_root.resolve().parent
        if not parent.is_dir():
            return {"files": 0, "directories": 0, "bytes": 0}
        now = time.time()
        targets: list[Path] = []
        pattern = f".{target_root.name}.migration-*.tmp"
        for path in parent.glob(pattern):
            if not path.is_dir():
                continue
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime >= STALE_TEMP_SECONDS:
                    _assert_exact_descendant(path, parent)
                    targets.append(path)
        deleted_bytes = sum(_directory_size(path) for path in targets)
        for path in targets:
            shutil.rmtree(path)
        return {
            "files": 0,
            "directories": len(targets),
            "bytes": deleted_bytes,
        }

    def _apply_retention(self, state: CurriculumState) -> dict[str, int]:
        totals = {"files": 0, "directories": 0, "bytes": 0}
        for stage in self.config.stages:
            runner = self._make_runner(stage)
            reports: list[dict[str, int]] = []
            if runner.run_root.is_dir():
                try:
                    with runner.lock():
                        reports.append(
                            self._retention_for_root(
                                runner.run_root, state.protected_models
                            )
                        )
                except KataGoRLRunnerError as error:
                    state.warnings.append(
                        {
                            "timestamp": _now(),
                            "kind": "retention_lock",
                            "message": str(error),
                        }
                    )
            reports.append(self._cleanup_stale_migrations(runner.run_root))
            for report in reports:
                for key in totals:
                    totals[key] += report[key]
        state.disk["last_cleanup"] = {
            **totals,
            "completed_at": _now(),
        }
        self.store.save(state)
        return totals

    def _update_disk(self, state: CurriculumState) -> str:
        status = self.disk_probe(
            self.state_root if self.state_root.exists() else self.project_root,
            cleanup_below_gib=self.config.disk_cleanup_gib,
            pause_below_gib=self.config.disk_pause_gib,
        )
        status_name = self.controller.update_disk(state, float(status.free_gib))
        state.disk["total_gib"] = float(status.total_gib)
        self.store.save(state)
        return status_name

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
                {"timestamp": _now(), "kind": "training", "message": str(error)}
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
        _atomic_write_text(self.dashboard_path, document)

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
    "parse_benchmark_output",
    "prepare_benchmark_npz",
]
