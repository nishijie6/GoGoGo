"""Windows-friendly orchestration for KataGo's official self-play loop.

The heavy neural-network implementation, training-data format, MCTS self-play,
and exporter all come from the pinned official KataGo source checkout.  This
module connects those stages to this project's validated RL configuration and
adds resumable paths, logging, process locking, and CUDA runtime discovery.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from .katago import katago_subprocess_environment
from .process_control import managed_popen
from .rl_curriculum import (
    CurriculumConfigError,
    CurriculumLockError,
    CurriculumMigrationError,
    CurriculumStateError,
)
from .rl_config import RLConfigError, RLTrainingConfig, load_rl_training_config
from .rl_metrics import RLMetricStore, parse_selfplay_output
from .replay_accounting import ReplayAccountingError, ReplayRowLedger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RTX_PROFILE = PROJECT_ROOT / "config" / "rl_training.rtx5070ti.json"
DEFAULT_CURRICULUM_PROFILE = (
    PROJECT_ROOT / "config" / "rl_curriculum.rtx5070ti.json"
)

# Official model names that have a direct correspondence to our simple
# residual-block/channel configuration.
KATAGO_MODEL_KINDS = {
    (6, 96): "b6c96",
    (10, 128): "b10c128",
    (15, 192): "b15c192",
    (20, 256): "b20c256",
}


class KataGoRLRunnerError(RuntimeError):
    """Raised when an external stage or local training prerequisite fails."""


def model_kind_for_config(config: RLTrainingConfig) -> str:
    """Map a validated project network shape to an official KataGo model."""

    shape = (config.network.residual_blocks, config.network.channels)
    try:
        return KATAGO_MODEL_KINDS[shape]
    except KeyError as error:
        supported = ", ".join(
            f"{blocks} blocks/{channels} channels ({name})"
            for (blocks, channels), name in KATAGO_MODEL_KINDS.items()
        )
        raise KataGoRLRunnerError(
            f"KataGo 没有与 {shape[0]} blocks/{shape[1]} channels 对应的官方模型；"
            f"可选：{supported}"
        ) from error


def _config_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_katago_overrides(overrides: Mapping[str, object]) -> str:
    """Format command-line config overrides accepted by KataGo."""

    return ",".join(f"{key}={_config_value(value)}" for key, value in overrides.items())


def build_selfplay_overrides(
    config: RLTrainingConfig,
    *,
    visits: Optional[int] = None,
    smoke: bool = False,
    has_model: bool = True,
) -> dict[str, object]:
    """Translate the project profile into official KataGo self-play options."""

    requested_visits = (
        config.search.simulations_per_move if visits is None else visits
    )
    if requested_visits < 1:
        raise KataGoRLRunnerError("自我对弈访问量必须至少为 1")
    cheap_visits = max(4, min(100, requested_visits // 4))
    board_size = config.game.board_size
    game_threads = 8 if smoke else max(32, min(256, config.self_play.workers * 16))
    batch_size = (
        min(16, config.self_play.inference_batch_size)
        if smoke
        else config.self_play.inference_batch_size
    )
    device_index = 0
    if ":" in config.hardware.device:
        device_index = int(config.hardware.device.rsplit(":", 1)[1])

    overrides: dict[str, object] = {
        "numGameThreads": game_threads,
        "maxVisits": requested_visits,
        "cheapSearchVisits": cheap_visits,
        "reducedVisitsMin": cheap_visits,
        "nnMaxBatchSize": batch_size,
        "nnCacheSizePowerOfTwo": 20 if not smoke else 18,
        "numNNServerThreadsPerModel": 2 if not smoke else 1,
        "cudaDeviceToUse": device_index,
        "cudaUseFP16": True,
        "dataBoardLen": board_size,
        "bSizes": board_size,
        "bSizeRelProbs": 1,
        "allowRectangleProb": 0.0,
        "koRules": "POSITIONAL",
        "scoringRules": "AREA",
        "taxRules": "NONE",
        "multiStoneSuicideLegals": False,
        "hasButtons": False,
        "komiAuto": False,
        "komiMean": config.game.komi,
        "komiStdev": 0.0,
        "komiBigStdevProb": 0.0,
        "handicapProb": 0.0,
        # Keep every generated position on the exact rule/komi domain used by
        # the desktop game. The stock training config otherwise varies komi
        # for forked and asymmetric games.
        "earlyForkGameProb": 0.0,
        "forkGameProb": 0.0,
        "sekiForkHackProb": 0.0,
        "compensateAfterPolicyInitProb": 0.0,
        "normalAsymmetricPlayoutProb": 0.0,
        "fancyKomiVarying": False,
        # Raw-policy initialization is especially destructive for a weak
        # small-board model: it can select pass twice before MCTS ever sees
        # the position. Exploration remains provided by root noise and the
        # move-temperature schedule below.
        "initGamesWithPolicy": False,
        # Translate the profile's exploration controls to KataGo's native
        # self-play names.
        "rootNoiseEnabled": config.search.dirichlet_epsilon > 0,
        "rootDirichletNoiseTotalConcentration": (
            config.search.dirichlet_alpha * board_size * board_size
        ),
        "rootDirichletNoiseWeight": config.search.dirichlet_epsilon,
        "chosenMoveTemperatureEarly": config.search.root_temperature,
        "chosenMoveTemperatureHalflife": max(
            1, config.search.temperature_moves
        ),
        "cpuctExploration": config.search.c_puct,
        "maxMovesPerGame": math.ceil(
            board_size * board_size * config.self_play.max_game_length_factor
        ),
        "logGamesEvery": 1,
    }
    # This option is consumed only after a real model replaces the random
    # bootstrap generator; omitting it initially avoids a noisy unused-key warning.
    if has_model:
        overrides["cudaUseNHWC"] = True
    return overrides


class KataGoRLRunner:
    """Run and resume one official single-machine KataGo training pipeline."""

    def __init__(
        self,
        config_path: Path = DEFAULT_RTX_PROFILE,
        project_root: Path = PROJECT_ROOT,
        run_root_override: Optional[Path] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = (
            config_path if config_path.is_absolute() else self.project_root / config_path
        ).resolve()
        self.config = load_rl_training_config(self.config_path)
        configured_output = Path(self.config.runtime.output_directory)
        self.run_root = (
            run_root_override
            if run_root_override is not None
            else (
                configured_output
                if configured_output.is_absolute()
                else self.project_root / configured_output
            )
        ).resolve()
        self.katago_executable = self.project_root / "katago" / (
            "katago.exe" if os.name == "nt" else "katago"
        )
        self.katago_source = self.project_root / "katago" / "source"
        self.python_source = self.katago_source / "python"
        self.selfplay_config = (
            self.katago_source / "cpp" / "configs" / "training" / "selfplay1.cfg"
        )
        self.model_kind = model_kind_for_config(self.config)
        self.metric_store = RLMetricStore(self.run_root)

    @property
    def models_dir(self) -> Path:
        return self.run_root / "models"

    @property
    def selfplay_dir(self) -> Path:
        return self.run_root / "selfplay"

    @property
    def shuffled_dir(self) -> Path:
        return self.run_root / "shuffleddata"

    @property
    def logs_dir(self) -> Path:
        return self.run_root / "logs"

    def _ensure_directories(self) -> None:
        for folder in (
            self.models_dir,
            self.selfplay_dir,
            self.shuffled_dir,
            self.logs_dir,
            self.run_root / "shufflescratch" / "train",
            self.run_root / "train" / "gogogo",
            self.run_root / "torchmodels_toexport",
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def _require_runtime(self) -> None:
        missing = [
            path
            for path in (
                self.katago_executable,
                self.python_source / "train.py",
                self.python_source / "shuffle.py",
                self.python_source / "export_model_pytorch.py",
                self.selfplay_config,
            )
            if not path.is_file()
        ]
        if missing:
            raise KataGoRLRunnerError(
                "缺少 KataGo 运行或训练文件：" + "、".join(str(path) for path in missing)
            )

    def _environment(self) -> dict[str, str]:
        environment = katago_subprocess_environment()
        environment["PYTHONUTF8"] = "1"
        return environment

    def _run(
        self,
        command: Sequence[object],
        *,
        cwd: Optional[Path] = None,
        log_name: Optional[str] = None,
    ) -> str:
        args = [str(item) for item in command]
        print("\n$ " + subprocess.list2cmdline(args), flush=True)
        log_file = None
        output_lines: list[str] = []
        try:
            if log_name is not None:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                log_file = (self.logs_dir / log_name).open(
                    "a", encoding="utf-8", newline=""
                )
                log_file.write("\n$ " + subprocess.list2cmdline(args) + "\n")
                log_file.flush()
            with managed_popen(
                args,
                cwd=str(cwd or self.project_root),
                env=self._environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    output_lines.append(line)
                    print(line, end="", flush=True)
                    if log_file is not None:
                        log_file.write(line)
                        log_file.flush()
                return_code = process.wait()
        finally:
            if log_file is not None:
                log_file.close()
        if return_code != 0:
            raise KataGoRLRunnerError(
                f"外部阶段退出，代码 {return_code}：{subprocess.list2cmdline(args)}"
            )
        return "".join(output_lines)

    def doctor(self) -> dict[str, object]:
        """Verify pinned sources, CUDA PyTorch, GPU access, and KataGo."""

        self._require_runtime()
        try:
            import torch
        except ImportError as error:
            raise KataGoRLRunnerError(
                "未安装 CUDA PyTorch；请运行 pip install -r requirements-rl.txt"
            ) from error
        if not torch.cuda.is_available():
            raise KataGoRLRunnerError("PyTorch 未检测到可用的 CUDA 显卡")
        device = torch.device(self.config.hardware.device.replace("auto", "cuda:0"))
        index = device.index or 0
        self._run(
            [self.katago_executable, "version"],
            log_name="doctor.log",
        )
        result = {
            "config": str(self.config_path),
            "run_root": str(self.run_root),
            "model_kind": self.model_kind,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(index),
            "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    def selfplay(
        self,
        *,
        games: Optional[int] = None,
        visits: Optional[int] = None,
        smoke: bool = False,
    ) -> dict[str, int | float]:
        """Generate training rows with the newest accepted model."""

        self._require_runtime()
        self._ensure_directories()
        requested_games = (
            4 if smoke else self.config.self_play.games_per_iteration
        ) if games is None else games
        if requested_games < 1:
            raise KataGoRLRunnerError("自我对弈局数必须至少为 1")
        has_model = any(self.models_dir.glob("*/model.bin.gz"))
        overrides = build_selfplay_overrides(
            self.config,
            visits=(8 if smoke else None) if visits is None else visits,
            smoke=smoke,
            has_model=has_model,
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = self._run(
            [
                self.katago_executable,
                "selfplay",
                "-max-games-total",
                requested_games,
                "-output-dir",
                self.selfplay_dir,
                "-models-dir",
                self.models_dir,
                "-config",
                self.selfplay_config,
                "-override-config",
                format_katago_overrides(overrides),
            ],
            log_name=f"selfplay-{timestamp}.log",
        )
        return parse_selfplay_output(output)

    def shuffle(self, *, min_rows: Optional[int] = None, smoke: bool = False) -> Path:
        """Shuffle the current replay window into consumable training batches."""

        self._require_runtime()
        self._ensure_directories()
        if not any(self.selfplay_dir.glob("**/tdata/*.npz")):
            raise KataGoRLRunnerError("没有找到自我对弈训练数据")
        try:
            replay_offset = ReplayRowLedger(self.run_root).sync().deleted_rows_offset
        except ReplayAccountingError as error:
            raise KataGoRLRunnerError(f"回放数据累计行号不可用：{error}") from error
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        temporary_parent = self.shuffled_dir / f"{stamp}.tmp"
        output_dir = temporary_parent / "train"
        output_dir.mkdir(parents=True, exist_ok=False)
        scratch = self.run_root / "shufflescratch" / "train"
        keep_rows: object = (
            "all" if smoke else self.config.optimizer.replay_buffer_capacity
        )
        required_rows = (
            1 if smoke else self.config.optimizer.minimum_replay_size
        ) if min_rows is None else min_rows
        if required_rows < 1:
            raise KataGoRLRunnerError("打乱窗口的最小行数必须至少为 1")
        workers = max(1, min(4, self.config.hardware.data_loader_workers))
        rows_per_file = max(4096, self.config.optimizer.batch_size * 16)
        self._run(
            [
                sys.executable,
                self.python_source / "shuffle.py",
                self.selfplay_dir,
                "-add-to-data-rows",
                replay_offset,
                "-min-rows",
                required_rows,
                "-keep-target-rows",
                keep_rows,
                "-expand-window-per-row",
                0.4,
                "-taper-window-exponent",
                0.65,
                "-taper-window-scale",
                max(1, required_rows),
                "-out-dir",
                output_dir,
                "-out-tmp-dir",
                scratch,
                "-approx-rows-per-out-file",
                rows_per_file,
                "-num-processes",
                workers,
            ],
            cwd=self.python_source,
            log_name=f"shuffle-{stamp}.log",
        )
        final_parent = self.shuffled_dir / stamp
        temporary_parent.replace(final_parent)
        print(f"已生成打乱数据：{final_parent}", flush=True)
        return final_parent

    def train(
        self,
        *,
        smoke: bool = False,
        batch_size: Optional[int] = None,
        initial_checkpoint: Optional[Path] = None,
    ) -> None:
        """Update the policy/value network from the newest shuffled data."""

        self._require_runtime()
        self._ensure_directories()
        batch_size = self.config.optimizer.batch_size if batch_size is None else batch_size
        if batch_size < 1:
            raise KataGoRLRunnerError("训练批次必须至少为 1")
        if not any(self.shuffled_dir.glob("*/train/*.npz")):
            raise KataGoRLRunnerError("没有找到已打乱的训练数据")
        max_samples = (
            batch_size
            if smoke
            else batch_size * self.config.optimizer.training_steps_per_iteration
        )
        command: list[object] = [
            sys.executable,
            self.python_source / "train.py",
            "-traindir",
            self.run_root / "train" / "gogogo",
            "-latestdatadir",
            self.shuffled_dir,
            "-exportdir",
            self.run_root / "torchmodels_toexport",
            "-exportprefix",
            "gogogo",
            "-pos-len",
            self.config.game.board_size,
            "-batch-size",
            batch_size,
            "-model-kind",
            self.model_kind,
            "-samples-per-epoch",
            max_samples,
            "-swa-period-samples",
            max(batch_size, min(80000, max_samples // 2)),
            "-lr-scale",
            1.0,
            "-use-adamw",
            "-epochs-per-export",
            1,
            "-max-epochs-this-instance",
            1,
            "-max-train-bucket-per-new-data",
            4,
            "-max-train-bucket-size",
            max_samples,
            "-stop-when-train-bucket-limited",
            "-quit-if-no-data",
            "-no-repeat-files",
            "-data-prefetch-depth",
            1,
        ]
        if initial_checkpoint is not None:
            resolved_initial = initial_checkpoint.resolve()
            if not resolved_initial.is_file():
                raise KataGoRLRunnerError(
                    f"找不到迁移初始检查点：{resolved_initial}"
                )
            command.extend(["-initial-checkpoint", resolved_initial])
        if self.config.hardware.precision == "amp_float16":
            command.append("-use-fp16")
        elif self.config.hardware.precision == "amp_bfloat16":
            command.append("-use-bf16")
        if not self.config.hardware.compile_model:
            command.append("-no-compile")
        if self.config.hardware.allow_tf32:
            command.append("-use-tf32-matmul")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._run(
            command,
            cwd=self.python_source,
            log_name=f"train-{timestamp}.log",
        )

    def export_new_models(self) -> list[Path]:
        """Export every new PyTorch checkpoint into KataGo's binary format."""

        self._require_runtime()
        self._ensure_directories()
        export_root = self.run_root / "torchmodels_toexport"
        candidates = sorted(
            (
                path
                for path in export_root.iterdir()
                if path.is_dir()
                and not path.name.endswith(".tmp")
                and (path / "model.ckpt").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        exported: list[Path] = []
        for checkpoint_dir in candidates:
            target = self.models_dir / checkpoint_dir.name
            if target.is_dir():
                continue
            temporary = self.models_dir / (
                checkpoint_dir.name + f".tmp-{os.getpid()}"
            )
            temporary.mkdir(parents=True, exist_ok=False)
            self._run(
                [
                    sys.executable,
                    self.python_source / "export_model_pytorch.py",
                    "-checkpoint",
                    checkpoint_dir / "model.ckpt",
                    "-export-dir",
                    temporary,
                    "-model-name",
                    f"GoGoGo-{checkpoint_dir.name}",
                    "-filename-prefix",
                    "model",
                    "-use-swa",
                ],
                cwd=self.python_source,
                log_name=f"export-{checkpoint_dir.name}.log",
            )
            raw_model = temporary / "model.bin"
            compressed_model = temporary / "model.bin.gz"
            with raw_model.open("rb") as source, gzip.open(
                compressed_model, "wb", compresslevel=6
            ) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            raw_model.unlink()
            temporary.replace(target)
            exported.append(target)
            print(f"已接纳新模型：{target}", flush=True)
        return exported

    def cycle(
        self,
        *,
        smoke: bool = False,
        games: Optional[int] = None,
        visits: Optional[int] = None,
        min_rows: Optional[int] = None,
        training_batch_size: Optional[int] = None,
        initial_checkpoint: Optional[Path] = None,
    ) -> None:
        """Run self-play, shuffle, one training epoch, and model export."""

        cycle_started = time.perf_counter()
        stage_started = cycle_started
        selfplay_metrics = self.selfplay(
            games=games, visits=visits, smoke=smoke
        )
        selfplay_metrics["selfplay_stage_seconds"] = (
            time.perf_counter() - stage_started
        )

        stage_started = time.perf_counter()
        self.shuffle(min_rows=min_rows, smoke=smoke)
        shuffle_seconds = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        self.train(
            smoke=smoke,
            batch_size=training_batch_size,
            initial_checkpoint=initial_checkpoint,
        )
        train_seconds = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        exported = self.export_new_models()
        export_seconds = time.perf_counter() - stage_started
        if not exported:
            raise KataGoRLRunnerError(
                "训练轮次没有产生更高水位的新模型；已停止以避免静默空转"
            )
        status = self.status()
        live_metrics: dict[str, object] = {
            **selfplay_metrics,
            "shuffle_seconds": shuffle_seconds,
            "train_seconds": train_seconds,
            "export_seconds": export_seconds,
            "cycle_seconds": time.perf_counter() - cycle_started,
        }
        live_model = exported[-1].name if exported else None
        records = self.metric_store.sync(
            live_model=live_model,
            live_metrics=live_metrics,
        )
        print(
            json.dumps(
                {
                    "metrics_recorded": bool(records),
                    "latest_cycle": records[-1]["cycle"] if records else None,
                    **self.metric_store.paths(),
                    "accepted_models": status["accepted_models"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    def dashboard(self) -> dict[str, object]:
        """Backfill checkpoints and rebuild the CSV/JSONL/HTML dashboard."""

        self._ensure_directories()
        records = self.metric_store.sync()
        result: dict[str, object] = {
            "records": len(records),
            "latest_model": records[-1]["model"] if records else None,
            **self.metric_store.paths(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    def status(self) -> dict[str, object]:
        """Print a compact, machine-readable summary of persisted progress."""

        self._ensure_directories()
        model_files = sorted(
            self.models_dir.glob("*/model.bin.gz"),
            key=lambda path: path.stat().st_mtime,
        )
        data_files = list(self.selfplay_dir.glob("**/tdata/*.npz"))
        shuffled_files = list(self.shuffled_dir.glob("*/train/*.npz"))
        checkpoint = self.run_root / "train" / "gogogo" / "checkpoint.ckpt"
        result = {
            "run_root": str(self.run_root),
            "model_kind": self.model_kind,
            "accepted_models": len(model_files),
            "latest_model": str(model_files[-1]) if model_files else None,
            "selfplay_files": len(data_files),
            "selfplay_bytes": sum(path.stat().st_size for path in data_files),
            "shuffled_files": len(shuffled_files),
            "checkpoint": str(checkpoint) if checkpoint.is_file() else None,
            **self.metric_store.paths(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Prevent two mutating training commands from sharing one run."""

        self.run_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.run_root / "training.lock"
        lock_file = lock_path.open("a+b")
        if lock_path.stat().st_size == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            lock_file.close()
            raise KataGoRLRunnerError(
                f"训练目录已被另一个进程占用：{lock_path}"
            ) from error
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(str(os.getpid()).encode("ascii"))
            lock_file.flush()
            yield
        finally:
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行可续训的 KataGo 单机强化学习闭环"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="cycle",
        choices=(
            "doctor",
            "status",
            "selfplay",
            "shuffle",
            "train",
            "export",
            "cycle",
            "continuous",
            "dashboard",
            "curriculum",
            "curriculum-status",
            "curriculum-dashboard",
            "curriculum-evaluate",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RTX_PROFILE)
    parser.add_argument(
        "--curriculum-config",
        type=Path,
        default=DEFAULT_CURRICULUM_PROFILE,
    )
    parser.add_argument("--games", type=int)
    parser.add_argument("--visits", type=int)
    parser.add_argument("--min-rows", type=int)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="continuous 的循环数；0 表示持续运行到 Ctrl+C",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用少量局数和访问量验证完整闭环",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.iterations < 0:
            raise KataGoRLRunnerError("--iterations 不能小于 0")
        if args.command in {"curriculum", "curriculum-status", "curriculum-dashboard", "curriculum-evaluate"}:
            # Import lazily because the runtime itself builds stage runners
            # from this module.
            from .rl_curriculum_runtime import CurriculumRuntime

            curriculum = CurriculumRuntime(args.curriculum_config)
            if args.command == "curriculum-status":
                curriculum.status()
            elif args.command == "curriculum-dashboard":
                curriculum.refresh_dashboard()
            elif args.command == "curriculum-evaluate":
                curriculum.evaluate_now()
            else:
                curriculum.run(iterations=args.iterations, smoke=args.smoke)
            return
        runner = KataGoRLRunner(args.config)
        if args.command == "doctor":
            runner.doctor()
            return
        if args.command == "status":
            runner.status()
            return
        if args.command == "dashboard":
            runner.dashboard()
            return
        with runner.lock():
            if args.command == "selfplay":
                runner.selfplay(
                    games=args.games,
                    visits=args.visits,
                    smoke=args.smoke,
                )
            elif args.command == "shuffle":
                runner.shuffle(min_rows=args.min_rows, smoke=args.smoke)
            elif args.command == "train":
                runner.train(smoke=args.smoke)
            elif args.command == "export":
                runner.export_new_models()
            elif args.command == "cycle":
                runner.cycle(
                    smoke=args.smoke,
                    games=args.games,
                    visits=args.visits,
                    min_rows=args.min_rows,
                )
            else:
                completed = 0
                while args.iterations <= 0 or completed < args.iterations:
                    print(f"\n===== 强化学习循环 {completed + 1} =====", flush=True)
                    runner.cycle(
                        smoke=args.smoke,
                        games=args.games,
                        visits=args.visits,
                        min_rows=args.min_rows,
                    )
                    completed += 1
    except KeyboardInterrupt:
        print("\n已收到中断，KataGo 会保留可续训产物。", file=sys.stderr)
        raise SystemExit(130)
    except (
        KataGoRLRunnerError,
        RLConfigError,
        CurriculumConfigError,
        CurriculumLockError,
        CurriculumMigrationError,
        CurriculumStateError,
        OSError,
    ) as error:
        print(f"强化学习失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error


__all__ = [
    "DEFAULT_CURRICULUM_PROFILE",
    "DEFAULT_RTX_PROFILE",
    "KATAGO_MODEL_KINDS",
    "KataGoRLRunner",
    "KataGoRLRunnerError",
    "build_selfplay_overrides",
    "format_katago_overrides",
    "main",
    "model_kind_for_config",
]
