"""Windows-friendly orchestration for KataGo's official self-play loop.

The heavy neural-network implementation, training-data format, MCTS self-play,
and exporter all come from the pinned official KataGo source checkout.  This
module connects those stages to this project's validated RL configuration and
adds resumable paths, logging, process locking, and CUDA runtime discovery.
"""

from __future__ import annotations

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

from .fileio import FileLockHeld, acquire_file_lock, release_file_lock
from .katago import katago_subprocess_environment
from .katago_hardware import HardwareProfile
from .process_control import stream_command
from .rl_config import RLTrainingConfig, load_rl_training_config
from .rl_handoff_guard import HandoffLockedError, ensure_training_available
from .rl_metric_store import RLMetricStore
from .rl_metrics import checkpoint_metrics, parse_selfplay_output
from .replay_accounting import ReplayAccountingError, ReplayRowLedger
from .rl_shuffle import align_training_files


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
    hardware_profile: Optional[HardwareProfile] = None,
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
        "earlyForkGameProb": config.self_play.early_fork_game_probability if has_model else 0.0,
        "earlyForkGameExpectedMoveProp": 0.025,
        "forkGameProb": config.self_play.fork_game_probability if has_model else 0.0,
        "forkGameMinChoices": 3,
        "earlyForkGameMaxChoices": 12,
        "forkGameMaxChoices": 36,
        "sekiForkHackProb": 0.0,
        "forkCompensateKomiProb": 0.0,
        "sgfCompensateKomiProb": 0.0,
        "handicapCompensateKomiProb": 0.0,
        "flipKomiProbWhenNoCompensate": 0.0,
        "compensateAfterPolicyInitProb": 0.0,
        "normalAsymmetricPlayoutProb": 0.0,
        "fancyKomiVarying": False,
        # Opt in per profile; weak small-board/bootstrap models keep policy
        # initialization disabled to avoid games ending before MCTS starts.
        "initGamesWithPolicy": config.self_play.init_games_with_policy and has_model,
        "policyInitAreaProp": config.self_play.policy_init_area_prop,
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
    if hardware_profile is not None:
        for key in ("cudaDeviceToUse", "cudaUseFP16", "cudaUseNHWC"):
            overrides.pop(key, None)
        overrides.update(hardware_profile.inference_overrides(has_model=has_model, smoke=smoke))
        if hardware_profile.game_threads is not None:
            overrides["numGameThreads"] = hardware_profile.game_threads
    return overrides


class KataGoRLRunner:
    """Run and resume one official single-machine KataGo training pipeline."""

    def __init__(
        self,
        config_path: Path = DEFAULT_RTX_PROFILE,
        project_root: Path = PROJECT_ROOT,
        run_root_override: Optional[Path] = None,
        hardware_profile: Optional[HardwareProfile] = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = (
            config_path if config_path.is_absolute() else self.project_root / config_path
        ).resolve()
        self.config = load_rl_training_config(self.config_path)
        self.hardware_profile = hardware_profile
        if hardware_profile is not None:
            self.config = hardware_profile.apply(self.config)
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
        if hardware_profile is not None and hardware_profile.backend == "eigen":
            self.katago_executable = self.katago_executable.parent / "cpu" / self.katago_executable.name
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
            if not self.katago_executable.is_file() and self.hardware_profile is not None and self.hardware_profile.tier == "cpu":
                raise KataGoRLRunnerError(
                    "缺少 CPU 版 KataGo。请先运行 powershell -File scripts/install_katago_cpu.ps1；"
                    "CPU 引擎安装在 katago/cpu，不会覆盖 CUDA 引擎。"
                )
            raise KataGoRLRunnerError(
                "缺少 KataGo 运行或训练文件：" + "、".join(str(path) for path in missing)
            )

    def _environment(self) -> dict[str, str]:
        environment = katago_subprocess_environment()
        environment["PYTHONUTF8"] = "1"
        if self.config.hardware.device == "cpu":
            # Official train.py auto-selects CUDA; hide it only in this child.
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        if self.hardware_profile is not None:
            environment["OMP_NUM_THREADS"] = str(self.hardware_profile.cpu_threads)
            environment["MKL_NUM_THREADS"] = str(self.hardware_profile.cpu_threads)
        return environment

    def training_batch_size(self, requested: int) -> int:
        """Clamp a persisted batch when resuming on a smaller machine."""
        if self.hardware_profile is None:
            return requested
        limit = self.hardware_profile.training_batch_limit(self.config.game.board_size)
        return min(requested, limit)

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
        try:
            if log_name is not None:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                log_file = (self.logs_dir / log_name).open(
                    "a", encoding="utf-8", newline=""
                )
                log_file.write("\n$ " + subprocess.list2cmdline(args) + "\n")
                log_file.flush()
            return_code, output = stream_command(
                args, cwd=str(cwd or self.project_root), env=self._environment(), log=log_file
            )
        finally:
            if log_file is not None:
                log_file.close()
        if return_code != 0:
            raise KataGoRLRunnerError(
                f"外部阶段退出，代码 {return_code}：{subprocess.list2cmdline(args)}"
            )
        return output

    def doctor(self) -> dict[str, object]:
        """Verify the selected PyTorch device and KataGo backend."""

        self._require_runtime()
        try:
            import torch
        except ImportError as error:
            raise KataGoRLRunnerError(
                "未安装 PyTorch；CPU 使用 requirements-rl-cpu.txt，CUDA 使用 requirements-rl.txt"
            ) from error
        cpu = self.config.hardware.device == "cpu"
        if not cpu and not torch.cuda.is_available():
            raise KataGoRLRunnerError("PyTorch 未检测到可用的 CUDA 显卡")
        device = torch.device(self.config.hardware.device.replace("auto", "cuda:0"))
        index = device.index or 0
        version = self._run(
            [self.katago_executable, "version"],
            log_name="doctor.log",
        )
        if self.hardware_profile is not None:
            expected = "eigen" if cpu else "cuda"
            if expected not in version.casefold():
                raise KataGoRLRunnerError(f"KataGo 引擎与选择的 {expected} 后端不符：{version.strip()}")
        result = {
            "config": str(self.config_path),
            "run_root": str(self.run_root),
            "model_kind": self.model_kind,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": None if cpu else torch.cuda.get_device_name(index),
            "capability": None if cpu else ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
        }
        if self.hardware_profile is not None:
            result["hardware"] = self.hardware_profile.to_dict()
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
            hardware_profile=self.hardware_profile,
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

    def shuffle(self, *, min_rows: Optional[int] = None, smoke: bool = False,
                training_batch_size: Optional[int] = None,
                new_data_rows: Optional[int] = None) -> Path:
        """Shuffle the current replay window into consumable training batches."""

        self._require_runtime()
        self._ensure_directories()
        if not any(self.selfplay_dir.glob("**/tdata/*.npz")):
            raise KataGoRLRunnerError("没有找到自我对弈训练数据")
        try:
            ledger = ReplayRowLedger(self.run_root).sync()
            replay_offset = ledger.deleted_rows_offset
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
        batch = self.training_batch_size(training_batch_size or self.config.optimizer.batch_size)
        if not smoke:
            budget = self.training_sample_budget(batch, new_data_rows=new_data_rows)
            # The output target alone does not enlarge KataGo's tapered input
            # window. Keep enough unique rows for the epoch plus per-file tails.
            available = min(ledger.physical_rows, self.config.optimizer.replay_buffer_capacity)
            required_rows = max(required_rows, min(available, math.ceil(budget * 1.5)))
        workers = max(1, min(4, self.config.hardware.data_loader_workers))
        rows_per_file = max(4096, batch * 16)
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
        if not smoke and batch >= 128 and available >= budget * 1.5 and any(output_dir.glob("*.npz")):
            align_training_files(temporary_parent, batch_size=batch, samples=budget)
        final_parent = self.shuffled_dir / stamp
        temporary_parent.replace(final_parent)
        print(f"已生成打乱数据：{final_parent}", flush=True)
        return final_parent

    def training_sample_budget(self, batch_size: int, *, new_data_rows: Optional[int] = None) -> int:
        budget = batch_size * self.config.optimizer.training_steps_per_iteration
        if new_data_rows is not None:
            if isinstance(new_data_rows, bool) or not isinstance(new_data_rows, int) or new_data_rows < 1:
                raise KataGoRLRunnerError("本轮新增训练行必须为正整数")
            budget = min(budget, (4 * new_data_rows // batch_size) * batch_size)
        if budget < batch_size:
            raise KataGoRLRunnerError("本轮新增数据不足以支持一个训练批次")
        return budget

    def train(
        self,
        *,
        smoke: bool = False,
        batch_size: Optional[int] = None,
        initial_checkpoint: Optional[Path] = None,
        new_data_rows: Optional[int] = None,
    ) -> dict[str, int]:
        """Update the policy/value network from the newest shuffled data."""

        self._require_runtime()
        self._ensure_directories()
        if batch_size is None:
            # Four smoke games produce too few rows for a production GPU batch.
            batch_size = min(32, self.config.optimizer.batch_size) if smoke else self.config.optimizer.batch_size
        if batch_size < 1:
            raise KataGoRLRunnerError("训练批次必须至少为 1")
        batch_size = self.training_batch_size(batch_size)
        if not any(self.shuffled_dir.glob("*/train/*.npz")):
            raise KataGoRLRunnerError("没有找到已打乱的训练数据")
        max_samples = batch_size if smoke else self.training_sample_budget(batch_size, new_data_rows=new_data_rows)
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
        if self.config.hardware.device.startswith("cuda:"):
            # A single entry chooses the device without enabling DDP.
            command.extend(["-multi-gpus", self.config.hardware.device.split(":", 1)[1]])
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint = self.run_root / "train" / "gogogo" / "checkpoint.ckpt"
        before = checkpoint_metrics(checkpoint) if checkpoint.is_file() else {}
        self._run(
            command,
            cwd=self.python_source,
            log_name=f"train-{timestamp}.log",
        )
        result = {"training_batch_size": batch_size, "training_requested_samples": max_samples}
        if before.get("trained_samples") is not None:
            after = checkpoint_metrics(checkpoint)
            if after.get("trained_samples") is None:
                raise KataGoRLRunnerError("训练结束后无法核对检查点样本数")
            actual = int(after["trained_samples"] - before["trained_samples"])
            result["training_actual_samples"] = actual
            if not smoke and batch_size >= 128 and actual < max_samples and ReplayRowLedger(self.run_root).sync().physical_rows >= max_samples * 1.5:
                raise KataGoRLRunnerError(f"训练实际只处理 {actual}/{max_samples} 条样本，拒绝发布不足预算的模型")
        return result

    def _batch_fitting_shuffle(self, shuffled: Path, requested: int) -> int:
        """Fit one training epoch into the shuffled files at a new stage."""

        metadata = [
            path.with_suffix(".json")
            for path in (shuffled / "train").glob("*.npz")
            if path.with_suffix(".json").is_file()
        ]
        if not metadata:
            return requested
        rows = [
            int(json.loads(path.read_text(encoding="utf-8"))["num_rows"])
            for path in metadata
        ]
        steps = self.config.optimizer.training_steps_per_iteration
        batch = requested
        while batch > 1 and sum(count // batch for count in rows) < steps:
            batch = max(1, batch // 2)
        if sum(count // batch for count in rows) < steps:
            raise KataGoRLRunnerError(
                f"打乱数据不足以完成 {steps} 个训练批次；各文件行数：{rows}"
            )
        if batch != requested:
            print(f"打乱数据较少，本轮训练批次从 {requested} 暂降到 {batch}", flush=True)
        return batch

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

        if self.hardware_profile is not None and not smoke:
            required_rows = self.config.optimizer.minimum_replay_size if min_rows is None else min_rows
            if required_rows < 1:
                raise KataGoRLRunnerError("打乱窗口的最小行数必须至少为 1")
            try:
                rows = ReplayRowLedger(self.run_root).sync().physical_rows
            except ReplayAccountingError as error:
                raise KataGoRLRunnerError(f"回放数据累计行号不可用：{error}") from error
            if rows == 0:
                raise KataGoRLRunnerError("自我对弈未产生有效训练样本")
            if rows < required_rows:
                # Small machines gather a few games per launch. Keep their data
                # and continue collecting rather than publishing an empty shuffle.
                self.metric_store.sync()
                print(f"回放预热：已累计 {rows}/{required_rows} 行；本轮保留对弈数据，下轮继续采集。", flush=True)
                return

        stage_started = time.perf_counter()
        shuffled = self.shuffle(min_rows=min_rows, smoke=smoke,
                                training_batch_size=training_batch_size,
                                new_data_rows=selfplay_metrics.get("selfplay_data_rows"))
        shuffle_seconds = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        batch = training_batch_size
        if not smoke and isinstance(shuffled, Path):
            batch = batch or self.config.optimizer.batch_size
            fitted = self._batch_fitting_shuffle(shuffled, batch)
            # Once enough replay exists, shrinking the batch would silently cut
            # the epoch budget too. Fail visibly instead of publishing that run.
            if fitted != batch and ReplayRowLedger(self.run_root).sync().physical_rows >= self.training_sample_budget(
                    batch, new_data_rows=selfplay_metrics.get("selfplay_data_rows")) * 1.5:
                raise KataGoRLRunnerError("打乱数据未覆盖完整训练预算，拒绝静默缩小批次")
            batch = fitted
        training_metrics = self.train(
            smoke=smoke,
            batch_size=batch,
            initial_checkpoint=initial_checkpoint,
            new_data_rows=selfplay_metrics.get("selfplay_data_rows"),
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
            **(training_metrics if isinstance(training_metrics, dict) else {}),
            "training_batch_size": batch or self.config.optimizer.batch_size,
            "training_requested_samples": (
                (batch or self.config.optimizer.batch_size) if smoke else self.training_sample_budget(
                    batch or self.config.optimizer.batch_size, new_data_rows=selfplay_metrics.get("selfplay_data_rows"))),
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
        try:
            handle = acquire_file_lock(lock_path)
        except FileLockHeld as error:
            raise KataGoRLRunnerError(
                f"训练目录已被另一个进程占用：{lock_path}"
            ) from error
        try:
            try:
                ensure_training_available(self.run_root)
            except HandoffLockedError as error:
                raise KataGoRLRunnerError(str(error)) from error
            yield
        finally:
            release_file_lock(handle)


__all__ = [
    "DEFAULT_CURRICULUM_PROFILE",
    "DEFAULT_RTX_PROFILE",
    "KATAGO_MODEL_KINDS",
    "KataGoRLRunner",
    "KataGoRLRunnerError",
    "build_selfplay_overrides",
    "format_katago_overrides",
    "model_kind_for_config",
]
