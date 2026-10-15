"""Resolve portable KataGo training settings without importing PyTorch eagerly.

Profiles change concurrency, batch sizes, and numerical precision.  They retain
the model shape, search effort, and promotion rules so an existing training run
can move between Windows computers without changing its learning objective.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import asdict, dataclass, replace
from typing import Optional

from .rl_config import RLTrainingConfig


HARDWARE_TIERS = ("auto", "cpu", "low", "high")


@dataclass(frozen=True)
class HardwareProfile:
    """An actually usable device and conservative throughput limits."""

    tier: str
    device: str
    precision: str
    gpu_name: Optional[str]
    memory_gib: float
    cpu_threads: int
    use_fp16: bool
    reason: str = ""

    @property
    def backend(self) -> str:
        return "eigen" if self.tier == "cpu" else "cuda"

    @property
    def gpu_index(self) -> Optional[int]:
        if not self.device.startswith("cuda"):
            return None
        return int(self.device.partition(":")[2] or "0")

    @property
    def game_threads(self) -> Optional[int]:
        if self.tier == "cpu":
            return min(4, max(1, self.cpu_threads))
        if self.tier == "low":
            return 32
        return None

    @property
    def inference_cache_power(self) -> int:
        return {"cpu": 16, "low": 18, "high": 20}[self.tier]

    @property
    def nn_server_threads(self) -> int:
        return 2 if self.tier == "high" else 1

    def training_batch_limit(self, board_size: int) -> int:
        """Bound training memory while preserving existing high-tier batches."""

        if self.tier == "cpu":
            limits = {9: 16, 13: 8, 19: 4}
        elif self.tier == "low" and self.memory_gib <= 4:
            limits = {9: 64, 13: 32, 19: 16}
        elif self.tier == "low":
            limits = {9: 256, 13: 128, 19: 64}
        else:
            limits = {9: 1024, 13: 2048, 19: 1280}
        try:
            return limits[board_size]
        except KeyError as error:
            raise ValueError("硬件适配只支持 9、13 或 19 路棋盘") from error

    def apply(self, config: RLTrainingConfig) -> RLTrainingConfig:
        """Cap resource use without modifying checkpoints or training rules."""

        hardware = replace(config.hardware, device=self.device, precision=self.precision)
        self_play = config.self_play
        if self.tier == "cpu":
            hardware = replace(
                hardware,
                allow_tf32=False,
                compile_model=False,
                data_loader_workers=0,
                pin_memory=False,
            )
            self_play = replace(
                self_play,
                workers=1,
                games_per_iteration=min(self_play.games_per_iteration, 4),
                inference_batch_size=1,
            )
        elif self.tier == "low":
            hardware = replace(
                hardware,
                compile_model=False,
                data_loader_workers=min(hardware.data_loader_workers, 2),
            )
            self_play = replace(
                self_play,
                workers=min(self_play.workers, 2),
                games_per_iteration=min(self_play.games_per_iteration, 32),
                inference_batch_size=min(self_play.inference_batch_size, 16),
            )
        optimizer = replace(
            config.optimizer,
            batch_size=min(
                config.optimizer.batch_size,
                self.training_batch_limit(config.game.board_size),
            ),
        )
        return replace(config, hardware=hardware, self_play=self_play, optimizer=optimizer)

    def inference_overrides(
        self, *, has_model: bool = True, smoke: bool = False
    ) -> dict[str, object]:
        """Backend-specific KataGo settings, including modest CPU caches."""

        overrides: dict[str, object] = {
            "nnCacheSizePowerOfTwo": min(self.inference_cache_power, 18)
            if smoke
            else self.inference_cache_power,
            "numNNServerThreadsPerModel": 1 if smoke else self.nn_server_threads,
        }
        if self.tier == "cpu":
            overrides["numEigenThreadsPerModel"] = max(1, self.cpu_threads)
        else:
            overrides["cudaDeviceToUse"] = self.gpu_index
            overrides["cudaUseFP16"] = self.use_fp16
            if has_model:
                overrides["cudaUseNHWC"] = self.use_fp16
        return overrides

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cpu_profile(cpu_threads: int, reason: str) -> HardwareProfile:
    return HardwareProfile(
        tier="cpu",
        device="cpu",
        precision="float32",
        gpu_name=None,
        memory_gib=0.0,
        cpu_threads=cpu_threads,
        use_fp16=False,
        reason=reason,
    )


def _unavailable(tier: str, cpu_threads: int, reason: str) -> HardwareProfile:
    if tier != "auto":
        raise ValueError(f"硬件档位 {tier} 需要可用的 CUDA 显卡：{reason}")
    return _cpu_profile(cpu_threads, reason)


def detect_hardware(tier: str = "auto") -> HardwareProfile:
    """Probe CUDA kernels and choose the working GPU with the most VRAM.

    CUDA discovery alone is insufficient: a wheel may list a new or old GPU
    whose architecture its kernels do not support.  A real, synchronized tensor
    operation is required before selecting that device.  Explicit GPU tiers
    fail clearly instead of silently using the CPU.
    """

    if tier not in HARDWARE_TIERS:
        raise ValueError(f"未知硬件档位 {tier!r}；可选：auto、cpu、low、high")
    cpu_threads = max(1, os.cpu_count() or 1)
    if tier == "cpu":
        return _cpu_profile(cpu_threads, "用户指定 CPU 档位")

    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        return _unavailable(tier, cpu_threads, f"无法加载 PyTorch：{error}")
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
    except Exception as error:
        return _unavailable(tier, cpu_threads, f"CUDA 初始化失败：{error}")
    if not available:
        return _unavailable(tier, cpu_threads, "PyTorch 未检测到可用 CUDA")

    candidates = []
    failures = []
    for index in range(count):
        try:
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            probe = torch.ones((2, 2), dtype=torch.float32, device=f"cuda:{index}")
            # Reading the result synchronizes the operation, including failures
            # that would otherwise only be raised during subsequent work.
            (probe * probe).sum().item()
            memory_gib = properties.total_memory / (1024 ** 3)
            candidates.append((memory_gib, index, properties.name, capability))
        except Exception as error:
            failures.append(f"cuda:{index} 无法运行张量运算：{error}")
    if not candidates:
        reason = "；".join(failures) or "未找到可用 CUDA 设备"
        return _unavailable(tier, cpu_threads, reason)

    memory_gib, index, name, capability = max(candidates, key=lambda item: item[0])
    use_fp16 = capability >= (7, 0)
    supports_bf16 = False
    if capability >= (8, 0):
        try:
            with torch.cuda.device(index):
                supports_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            # Older PyTorch versions may omit the BF16 capability helper.
            # These devices can still use their established FP16 kernels.
            supports_bf16 = False
    precision = (
        "amp_bfloat16" if supports_bf16 else "amp_float16" if use_fp16 else "float32"
    )
    resolved_tier = tier
    if resolved_tier == "auto":
        resolved_tier = "high" if memory_gib >= 12 and capability >= (8, 0) else "low"
    reason = (
        f"使用 {name}（{memory_gib:.1f} GiB，CUDA {capability[0]}.{capability[1]}）；"
        f"{'自动检测' if tier == 'auto' else '用户指定'} {resolved_tier} 档位"
    )
    if failures:
        reason += "；" + "；".join(failures)
    return HardwareProfile(
        tier=resolved_tier,
        device=f"cuda:{index}",
        precision=precision,
        gpu_name=name,
        memory_gib=memory_gib,
        cpu_threads=cpu_threads,
        use_fp16=use_fp16,
        reason=reason,
    )
