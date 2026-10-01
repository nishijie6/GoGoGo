"""Training-batch autotuning with KataGo's ``benchmark_fresh_model.py``."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from .fileio import atomic_open, atomic_write_json, local_timestamp


THROUGHPUT_RE = re.compile(
    r"Throughput:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+samples/s", re.IGNORECASE
)
PEAK_MEMORY_RE = re.compile(
    r"Peak GPU memory(?:\s*\([^)]*\))?:\s*([0-9]+(?:\.[0-9]+)?)\s*GiB",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BenchmarkMeasurement:
    batch_size: int
    success: bool
    throughput: float = 0.0
    peak_gpu_gib: float = math.inf
    oom: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class AutotuneResult:
    selected_batch_size: int
    measurements: tuple[BenchmarkMeasurement, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_batch_size": self.selected_batch_size,
            "measurements": [asdict(item) for item in self.measurements],
            "completed_at": local_timestamp(),
        }


def choose_autotune_batch(
    candidates: Sequence[int],
    benchmark: Callable[[int], BenchmarkMeasurement],
    *,
    max_gpu_memory_gib: float,
    persist_path: Optional[Path] = None,
) -> AutotuneResult:
    """Benchmark candidates, exclude OOM/>limit runs, and persist the fastest."""

    measurements: list[BenchmarkMeasurement] = []
    for batch in candidates:
        try:
            measurement = benchmark(batch)
            if measurement.batch_size != batch:
                raise ValueError("benchmark 返回了不同的 batch_size")
        except Exception as error:  # a failed candidate must not abort tuning
            message = str(error)
            measurement = BenchmarkMeasurement(
                batch_size=batch,
                success=False,
                oom="out of memory" in message.casefold() or "oom" in message.casefold(),
                error=message,
            )
        measurements.append(measurement)
    eligible = [
        item
        for item in measurements
        if item.success
        and not item.oom
        and item.throughput > 0
        and item.peak_gpu_gib <= max_gpu_memory_gib
    ]
    if not eligible:
        raise RuntimeError("没有批次通过 OOM、吞吐和峰值显存筛选")
    winner = max(eligible, key=lambda item: (item.throughput, item.batch_size))
    result = AutotuneResult(winner.batch_size, tuple(measurements))
    if persist_path is not None:
        atomic_write_json(persist_path, result.to_dict())
    return result


def build_benchmark_command(
    *,
    python_executable: Path,
    benchmark_script: Path,
    data_file: Path,
    board_size: int,
    batch_size: int,
    gpu_index: int = 0,
    model_kind: str = "b10c128",
) -> list[str]:
    return [
        str(python_executable),
        str(benchmark_script),
        "-model-kind",
        model_kind,
        "-optimizer",
        "adam",
        "-batch-size",
        str(batch_size),
        "-data",
        str(data_file),
        "-gpu",
        str(gpu_index),
        "-pos-len",
        str(board_size),
        "-num-iters",
        "3",
        "-warmup-iters",
        "1",
        "-mode",
        "trainloop",
        "-use-bf16",
        "-use-tf32-matmul",
        "-no-compile",
    ]


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
        with atomic_open(target, "wb") as output:
            np.savez_compressed(output, **arrays)
    return target


__all__ = [
    "AutotuneResult",
    "BenchmarkMeasurement",
    "build_benchmark_command",
    "choose_autotune_batch",
    "parse_benchmark_output",
    "prepare_benchmark_npz",
]
