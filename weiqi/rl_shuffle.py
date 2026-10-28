"""Align shuffled arrays to uniform files without changing training targets."""

import math
from pathlib import Path

from .fileio import atomic_write_json


def align_training_files(root: Path, *, batch_size: int, samples: int) -> dict[str, int]:
    """Make whole-file epoch selection exact for the requested batch count.

    KataGo randomly skips an overflowing final file. Uniform files whose batch
    counts divide the epoch avoid both that undershoot and partial-batch tails.
    Original files and the logical data-row watermark remain intact.
    """
    import numpy as np

    if batch_size < 1 or samples < batch_size or samples % batch_size:
        raise ValueError("训练样本预算必须为整批大小")
    batches_per_file = math.gcd(16, samples // batch_size)
    rows_per_file = batch_size * batches_per_file
    source = root / "train"
    target = root / "aligned-train.tmp"
    target.mkdir()
    carry = None
    input_rows = written = files = 0
    for path in sorted(source.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        count = arrays["globalInputNC"].shape[0]
        if any(value.shape[0] != count for value in arrays.values()):
            raise ValueError("训练数组行数不一致")
        input_rows += count
        if carry is not None:
            if arrays.keys() != carry.keys():
                raise ValueError("训练数组字段不一致")
            arrays = {key: np.concatenate((carry[key], value), axis=0) for key, value in arrays.items()}
        while arrays["globalInputNC"].shape[0] >= rows_per_file:
            output = target / f"batch-{files:05d}.npz"
            np.savez_compressed(output, **{key: value[:rows_per_file] for key, value in arrays.items()})
            atomic_write_json(output.with_suffix(".json"), {"num_rows": rows_per_file})
            arrays = {key: value[rows_per_file:] for key, value in arrays.items()}
            files += 1
            written += rows_per_file
        carry = {key: value.copy() for key, value in arrays.items()}
    if written < samples:
        raise ValueError(f"整批训练文件不足：{written}/{samples} 行")
    result = {"batch_size": batch_size, "batches_per_file": batches_per_file,
              "requested_samples": samples, "input_rows": input_rows,
              "training_rows": written, "remainder_rows": input_rows - written, "files": files}
    source.replace(root / "unaligned-train")
    target.replace(source)
    atomic_write_json(root / "batch-alignment.json", result)
    return result
