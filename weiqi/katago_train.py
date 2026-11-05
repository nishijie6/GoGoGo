"""Run an unmodified official KataGo trainer on the explicitly selected device.

This launcher intentionally imports Torch only after configuring MPS fallback.
Official trainer arguments are forwarded verbatim after the script path.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys
from typing import Sequence


def run_training(
    script: Path,
    device: str,
    argv: Sequence[str],
    memory_fraction: float = 1.0,
) -> None:
    """Execute the official trainer in this interpreter with enforced selection.

    CUDA indices must also match the official ``-multi-gpus`` argument (whose
    default is 0). Multi-process training is deliberately outside this launcher's
    single-device contract. Discovery patches and interpreter state are restored
    even if the official trainer exits or raises.
    """
    if device not in ("cpu", "mps") and re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError(f"Unsupported training device: {device!r}")
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be finite and in (0, 1]")
    script = Path(script).absolute()
    if not script.is_file():
        raise FileNotFoundError(script)
    arguments = list(argv)
    official_gpu_ids = "0"
    for index, argument in enumerate(arguments):
        if argument == "-multi-gpus":
            if index + 1 >= len(arguments):
                raise ValueError("-multi-gpus requires a device index")
            official_gpu_ids = arguments[index + 1]
        elif argument.startswith("-multi-gpus="):
            official_gpu_ids = argument.partition("=")[2]
    expected_gpu_id = device.partition(":")[2] if device.startswith("cuda:") else "0"

    with ExitStack() as cleanup:
        if device == "mps":
            previous_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            if previous_fallback is None:
                cleanup.callback(os.environ.pop, "PYTORCH_ENABLE_MPS_FALLBACK", None)
            else:
                cleanup.callback(os.environ.__setitem__, "PYTORCH_ENABLE_MPS_FALLBACK", previous_fallback)
        torch = importlib.import_module("torch")
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Requested MPS training device is unavailable")
            torch.mps.set_per_process_memory_fraction(memory_fraction)
            # Allocation alone does not prove the accelerator can execute kernels.
            probe = torch.ones(4, device="mps")
            (probe * probe).sum().item()
        elif device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA training device is unavailable")
            cuda_index = int(device.partition(":")[2])
            if cuda_index >= torch.cuda.device_count():
                raise ValueError(f"CUDA index {cuda_index} is outside the available device count")

        if official_gpu_ids != expected_gpu_id:
            raise ValueError(
                f"Official -multi-gpus {official_gpu_ids!r} disagrees with selected device {device}; "
                f"use -multi-gpus {expected_gpu_id}"
            )

        def suppress_discovery(owner, name):
            original = getattr(owner, name)
            setattr(owner, name, lambda: False if name == "is_available" else 0)
            cleanup.callback(setattr, owner, name, original)

        if device in ("cpu", "mps"):
            suppress_discovery(torch.cuda, "is_available")
            suppress_discovery(torch.cuda, "device_count")
        if device != "mps":
            suppress_discovery(torch.backends.mps, "is_available")

        def verify_training_device(module, inputs):
            if not (module.training and type(module).__name__ == "Model"
                    and type(module).__module__ == "katago.train.model_pytorch"):
                return
            if not inputs or not hasattr(inputs[0], "device"):
                raise RuntimeError("Cannot verify official Model training input device")
            actual = inputs[0].device
            matches = actual.type == device.partition(":")[0]
            if device.startswith("cuda:"):
                matches = matches and actual.index == int(device.partition(":")[2])
            if not matches:
                raise RuntimeError(f"Official training input device mismatch: expected {device}, got {actual}")
            print("KATAGO_TRAIN_FORWARD_DEVICE=" + json.dumps({"device": device}), flush=True)
            hook.remove()

        hook = torch.nn.modules.module.register_module_forward_pre_hook(verify_training_device)
        cleanup.callback(hook.remove)
        previous_argv, previous_path = sys.argv, sys.path
        cleanup.callback(setattr, sys, "argv", previous_argv)
        cleanup.callback(setattr, sys, "path", previous_path)
        cleanup.callback(previous_path.__setitem__, slice(None), list(previous_path))
        sys.argv = [str(script), *arguments]
        sys.path.insert(0, str(script.parent))
        print("KATAGO_TRAIN_DEVICE=" + json.dumps({"device": device}), flush=True)
        runpy.run_path(str(script), run_name="__main__")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--memory-fraction", type=float, default=1.0)
    parser.add_argument("script", type=Path)
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    run_training(args.script, args.device, args.official_args, args.memory_fraction)


if __name__ == "__main__":
    main()
