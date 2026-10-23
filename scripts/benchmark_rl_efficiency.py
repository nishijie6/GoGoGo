"""Compare fixed-protocol evaluation and self-play throughput in isolated folders."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weiqi.fileio import atomic_write_json, local_timestamp, sha256_file
from weiqi.katago_hardware import detect_hardware
from weiqi.katago_rl import DEFAULT_CURRICULUM_PROFILE, KataGoRLRunner
from weiqi.rl_curriculum import GlobalCurriculumLock
from weiqi.rl_curriculum_runtime import CurriculumRuntime
from weiqi.rl_metrics import parse_sgfs
from weiqi.rl_metrics import checkpoint_metrics, parse_model_name


class GpuMonitor:
    def __init__(self, gpu_index):
        self.index = gpu_index
        self.samples = []
        self.stop = threading.Event()

    def __enter__(self):
        def poll():
            if self.index is None:
                return
            while not self.stop.is_set():
                try:
                    command = ["nvidia-smi", "-i", str(self.index),
                               "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
                    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
                    output = subprocess.check_output(command, text=True, timeout=5, **options)
                    self.samples.append([float(x.strip()) for x in output.split(",")])
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                self.stop.wait(1)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)

    def metrics(self):
        if not self.samples:
            return {}
        return {
            "peak_device_memory_gib": max(row[0] for row in self.samples) / 1024,
            "mean_gpu_utilization": sum(row[1] for row in self.samples) / len(self.samples),
        }


def optimizer_steps(checkpoint):
    """Inspect Adam state on CPU without allocating GPU memory."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    return {str(key): int(value["step"]) for key, value in payload["optimizer"]["state"].items()}


def validate_checkpoint_update(before_path, after_path):
    """Require finite model/optimizer tensors and an actual parameter update."""
    import torch

    before = torch.load(before_path, map_location="cpu", weights_only=False, mmap=True)
    after = torch.load(after_path, map_location="cpu", weights_only=False, mmap=True)
    checked = 0

    def inspect(value, name):
        nonlocal checked
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                checked += 1
                if not torch.isfinite(value).all().item():
                    raise RuntimeError(f"nonfinite checkpoint tensor: {name}")
        elif isinstance(value, dict):
            for key, child in value.items():
                inspect(child, f"{name}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                inspect(child, f"{name}.{index}")

    for key in ("model", "swa_model", "optimizer"):
        inspect(after[key], key)
    if before["model"].keys() != after["model"].keys():
        raise RuntimeError("checkpoint model structure changed during resume")
    changed = sum(not torch.equal(value, after["model"][key]) for key, value in before["model"].items())
    if not changed:
        raise RuntimeError("checkpoint model parameters did not change")
    return {"finite_tensors_checked": checked, "changed_model_tensors": changed}


def validate_training(runner, state, candidate, profile, selfplay_input, output, training_steps=None):
    """Resume the real optimizer in an isolated copy using new benchmark data."""
    bench = KataGoRLRunner(runner.config_path, run_root_override=output / "run",
                           hardware_profile=profile)
    if training_steps is not None:
        bench.config = replace(bench.config, optimizer=replace(
            bench.config.optimizer, training_steps_per_iteration=training_steps))
    # Copy the retained replay and its ledger together, preserving the logical
    # row offset and allowing the official train bucket to advance normally.
    ledger_path = runner.run_root / "replay_rows.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    sources = [runner.run_root / name for name in ledger["files"]]
    for source in sources:
        relative = source.resolve().relative_to(runner.run_root)
        target = bench.run_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(ledger_path, bench.run_root / "replay_rows.json")
    source_model = selfplay_input / "models" / str(state.active.latest_model) / "model.bin.gz"
    if sha256_file(source_model) != sha256_file(candidate):
        raise RuntimeError("self-play benchmark used a different model")
    for source in selfplay_input.glob("selfplay/*/tdata/*.npz"):
        relative = source.relative_to(selfplay_input)
        target = bench.run_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    source_checkpoint = runner.run_root / "train/gogogo/checkpoint.ckpt"
    source_hash = sha256_file(source_checkpoint)
    before = checkpoint_metrics(source_checkpoint)
    expected = parse_model_name(str(state.active.latest_model))
    if (before.get("trained_samples"), before.get("data_rows")) != expected:
        raise RuntimeError("source checkpoint and curriculum model counters disagree")
    steps_before = optimizer_steps(source_checkpoint)
    checkpoint = bench.run_root / "train/gogogo/checkpoint.ckpt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copy2(source_checkpoint, checkpoint)
    target_model = bench.models_dir / str(state.active.latest_model) / "model.bin.gz"
    target_model.parent.mkdir(parents=True)
    shutil.copy2(candidate, target_model)
    batch_size = state.active.autotune_batch or runner.config.optimizer.batch_size
    timings = {}
    gpu_metrics = {}
    started = time.perf_counter()
    bench.shuffle()
    timings["shuffle_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    new_rows = json.loads((bench.run_root / "replay_rows.json").read_text(encoding="utf-8"))["logical_rows"] - before["data_rows"]
    with GpuMonitor(profile.gpu_index) as monitor:
        bench.train(batch_size=batch_size, new_data_rows=new_rows)
    gpu_metrics.update(monitor.metrics())
    timings["train_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    exported = bench.export_new_models()
    timings["export_seconds"] = time.perf_counter() - started
    after = checkpoint_metrics(checkpoint)
    if not exported or after["trained_samples"] <= before["trained_samples"] or after["data_rows"] <= before["data_rows"]:
        raise RuntimeError("isolated resume failed to advance training and data counters")
    if sha256_file(source_checkpoint) != source_hash:
        raise RuntimeError("source checkpoint changed during isolated validation")
    sample_delta = after["trained_samples"] - before["trained_samples"]
    batches, remainder = divmod(sample_delta, batch_size)
    steps_after = optimizer_steps(checkpoint)
    if remainder or not steps_before or steps_before.keys() != steps_after.keys() or any(
        steps_after[key] - step != batches for key, step in steps_before.items()
    ):
        raise RuntimeError("isolated resume did not preserve and advance every optimizer step")
    health = validate_checkpoint_update(source_checkpoint, checkpoint)
    if any(not math.isfinite(value) for value in after.values()):
        raise RuntimeError("isolated resume produced nonfinite training metrics")
    return {
        "mode": "training_validation", "source_checkpoint_sha256": source_hash,
        "trained_samples_before": before["trained_samples"], "trained_samples_after": after["trained_samples"],
        "new_data_rows": after["data_rows"] - before["data_rows"],
        "sample_reuse_ratio": sample_delta / (after["data_rows"] - before["data_rows"]),
        "batch_size": batch_size, "requested_training_steps": bench.config.optimizer.training_steps_per_iteration,
        "actual_training_batches": batches, "optimizer_state_entries": len(steps_before),
        "optimizer_step_before": min(steps_before.values()), "optimizer_step_after": min(steps_after.values()),
        "exported_models": [p.name for p in exported], "checkpoint_metrics": after,
        **health, **timings, **gpu_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluation", "selfplay", "training", "all"), default="all")
    parser.add_argument("--curriculum-config", type=Path, default=DEFAULT_CURRICULUM_PROFILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--threads", type=int, nargs="+", default=[128, 256],
                        help="self-play game threads to compare (multiples of 16, 32 to 256)")
    parser.add_argument("--training-steps", type=int,
                        help="override training budget only in the isolated validation")
    parser.add_argument("--selfplay-input", type=Path, help="isolated self-play run for --mode training")
    args = parser.parse_args()
    if args.pairs < 1 or args.games < 1:
        parser.error("pairs and games must be positive")
    if len(set(args.threads)) != len(args.threads) or any(
        threads < 32 or threads > 256 or threads % 16 for threads in args.threads
    ):
        parser.error("threads must be distinct multiples of 16 from 32 to 256")
    if args.training_steps is not None and args.training_steps < 1:
        parser.error("training-steps must be positive")
    if args.mode == "training" and args.selfplay_input is None:
        parser.error("--mode training requires --selfplay-input")
    runtime = CurriculumRuntime(args.curriculum_config)
    # Refuse to contend with the real training process or its retention pass.
    with GlobalCurriculumLock(runtime.state_root):
        state = runtime.store.load()
        if state is None:
            parser.error("curriculum state does not exist")
        profile = detect_hardware()
        runtime.hardware_profile = profile
        runner = runtime._stage_runner(state)
        with runner.lock():
            candidate = runtime._model_path(runner, state.active.latest_model)
            candidate_hash = sha256_file(candidate)
            output = args.output or PROJECT_ROOT / "training_runs" / "efficiency-benchmarks" / datetime.now().strftime("%Y%m%d-%H%M%S")
            output = output.resolve()
            output.mkdir(parents=True, exist_ok=False)
            summary = {
                "created_at": local_timestamp(), "candidate_model": state.active.latest_model,
                "candidate_sha256": candidate_hash, "board_size": state.active.board_size,
                "hardware": profile.to_dict(), "results": [],
            }

            def record(row):
                summary["results"].append(row)
                atomic_write_json(output / "summary.json", summary)
                print("BENCHMARK_RESULT " + json.dumps(row, ensure_ascii=False), flush=True)

            if args.mode in {"all", "evaluation"}:
                # The deterministic generator produces the same first N openings
                # in a separate suite; never rewrite the live 100-opening suite.
                runtime.state_root = output / "evaluation-state"
                baseline = state.active.champion_model or state.active.baseline_model
                for parallel in (1, 4):
                    runtime.config = replace(runtime.config, evaluation_games=2 * args.pairs,
                                             evaluation_parallel_pairs=parallel)
                    started = time.perf_counter()
                    with GpuMonitor(profile.gpu_index) as monitor:
                        result = runtime._match_opponent(
                            state, runner, state.active.latest_model, baseline, ["benchmark"],
                            output / f"evaluation-p{parallel}",
                        )
                    seconds = time.perf_counter() - started
                    record({"mode": "evaluation", "parallel_pairs": runtime._parallel_pairs(),
                            "seconds": seconds, "games_per_hour": result.games * 3600 / seconds,
                            "visits": runner.config.evaluation.simulations_per_move,
                            **result.to_dict(), **monitor.metrics()})
            if args.mode in {"all", "selfplay"}:
                for threads in args.threads:
                    bench = KataGoRLRunner(runner.config_path, run_root_override=output / f"selfplay-t{threads}",
                                           hardware_profile=profile)
                    bench.config = replace(bench.config, self_play=replace(
                        bench.config.self_play, workers=threads // 16))
                    target = bench.models_dir / str(state.active.latest_model) / "model.bin.gz"
                    target.parent.mkdir(parents=True)
                    shutil.copy2(candidate, target)
                    started = time.perf_counter()
                    with GpuMonitor(profile.gpu_index) as monitor:
                        metrics = bench.selfplay(games=args.games)
                    seconds = time.perf_counter() - started
                    health = parse_sgfs(list(bench.selfplay_dir.glob("*/sgfs/*.sgfs")))
                    if metrics.get("selfplay_games") != args.games or health["sgf_games"] != args.games:
                        raise RuntimeError("benchmark did not finish the requested self-play games")
                    if health["damaged_games"] or health["no_result_games"]:
                        raise RuntimeError("benchmark produced damaged or unfinished games")
                    record({"mode": "selfplay", "game_threads": threads, "requested_games": args.games,
                            "visits": bench.config.search.simulations_per_move, "seconds": seconds,
                            "new_data_rows_per_hour": metrics["selfplay_data_rows"] * 3600 / seconds,
                            **metrics, **health, **monitor.metrics()})
            if args.mode == "training":
                record(validate_training(runner, state, candidate, profile,
                                         args.selfplay_input.resolve(), output, args.training_steps))
            if sha256_file(candidate) != candidate_hash:
                raise RuntimeError("source model changed during benchmark")
            print(f"Benchmark saved: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
