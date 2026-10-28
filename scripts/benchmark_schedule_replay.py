"""Auditable, isolated replay-window and evaluation-schedule experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from weiqi.fileio import atomic_write_json, local_timestamp, sha256_file
from weiqi.katago_hardware import detect_hardware
from weiqi.katago_rl import KataGoRLRunner
from weiqi.replay_accounting import ReplayRowLedger
from weiqi.rl_curriculum import GlobalCurriculumLock
from weiqi.rl_curriculum_runtime import CurriculumRuntime
from weiqi.rl_metrics import checkpoint_metrics
from scripts.benchmark_rl_efficiency import GpuMonitor, optimizer_steps, validate_checkpoint_update


class MemoryMonitor:
    """Sample the process tree and system headroom, including shuffle workers."""

    def __enter__(self):
        import psutil
        self.stop = threading.Event()
        self.peak = 0
        self.minimum_available = psutil.virtual_memory().available
        process = psutil.Process()

        def poll():
            while not self.stop.is_set():
                total = 0
                for child in [process, *process.children(recursive=True)]:
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, total)
                self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
                self.stop.wait(0.5)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)

    def metrics(self):
        return {"peak_process_tree_working_set_gib": self.peak / 1024**3,
                "minimum_system_available_gib": self.minimum_available / 1024**3}


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in sorted(data.files)}


def game_id(target):
    # Six exactly represented float32 words encode the C++ writer's 128-bit ID.
    return np.ascontiguousarray(target[41:47], dtype=np.float32).tobytes()


def row_digest(arrays, index):
    digest = hashlib.blake2b(digest_size=16)
    for key in sorted(arrays):
        digest.update(arrays[key][index].tobytes())
    return digest.digest()


def held_out_game(target, threshold):
    return hashlib.blake2b(game_id(target), digest_size=8).digest()[0] < threshold


class ArrayWriter:
    def __init__(self, directory, rows_per_file=20000):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.rows_per_file = rows_per_file
        self.parts = []
        self.pending = self.rows = self.files = 0

    def add(self, arrays, indices):
        if len(indices):
            self.parts.append({key: value[indices].copy() for key, value in arrays.items()})
            self.pending += len(indices)
        if self.pending >= self.rows_per_file:
            self.flush()

    def flush(self):
        if not self.parts:
            return
        arrays = {key: np.concatenate([part[key] for part in self.parts]) for key in self.parts[0]}
        for start in range(0, self.pending, self.rows_per_file):
            path = self.directory / f"data-{self.files:05d}.npz"
            count = min(self.rows_per_file, self.pending - start)
            np.savez_compressed(path, **{key: value[start:start + count] for key, value in arrays.items()})
            # Newest source rows are emitted first; preserve ordering for shuffle.
            stamp = 1700000000 - self.files
            os.utime(path, (stamp, stamp))
            self.files += 1
            self.rows += count
        self.parts.clear()
        self.pending = 0


def prepare_pool(output):
    """Recover unique historical rows from retained raw/shuffle snapshots."""
    if (output / "pool/history").exists():
        raise FileExistsError("use a fresh experiment directory; do not overwrite a replay pool")
    source = ROOT / "training_runs/19x19"
    ledger = json.loads((source / "replay_rows.json").read_text(encoding="utf-8"))
    paths = [source / name for name, _ in sorted(
        ledger["files"].items(), key=lambda pair: pair[1]["mtime_ns"], reverse=True)]
    for snapshot in sorted((source / "shuffleddata").iterdir(), reverse=True):
        # Alignment preserves originals; never count aligned copies twice.
        paths.extend(sorted((snapshot / "unaligned-train").glob("*.npz")))
    history = ArrayWriter(output / "pool/history")
    validation = ArrayWriter(output / "pool/historical-holdout")
    seen = set()
    used = []
    signature = None
    accepted = duplicates = withheld = 0
    for path in paths:
        arrays = read_arrays(path)
        current = {key: (str(value.dtype), value.shape[1:]) for key, value in arrays.items()}
        if signature is None:
            signature = current
        if current != signature:
            raise RuntimeError(f"incompatible historical data: {path}")
        keep, val = [], []
        for index, target in enumerate(arrays["globalTargetsNC"]):
            key = row_digest(arrays, index)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            if held_out_game(target, 3):
                if withheld < 15000:
                    val.append(index)
                    withheld += 1
            elif accepted < 1000000:
                keep.append(index)
                accepted += 1
        history.add(arrays, keep)
        validation.add(arrays, val)
        used.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                     "training_rows": len(keep), "holdout_rows": len(val)})
        print(f"POOL unique={accepted} duplicates={duplicates} holdout={withheld}", flush=True)
        if accepted >= 1000000:
            break
    history.flush()
    validation.flush()
    if accepted != 1000000:
        raise RuntimeError(f"only {accepted} distinct historical rows available")
    manifest = {"created_at": local_timestamp(), "unique_training_rows": accepted,
                "duplicate_rows_removed": duplicates, "historical_holdout_rows": withheld,
                "holdout_group": "128-bit game hash", "sources": used,
                "note": "historical holdout may have been seen by the starting checkpoint"}
    atomic_write_json(output / "pool/manifest.json", manifest)


def prepare_fresh(output):
    source = output / "fresh-data/selfplay-t256/selfplay"
    train = ArrayWriter(output / "pool/fresh-train")
    validation = ArrayWriter(output / "pool/fresh-holdout")
    groups = {"train": set(), "holdout": set()}
    for path in sorted(source.glob("*/tdata/*.npz")):
        arrays = read_arrays(path)
        keep, val = [], []
        for i, target in enumerate(arrays["globalTargetsNC"]):
            split = "holdout" if held_out_game(target, 26) else "train"
            groups[split].add(game_id(target))
            (val if split == "holdout" else keep).append(i)
        train.add(arrays, keep)
        validation.add(arrays, val)
    train.flush()
    validation.flush()
    if groups["train"] & groups["holdout"] or validation.rows < 2560:
        raise RuntimeError("fresh validation split is too small or leaks games")
    result = {"training_rows": train.rows, "holdout_rows": validation.rows,
              "training_games": len(groups["train"]), "holdout_games": len(groups["holdout"]),
              "game_overlap": 0}
    atomic_write_json(output / "pool/fresh-manifest.json", result)
    return result


def train_windows(output):
    runtime = CurriculumRuntime(ROOT / "config/rl_curriculum.rtx5070ti.json")
    with GlobalCurriculumLock(runtime.state_root):
        state = runtime.store.load()
        profile = detect_hardware()
        runtime.hardware_profile = profile
        source = runtime._stage_runner(state)
        with source.lock():
            frozen = json.loads((output / "fresh-data/summary.json").read_text(encoding="utf-8"))
            if (frozen["candidate_model"] != state.active.latest_model or frozen["candidate_sha256"] !=
                    sha256_file(runtime._model_path(source, state.active.latest_model))):
                raise RuntimeError("fresh data and source checkpoint used different models")
            fresh = prepare_fresh(output)
            source_checkpoint = source.run_root / "train/gogogo/checkpoint.ckpt"
            source_hash = sha256_file(source_checkpoint)
            before = checkpoint_metrics(source_checkpoint)
            old_steps = optimizer_steps(source_checkpoint)
            batch = state.active.autotune_batch
            steps = min(96, (4 * fresh["training_rows"] // batch) // 16 * 16)
            if steps < 64:
                raise RuntimeError("insufficient fresh data for a useful comparison")
            results = []
            for window in (500000, 1000000):
                runner = KataGoRLRunner(source.config_path, run_root_override=output / f"window-{window}",
                                        hardware_profile=profile)
                runner.config = replace(runner.config, optimizer=replace(
                    runner.config.optimizer, replay_buffer_capacity=window, training_steps_per_iteration=steps))
                checkpoint = runner.run_root / "train/gogogo/checkpoint.ckpt"
                checkpoint.parent.mkdir(parents=True)
                shutil.copy2(source_checkpoint, checkpoint)
                model = runner.models_dir / state.active.latest_model / "model.bin.gz"
                model.parent.mkdir(parents=True)
                shutil.copy2(runtime._model_path(source, state.active.latest_model), model)
                records = {}
                physical = 0
                for path in sorted((output / "pool/history").glob("*.npz")):
                    arrays = read_arrays(path)
                    count = len(arrays["globalInputNC"])
                    if physical >= window:
                        break
                    count = min(count, window - physical)
                    target = runner.selfplay_dir / "historical/tdata" / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if count == len(arrays["globalInputNC"]):
                        shutil.copy2(path, target)
                    else:
                        np.savez_compressed(target, **{key: value[:count] for key, value in arrays.items()})
                        os.utime(target, (path.stat().st_mtime, path.stat().st_mtime))
                    stat = target.stat()
                    records[target.relative_to(runner.run_root).as_posix()] = {
                        "rows": count, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                    physical += count
                atomic_write_json(runner.run_root / "replay_rows.json", {
                    "schema_version": 1, "deleted_rows_offset": before["data_rows"] - physical,
                    "physical_rows": physical, "logical_rows": before["data_rows"],
                    "initialized_at": local_timestamp(), "files": records})
                for path in sorted((output / "pool/fresh-train").glob("*.npz")):
                    target = runner.selfplay_dir / "fresh/tdata" / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    os.utime(target, None)
                snapshot = ReplayRowLedger(runner.run_root).sync()
                assert snapshot.logical_rows - before["data_rows"] == fresh["training_rows"]
                started = time.perf_counter()
                with GpuMonitor(profile.gpu_index) as monitor, MemoryMonitor() as memory:
                    shuffled = runner.shuffle(training_batch_size=batch, new_data_rows=fresh["training_rows"])
                shuffle_seconds = time.perf_counter() - started
                val = shuffled / "val"
                val.mkdir()
                for path in (output / "pool/fresh-holdout").glob("*.npz"):
                    shutil.copy2(path, val / path.name)
                started = time.perf_counter()
                with GpuMonitor(profile.gpu_index) as train_monitor:
                    runner.train(batch_size=batch, new_data_rows=fresh["training_rows"])
                train_seconds = time.perf_counter() - started
                after = checkpoint_metrics(checkpoint)
                new_steps = optimizer_steps(checkpoint)
                if after["trained_samples"] - before["trained_samples"] != steps * batch or any(
                    new_steps[key] - value != steps for key, value in old_steps.items()
                ):
                    raise RuntimeError("comparison did not preserve equal training budgets/optimizer state")
                health = validate_checkpoint_update(source_checkpoint, checkpoint)
                exported = runner.export_new_models()
                result = {"window_rows": window, "batch_size": batch, "training_batches": steps,
                          "new_training_rows": fresh["training_rows"], "shuffle_seconds": shuffle_seconds,
                          "train_seconds": train_seconds, "checkpoint_metrics": after,
                          "shuffle_memory": memory.metrics(),
                          "exported_models": [p.name for p in exported], "optimizer_entries": len(old_steps),
                          "train_gpu": train_monitor.metrics(), **health}
                results.append(result)
                atomic_write_json(output / "replay-results.json", {"results": results, "fresh": fresh})
                print("REPLAY_RESULT " + json.dumps(result), flush=True)
                # A second independent training seed on the identical shuffle
                # distinguishes a window effect from one lucky short update.
                replica = KataGoRLRunner(source.config_path,
                    run_root_override=output / f"window-{window}-replica2", hardware_profile=profile)
                replica.config = runner.config
                replica_checkpoint = replica.run_root / "train/gogogo/checkpoint.ckpt"
                replica_checkpoint.parent.mkdir(parents=True)
                shutil.copy2(source_checkpoint, replica_checkpoint)
                shutil.copytree(shuffled, replica.shuffled_dir / "frozen-shuffle", copy_function=os.link)
                started = time.perf_counter()
                replica.train(batch_size=batch, new_data_rows=fresh["training_rows"])
                replica_after = checkpoint_metrics(replica_checkpoint)
                if replica_after["trained_samples"] - before["trained_samples"] != steps * batch:
                    raise RuntimeError("replica training did not complete the same sample budget")
                replica_steps = optimizer_steps(replica_checkpoint)
                if any(replica_steps[key] - value != steps for key, value in old_steps.items()):
                    raise RuntimeError("replica optimizer steps disagree")
                replica_health = validate_checkpoint_update(source_checkpoint, replica_checkpoint)
                result["replica2"] = {"train_seconds": time.perf_counter() - started,
                                      "checkpoint_metrics": replica_after, **replica_health}
                atomic_write_json(output / "replay-results.json", {"results": results, "fresh": fresh})
            if sha256_file(source_checkpoint) != source_hash:
                raise RuntimeError("source checkpoint changed during experiment")


def evaluate_holdouts(output):
    """Compare deployed SWA networks on identical game-disjoint cohorts."""
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(ROOT / "katago/source/python"))
    from katago.train import load_model, data_processing_pytorch

    torch.set_num_threads(4)
    checkpoints = {"starting_checkpoint": output / "before/training_runs/19x19/train/gogogo/checkpoint.ckpt"}
    checkpoints.update({str(window): output / f"matched-{window}-r1/train/gogogo/checkpoint.ckpt"
                        for window in (500000, 1000000)})
    checkpoints.update({f"{window}-replica2": output / f"matched-{window}-r2/train/gogogo/checkpoint.ckpt"
                        for window in (500000, 1000000)})
    results = {}
    for name, checkpoint in checkpoints.items():
        raw, model, _ = load_model.load_model(str(checkpoint), use_swa=True, device="cuda", pos_len=19)
        model.eval()
        cohorts = {}
        for cohort in ("fresh-holdout", "historical-holdout"):
            sums = dict(policy_ce=0.0, value_ce=0.0, policy_accuracy=0.0, score_mse=0.0,
                        policy_weight=0.0, value_weight=0.0, score_weight=0.0)
            rows = 0
            games = {}
            files = sorted((output / "pool" / cohort).glob("*.npz"))
            for batch in data_processing_pytorch.read_npz_training_data(
                [str(path) for path in files], batch_size=256, world_size=1, rank=0,
                pos_len=19, device="cuda", randomize_symmetries=False,
                include_meta=model.get_has_metadata_encoder(), model_config=model.config,
            ):
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model.postprocess_output(model(batch["binaryInputNCHW"], batch["globalInputNC"]))[0]
                policy, value = prediction[:2]
                target = batch["globalTargetsNC"]
                target_policy = batch["policyTargetsNCMove"][:, 0, :].float()
                target_policy /= target_policy.sum(dim=1, keepdim=True).clamp_min(1e-20)
                pw = target[:, 25] * target[:, 26]
                vw = target[:, 25] * (1 - target[:, 35])
                sw = target[:, 25] * target[:, 27]
                policy_ce = -(target_policy * F.log_softmax(policy[:, 0, :].float(), dim=1)).sum(dim=1)
                value_ce = -(target[:, :3] * F.log_softmax(value.float(), dim=1)).sum(dim=1)
                correct = (policy[:, 0, :].argmax(dim=1) == target_policy.argmax(dim=1)).float()
                score_mse = (prediction[8].float() - target[:, 3]).square()
                targets_cpu = target.cpu().numpy()
                p_num, p_den = (policy_ce * pw).cpu().numpy(), pw.cpu().numpy()
                v_num, v_den = (value_ce * vw).cpu().numpy(), vw.cpu().numpy()
                for i, row in enumerate(targets_cpu):
                    entry = games.setdefault(game_id(row).hex(), [0.0, 0.0, 0.0, 0.0])
                    for j, amount in enumerate((p_num[i], p_den[i], v_num[i], v_den[i])):
                        entry[j] += float(amount)
                for key, values, weights in (("policy_ce", policy_ce, pw), ("value_ce", value_ce, vw),
                                             ("policy_accuracy", correct, pw), ("score_mse", score_mse, sw)):
                    sums[key] += (values * weights).sum().item()
                sums["policy_weight"] += pw.sum().item()
                sums["value_weight"] += vw.sum().item()
                sums["score_weight"] += sw.sum().item()
                rows += len(target)
            cohorts[cohort] = {"rows": rows,
                "policy_ce": sums["policy_ce"] / sums["policy_weight"],
                "value_ce": sums["value_ce"] / sums["value_weight"],
                "policy_accuracy": sums["policy_accuracy"] / sums["policy_weight"],
                "score_rmse": (sums["score_mse"] / sums["score_weight"])**0.5,
                "per_game_policy_value_sums_weights": games}
        results[name] = cohorts
        atomic_write_json(output / "holdout-results.json", results)
        print("HOLDOUT_RESULT " + json.dumps({name: {key: {k: v for k, v in value.items()
            if k != "per_game_policy_value_sums_weights"} for key, value in cohorts.items()}}), flush=True)
        del raw, model
        torch.cuda.empty_cache()


def train_matched_probes(output):
    """Match production updates/game: 512/2048 games imply 64/256 batches.

    The preceding 96-batch runs provide an additional higher-reuse stress test.
    Reusing immutable shuffles avoids timing another CPU preparation pass.
    """
    runtime = CurriculumRuntime(ROOT / "config/rl_curriculum.rtx5070ti.json")
    with GlobalCurriculumLock(runtime.state_root):
        state = runtime.store.load()
        profile = detect_hardware()
        runtime.hardware_profile = profile
        source = runtime._stage_runner(state)
        with source.lock():
            checkpoint = source.run_root / "train/gogogo/checkpoint.ckpt"
            source_hash = sha256_file(checkpoint)
            before = checkpoint_metrics(checkpoint)
            steps_before = optimizer_steps(checkpoint)
            fresh = json.loads((output / "pool/fresh-manifest.json").read_text(encoding="utf-8"))
            batch = state.active.autotune_batch
            results = []
            for window in (500000, 1000000):
                frozen = next((output / f"window-{window}/shuffleddata").iterdir())
                for replica in (1, 2):
                    runner = KataGoRLRunner(source.config_path,
                        run_root_override=output / f"matched-{window}-r{replica}", hardware_profile=profile)
                    runner.config = replace(runner.config, optimizer=replace(runner.config.optimizer,
                        replay_buffer_capacity=window, training_steps_per_iteration=64))
                    target = runner.run_root / "train/gogogo/checkpoint.ckpt"
                    target.parent.mkdir(parents=True)
                    shutil.copy2(checkpoint, target)
                    shutil.copytree(frozen, runner.shuffled_dir / "frozen-shuffle", copy_function=os.link)
                    started = time.perf_counter()
                    runner.train(batch_size=batch, new_data_rows=fresh["training_rows"])
                    after = checkpoint_metrics(target)
                    steps_after = optimizer_steps(target)
                    if (after["trained_samples"] - before["trained_samples"] != 64 * batch or any(
                        steps_after[key] - value != 64 for key, value in steps_before.items())):
                        raise RuntimeError("matched probe did not preserve production updates/game")
                    row = {"window_rows": window, "replica": replica, "training_batches": 64,
                           "training_samples": 64 * batch, "train_seconds": time.perf_counter() - started,
                           **validate_checkpoint_update(checkpoint, target)}
                    results.append(row)
                    atomic_write_json(output / "matched-results.json", results)
                    print("MATCHED_RESULT " + json.dumps(row), flush=True)
            if sha256_file(checkpoint) != source_hash:
                raise RuntimeError("source checkpoint changed during matched probes")


def benchmark_schedule(output, pairs):
    """Measure unchanged match work, then replay actual sample boundaries."""
    if (output / "schedule/match").exists():
        raise FileExistsError("timing requires new matches, not reused opening-pair results")
    runtime = CurriculumRuntime(ROOT / "config/rl_curriculum.rtx5070ti.json")
    with GlobalCurriculumLock(runtime.state_root):
        state = runtime.store.load()
        profile = detect_hardware()
        runtime.hardware_profile = profile
        runner = runtime._stage_runner(state)
        with runner.lock():
            # Identical deterministic opening/color pairs; never consume live cache.
            runtime.state_root = output / "schedule/opening-state"
            runtime.config = replace(runtime.config, evaluation_games=pairs * 2)
            started = time.perf_counter()
            with GpuMonitor(profile.gpu_index) as gpu, MemoryMonitor() as memory:
                match = runtime._match_opponent(state, runner, state.active.latest_model,
                    state.active.champion_model, ["benchmark"], output / "schedule/match")
            seconds = time.perf_counter() - started
            records = [json.loads(line) for line in runner.metric_store.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            records = [row for row in records if row.get("selfplay_games", 0) >= 2000][-5:]
            scheduled = []
            for every_cycle in (True, False):
                initial = int(records[0]["trained_samples"] - records[0]["sample_delta"])
                boundary = runtime.controller.next_evaluation_sample(state.active_stage_index, initial)
                opponents = 0
                cycles = []
                for record in records:
                    samples = record["trained_samples"]
                    due = samples >= boundary
                    matches = 2 if due else int(every_cycle)
                    if due:
                        boundary = runtime.controller.next_evaluation_sample(state.active_stage_index, samples)
                    opponents += matches
                    cycles.append({"trained_samples": samples, "opponent_matches": matches,
                                   "fixed_gate_due": due, "next_fixed_gate": boundary})
                training = sum(row["cycle_seconds"] for row in records)
                evaluation = opponents * seconds * 200 / (pairs * 2)
                scheduled.append({"champion_every_cycle": every_cycle, "cycles": cycles,
                    "opponent_matches": opponents, "training_seconds": training,
                    "projected_evaluation_seconds": evaluation, "projected_total_seconds": training + evaluation})
            report = {"created_at": local_timestamp(), "short_match": match.to_dict(),
                      "short_match_seconds": seconds, "games": pairs * 2,
                      "visits": runner.config.evaluation.simulations_per_move,
                      "parallel_pairs": runtime._parallel_pairs(), "inference_batch_size": runtime.config.evaluation_inference_batch_size,
                      "projection_note": "200-game costs extrapolated from short match; schedule replay uses actual sample counters",
                      "schedule_replay": scheduled, "gpu": gpu.metrics(), "memory": memory.metrics()}
            atomic_write_json(output / "schedule-results.json", report)
            print("SCHEDULE_RESULT " + json.dumps(report), flush=True)


def analyze_holdouts(output):
    results = json.loads((output / "holdout-results.json").read_text(encoding="utf-8"))
    report = {"note": "Paired game-bootstrap conditional on two trained replicas per window; short-run proxy, not an Elo test",
              "cohorts": {}}
    for cohort in ("fresh-holdout", "historical-holdout"):
        names = ("500000", "500000-replica2", "1000000", "1000000-replica2")
        cohorts = [results[name][cohort] for name in names]
        games = sorted(cohorts[0]["per_game_policy_value_sums_weights"])
        if any(set(item["per_game_policy_value_sums_weights"]) != set(games) for item in cohorts):
            raise RuntimeError("models were not evaluated on the same held-out games")
        matrices = [np.array([item["per_game_policy_value_sums_weights"][game] for game in games]) for item in cohorts]
        a, b = (matrices[0] + matrices[1]) / 2, (matrices[2] + matrices[3]) / 2
        rng = np.random.default_rng(20261027)
        choices = rng.integers(0, len(games), size=(2000, len(games)))
        comparison = {"games": len(games), "rows": cohorts[0]["rows"]}
        for key, numerator, denominator in (("policy_ce", 0, 1), ("value_ce", 2, 3)):
            mean_a = a[:, numerator].sum() / a[:, denominator].sum()
            mean_b = b[:, numerator].sum() / b[:, denominator].sum()
            bootstrap_a = a[choices, numerator].sum(axis=1) / a[choices, denominator].sum(axis=1)
            bootstrap_b = b[choices, numerator].sum(axis=1) / b[choices, denominator].sum(axis=1)
            changes = (bootstrap_b / bootstrap_a - 1) * 100
            comparison[key] = {"500000": mean_a, "1000000": mean_b,
                "relative_change_percent": (mean_b / mean_a - 1) * 100,
                "game_bootstrap_95_percent": np.quantile(changes, [0.025, 0.975]).tolist(),
                "replica_values": [item[key] for item in cohorts]}
        report["cohorts"][cohort] = comparison
    atomic_write_json(output / "holdout-comparison.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pool", "train", "matched", "evaluate", "analyze", "schedule"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "pool":
        prepare_pool(args.output.resolve())
    elif args.mode == "train":
        train_windows(args.output.resolve())
    elif args.mode == "evaluate":
        evaluate_holdouts(args.output.resolve())
    elif args.mode == "matched":
        train_matched_probes(args.output.resolve())
    elif args.mode == "analyze":
        analyze_holdouts(args.output.resolve())
    else:
        if args.pairs < 1:
            parser.error("pairs must be positive")
        benchmark_schedule(args.output.resolve(), args.pairs)


if __name__ == "__main__":
    main()
