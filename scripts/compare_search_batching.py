"""Direct, fixed-sample paired matches of two search batches using identical weights."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_saved_games(output: Path, seed: int, games: int, candidate_batch: int,
                     opponent_batch: int, model_sha: str):
    """Restore complete pairs; preserve and replay an interrupted trailing orphan."""
    from weiqi.rl.selfplay import GameResult
    known_fields = {field.name for field in fields(GameResult)}
    results = []
    for path in sorted((output / "games").glob("game_*.json")):
        row = json.loads(path.read_text())
        result = GameResult(**{key:value for key,value in row.items() if key in known_fields})
        if result.index != len(results) or result.seed != seed + result.index // 2:
            raise ValueError("Saved game indices or paired seeds do not match")
        if result.candidate_color != (1 if result.index % 2 == 0 else 2):
            raise ValueError("Saved game colors do not match")
        if (row.get("model_sha256") != model_sha or row.get("search_leaf_batches") != {
                "candidate":candidate_batch, "opponent":opponent_batch}):
            raise ValueError("Saved game model or search batches do not match")
        results.append(result)
    if len(results) > games:
        raise ValueError("Saved match has more games than planned")
    if len(results) % 2:
        orphan = results.pop()
        preserved = output / "interrupted" / str(time.time_ns())
        preserved.mkdir(parents=True)
        for suffix in ("json", "sgf"):
            path = output / "games" / f"game_{orphan.index:04d}.{suffix}"
            if path.exists():
                path.rename(preserved / path.name)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--visits", type=int, default=96)
    parser.add_argument("--candidate-batch", type=int, default=4)
    parser.add_argument("--opponent-batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=940261020)
    parser.add_argument("--chunk-games", type=int, default=20)
    args = parser.parse_args()
    if (args.games < 2 or args.games % 2 or args.chunk_games < 2 or args.chunk_games % 2
            or min(args.visits, args.candidate_batch, args.opponent_batch) < 1
            or not 0 <= args.seed < 2**32 - args.games):
        parser.error("Games/chunks must be positive even counts; visits/batches positive; seed in range")

    import numpy as np
    from weiqi.fileio import sha256_file
    from weiqi.rl.control import Progress, RunLock, TrainingControl
    from weiqi.rl.eval_stats import paired_confidence, paired_sign_test
    from weiqi.rl.network import PolicyValueNet, Runtime
    from weiqi.rl.opening_experiment import make_evaluation_config
    from weiqi.rl.selfplay import GameJob, evaluation_summary, run_games
    from weiqi.rl.storage import atomic_json, atomic_torch_save, fingerprint, load_checkpoint, write_games

    output = args.output.resolve()
    source = args.checkpoint.resolve()
    source_checksum = sha256_file(source)
    payload, original = load_checkpoint(source)
    model_sha = fingerprint(payload["model"])
    config = make_evaluation_config(original, games=args.games, visits=args.visits,
                                    seed=args.seed, length_factor=6.0)
    config = replace(config, runtime=replace(config.runtime, output_directory=str(output)))
    source_files = (Path(__file__), ROOT / "weiqi/rl/selfplay.py", ROOT / "weiqi/rl/search.py",
                    ROOT / "weiqi/rl/network.py", ROOT / "weiqi/rl/state.py",
                    ROOT / "weiqi/rl/storage.py", ROOT / "weiqi/rl/eval_stats.py")
    manifest = {
        "source": str(source), "source_sha256": source_checksum,
        "source_iteration": payload["iteration"], "model_sha256": model_sha,
        "candidate_model_sha256": model_sha, "opponent_model_sha256": model_sha,
        "candidate_leaf_batch_size": args.candidate_batch,
        "opponent_leaf_batch_size": args.opponent_batch,
        "planned_games": args.games, "planned_pairs": args.games // 2,
        "visits_per_player_per_move": args.visits, "opening_seed_base": args.seed,
        "opening": "two uniformly sampled legal board moves; identical within each color-swapped pair",
        "chunk_games": args.chunk_games, "config": config.to_dict(),
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files},
        "primary_test": "fixed_sample_two_sided_exact_sign_test_on_opening_pairs",
        "alpha": 0.05, "stop_rule": "Complete all planned games; no significance-based early stopping",
        "strength_scope": "One frozen model and fixed visit budget; nonsignificance does not establish equivalence",
    }
    with RunLock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing experiment has a different manifest")
            if (output / "summary.json").exists():
                print((output / "summary.json").read_text(), flush=True)
                return 0
        else:
            atomic_json(manifest, manifest_path)
            atomic_torch_save({
                "checkpoint_version": payload["checkpoint_version"],
                "feature_version": payload["feature_version"], "kind": "model",
                "iteration": payload["iteration"], "training_steps": payload["training_steps"],
                "best_iteration": payload["best_iteration"], "config": config.to_dict(),
                "model": payload["model"],
            }, output / "frozen_model.pt")
        frozen, frozen_config = load_checkpoint(output / "frozen_model.pt")
        if fingerprint(frozen["model"]) != model_sha:
            raise ValueError("Frozen model weights changed")
        results = load_saved_games(output, args.seed, args.games, args.candidate_batch,
                                   args.opponent_batch, model_sha)
        game_metadata = {"search_leaf_batches":{"candidate":args.candidate_batch,
                                                 "opponent":args.opponent_batch},
                         "model_sha256":model_sha}
        if results:
            write_games(results, config, output / "games", metadata=game_metadata)

        with Progress(output, operation="batch_direct_comparison") as progress:
            progress.phase = "evaluation"
            control = TrainingControl(output, config.runtime.pause_while_game_is_active, progress)
            runtime = Runtime(frozen_config)
            model = PolicyValueNet(frozen_config).to(runtime.device)
            model.load_state_dict(frozen["model"])
            predict = runtime.evaluator(model)
            for size in sorted({1, args.candidate_batch, args.opponent_batch}):
                predict(np.zeros((size, 20, config.game.board_size, config.game.board_size), dtype=np.float32))
            counts = {0: Counter(), 1: Counter()}
            seconds = {0: 0.0, 1: 0.0}
            evaluators = {}
            for model_id in (0, 1):
                def tracked(features, model_id=model_id):
                    started = time.perf_counter()
                    answer = predict(features)
                    seconds[model_id] += time.perf_counter() - started
                    counts[model_id][len(features)] += 1
                    return answer
                evaluators[model_id] = tracked
            started = time.monotonic()
            for start in range(len(results), args.games, args.chunk_games):
                jobs = [GameJob(index, args.seed + index // 2, 1 if index % 2 == 0 else 2,
                                opponent_name=f"same_weights_batch_{args.opponent_batch}",
                                opponent_sha256=model_sha,
                                candidate_leaf_batch_size=args.candidate_batch,
                                opponent_leaf_batch_size=args.opponent_batch)
                        for index in range(start, min(args.games, start + args.chunk_games))]
                games = run_games(config, jobs, evaluators, training=False, check=control,
                                  progress=progress,
                                  metadata={"candidate_sha256":model_sha,
                                            "opponent_sha256":model_sha,
                                            "candidate_leaf_batch_size":args.candidate_batch,
                                            "opponent_leaf_batch_size":args.opponent_batch})
                write_games(games, config, output / "games", metadata=game_metadata)
                results.extend(games)
                progress({"event":"chunk_completed", "completed_games":len(results),
                          "planned_games":args.games})
            for index in range(0, len(results), 2):
                if results[index].moves[:2] != results[index + 1].moves[:2]:
                    raise ValueError("Color-swapped games did not use identical opening moves")
            sign = paired_sign_test(results)
            decisive = sign["wins"] + sign["losses"]
            reverse_p = (sum(math.comb(decisive, k) for k in range(sign["losses"], decisive + 1))
                         / 2**decisive) if decisive else 1.0
            two_sided = min(1.0, 2 * min(sign["p_value"], reverse_p))
            conclusion = "inconclusive"
            if not sign["truncated_pairs"] and two_sided <= 0.05:
                conclusion = "batch4_stronger" if sign["wins"] > sign["losses"] else "batch4_weaker"
                if args.candidate_batch != 4:
                    conclusion = "candidate_stronger" if sign["wins"] > sign["losses"] else "candidate_weaker"
            inference = {str(model_id):{"batch_histogram":dict(counts[model_id]),
                                      "network_seconds_this_session":seconds[model_id]}
                         for model_id in (0,1)}
            summary = {"status":"complete", **manifest, **evaluation_summary(results),
                       "paired":paired_confidence(results), "paired_sign":sign,
                       "two_sided_p_value":two_sided, "conclusion":conclusion,
                       "elapsed_seconds_this_session":time.monotonic()-started,
                       "network_inference_this_session":inference,
                       "equivalence_established":False}
            if fingerprint(model.state_dict()) != model_sha or sha256_file(source) != source_checksum:
                raise ValueError("Model weights or source checkpoint changed during comparison")
            atomic_json(summary, output / "summary.json")
            progress({"event":"comparison_completed", "games":len(results),
                      "conclusion":conclusion, "two_sided_p_value":two_sided})
            print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
