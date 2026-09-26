"""Optional CLI entrypoint: python train.py --help (no GUI or engine startup)."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def with_evaluation_overrides(config, path: Path):
    """Change only measurement budgets when continuing an existing model."""
    from weiqi.rl_config import resolve_rl_training_config

    changes = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(changes, dict):
        raise ValueError("Evaluation overrides must be a JSON object")
    resolved = config.to_dict()
    resolved["evaluation"].update(changes)
    return resolve_rl_training_config(
        config.preset, {key: value for key, value in resolved.items()
                        if key not in ("schema_version", "preset")})


def with_training_policy(config, path: Path):
    """Opt into opponent/promotion policy without changing model or replay settings."""
    from weiqi.rl_config import resolve_rl_training_config

    changes = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "self_play": {"champion_fraction", "milestone_fraction"},
        "evaluation": {"games", "promotion_win_rate", "promotion_test",
                       "confirmation_max_game_length_factor", "confidence_level"},
    }
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Policy overrides must be a nonempty JSON object")
    for section, fields in changes.items():
        if section not in allowed or not isinstance(fields, dict) or not fields:
            raise ValueError(f"Unsupported policy section: {section}")
        unknown = set(fields) - allowed[section]
        if unknown:
            raise ValueError(f"Unsupported policy field: {section}.{sorted(unknown)[0]}")
    resolved = config.to_dict()
    for section, fields in changes.items():
        resolved[section].update(fields)
    return resolve_rl_training_config(
        config.preset, {key: value for key, value in resolved.items()
                        if key not in ("schema_version", "preset")})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="9x9 Go self-play training and evaluation")
    actions = parser.add_subparsers(dest="action", required=True)
    train_parser = actions.add_parser("train", help="Run a bounded number of additional training iterations")
    train_parser.add_argument("--config", type=Path)
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--iterations", type=int, default=1)
    train_parser.add_argument("--resume", type=Path, help="Restore model, optimizer, replay and RNG from a full checkpoint")
    train_parser.add_argument("--evaluation-overrides", type=Path,
                              help="JSON overrides for evaluation budgets only, including on resume")
    train_parser.add_argument("--policy-overrides", type=Path,
                              help="Safe opponent and promotion policy overrides, including on resume")
    evaluate_parser = actions.add_parser("evaluate", help="Play color-swapped paired games")
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--opponent", type=Path, help="Other model; default is uniform-policy MCTS")
    evaluate_parser.add_argument("--games", type=int)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser = actions.add_parser("benchmark", help="Compare against fixed KataGo positions and model references")
    benchmark_parser.add_argument("checkpoint", type=Path)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--positions", type=Path)
    benchmark_parser.add_argument("--teacher", type=Path)
    benchmark_parser.add_argument("--pool", type=Path)
    benchmark_parser.add_argument("--opponent", type=Path, action="append")
    benchmark_parser.add_argument("--games", type=int)
    benchmark_parser.add_argument("--visits", type=int)
    benchmark_parser.add_argument("--position-visits", type=int)
    inspect_parser = actions.add_parser("inspect", help="Inspect a checkpoint without starting games")
    inspect_parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        from weiqi.rl.control import TrainingStopped
        from weiqi.rl.eval_positions import file_hash
        from weiqi.rl.runner import benchmark, evaluate, train
        from weiqi.rl.storage import fingerprint, load_checkpoint
        from weiqi.rl_config import DEFAULT_RL_CONFIG_PATH, load_rl_training_config
    except ImportError as error:
        print(f"Training dependencies unavailable: {error}\n"
              "Use the WSL PyTorch environment, or install requirements-training.txt in a dedicated environment.",
              file=sys.stderr)
        return 2
    try:
        if args.action == "inspect":
            payload, config = load_checkpoint(args.checkpoint)
            result = {"kind": payload["kind"], "iteration": payload["iteration"],
                      "training_steps": payload["training_steps"],
                      "best_iteration": payload["best_iteration"],
                      "model_sha256": fingerprint(payload["model"]), "config": config.to_dict()}
            if payload["kind"] == "training":
                result["replay_samples"] = len(payload["replay"]["values"])
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.action == "evaluate":
            evaluate(args.checkpoint.resolve(), args.opponent.resolve() if args.opponent else None,
                     args.output.resolve(), args.games)
        elif args.action == "benchmark":
            benchmark(args.checkpoint.resolve(), args.output.resolve(),
                      positions=args.positions.resolve() if args.positions else None,
                      teacher=args.teacher.resolve() if args.teacher else None,
                      pool=args.pool.resolve() if args.pool else None,
                      opponents=[path.resolve() for path in args.opponent or []],
                      games=args.games, simulations=args.visits,
                      position_simulations=args.position_visits)
        else:
            if args.iterations < 1:
                raise ValueError("--iterations must be at least 1")
            resume = None
            resume_checksum = None
            if args.resume:
                resume_checksum = file_hash(args.resume)
                resume, config = load_checkpoint(args.resume)
                if args.config:
                    requested = load_rl_training_config(args.config)
                    # A changed output location is fine; all training parameters must match.
                    original = replace(config, runtime=replace(config.runtime, output_directory=""))
                    compare = replace(requested, runtime=replace(requested.runtime, output_directory=""))
                    if original != compare:
                        raise ValueError("Resume config differs from checkpoint; omit --config to use saved settings")
            else:
                config = load_rl_training_config(args.config or DEFAULT_RL_CONFIG_PATH)
            if args.evaluation_overrides:
                config = with_evaluation_overrides(config, args.evaluation_overrides)
            if args.policy_overrides:
                config = with_training_policy(config, args.policy_overrides)
            output = args.output or (args.resume.resolve().parent if args.resume else ROOT / config.runtime.output_directory)
            output = output.resolve()
            config = replace(config, runtime=replace(config.runtime, output_directory=str(output)))
            train(config, output, args.iterations, resume,
                  resume_path=args.resume.resolve() if args.resume else None,
                  resume_checksum=resume_checksum)
    except (KeyboardInterrupt, TrainingStopped):
        print("Stopped. Resume the last completed iteration with --resume <output>/latest.pt.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"Training error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
