"""Run a predeclared multi-seed replication of the native joseki A/B experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-study", action="store_true")
    parser.add_argument("--training-seeds", nargs="+", type=int,
                        default=list(range(2026101701, 2026101706)))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=.25)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--visits", type=int, default=32)
    parser.add_argument("--training-games", type=int, default=16)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--training-visits", type=int, default=64)
    parser.add_argument("--training-leaf-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-leaf-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-length-factor", type=float, default=6)
    parser.add_argument("--evaluation-seed-base", type=int, default=810000001)
    parser.add_argument("--checkpoint-retention", type=int, default=2)
    parser.add_argument("--hardware-overrides", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=90720261017)
    args = parser.parse_args()
    from weiqi.rl.opening_study import run_study
    try:
        result = run_study(**vars(args))
    except (ValueError, RuntimeError, OSError) as error:
        print(f"Opening study error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
