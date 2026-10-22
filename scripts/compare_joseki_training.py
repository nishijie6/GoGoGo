"""Run an isolated native trainer control/joseki pilot from one full checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True, help="Full latest.pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the source run")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=.25)
    parser.add_argument("--games", type=int, default=80, help="Even evaluation games per arm")
    parser.add_argument("--visits", type=int, default=32, help="Evaluation MCTS visits per move")
    parser.add_argument("--hardware-overrides", type=Path)
    parser.add_argument("--training-games", type=int)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--training-visits", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--training-leaf-batch-size", type=int)
    parser.add_argument("--evaluation-leaf-batch-size", type=int)
    parser.add_argument("--evaluation-length-factor", type=float)
    parser.add_argument("--freeze-training-opponents", action="store_true")
    parser.add_argument("--checkpoint-retention", type=int)
    parser.add_argument("--skip-training-evaluation", action="store_true",
                        help="Both arms skip diagnostics, screening, pool matches, and promotion")
    args = parser.parse_args()
    from weiqi.rl.opening_experiment import compare_training
    try:
        report = compare_training(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Opening experiment error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
