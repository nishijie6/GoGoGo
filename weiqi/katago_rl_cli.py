"""Command line for the KataGo training loop and the board-size curriculum (train_rl.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .katago_rl import (
    DEFAULT_CURRICULUM_PROFILE,
    DEFAULT_RTX_PROFILE,
    KataGoRLRunner,
    KataGoRLRunnerError,
)
from .rl_config import RLConfigError
from .rl_curriculum import (
    CurriculumConfigError,
    CurriculumLockError,
    CurriculumMigrationError,
    CurriculumStateError,
)
from .rl_curriculum_runtime import CurriculumRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行可续训的 KataGo 单机强化学习闭环"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="cycle",
        choices=(
            "doctor",
            "status",
            "selfplay",
            "shuffle",
            "train",
            "export",
            "cycle",
            "continuous",
            "dashboard",
            "curriculum",
            "curriculum-status",
            "curriculum-dashboard",
            "curriculum-evaluate",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RTX_PROFILE)
    parser.add_argument(
        "--curriculum-config",
        type=Path,
        default=DEFAULT_CURRICULUM_PROFILE,
    )
    parser.add_argument("--games", type=int)
    parser.add_argument("--visits", type=int)
    parser.add_argument("--min-rows", type=int)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="continuous 的循环数；0 表示持续运行到 Ctrl+C",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用少量局数和访问量验证完整闭环",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.iterations < 0:
            raise KataGoRLRunnerError("--iterations 不能小于 0")
        if args.command in {"curriculum", "curriculum-status", "curriculum-dashboard", "curriculum-evaluate"}:
            curriculum = CurriculumRuntime(args.curriculum_config)
            if args.command == "curriculum-status":
                curriculum.status()
            elif args.command == "curriculum-dashboard":
                curriculum.refresh_dashboard()
            elif args.command == "curriculum-evaluate":
                curriculum.evaluate_now()
            else:
                curriculum.run(iterations=args.iterations, smoke=args.smoke)
            return
        runner = KataGoRLRunner(args.config)
        if args.command == "doctor":
            runner.doctor()
            return
        if args.command == "status":
            runner.status()
            return
        if args.command == "dashboard":
            runner.dashboard()
            return
        with runner.lock():
            if args.command == "selfplay":
                runner.selfplay(
                    games=args.games,
                    visits=args.visits,
                    smoke=args.smoke,
                )
            elif args.command == "shuffle":
                runner.shuffle(min_rows=args.min_rows, smoke=args.smoke)
            elif args.command == "train":
                runner.train(smoke=args.smoke)
            elif args.command == "export":
                runner.export_new_models()
            elif args.command == "cycle":
                runner.cycle(
                    smoke=args.smoke,
                    games=args.games,
                    visits=args.visits,
                    min_rows=args.min_rows,
                )
            else:
                completed = 0
                while args.iterations <= 0 or completed < args.iterations:
                    print(f"\n===== 强化学习循环 {completed + 1} =====", flush=True)
                    runner.cycle(
                        smoke=args.smoke,
                        games=args.games,
                        visits=args.visits,
                        min_rows=args.min_rows,
                    )
                    completed += 1
    except KeyboardInterrupt:
        print("\n已收到中断，KataGo 会保留可续训产物。", file=sys.stderr)
        raise SystemExit(130)
    except (
        KataGoRLRunnerError,
        RLConfigError,
        CurriculumConfigError,
        CurriculumLockError,
        CurriculumMigrationError,
        CurriculumStateError,
        OSError,
    ) as error:
        print(f"强化学习失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error


__all__ = ["build_parser", "main"]
