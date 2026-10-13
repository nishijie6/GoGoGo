"""Command line for the KataGo training loop and the board-size curriculum (train_rl.py)."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

from .katago_rl import (
    DEFAULT_CURRICULUM_PROFILE,
    DEFAULT_RTX_PROFILE,
    PROJECT_ROOT,
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
from .rl_handoff import (
    HandoffError,
    cancel_handoff,
    check_import,
    export_handoff,
    import_handoff,
    verify_handoff,
)


HANDOFF_COMMANDS = {
    "handoff-export",
    "handoff-verify",
    "handoff-import",
    "handoff-check",
    "handoff-cancel",
}


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
            *sorted(HANDOFF_COMMANDS),
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
    parser.add_argument(
        "--bundle",
        type=Path,
        help="接力包路径；handoff-export、handoff-verify 和 handoff-import 必填",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="接力恢复目录；handoff-import 必须指定尚不存在的新目录",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="handoff-export/handoff-cancel 仅处理 --config 指定的单棋盘训练",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="导入时显式允许代码版本差异；随后必须自行验证恢复兼容性",
    )
    parser.add_argument(
        "--hardware-profile",
        type=Path,
        help="导入时覆盖本机设备、精度、批次和并发设置的 JSON 文件",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="handoff-check 使用 PyTorch 在 CPU 上读取检查点并验证续训状态",
    )
    return parser


def _resume_command(report: dict) -> str | None:
    curriculum_config = report.get("curriculum_config")
    training_config = report.get("training_config")
    if curriculum_config:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train_rl.py"),
            "curriculum",
            "--curriculum-config",
            str(curriculum_config),
        ]
    elif training_config:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train_rl.py"),
            "continuous",
            "--config",
            str(training_config),
        ]
    else:
        return None
    if sys.platform == "win32":
        # PowerShell's call operator also works when the executable contains spaces.
        return "& " + " ".join("'" + value.replace("'", "''") + "'" for value in command)
    return shlex.join(command)


def _run_handoff(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Handle local package operations without constructing a training runner."""

    if args.command in {"handoff-export", "handoff-verify", "handoff-import"} and args.bundle is None:
        parser.error(f"{args.command} 必须指定 --bundle")
    if args.command in {"handoff-import", "handoff-check"} and args.destination is None:
        parser.error(f"{args.command} 必须指定 --destination")
    if args.single_run and args.command not in {"handoff-export", "handoff-cancel"}:
        parser.error("--single-run 仅用于 handoff-export 或 handoff-cancel")
    if (args.allow_version_mismatch or args.hardware_profile is not None) and args.command != "handoff-import":
        parser.error("--allow-version-mismatch 和 --hardware-profile 仅用于 handoff-import")
    if args.load_checkpoint and args.command != "handoff-check":
        parser.error("--load-checkpoint 仅用于 handoff-check")

    selection = {
        "training_config": args.config,
    } if args.single_run else {
        "curriculum_config": args.curriculum_config,
    }
    if args.command == "handoff-export":
        report = export_handoff(PROJECT_ROOT, args.bundle, **selection)
    elif args.command == "handoff-verify":
        report = verify_handoff(args.bundle, expected_trainer="katago")
    elif args.command == "handoff-import":
        report = dict(import_handoff(
            PROJECT_ROOT,
            args.bundle,
            args.destination,
            allow_version_mismatch=args.allow_version_mismatch,
            hardware_profile=args.hardware_profile,
            expected_trainer="katago",
        ))
        resume = _resume_command(report)
        if resume:
            report["resume_command"] = resume
    elif args.command == "handoff-check":
        report = check_import(args.destination, load_checkpoint=args.load_checkpoint,
                              expected_trainer="katago")
    else:
        report = cancel_handoff(PROJECT_ROOT, **selection)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in HANDOFF_COMMANDS:
            _run_handoff(args, parser)
            return
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
        HandoffError,
        OSError,
    ) as error:
        print(f"强化学习失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error


__all__ = ["build_parser", "main"]
