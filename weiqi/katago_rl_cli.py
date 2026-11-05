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
from .katago_hardware import detect_hardware
from .katago_mixed import DEFAULT_MIXED_PROFILE, MixedBoardRunner
from .katago_mixed_health import build_mixed_health
from .katago_mixed_eval import evaluate_mixed
from .katago_mixed_experiments import prepare_mixed_experiment
from .katago_mixed_trials import run_prepared_experiment
from .katago_mixed_budget_profiles import prepare_budget_candidates
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
            "hardware",
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
            "mixed", "mixed-status", "mixed-doctor", "mixed-health", "mixed-evaluate",
            "mixed-experiment-prepare", "mixed-experiment-run",
            "mixed-budget-prepare",
            *sorted(HANDOFF_COMMANDS),
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RTX_PROFILE)
    parser.add_argument(
        "--hardware", choices=("auto", "cpu", "mps", "low", "high", "configured"), default="auto",
        help="启动时检测硬件，或手动选择 CPU/低端 CUDA/高端 CUDA；configured 沿用原配置",
    )
    parser.add_argument("--mixed-config", type=Path, default=DEFAULT_MIXED_PROFILE,
                        help="KataGo 混合棋盘配置；mixed 系列命令使用")
    parser.add_argument("--run-root", type=Path,
                        help="mixed 系列命令的独立输出目录；省略时使用混训配置")
    parser.add_argument("--baseline-model", type=Path, help="mixed-evaluate 的冻结对手模型")
    parser.add_argument("--candidate-model", type=Path, help="评测候选模型；省略时首次选定最新模型")
    parser.add_argument("--evaluation-output", type=Path, help="独立、可恢复的评测输出目录")
    parser.add_argument("--evaluation-games", type=int, default=12, help="每种棋盘预先固定的偶数评测局数")
    parser.add_argument("--evaluation-visits", type=int, default=32)
    parser.add_argument("--evaluation-parallel-pairs", type=int, default=1,
                        help="mixed-evaluate 同时运行的换色开局组数；改变时使用新评测目录")
    parser.add_argument("--evaluation-max-moves-factor", type=float, default=2.5,
                        help="mixed-evaluate 每局新增手数上限为棋盘面积的此倍数；改变时使用新评测目录")
    parser.add_argument("--evaluation-boards", type=int, nargs="+", help="要评测的棋盘尺寸；省略时使用全部混训棋盘")
    parser.add_argument("--experiment-output", type=Path, help="独立的混训试验目录")
    parser.add_argument("--budget-seconds", type=int, default=180, help="准备试验时固定每个分支的训练墙钟秒数")
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
        help="接力恢复或 mixed-budget-prepare 候选目录；必须是新目录",
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
        if args.run_root is not None and not args.command.startswith("mixed"):
            parser.error("--run-root 仅用于 mixed 系列命令")
        if args.command == "mixed-evaluate" and (args.baseline_model is None or args.evaluation_output is None):
            parser.error("mixed-evaluate 必须指定 --baseline-model 和 --evaluation-output")
        if args.command in {"mixed-experiment-prepare", "mixed-experiment-run", "mixed-budget-prepare"} and args.experiment_output is None:
            parser.error("混训试验必须指定 --experiment-output")
        if args.command == "mixed-budget-prepare" and args.destination is None:
            parser.error("mixed-budget-prepare 必须指定 --destination")
        if args.command == "mixed-experiment-run" and args.hardware == "configured":
            parser.error("试验执行必须显式检测硬件，请使用 --hardware mps、cpu 或 auto")
        profile = None
        read_only = {"status", "dashboard", "curriculum-status", "curriculum-dashboard", "mixed-status", "mixed-health", "mixed-experiment-prepare", "mixed-budget-prepare"}
        if args.command == "hardware" or (args.hardware != "configured" and args.command not in read_only):
            profile = detect_hardware("auto" if args.hardware == "configured" else args.hardware)
            print(json.dumps({"hardware": profile.to_dict()}, ensure_ascii=False, indent=2), flush=True)
        if args.command == "hardware":
            return
        if args.command == "mixed-budget-prepare":
            result = prepare_budget_candidates(args.experiment_output, args.destination)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command == "mixed-experiment-run":
            result = run_prepared_experiment(args.experiment_output, hardware_profile=profile,
                                            games_per_board=args.evaluation_games, visits=args.evaluation_visits)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command.startswith("mixed"):
            runner = MixedBoardRunner(args.mixed_config, run_root_override=args.run_root,
                                      hardware_profile=profile)
            if args.command == "mixed-status":
                runner.status()
                return
            if args.command == "mixed-experiment-prepare":
                with runner.lock():
                    result = prepare_mixed_experiment(runner.run_root, args.mixed_config,
                                                      args.experiment_output, wall_seconds=args.budget_seconds)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return
            if args.command == "mixed-health":
                print(json.dumps(build_mixed_health(runner.run_root, runner.mixed.board_sizes), ensure_ascii=False, indent=2))
                return
            if args.command == "mixed-evaluate":
                result = evaluate_mixed(runner, args.evaluation_output, candidate=args.candidate_model,
                                        baseline=args.baseline_model, boards=args.evaluation_boards,
                                        games_per_board=args.evaluation_games, visits=args.evaluation_visits,
                                        max_moves_factor=args.evaluation_max_moves_factor,
                                        parallel_pairs=args.evaluation_parallel_pairs)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return
            if args.command == "mixed-doctor":
                runner.doctor()
                return
            with runner.lock():
                completed = 0
                limit = args.iterations or (1 if args.smoke else 0)
                while limit <= 0 or completed < limit:
                    print(f"\n===== 混合棋盘循环 {completed + 1} =====", flush=True)
                    runner.cycle(smoke=args.smoke, games=args.games, visits=args.visits,
                                 min_rows=args.min_rows)
                    completed += 1
            return
        if args.command in {"curriculum", "curriculum-status", "curriculum-dashboard", "curriculum-evaluate"}:
            curriculum = CurriculumRuntime(args.curriculum_config, hardware_profile=profile)
            if args.command == "curriculum-status":
                curriculum.status()
            elif args.command == "curriculum-dashboard":
                curriculum.refresh_dashboard()
            elif args.command == "curriculum-evaluate":
                curriculum.evaluate_now()
            else:
                curriculum.run(iterations=args.iterations, smoke=args.smoke)
            return
        runner = KataGoRLRunner(args.config, hardware_profile=profile)
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
        ValueError,
    ) as error:
        print(f"强化学习失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error


__all__ = ["build_parser", "main"]
