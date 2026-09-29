"""Run isolated 13x13 and 19x19 migration smoke checks on the local GPU."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from weiqi.katago_rl import PROJECT_ROOT, KataGoRLRunner
from weiqi.rl_curriculum import create_clean_swa_checkpoint, verify_checkpoint_loads


PROFILES = {
    13: PROJECT_ROOT / "config" / "rl_training.rtx5070ti.13x13.json",
    19: PROJECT_ROOT / "config" / "rl_training.rtx5070ti.curriculum.19x19.json",
}


def validate_board(
    board_size: int, source_checkpoint: Path, output_root: Path, batch_size: int
) -> dict[str, object]:
    stage_root = output_root / f"{board_size}x{board_size}"
    checkpoint = stage_root / "train" / "gogogo" / "checkpoint.ckpt"
    create_clean_swa_checkpoint(source_checkpoint, checkpoint)
    export_checkpoint = (
        stage_root / "torchmodels_toexport" / "gogogo-s0-d0" / "model.ckpt"
    )
    export_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, export_checkpoint)
    runner = KataGoRLRunner(PROFILES[board_size], run_root_override=stage_root)
    verify_checkpoint_loads(
        checkpoint, boards=(9, 13, 19), python_source=runner.python_source
    )
    exported = runner.export_new_models()
    runner.selfplay(games=4, visits=8, smoke=True)
    npz_files = sorted(stage_root.glob("selfplay/**/tdata/*.npz"))
    sgf_files = sorted(stage_root.glob("selfplay/**/sgfs/*.sgfs"))
    if not npz_files or not sgf_files:
        raise RuntimeError(f"{board_size}x{board_size} 冒烟未生成 NPZ/SGF")
    runner.shuffle(min_rows=1, smoke=True)
    runner.train(smoke=True, batch_size=batch_size)
    if not checkpoint.is_file():
        raise RuntimeError(f"{board_size}x{board_size} 单批训练后检查点丢失")
    return {
        "board_size": board_size,
        "profile": str(PROFILES[board_size]),
        "run_root": str(stage_root),
        "exported_models": [str(path) for path in exported],
        "selfplay_games": 4,
        "selfplay_npz": len(npz_files),
        "selfplay_sgf_files": len(sgf_files),
        "training_batches": 1,
        "batch_size": batch_size,
        "checkpoint": str(checkpoint),
        "status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    source = args.source_checkpoint.resolve()
    if not source.is_file():
        parser.error(f"source checkpoint does not exist: {source}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    parent = (
        args.output_parent.resolve()
        if args.output_parent
        else PROJECT_ROOT / "training_runs" / "curriculum" / "smoke-validation"
    )
    token = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = parent / token
    output_root.mkdir(parents=True, exist_ok=False)
    results = [
        validate_board(board, source, output_root, args.batch_size)
        for board in (13, 19)
    ]
    summary = {
        "source_checkpoint": str(source),
        "output_root": str(output_root),
        "results": results,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
