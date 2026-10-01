"""Fixed, color-swapped evaluation openings for curriculum matches.

Each opening is a deterministic KataGo ``PositionSample`` written once into its
own directory, so every candidate faces exactly the same starts and both
colors of each.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .fileio import atomic_write_json
from .rl_curriculum import CurriculumStateError


EVALUATION_OPENING_SUITE_VERSION = 1


def _gtp_coordinate(x: int, y: int, board_size: int) -> str:
    columns = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return f"{columns[x]}{board_size - y}"


def _fixed_opening_sample(board_size: int, index: int) -> str:
    """Generate one legal, deterministic early-game PositionSample."""

    candidates = [
        (x, y)
        for y in range(1, board_size - 1)
        for x in range(1, board_size - 1)
        if (x + y) % 2 == 0
    ]
    candidates.sort(
        key=lambda point: hashlib.sha256(
            (
                f"gogogo-eval-v{EVALUATION_OPENING_SUITE_VERSION}:"
                f"{board_size}:{index}:{point[0]}:{point[1]}"
            ).encode("ascii")
        ).digest()
    )
    move_count = 4 + 2 * (index % 4)
    selected = candidates[:move_count]
    if len(selected) != move_count:
        raise CurriculumStateError(f"无法为 {board_size}x{board_size} 生成固定评测开局")
    sample = {
        "board": ("." * board_size + "/") * board_size,
        "hintLoc": "null",
        "initialTurnNumber": 0,
        "metadata": (
            f"gogogo-evaluation-v{EVALUATION_OPENING_SUITE_VERSION}-"
            f"{board_size}x{board_size}-{index:03d}"
        ),
        "moveLocs": [_gtp_coordinate(x, y, board_size) for x, y in selected],
        "movePlas": ["B" if move % 2 == 0 else "W" for move in range(move_count)],
        "nextPla": "B",
        "trainingWeight": 1.0,
        "weight": 1.0,
        "xSize": board_size,
        "ySize": board_size,
    }
    return json.dumps(sample, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


def ensure_evaluation_opening_suite(
    state_root: Path, *, board_size: int, opening_count: int
) -> tuple[Path, ...]:
    """Create or verify immutable singleton opening directories for evaluation."""

    if board_size not in {9, 13, 19} or opening_count < 1:
        raise ValueError("固定评测开局参数无效")
    samples = [_fixed_opening_sample(board_size, index) for index in range(opening_count)]
    hashes = [hashlib.sha256(sample.encode("utf-8")).hexdigest() for sample in samples]
    if len(set(hashes)) != opening_count:
        raise CurriculumStateError("固定评测开局生成器产生了重复局面")
    suite_root = (
        state_root.resolve()
        / "evaluation_openings"
        / f"{board_size}x{board_size}"
        / f"v{EVALUATION_OPENING_SUITE_VERSION}"
    )
    manifest = {
        "schema_version": 1,
        "suite_version": EVALUATION_OPENING_SUITE_VERSION,
        "board_size": board_size,
        "opening_count": opening_count,
        "format": "KataGo PositionSample JSONL; one singleton directory per color-swapped pair",
        "sha256": hashes,
    }

    def is_valid() -> bool:
        try:
            stored = json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))
            files_match = all(
                (
                    suite_root
                    / f"opening-{index:03d}"
                    / "opening.startposes.txt"
                ).read_text(encoding="utf-8")
                == samples[index]
                for index in range(opening_count)
            )
        except (OSError, json.JSONDecodeError):
            return False
        return stored == manifest and files_match

    if suite_root.is_dir() and is_valid():
        return tuple(suite_root / f"opening-{index:03d}" for index in range(opening_count))

    suite_root.parent.mkdir(parents=True, exist_ok=True)
    if suite_root.exists():
        quarantine = suite_root.with_name(
            f".{suite_root.name}.invalid-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        suite_root.replace(quarantine)
    temporary = suite_root.with_name(f".{suite_root.name}.build-{uuid.uuid4().hex}.tmp")
    try:
        for index, sample in enumerate(samples):
            opening = temporary / f"opening-{index:03d}"
            opening.mkdir(parents=True, exist_ok=False)
            (opening / "opening.startposes.txt").write_text(sample, encoding="utf-8")
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, suite_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(suite_root / f"opening-{index:03d}" for index in range(opening_count))


__all__ = ["EVALUATION_OPENING_SUITE_VERSION", "ensure_evaluation_opening_suite"]
