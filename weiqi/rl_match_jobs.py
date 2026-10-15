"""Bounded, resumable execution of strictly paired fixed-opening matches."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Sequence

from .fileio import atomic_write_json, atomic_write_text, local_timestamp
from .rl_match import MatchSummary, summarize_match_sgfs
from .sgf import split_sgf_collection


def _complete_pair(paths: Sequence[Path]) -> bool:
    for path in paths:
        try:
            if split_sgf_collection(path.read_text(encoding="utf-8"))[1]:
                return False
        except (OSError, UnicodeError):
            return False
    summary = summarize_match_sgfs(paths, candidate_name="candidate", requested_games=2)
    return (
        summary.games == 2
        and summary.candidate_black_games == summary.candidate_white_games == 1
        and not summary.no_result_games
        and not summary.damaged_games
    )


def run_opening_pairs(
    openings: Sequence[Path],
    output_dir: Path,
    *,
    parallel_pairs: int,
    config_builder: Callable[[Path], str],
    execute: Callable[[Path, Path, Path], int],
) -> MatchSummary:
    """Reuse only complete, matching pairs; retain incomplete attempts for audit.

    Each process still plays exactly two games from a singleton opening with
    opposite colors. Only the execution order changes. At most ``parallel_pairs``
    jobs are submitted, so a failure cannot start a backlog of extra matches.
    """
    if not openings or parallel_pairs < 1:
        raise ValueError("固定开局与评测并发数必须为正")
    started = time.perf_counter()

    def run_pair(index: int, opening: Path) -> tuple[int, list[Path], bool]:
        pair_root = output_dir / "pairs" / f"opening-{index:03d}"
        config_text = config_builder(opening)
        attempts = [pair_root, *sorted(pair_root.glob("retry-*"))]
        for attempt in attempts:
            config_path = attempt / "match.cfg"
            paths = sorted((attempt / "sgfs").glob("*.sgfs"))
            if (
                config_path.is_file()
                and config_path.read_text(encoding="utf-8") == config_text
                and _complete_pair(paths)
            ):
                return index, paths, True
        attempt = pair_root if not pair_root.exists() else pair_root / f"retry-{uuid.uuid4().hex[:12]}"
        sgf_dir = attempt / "sgfs"
        sgf_dir.mkdir(parents=True, exist_ok=False)
        config_path = attempt / "match.cfg"
        atomic_write_text(config_path, config_text)
        returncode = execute(config_path, sgf_dir, attempt / "match.log")
        if returncode:
            raise RuntimeError(f"固定开局 {index:03d} 的 KataGo match 退出代码 {returncode}")
        paths = sorted(sgf_dir.glob("*.sgfs"))
        if not _complete_pair(paths):
            raise RuntimeError(f"固定开局 {index:03d} 未完成两局有效且严格黑白互换的对局")
        return index, paths, False

    results: dict[int, list[Path]] = {}
    reused = 0
    workers = min(parallel_pairs, len(openings))

    def progress() -> None:
        atomic_write_json(output_dir / "progress.json", {
            "completed_pairs": len(results), "total_pairs": len(openings),
            "parallel_pairs": workers, "reused_pairs": reused,
            "elapsed_seconds": time.perf_counter() - started,
            "updated_at": local_timestamp(),
        })

    progress()
    remaining = iter(enumerate(openings))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="match-pair") as pool:
        pending = {}

        def submit_next() -> None:
            item = next(remaining, None)
            if item is not None:
                pending[pool.submit(run_pair, *item)] = item[0]

        for _ in range(workers):
            submit_next()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    del pending[future]
                    index, paths, cached = future.result()
                    results[index] = paths
                    reused += int(cached)
                progress()
                for _ in done:
                    submit_next()
        finally:
            for future in pending:
                future.cancel()

    summary = summarize_match_sgfs(
        [path for index in sorted(results) for path in results[index]],
        candidate_name="candidate", requested_games=2 * len(openings),
    )
    atomic_write_json(output_dir / "match-result.json", summary.to_dict())
    return summary
