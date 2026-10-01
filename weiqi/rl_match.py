"""Fixed-protocol KataGo matches: configuration, SGF results, and confidence."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from .engine import GoGame
from .sgf import split_sgf_collection


def wilson_interval(
    successes: float, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """95% Wilson score interval, supporting half a success for a draw."""

    if trials < 1 or successes < 0 or successes > trials:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def approximate_elo(win_rate: float) -> float:
    """Convert a match score to the conventional logistic Elo approximation."""

    clipped = min(1.0 - 1e-9, max(1e-9, win_rate))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class MatchGame:
    black: Optional[str]
    white: Optional[str]
    result: Optional[str]
    winner: Optional[str]
    margin: Optional[float]
    damaged: bool = False


@dataclass(frozen=True)
class MatchSummary:
    requested_games: int
    games: int
    candidate_wins: int
    baseline_wins: int
    draws: int
    candidate_black_games: int
    candidate_white_games: int
    black_wins: int
    white_wins: int
    no_result_games: int
    damaged_games: int
    win_rate: float
    wilson_lower: float
    wilson_upper: float
    elo: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sgf_value(tree: str, key: str) -> Optional[str]:
    match = re.search(rf"(?<![A-Z]){re.escape(key)}\[((?:\\.|[^\\\]])*)\]", tree)
    if match is None:
        return None
    return re.sub(r"\\(.)", r"\1", match.group(1)).strip()


def parse_match_game(tree: str) -> MatchGame:
    black = _sgf_value(tree, "PB")
    white = _sgf_value(tree, "PW")
    result = _sgf_value(tree, "RE")
    damaged = black is None or white is None or result is None
    winner: Optional[str] = None
    margin: Optional[float] = None
    if result:
        result_text = result.strip()
        upper = result_text.upper()
        if upper.startswith("B+"):
            winner = "B"
        elif upper.startswith("W+"):
            winner = "W"
        elif upper in {"0", "DRAW", "JIGO"}:
            winner = "D"
        elif upper not in {"?", "VOID", "NO RESULT"}:
            damaged = True
        if winner in {"B", "W"}:
            score = result_text.split("+", 1)[1]
            try:
                margin = float(score)
            except ValueError:
                margin = None
    return MatchGame(black, white, result, winner, margin, damaged)


def summarize_match_sgfs(
    paths: Iterable[Path], *, candidate_name: str, requested_games: int
) -> MatchSummary:
    """Parse KataGo ``match`` SGFs with candidate color accounting."""

    games: list[MatchGame] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        games.extend(parse_match_game(tree) for tree in split_sgf_collection(text)[0])
    candidate_key = candidate_name.casefold()
    candidate_wins = baseline_wins = draws = 0
    candidate_black = candidate_white = 0
    black_wins = white_wins = no_result = damaged = 0
    valid_candidate_games = 0
    for game in games:
        is_black = bool(game.black and game.black.casefold() == candidate_key)
        is_white = bool(game.white and game.white.casefold() == candidate_key)
        if game.damaged:
            damaged += 1
        if game.winner is None:
            no_result += 1
            continue
        if game.damaged:
            continue
        if not (is_black or is_white):
            damaged += 1
            continue
        if is_black:
            candidate_black += 1
        if is_white:
            candidate_white += 1
        valid_candidate_games += 1
        if game.winner == "D":
            draws += 1
        elif game.winner == "B":
            black_wins += 1
            if is_black:
                candidate_wins += 1
            else:
                baseline_wins += 1
        elif game.winner == "W":
            white_wins += 1
            if is_white:
                candidate_wins += 1
            else:
                baseline_wins += 1
    successes = candidate_wins + 0.5 * draws
    rate = successes / valid_candidate_games if valid_candidate_games else 0.0
    lower, upper = wilson_interval(successes, valid_candidate_games)
    return MatchSummary(
        requested_games=requested_games,
        games=valid_candidate_games,
        candidate_wins=candidate_wins,
        baseline_wins=baseline_wins,
        draws=draws,
        candidate_black_games=candidate_black,
        candidate_white_games=candidate_white,
        black_wins=black_wins,
        white_wins=white_wins,
        no_result_games=no_result,
        damaged_games=damaged,
        win_rate=rate,
        wilson_lower=lower,
        wilson_upper=upper,
        elo=approximate_elo(rate),
    )


def build_match_config(
    *,
    candidate_model: Path,
    baseline_model: Path,
    board_size: int,
    visits: int,
    games: int = 200,
    game_threads: int = 128,
    inference_batch_size: int = 128,
    gpu_index: int = 0,
    opening_directory: Optional[Path] = None,
) -> str:
    """Build a fixed-protocol, mirrored-color KataGo match configuration."""

    checked_paths = [candidate_model, baseline_model]
    if opening_directory is not None:
        checked_paths.append(opening_directory)
    for path in checked_paths:
        if "\n" in str(path) or "\r" in str(path):
            raise ValueError("评测路径不能包含换行")
    if board_size not in GoGame.SUPPORTED_SIZES or visits < 1 or games < 2 or games % 2:
        raise ValueError("棋盘、visits 或偶数局数无效")
    values: list[tuple[str, object]] = [
        ("logSearchInfo", "false"),
        ("logMoves", "false"),
        ("logGamesEvery", 1),
        ("logToStdout", "true"),
        ("numBots", 2),
        ("botName0", "candidate"),
        ("botName1", "baseline"),
        ("nnModelFile0", candidate_model),
        ("nnModelFile1", baseline_model),
        ("numGameThreads", game_threads),
        ("numGamesTotal", games),
        ("maxMovesPerGame", math.ceil(board_size * board_size * 2.5)),
        ("allowResignation", "false"),
        # KataGo's match loader requires these companion keys even when
        # resignation itself is disabled.
        ("resignThreshold", -1.0),
        ("resignConsecTurns", 3),
        ("koRules", "POSITIONAL"),
        ("scoringRules", "AREA"),
        ("taxRules", "NONE"),
        ("multiStoneSuicideLegals", "false"),
        ("hasButtons", "false"),
        ("bSizes", board_size),
        ("bSizeRelProbs", 1),
        ("komiAuto", "false"),
        ("komiMean", 6.5),
        ("komiStdev", 0.0),
        ("handicapProb", 0.0),
        ("maxVisits", visits),
        ("numSearchThreads", 1),
        ("rootNoiseEnabled", "false"),
        ("chosenMoveTemperatureEarly", 0.0),
        ("chosenMoveTemperature", 0.0),
        ("chosenMoveTemperatureOnlyBelowProb", 1.0),
        ("rootNumSymmetriesToSample", 1),
        ("rootSymmetryPruning", "false"),
        ("useGraphSearch", "false"),
        ("initGamesWithPolicy", "false"),
        ("nnMaxBatchSize", inference_batch_size),
        ("nnCacheSizePowerOfTwo", 20),
        ("numNNServerThreadsPerModel", 2),
        ("cudaDeviceToUse", gpu_index),
        ("cudaUseFP16", "true"),
        ("cudaUseNHWC", "true"),
        ("nnRandomize", "false"),
        ("nnRandSeed", "gogogo-curriculum-evaluation-v1"),
    ]
    if opening_directory is not None:
        values.extend(
            [
                ("hintPosesDir", opening_directory),
                ("hintPosesProb", 1.0),
            ]
        )
    return "\n".join(f"{key} = {value}" for key, value in values) + "\n"


__all__ = [
    "MatchGame",
    "MatchSummary",
    "approximate_elo",
    "build_match_config",
    "parse_match_game",
    "summarize_match_sgfs",
    "wilson_interval",
]
