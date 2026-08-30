"""Local Tkinter Go game package."""

from .ai import AI_DIFFICULTIES, AIMove, GoAI
from .engine import BLACK, EMPTY, WHITE, GoGame, MoveAnalysis, ScoreResult
from .training import (
    TRAINING_CATEGORIES,
    TRAINING_LESSONS,
    TrainingLesson,
    TrainingMove,
)
from .winrate import WinRateEstimate, WinRateEstimator

__all__ = [
    "AIMove",
    "AI_DIFFICULTIES",
    "BLACK",
    "EMPTY",
    "GoAI",
    "GoGame",
    "MoveAnalysis",
    "ScoreResult",
    "TRAINING_CATEGORIES",
    "TRAINING_LESSONS",
    "TrainingLesson",
    "TrainingMove",
    "WHITE",
    "WinRateEstimate",
    "WinRateEstimator",
]
