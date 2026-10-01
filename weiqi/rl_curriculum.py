"""Resumable curriculum orchestration primitives for KataGo reinforcement learning.

This module is the curriculum's public entry point: it holds the health
window, quality gate and controller, and re-exports the configuration
(``rl_curriculum_config``) and durable state (``rl_curriculum_state``) API.

The module deliberately keeps process execution behind small callbacks.  The CLI
runner can therefore use the real KataGo executable while the safety-critical
state machine, promotion gates, retention rules, and migrations remain fast to
unit test.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from .fileio import local_timestamp
from .rl_curriculum_config import (
    CURRICULUM_SCHEMA_VERSION,
    EXPECTED_BOARDS,
    CurriculumConfig,
    CurriculumConfigError,
    CurriculumStage,
    HealthThresholds,
    load_curriculum_config,
    validate_training_profiles,
)
from .rl_curriculum_state import (
    STATE_SCHEMA_VERSION,
    CurriculumLockError,
    CurriculumState,
    CurriculumStateError,
    CurriculumStateStore,
    GlobalCurriculumLock,
    StageProgress,
    progress_key,
)
from .rl_match import MatchSummary


DEFAULT_HEALTH_GAMES_PER_CYCLE = 128


class CurriculumMigrationError(RuntimeError):
    """Raised when a stage seed cannot be prepared and verified safely."""


def next_evaluation_boundary(samples: int, interval: int) -> int:
    """Return the first evaluation boundary strictly after ``samples``."""

    if samples < 0 or interval < 1:
        raise ValueError("samples 不能为负且 interval 必须为正")
    return (samples // interval + 1) * interval


@dataclass(frozen=True)
class HealthSummary:
    cycles: int
    games: int
    black_wins: int
    white_wins: int
    draws: int
    immediate_double_pass_games: int
    extreme_games: int
    invalid_games: int
    black_win_rate: float
    immediate_double_pass_rate: float
    extreme_result_rate: float
    invalid_rate: float
    complete: bool
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def _record_int(record: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0, int(value))
    return 0


def compute_health_window(
    records: Sequence[Mapping[str, object]],
    *,
    window_cycles: int = 10,
    games_per_cycle: int = DEFAULT_HEALTH_GAMES_PER_CYCLE,
    thresholds: HealthThresholds = HealthThresholds(),
) -> HealthSummary:
    """Aggregate the most recent self-play cycles and apply all health gates."""

    selected = list(records[-window_cycles:])
    games = sum(
        _record_int(record, "games", "sgf_games", "selfplay_games")
        for record in selected
    )
    black = sum(_record_int(record, "black_wins") for record in selected)
    white = sum(_record_int(record, "white_wins") for record in selected)
    draws = sum(_record_int(record, "draws") for record in selected)
    double_pass = sum(
        _record_int(
            record,
            "immediate_double_pass_games",
            "opening_double_pass_games",
            "double_pass_games",
        )
        for record in selected
    )
    extreme = sum(
        _record_int(record, "extreme_games", "extreme_result_games")
        for record in selected
    )
    invalid = sum(
        _record_int(record, "invalid_games", "no_result_games")
        + _record_int(record, "damaged_games", "damaged_sgfs")
        for record in selected
    )
    complete = len(selected) >= window_cycles and games >= window_cycles * games_per_cycle
    decided = black + white + draws
    black_rate = (black + 0.5 * draws) / decided if decided else 0.0
    double_rate = double_pass / games if games else 1.0
    extreme_rate = extreme / games if games else 1.0
    invalid_rate = min(games, invalid) / games if games else 1.0
    reasons: list[str] = []
    if not complete:
        reasons.append("健康窗口不足")
    if double_rate > thresholds.immediate_double_pass_rate:
        reasons.append("开局双方立即停一手比例超标")
    if not thresholds.black_win_rate_min <= black_rate <= thresholds.black_win_rate_max:
        reasons.append("黑方胜率超出 45%–55%")
    if invalid_rate > thresholds.invalid_rate:
        reasons.append("无结果或损坏棋谱比例超标")
    return HealthSummary(
        cycles=len(selected),
        games=games,
        black_wins=black,
        white_wins=white,
        draws=draws,
        immediate_double_pass_games=double_pass,
        extreme_games=extreme,
        invalid_games=min(games, invalid),
        black_win_rate=black_rate,
        immediate_double_pass_rate=double_rate,
        extreme_result_rate=extreme_rate,
        invalid_rate=invalid_rate,
        complete=complete,
        passed=not reasons,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class QualityGate:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_quality_gate(
    match: MatchSummary,
    health: HealthSummary,
    config: CurriculumConfig,
    *,
    stage_samples: int,
    minimum_samples: Optional[int],
    rolling_champion: bool = False,
) -> QualityGate:
    reasons: list[str] = []
    if minimum_samples is not None and stage_samples < minimum_samples:
        reasons.append("尚未达到阶段最低样本数")
    if match.games != config.evaluation_games:
        reasons.append("固定评测局数不完整")
    if match.win_rate < config.promotion_win_rate:
        reasons.append("评测胜率低于晋级门槛")
    if not rolling_champion and match.wilson_lower <= config.wilson_lower_bound:
        reasons.append("Wilson 置信区间下界未高于门槛")
    if not health.passed:
        reasons.extend(health.reasons)
    return QualityGate(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))


class CurriculumController:
    """Persistent 9 -> 13 -> 19 state transitions and quality decisions."""

    def __init__(self, config: CurriculumConfig, store: CurriculumStateStore):
        self.config = config
        self.store = store

    def adopt_existing_9x9(
        self,
        *,
        trained_samples: int,
        latest_model: str,
        entry_model: Optional[str] = None,
    ) -> CurriculumState:
        existing = self.store.load()
        if existing is not None:
            return existing
        if trained_samples < 0 or not latest_model:
            raise CurriculumStateError("无法接管无效的 9x9 训练状态")
        baseline = entry_model or latest_model
        progress = StageProgress(
            board_size=9,
            samples=trained_samples,
            entry_model=baseline,
            baseline_model=baseline,
            latest_model=latest_model,
            next_evaluation_sample=next_evaluation_boundary(
                trained_samples, self.config.evaluation_interval_samples
            ),
        )
        state = CurriculumState(
            schema_version=STATE_SCHEMA_VERSION,
            active_stage_index=0,
            phase="training",
            stages={progress_key(9): progress},
            protected_models=list(dict.fromkeys([baseline, latest_model])),
        )
        self.store.save(state)
        return state

    def mark_cycle(
        self,
        state: CurriculumState,
        *,
        trained_samples: int,
        latest_model: str,
    ) -> bool:
        if trained_samples < state.active.samples:
            raise CurriculumStateError("阶段训练样本数不能倒退")
        state.active.samples = trained_samples
        state.active.latest_model = latest_model
        due = trained_samples >= state.active.next_evaluation_sample
        if state.phase != "paused_disk":
            state.phase = "evaluating" if due else "training"
        self.store.save(state)
        return due

    def evaluation_baseline(self, state: CurriculumState) -> Optional[str]:
        return state.active.champion_model or state.active.baseline_model

    def evaluation_opponents(self, state: CurriculumState) -> dict[str, list[str]]:
        """One match per distinct opponent, even when it has both pool roles."""
        active = state.active
        opponents: dict[str, list[str]] = {}
        champion = self.evaluation_baseline(state)
        if champion and champion != active.latest_model:
            opponents.setdefault(champion, []).append("champion")
        for name in active.fixed_baseline_models or [active.baseline_model]:
            if name and name != active.latest_model:
                opponents.setdefault(name, []).append("fixed")
        return opponents

    def record_evaluation(
        self,
        state: CurriculumState,
        *,
        candidate_model: str,
        match: MatchSummary,
        health: HealthSummary,
        opponent_results: Optional[Mapping[str, MatchSummary]] = None,
    ) -> QualityGate:
        stage_definition = self.config.stages[state.active_stage_index]
        # Stage advancement uses a stable historical anchor. Champion promotion
        # measures playing strength independently of self-play health and samples.
        baseline = state.active.baseline_model
        gate = evaluate_quality_gate(
            match,
            health,
            self.config,
            stage_samples=state.active.samples,
            minimum_samples=stage_definition.min_stage_samples,
        )
        candidate = copy.deepcopy(state)
        record: dict[str, object] = {
            "timestamp": local_timestamp(),
            "candidate_model": candidate_model,
            "baseline_model": baseline,
            "stage_samples": state.active.samples,
            "passed": gate.passed,
            "reasons": list(gate.reasons),
            "match": match.to_dict(),
            "health": health.to_dict(),
        }
        champion = state.active.champion_model
        record["champion_before"] = champion
        if opponent_results is not None:
            expected = self.evaluation_opponents(state)
            if set(opponent_results) != set(expected):
                raise CurriculumStateError("两类评测对手结果不完整")
            if baseline not in opponent_results or opponent_results[baseline] != match:
                raise CurriculumStateError("课程门槛必须使用首个固定历史基准的成绩")
            if any(
                result.games != self.config.evaluation_games
                or result.candidate_black_games != self.config.evaluation_games // 2
                or result.candidate_white_games != self.config.evaluation_games // 2
                or result.damaged_games or result.no_result_games
                for result in opponent_results.values()
            ):
                raise CurriculumStateError("评测局数不完整或没有严格交换黑白")
            record["opponents"] = [
                {"model": name, "roles": roles, "match": opponent_results[name].to_dict()}
                for name, roles in expected.items()
            ]
            champion_match = opponent_results.get(champion) if champion else None
            if champion_match is not None and (
                champion_match.games == self.config.evaluation_games
                and champion_match.candidate_black_games == self.config.evaluation_games // 2
                and champion_match.candidate_white_games == self.config.evaluation_games // 2
                and not champion_match.damaged_games
                and not champion_match.no_result_games
                and champion_match.win_rate >= self.config.promotion_win_rate
                and champion_match.wilson_lower > self.config.wilson_lower_bound
            ):
                candidate.active.champion_model = candidate_model
                candidate.active.champion_established = True
                record["champion_promoted"] = True
        record["champion_after"] = candidate.active.champion_model
        if stage_definition.indefinite:
            candidate.active.consecutive_passes = 0
            candidate.phase = "training"
        else:
            candidate.active.consecutive_passes = (
                candidate.active.consecutive_passes + 1 if gate.passed else 0
            )
            candidate.phase = (
                "transition_ready"
                if candidate.active.consecutive_passes
                >= self.config.required_consecutive_passes
                else "training"
            )
        candidate.active.evaluations.append(record)
        while candidate.active.next_evaluation_sample <= candidate.active.samples:
            candidate.active.next_evaluation_sample += self.config.evaluation_interval_samples
        candidate.protected_models = list(dict.fromkeys(
            candidate.protected_models + candidate.active.fixed_baseline_models
            + ([candidate.active.champion_model] if candidate.active.champion_model else [])
        ))
        if champion and champion != candidate.active.champion_model:
            still_needed = {
                name for progress in candidate.stages.values()
                for name in [progress.entry_model, progress.baseline_model, progress.latest_model,
                             progress.champion_model, *progress.fixed_baseline_models]
            }
            if champion not in still_needed:
                candidate.protected_models = [name for name in candidate.protected_models if name != champion]
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)
        return gate

    def record_evaluation_error(
        self, state: CurriculumState, message: str
    ) -> None:
        candidate = copy.deepcopy(state)
        candidate.warnings.append(
            {"timestamp": local_timestamp(), "kind": "evaluation", "message": message}
        )
        candidate.active.evaluations.append(
            {
                "timestamp": local_timestamp(),
                "stage_samples": candidate.active.samples,
                "passed": False,
                "error": message,
            }
        )
        candidate.active.consecutive_passes = 0
        while candidate.active.next_evaluation_sample <= candidate.active.samples:
            candidate.active.next_evaluation_sample += self.config.evaluation_interval_samples
        candidate.phase = "training"
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)

    def begin_migration(self, state: CurriculumState) -> None:
        if state.phase != "transition_ready":
            raise CurriculumStateError("只有通过连续质量门槛后才能迁移")
        if state.active_stage_index >= len(self.config.stages) - 1:
            raise CurriculumStateError("19x19 是最终持续阶段")
        state.phase = "migrating"
        self.store.save(state)

    def migration_failed(self, state: CurriculumState, message: str) -> None:
        state.warnings.append(
            {"timestamp": local_timestamp(), "kind": "migration", "message": message}
        )
        # Keep training the old board and allow the next boundary to retry.
        state.active.consecutive_passes = 0
        state.phase = "training"
        state.migration = {}
        self.store.save(state)

    def complete_migration(
        self,
        state: CurriculumState,
        *,
        seed_model: str,
        autotune_batch: Optional[int] = None,
    ) -> None:
        if state.phase != "migrating":
            raise CurriculumStateError("迁移尚未开始")
        # Mutate a private candidate and only publish it to the caller after
        # the atomic state write succeeds.  This prevents a failed save from
        # leaving the in-memory object one stage ahead of durable state.
        candidate = copy.deepcopy(state)
        old = candidate.active
        old.completed_at = local_timestamp()
        candidate.active_stage_index += 1
        board = EXPECTED_BOARDS[candidate.active_stage_index]
        progress = StageProgress(
            board_size=board,
            samples=0,
            entry_model=seed_model,
            baseline_model=seed_model,
            latest_model=seed_model,
            champion_model=seed_model,
            fixed_baseline_models=[seed_model],
            next_evaluation_sample=self.config.evaluation_interval_samples,
            autotune_batch=autotune_batch,
        )
        candidate.stages[progress_key(board)] = progress
        candidate.protected_models.append(seed_model)
        candidate.protected_models = list(dict.fromkeys(candidate.protected_models))
        candidate.phase = "training"
        candidate.migration = {}
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)

    def update_disk(self, state: CurriculumState, free_gib: float) -> str:
        if free_gib < 0:
            raise ValueError("free_gib 不能为负")
        if free_gib < self.config.disk_pause_gib:
            status = "paused"
            if state.phase != "paused_disk":
                state.resume_phase = state.phase
            state.phase = "paused_disk"
        elif free_gib < self.config.disk_cleanup_gib:
            status = "cleanup"
            if state.phase == "paused_disk":
                state.phase = state.resume_phase or "training"
                state.resume_phase = None
        else:
            status = "healthy"
            if state.phase == "paused_disk":
                state.phase = state.resume_phase or "training"
                state.resume_phase = None
        state.disk = {"free_gib": free_gib, "status": status, "checked_at": local_timestamp()}
        self.store.save(state)
        return status


__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "CurriculumConfig",
    "CurriculumConfigError",
    "CurriculumController",
    "CurriculumLockError",
    "CurriculumMigrationError",
    "CurriculumStage",
    "CurriculumState",
    "CurriculumStateError",
    "CurriculumStateStore",
    "GlobalCurriculumLock",
    "HealthSummary",
    "HealthThresholds",
    "QualityGate",
    "StageProgress",
    "compute_health_window",
    "evaluate_quality_gate",
    "load_curriculum_config",
    "next_evaluation_boundary",
    "validate_training_profiles",
]
