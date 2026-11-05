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
from .training_standard import get_training_standard


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
    short_game_rate: Optional[float] = None
    pass_move_rate: Optional[float] = None

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
    minimum_games: Optional[int] = None,
    training_standard: Optional[str] = None,
) -> HealthSummary:
    """Aggregate the most recent self-play cycles and apply all health gates."""

    standard = get_training_standard(training_standard)
    selected = list(records[-window_cycles:])
    required_games = window_cycles * games_per_cycle if minimum_games is None else minimum_games
    if standard is not None:
        required_games = standard.health_minimum_games if minimum_games is None else minimum_games
        selected = []
        total = 0
        for record in reversed(records):
            selected.insert(0, record)
            total += _record_int(record, "games", "sgf_games", "selfplay_games")
            if total >= required_games:
                break
    if required_games < 1:
        raise ValueError("健康窗口的最低局数必须为正数")
    if minimum_games is not None:
        # Hardware tiers produce different games per cycle. Retain at least the
        # original cycle window, then include older cycles until evidence is full.
        count = len(selected)
        total = sum(_record_int(record, "games", "sgf_games", "selfplay_games") for record in selected)
        while count < len(records) and total < required_games:
            count += 1
            record = records[-count]
            selected.insert(0, record)
            total += _record_int(record, "games", "sgf_games", "selfplay_games")
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
    complete = games >= required_games and (standard is not None or len(selected) >= window_cycles)
    decided = black + white + draws
    black_rate = (black + 0.5 * draws) / decided if decided else 0.0
    double_rate = double_pass / games if games else 1.0
    extreme_rate = extreme / games if games else 1.0
    invalid_rate = min(games, invalid) / games if games else 1.0
    short_rate = (
        sum(_record_int(record, "short_games") for record in selected) / games
        if games and all("short_games" in record for record in selected) else None
    )
    total_moves = sum(_record_int(record, "total_moves") for record in selected)
    pass_rate = (
        sum(_record_int(record, "pass_moves") for record in selected) / total_moves
        if total_moves and all("pass_moves" in record and "total_moves" in record for record in selected)
        else None
    )
    reasons: list[str] = []
    if standard is not None:
        reasons.extend(standard.health_reasons(
            games=games, black_score_rate=black_rate,
            early_double_pass_rate=double_rate, invalid_rate=invalid_rate,
            short_game_rate=short_rate, pass_move_rate=pass_rate,
            minimum_games=required_games,
        ))
    elif not complete:
        reasons.append("健康窗口不足")
    if standard is None and double_rate > thresholds.immediate_double_pass_rate:
        reasons.append("开局双方立即停一手比例超标")
    if standard is None and not thresholds.black_win_rate_min <= black_rate <= thresholds.black_win_rate_max:
        reasons.append("黑方胜率超出 45%–55%")
    if standard is None and invalid_rate > thresholds.invalid_rate:
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
        short_game_rate=short_rate,
        pass_move_rate=pass_rate,
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
    standard = get_training_standard(config.training_standard)
    if standard is not None:
        reasons.extend(standard.promotion_reasons(
            score=match.win_rate, by_board={"active": match.win_rate},
            complete=(match.games == config.evaluation_games
                      and match.requested_games == config.evaluation_games
                      and match.candidate_black_games == config.evaluation_games // 2
                      and match.candidate_white_games == config.evaluation_games // 2
                      and not match.damaged_games and not match.no_result_games),
            confidence_lower=match.wilson_lower, health_reasons=health.reasons,
        ))
        if not health.passed and not health.reasons:
            reasons.append("health_failed")
        return QualityGate(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))
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

    def apply_training_standard(self, state: CurriculumState) -> None:
        """Adopt future sample gates without reinterpreting recorded decisions."""
        name = self.config.training_standard
        if state.active.training_standard == name:
            return
        candidate = copy.deepcopy(state)
        active = candidate.active
        evaluation_root = self.store.state_root / "evaluations" / progress_key(active.board_size)
        prefix = f"s{active.samples}-{active.latest_model or 'unknown'}-"
        previous_paths = (
            sorted(str(path) for path in evaluation_root.iterdir()
                   if path.is_dir() and path.name.startswith(prefix)
                   and not (path / "summary.json").exists())
            if evaluation_root.is_dir() else []
        )
        if (candidate.phase == "evaluating" or candidate.resume_phase == "evaluating"
                or active.pending_champion_evaluation or previous_paths):
            # Old manifests bind engine evidence but omit the complete gate
            # policy. Keep their bytes and decisions; never infer old rules or
            # reuse their outcomes as evidence under a newly selected standard.
            standard = get_training_standard(name)
            candidate.warnings.append({
                "timestamp": local_timestamp(), "kind": "evaluation_deferred",
                "reason": "training_standard_changed",
                "message": "训练标准已改变；旧评测缺少完整门槛快照，保留原证据并延期，按新样本边界启动独立评测",
                "candidate_model": active.latest_model,
                "stage_samples": active.samples,
                "previous_training_standard": active.training_standard,
                "training_standard": standard.to_dict() if standard is not None else None,
                "previous_phase": candidate.phase,
                "previous_resume_phase": candidate.resume_phase,
                "previous_pending_champion_evaluation": active.pending_champion_evaluation,
                "previous_next_evaluation_sample": active.next_evaluation_sample,
                "evaluation_paths": previous_paths,
            })
            active.pending_champion_evaluation = False
            if candidate.phase == "paused_disk":
                candidate.resume_phase = "training"
            else:
                candidate.phase = "training"
        candidate.active.training_standard = name
        candidate.active.next_evaluation_sample = self.next_evaluation_sample(
            candidate.active_stage_index, candidate.active.samples)
        candidate.active.consecutive_passes = 0
        if candidate.phase == "transition_ready":
            candidate.phase = "training"
        if candidate.resume_phase == "transition_ready":
            candidate.resume_phase = "training"
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)

    def next_evaluation_sample(self, stage_index: int, samples: int) -> int:
        """Use a wider interval mid-stage and dense checks near a finite gate."""
        interval = self.config.evaluation_interval_samples
        midstage = self.config.evaluation_midstage_interval_samples
        if midstage is None:
            return next_evaluation_boundary(samples, interval)
        minimum = self.config.stages[stage_index].min_stage_samples
        if minimum is None:
            return next_evaluation_boundary(samples, midstage)
        near_gate = max(0, minimum - self.config.evaluation_near_gate_samples)
        if samples >= near_gate:
            return next_evaluation_boundary(samples, interval)
        return min(next_evaluation_boundary(samples, midstage), near_gate)

    def adopt_existing_9x9(
        self,
        *,
        trained_samples: int,
        latest_model: str,
        entry_model: Optional[str] = None,
    ) -> CurriculumState:
        existing = self.store.load()
        if existing is not None:
            self.apply_training_standard(existing)
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
            next_evaluation_sample=self.next_evaluation_sample(0, trained_samples),
            training_standard=self.config.training_standard,
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
        self.apply_training_standard(state)
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

    def record_champion_evaluation(self, state: CurriculumState, *, match: MatchSummary, training_cycle=None,
                                   health: Optional[HealthSummary] = None) -> None:
        """Commit a full champion challenge without advancing the fixed-pool gate."""
        if (match.games != self.config.evaluation_games
                or match.requested_games != self.config.evaluation_games
                or match.candidate_black_games != self.config.evaluation_games // 2
                or match.candidate_white_games != self.config.evaluation_games // 2
                or match.damaged_games or match.no_result_games):
            raise CurriculumStateError("冠军评测必须完整完成指定局数并严格交换黑白")
        candidate = copy.deepcopy(state)
        active = candidate.active
        before = active.champion_model
        if not before or before == active.latest_model:
            raise CurriculumStateError("冠军评测需要不同于候选的冠军模型")
        promoted = (match.win_rate >= self.config.promotion_win_rate
                    and match.wilson_lower > self.config.wilson_lower_bound)
        standard = get_training_standard(self.config.training_standard)
        champion_reasons = ()
        if standard is not None:
            health_reasons = ("health_missing",) if health is None else health.reasons
            if health is not None and not health.passed and not health_reasons:
                health_reasons = ("health_failed",)
            champion_reasons = standard.promotion_reasons(
                score=match.win_rate, by_board={active.board_size: match.win_rate},
                complete=True, confidence_lower=match.wilson_lower,
                health_reasons=health_reasons,
            )
            promoted = not champion_reasons
        if promoted:
            active.champion_model = active.latest_model
            active.champion_established = True
        active.evaluations.append({
            "evaluation_scope": "champion_cycle", "timestamp": local_timestamp(),
            "training_cycle": training_cycle,
            "candidate_model": active.latest_model, "stage_samples": active.samples,
            "champion_before": before, "champion_after": active.champion_model,
            "champion_promoted": promoted,
            "opponents": [{"model": before, "roles": ["champion"], "match": match.to_dict()}],
        })
        if standard is not None:
            active.evaluations[-1].update(training_standard=standard.to_dict(),
                                          champion_reasons=list(champion_reasons),
                                          health=health.to_dict() if health is not None else None)
        active.pending_champion_evaluation = False
        candidate.phase = "training"
        self.store.save(candidate)
        state.__dict__.update(candidate.__dict__)

    def record_evaluation(
        self,
        state: CurriculumState,
        *,
        candidate_model: str,
        match: MatchSummary,
        health: HealthSummary,
        opponent_results: Optional[Mapping[str, MatchSummary]] = None,
        training_cycle: Optional[int] = None,
    ) -> QualityGate:
        stage_definition = self.config.stages[state.active_stage_index]
        # Stage advancement uses every frozen historical anchor. Champion promotion
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
            "training_cycle": training_cycle,
            "stage_gate_version": 2,
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
        standard = get_training_standard(self.config.training_standard)
        if standard is not None:
            record.update(training_standard=standard.to_dict(), champion_promoted=False)
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
            reasons = list(gate.reasons)
            fixed_gates = []
            for name in state.active.fixed_baseline_models or [baseline]:
                if name not in opponent_results:
                    raise CurriculumStateError("固定历史基准必须完成真实对弈，不能与候选模型相同")
                fixed_gate = evaluate_quality_gate(
                    opponent_results[name], health, self.config,
                    stage_samples=state.active.samples,
                    minimum_samples=stage_definition.min_stage_samples,
                )
                fixed_gates.append({
                    "model": name, "passed": fixed_gate.passed,
                    "reasons": list(fixed_gate.reasons),
                })
                if name != baseline:
                    # Shared sample/health failures are already reported above.
                    reasons.extend(
                        f"固定基准 {name}：{reason}"
                        for reason in fixed_gate.reasons if reason not in gate.reasons
                    )
            gate = QualityGate(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))
            record.update(passed=gate.passed, reasons=list(gate.reasons), fixed_gates=fixed_gates)
            champion_match = opponent_results.get(champion) if champion else None
            champion_passed = champion_match is not None and (
                champion_match.games == self.config.evaluation_games
                and champion_match.candidate_black_games == self.config.evaluation_games // 2
                and champion_match.candidate_white_games == self.config.evaluation_games // 2
                and not champion_match.damaged_games
                and not champion_match.no_result_games
                and champion_match.win_rate >= self.config.promotion_win_rate
                and champion_match.wilson_lower > self.config.wilson_lower_bound
            )
            if standard is not None and champion_match is not None:
                champion_gate = evaluate_quality_gate(
                    champion_match, health, self.config, stage_samples=state.active.samples,
                    minimum_samples=None, rolling_champion=True)
                champion_passed = champion_gate.passed
                record["champion_reasons"] = list(champion_gate.reasons)
            if champion_passed:
                candidate.active.champion_model = candidate_model
                candidate.active.champion_established = True
                record["champion_promoted"] = True
        elif len(state.active.fixed_baseline_models) > 1:
            raise CurriculumStateError("多个固定历史基准的晋级必须提供全部对手结果")
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
        candidate.active.next_evaluation_sample = self.next_evaluation_sample(
            candidate.active_stage_index, candidate.active.samples
        )
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
        standard = get_training_standard(self.config.training_standard)
        if standard is not None:
            candidate.active.evaluations[-1]["training_standard"] = standard.to_dict()
        candidate.active.consecutive_passes = 0
        candidate.active.next_evaluation_sample = self.next_evaluation_sample(
            candidate.active_stage_index, candidate.active.samples
        )
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
            next_evaluation_sample=self.next_evaluation_sample(candidate.active_stage_index, 0),
            autotune_batch=autotune_batch,
            training_standard=self.config.training_standard,
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
