"""Keep a frozen champion in the actor pool and evaluate exports separately."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import shutil
import uuid

from .fileio import atomic_open, atomic_write_json, local_timestamp, sha256_file
from .katago_mixed_eval import _select_candidate, evaluate_mixed
from .katago_mixed_health import build_mixed_health
from .katago_rl import KataGoRLRunnerError
from .rl_metrics import parse_model_name
from .training_standard import TrainingStandard, get_training_standard


@dataclass(frozen=True)
class PromotionConfig:
    every_cycles: int = 50
    screen_games_per_board: int = 12
    confirmation_games_per_board: int = 40
    visits: int = 32
    win_rate: float = .55
    evaluation_boards: tuple[int, ...] | None = None
    require_confidence: bool = True
    parallel_pairs: int = 1
    standard: str | None = None

    @classmethod
    def load(cls, raw, board_sizes):
        if not isinstance(raw, dict) or set(raw) - cls.__dataclass_fields__.keys():
            raise ValueError("promotion 配置包含未知字段或格式无效")
        config = cls(**raw)
        get_training_standard(config.standard)
        for name in ("every_cycles", "visits"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"promotion.{name} 必须为正整数")
        for name in ("screen_games_per_board", "confirmation_games_per_board"):
            value = getattr(config, name)
            if type(value) is not int or value < 2 or value % 2:
                raise ValueError(f"promotion.{name} 必须为正偶数")
        if type(config.win_rate) not in (int, float) or not .5 < config.win_rate <= 1:
            raise ValueError("promotion.win_rate 必须大于 0.5 且不超过 1")
        if type(config.require_confidence) is not bool:
            raise ValueError("promotion.require_confidence 必须是布尔值")
        if type(config.parallel_pairs) is not int or not 1 <= config.parallel_pairs <= 16:
            raise ValueError("promotion.parallel_pairs 必须是 1 到 16 的整数")
        if "evaluation_boards" in raw:
            boards = config.evaluation_boards
            if (not isinstance(boards, list) or not boards
                    or any(type(board) is not int or board not in board_sizes for board in boards)
                    or len(set(boards)) != len(boards)):
                raise ValueError("promotion.evaluation_boards 必须是非空且无重复的混训棋盘子集")
            config = replace(config, evaluation_boards=tuple(boards))
        return config


def _match_scores(report, boards, games):
    if report["status"] != "complete":
        raise ValueError("晋级对战未完成")
    scores = []
    for board in boards:
        row = report["results_by_board"][str(board)]
        pairs = row["pair_statistics"]["scores"]
        if (row["games"] != games or row["no_result_games"] or row["damaged_games"]
                or len(pairs) != games // 2 or any(score not in (0, .25, .5, .75, 1) for score in pairs)):
            raise ValueError(f"{board} 路晋级对战缺少有效配对结果")
        scores.append(pairs)
    return scores


class PromotionManager:
    HEALTH_GAMES_PER_BOARD = 32
    LIVE_MINIMUM_GAMES = 24

    def __init__(self, runner, config: PromotionConfig):
        self.runner = runner
        self.config = config
        self.root = runner.run_root / "promotion"
        self.path = self.root / "state.json"

    def read_state(self):
        if not self.path.exists():
            return None
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state["schema_version"] != 1 or state["boards"] != list(self.runner.mixed.board_sizes):
            raise KataGoRLRunnerError("晋级状态与当前混训配置不兼容")
        return state

    def _save(self, state):
        atomic_write_json(self.path, state, allow_nan=False)

    def _snapshot_path(self, name):
        if parse_model_name(name) is None:
            raise KataGoRLRunnerError("晋级状态中的模型名称无效")
        return self.root / "snapshots" / name / "models" / name / "model.bin.gz"

    def _freeze(self, source):
        target = self._snapshot_path(source.parent.name)
        expected = sha256_file(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with atomic_open(target, mode="wb") as stream, source.open("rb") as original:
                shutil.copyfileobj(original, stream)
        if sha256_file(target) != expected:
            raise KataGoRLRunnerError("冻结的晋级模型与训练导出不一致")
        return target, expected

    def _verified_snapshot(self, name, expected):
        model = self._snapshot_path(name)
        if not model.is_file() or sha256_file(model) != expected:
            raise KataGoRLRunnerError("晋级模型快照缺失或校验失败")
        if list(model.parent.parent.iterdir()) != [model.parent] or list(model.parent.iterdir()) != [model]:
            raise KataGoRLRunnerError("冻结模型池出现额外文件或模型")
        return model

    def actor_models(self):
        state = self.read_state()
        if state is None:
            return None
        if state.get("halted"):
            raise KataGoRLRunnerError(f"冠军自对弈触发健康停机，请检查 {self.path}")
        model = self._verified_snapshot(state["champion"], state["champion_sha256"])
        return model.parent.parent

    def _health(self, model, root, *, standard=None):
        # A retry uses a fresh sample. Partial health probes never count toward admission.
        root = root / f"health-{uuid.uuid4().hex}"
        boards = self.runner.mixed.board_sizes
        games = standard.health_minimum_games if standard else self.HEALTH_GAMES_PER_BOARD
        for board in boards:
            self.runner._promotion_probe(model, root, board, games)
        health = build_mixed_health(root, boards, window_games=games, minimum_games=games,
                                    standard=standard.name if standard else None)
        inventory = health["inventory"]
        reasons = [key for key in ("pending_games", "damaged_games", "read_errors",
                                   "unconfigured_board_games") if inventory[key]]
        for board, row in health["by_board"].items():
            if row["cumulative_valid_games"] != games:
                reasons.append(f"{board}:unexpected_game_count")
            # Score margins remain visible diagnostics, matching the curriculum policy.
            warnings = (standard.health_reasons(
                games=row["window_games"], black_score_rate=row["black_score_rate"],
                early_double_pass_rate=row["early_double_pass_rate"],
                short_game_rate=row["short_game_rate"], invalid_rate=row["no_result_rate"],
                pass_move_rate=row["pass_move_rate"])
                if standard else [key for key in row["warnings"] if key != "extreme_margin_rate"])
            reasons.extend(f"{board}:{key}" for key in warnings)
            if row["pass_move_rate"] is None or (not standard and row["pass_move_rate"] > .20):
                reasons.append(f"{board}:pass_move_rate")
        return str(root / "health_by_board.json"), reasons

    def inspect_live_selfplay(self):
        """Stop before shuffling if the current champion starts producing collapsed games."""
        state = self.read_state()
        if state is None:
            return None
        standard = get_training_standard(self.config.standard)
        minimum = standard.live_minimum_games if standard else self.LIVE_MINIMUM_GAMES
        report = build_mixed_health(self.runner.run_root, self.runner.mixed.board_sizes,
                                    minimum_games=minimum, standard=self.config.standard)
        reasons = []
        for board, row in report["by_board"].items():
            champion = row["by_source"].get(state["champion"])
            if champion and champion["estimable"]:
                warnings = (standard.health_reasons(
                    games=champion["window_games"], black_score_rate=champion["black_score_rate"],
                    early_double_pass_rate=champion["early_double_pass_rate"],
                    short_game_rate=champion["short_game_rate"], invalid_rate=champion["no_result_rate"],
                    pass_move_rate=champion["pass_move_rate"], minimum_games=minimum)
                    if standard else [code for code in champion["warnings"]
                                      if code in ("black_score_rate", "short_game_rate", "early_double_pass_rate")])
                reasons.extend(f"{board}:{code}" for code in warnings)
                if champion["pass_move_rate"] is None or (not standard and champion["pass_move_rate"] > .20):
                    reasons.append(f"{board}:pass_move_rate")
        if reasons:
            state["halted"] = {"at": local_timestamp(), "reasons": reasons}
            self._save(state)
            raise KataGoRLRunnerError(f"冠军自对弈触发健康停机：{', '.join(reasons)}")
        return report

    def _migrate_schedule(self, state, *, cycle, samples, standard, legacy_cycle_samples=None):
        """Translate the remaining legacy interval once, without changing a round."""
        if "next_samples" in state or state["pending"]:
            return
        remaining_cycles = max(0, state["next_cycle"] - cycle)
        old_optimizer = self.runner.mixed.training.optimizer
        old_cycle_samples = (legacy_cycle_samples
                             if type(legacy_cycle_samples) is int and legacy_cycle_samples > 0
                             else old_optimizer.batch_size * old_optimizer.training_steps_per_iteration)
        state.update(next_samples=samples + remaining_cycles * old_cycle_samples,
                     training_standard=standard.name,
                     schedule_migration={"from_next_cycle": state["next_cycle"],
                                         "at_cycle": cycle, "at_samples": samples,
                                         "legacy_samples_per_cycle": old_cycle_samples})
        self._save(state)

    def advance(self):
        """Run under the training lock at cycle boundaries."""
        state = self.read_state()
        if state is not None:
            self.actor_models()
        bootstrap = self.runner.bootstrap.read_state()
        if bootstrap is not None and bootstrap["phase"] != "ready":
            return
        if state is None:
            if not any(self.runner.models_dir.glob("*/model.bin.gz")):
                if (bootstrap is not None
                        or (self.runner.run_root / "train/gogogo/checkpoint.ckpt").exists()):
                    raise KataGoRLRunnerError("找不到可参加晋级的训练导出")
                return
            if not parse_model_name(_select_candidate(self.runner.run_root).parent.name)[0]:
                return
        latest = self.runner.metric_store.latest_path
        metrics = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
        cycle = metrics.get("cycle", 0)
        standard = get_training_standard(self.config.standard)
        if standard is None and state is not None and any(
                key in state for key in ("training_standard", "next_samples")):
            for key in ("training_standard", "next_samples", "schedule_migration"):
                state.pop(key, None)
            self._save(state)
        samples = metrics.get("trained_samples")
        # A frozen round can finish even when an old metrics file lacks samples.
        if standard is not None and not (state and state["pending"]):
            if type(samples) is not int or samples < 0:
                raise KataGoRLRunnerError("共享晋级标准需要最近训练记录中的 trained_samples")
            if state is not None:
                self._migrate_schedule(state, cycle=cycle, samples=samples, standard=standard,
                                       legacy_cycle_samples=metrics.get("training_requested_samples"))
        due = state is not None and not state["pending"] and (
            samples >= state["next_samples"] if standard else cycle >= state["next_cycle"])
        if state is None or due:
            if not any(self.runner.models_dir.glob("*/model.bin.gz")):
                if (state is not None or bootstrap is not None
                        or (self.runner.run_root / "train/gogogo/checkpoint.ckpt").exists()):
                    raise KataGoRLRunnerError("找不到可参加晋级的训练导出")
                return
            candidate = _select_candidate(self.runner.run_root)
            name = candidate.parent.name
            if not parse_model_name(name)[0]:
                return
            if state is not None and name == state["champion"]:
                return
            frozen, checksum = self._freeze(candidate)
            if state is None:
                self._verified_snapshot(name, checksum)
                report, reasons = self._health(frozen, self.root / "initial" / name, standard=standard)
                if reasons:
                    raise KataGoRLRunnerError(f"初始冠军未通过自对弈健康检查：{report}（{', '.join(reasons)}）")
                initial = {"schema_version": 1, "boards": list(self.runner.mixed.board_sizes),
                            "champion": name, "champion_sha256": checksum, "initial_health": report,
                            "promotions": 0, "next_cycle": cycle + self.config.every_cycles,
                            "pending": None, "last_result": None}
                if standard:
                    initial.update(training_standard=standard.name,
                                   next_samples=samples + standard.evaluation_interval_samples)
                self._save(initial)
                return
            state["pending"] = {
                "cycle": cycle, "candidate": name, "sha256": checksum,
                "evaluation_boards": list(self.config.evaluation_boards or self.runner.mixed.board_sizes),
                "require_confidence": standard.require_confidence if standard else self.config.require_confidence,
                "win_rate": standard.promotion_win_rate if standard else self.config.win_rate,
                "parallel_pairs": self.config.parallel_pairs,
            }
            if standard:
                state["pending"].update(
                    trained_samples=samples, training_standard=standard.to_dict(),
                    protocol={"screen_games_per_board": standard.screen_games_per_board,
                              "confirmation_games_per_board": standard.confirmation_games_per_board,
                              "visits": standard.evaluation_visits, "max_moves_factor": 3.0,
                              "screen_opening_offset": 0,
                              "confirmation_opening_offset": standard.screen_games_per_board // 2})
            self._save(state)
        if state["pending"]:
            self._evaluate(state)

    def _evaluate(self, state):
        pending = state["pending"]
        config = self.config
        # Legacy pending rounds started with all training boards and the confidence gate.
        legacy_rules = {"evaluation_boards": list(self.runner.mixed.board_sizes),
                        "require_confidence": True, "win_rate": config.win_rate, "parallel_pairs": 1}
        rules = {key: pending.get(key, default) for key, default in legacy_rules.items()}
        standard = (TrainingStandard(**pending["training_standard"])
                    if "training_standard" in pending else None)
        protocol = pending.get("protocol", {
            "screen_games_per_board": config.screen_games_per_board,
            "confirmation_games_per_board": config.confirmation_games_per_board,
            "visits": config.visits, "max_moves_factor": 3.0,
            "screen_opening_offset": 0,
            "confirmation_opening_offset": config.screen_games_per_board // 2})
        boards = rules["evaluation_boards"]
        name, cycle = pending["candidate"], pending["cycle"]
        candidate = self._verified_snapshot(name, pending["sha256"])
        baseline = self._verified_snapshot(state["champion"], state["champion_sha256"])
        root = self.root / "evaluations" / f"cycle-{cycle}-{name}"
        result = {"cycle": cycle, "candidate": name, "baseline": state["champion"],
                  "candidate_sha256": pending["sha256"], "baseline_sha256": state["champion_sha256"],
                  "evaluation_dir": str(root), "at": local_timestamp()}
        if standard:
            result.update(training_standard=standard.to_dict(), protocol=protocol,
                          trained_samples=pending["trained_samples"])
        decision = root / "decision.json"
        if decision.exists():
            saved = json.loads(decision.read_text(encoding="utf-8"))
            if (any(saved.get(key) != value for key, value in result.items() if key != "at")
                    or any(saved.get(key, legacy_rules[key]) != value for key, value in rules.items())):
                raise KataGoRLRunnerError("晋级决策与待评测模型不一致")
            self._finish(state, saved)
            return
        result.update(rules)

        def scores(stage, games, offset):
            report = evaluate_mixed(self.runner, root / stage, candidate=candidate, baseline=baseline,
                                    boards=boards, games_per_board=games, visits=protocol["visits"],
                                    max_moves_factor=protocol["max_moves_factor"], opening_offset=offset,
                                    parallel_pairs=rules["parallel_pairs"])
            return _match_scores(report, boards, games)

        try:
            screen = scores("screen", protocol["screen_games_per_board"], protocol["screen_opening_offset"])
            result["screen_score"] = sum(map(sum, screen)) / sum(map(len, screen))
            result["outcome"] = "screen_rejected"
            if result["screen_score"] >= (standard.screen_min_score if standard else .5):
                confirmation = scores("confirmation", protocol["confirmation_games_per_board"],
                                      protocol["confirmation_opening_offset"])
                count = sum(map(len, confirmation))
                mean = sum(map(sum, confirmation)) / count
                lower = mean - math.sqrt(math.log(1 / .05) / (2 * count))
                by_board = {str(board): sum(pairs) / len(pairs)
                            for board, pairs in zip(boards, confirmation)}
                result.update(outcome="confirmation_rejected", confirmation_score=mean,
                              confirmation_lower=max(0., lower), confirmation_by_board=by_board)
                if standard:
                    promotion_reasons = standard.promotion_reasons(
                        score=mean, by_board=by_board, complete=True, confidence_lower=lower)
                    result["promotion_reasons"] = list(promotion_reasons)
                    accepted = not promotion_reasons
                else:
                    accepted = (mean >= rules["win_rate"] and min(by_board.values()) >= .5
                                and (not rules["require_confidence"] or lower > .5))
                if accepted:
                    report, reasons = self._health(candidate, root, standard=standard)
                    result.update(outcome="health_rejected" if reasons else "promoted",
                                  health_report=report, health_reasons=reasons)
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            result.update(outcome="evaluation_failed", error=str(error))
        atomic_write_json(decision, result, allow_nan=False)
        self._finish(state, result)

    def _finish(self, state, result):
        if result["outcome"] == "promoted":
            state.update(champion=result["candidate"], champion_sha256=result["candidate_sha256"],
                         promotions=state["promotions"] + 1)
        state.update(pending=None, next_cycle=result["cycle"] + self.config.every_cycles,
                     last_result=result)
        # A completed round retains its frozen policy in the decision. Future
        # rounds follow the currently selected configuration and actual progress.
        standard = get_training_standard(self.config.standard)
        if standard:
            latest = self.runner.metric_store.latest_path
            metrics = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
            samples = metrics.get("trained_samples")
            frozen_samples = result.get("trained_samples", parse_model_name(result["candidate"])[0])
            samples = max(samples, frozen_samples) if type(samples) is int and samples >= 0 else frozen_samples
            state.update(training_standard=standard.name,
                         next_samples=samples + standard.evaluation_interval_samples)
        else:
            for key in ("training_standard", "next_samples", "schedule_migration"):
                state.pop(key, None)
        self._save(state)
        print(json.dumps({"promotion": result, "promotions": state["promotions"]}, ensure_ascii=False), flush=True)
