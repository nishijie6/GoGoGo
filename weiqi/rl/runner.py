"""Synchronous self-play -> replay -> learning -> evaluation -> checkpoint loop."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..engine import BLACK, WHITE
from ..fileio import sha256_file
from ..rl_config import RLTrainingConfig
from .control import Progress, RunLock, TrainingControl
from .eval_pool import evaluate_pool, save_anchor, select_anchors
from .eval_quality import evaluate_positions
from .eval_stats import confirmed_improvement, paired_confidence, paired_sign_test, paired_sprt
from .network import PolicyValueNet, Runtime
from .selfplay import GameJob, evaluation_summary, run_games
from .state import FEATURE_VERSION
from .storage import (
    CHECKPOINT_VERSION, ReplayBuffer, atomic_json, atomic_torch_save,
    cpu_state, fingerprint, load_checkpoint, write_games,
)


class Trainer:
    def __init__(self, config: RLTrainingConfig, output: Path, resume: dict | None = None,
                 *, progress: Progress | None = None):
        self.config, self.output = config, output
        self.progress = progress if progress is not None else Progress(output)
        self.control = TrainingControl(output, config.runtime.pause_while_game_is_active, self.progress)
        self.control()
        self.runtime = Runtime(config)
        torch.manual_seed(config.runtime.random_seed)
        self.rng = np.random.default_rng(config.runtime.random_seed)
        self.model = PolicyValueNet(config).to(self.runtime.device)
        self.best = PolicyValueNet(config).to(self.runtime.device)
        self.best.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.runtime.precision == "amp_float16")
        self.replay = ReplayBuffer(config.optimizer.replay_buffer_capacity, config.game.board_size,
                                   config.optimizer.use_board_symmetry_augmentation)
        self.iteration = self.training_steps = self.best_iteration = 0
        self.untrained_samples = 0
        if resume is not None:
            if resume.get("kind") != "training":
                raise ValueError("Resume requires latest.pt or a full checkpoint, not a model export")
            self.model.load_state_dict(resume["model"])
            self.best.load_state_dict(resume["best_model"])
            self.optimizer.load_state_dict(resume["optimizer"])
            self.scaler.load_state_dict(resume["scaler"])
            self.replay.restore(resume["replay"])
            self.iteration, self.training_steps = resume["iteration"], resume["training_steps"]
            self.best_iteration = resume["best_iteration"]
            self.untrained_samples = resume.get("untrained_samples", 0)
            self.rng.bit_generator.state = resume["numpy_rng"]
            torch.set_rng_state(resume["torch_rng"])
            if self.runtime.device.type == "cuda" and resume["cuda_rng"]:
                torch.cuda.set_rng_state_all(resume["cuda_rng"])
        self.training_model = torch.compile(self.model) if config.hardware.compile_model else self.model
        self.anchors = output / "anchors"
        save_anchor(self.anchors, self.best, config, kind="accepted", iteration=self.best_iteration)
        if (self.iteration > 0 and
                self.iteration % config.evaluation.milestone_every_iterations == 0):
            save_anchor(self.anchors, self.model, config, kind="milestone", iteration=self.iteration)
        if resume is not None and config.self_play.milestone_fraction > 0:
            self._seed_previous_milestone()
        atomic_json(config.to_dict(), output / "config.resolved.json")
        self.progress({
            "event": "ready", "resumed": resume is not None, "iteration": self.iteration,
            "training_steps": self.training_steps, "best_iteration": self.best_iteration,
            "replay_samples": len(self.replay),
            "device": str(self.runtime.device), "precision": self.runtime.precision,
            "torch": torch.__version__, "parameters": sum(p.numel() for p in self.model.parameters()),
            "model_sha256": fingerprint(cpu_state(self.model)),
            "gpu": torch.cuda.get_device_name(self.runtime.device) if self.runtime.device.type == "cuda" else None,
        })

    def _seed_previous_milestone(self):
        """Expose a distinct saved candidate when a legacy run adopts milestone sparring."""
        candidate_sha = fingerprint(cpu_state(self.model))
        if select_anchors(self.anchors, self.config, candidate_sha256=candidate_sha,
                          kinds=("milestone",), recent=True):
            return
        for path in sorted((self.output / "checkpoints").glob("iteration_*.pt"), reverse=True):
            previous, old_config = load_checkpoint(path)
            if previous.get("kind") != "training" or previous["iteration"] >= self.iteration:
                continue
            if old_config.game != self.config.game or old_config.network != self.config.network:
                raise ValueError(f"Historical checkpoint has incompatible model: {path}")
            if fingerprint(previous["model"]) == candidate_sha:
                continue
            frozen = PolicyValueNet(old_config)
            frozen.load_state_dict(previous["model"])
            save_anchor(self.anchors, frozen, self.config,
                        kind="milestone", iteration=previous["iteration"])
            return

    def _base_payload(self):
        return {"checkpoint_version": CHECKPOINT_VERSION, "feature_version": FEATURE_VERSION,
                "config": self.config.to_dict(), "iteration": self.iteration,
                "training_steps": self.training_steps, "best_iteration": self.best_iteration}

    def save(self, archive: bool = True):
        payload = {
            **self._base_payload(), "kind": "training", "model": cpu_state(self.model),
            "best_model": cpu_state(self.best), "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(), "replay": self.replay.state(),
            "untrained_samples": self.untrained_samples,
            "numpy_rng": deepcopy(self.rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.runtime.device.type == "cuda" else [],
        }
        atomic_torch_save(payload, self.output / "latest.pt")
        for name, model in (("candidate", self.model), ("best", self.best)):
            export = {**self._base_payload(), "kind": "model", "model": cpu_state(model),
                      "model_role": name}
            atomic_torch_save(export, self.output / f"{name}.pt")
        if archive and self.iteration % self.config.runtime.checkpoint_every_iterations == 0:
            path = self.output / "checkpoints" / f"iteration_{self.iteration:06d}.pt"
            atomic_torch_save(payload, path)
            older = sorted(path.parent.glob("iteration_*.pt"))[:-self.config.runtime.keep_checkpoints]
            for old in older:
                old.unlink()

    def training_step_count(self) -> int:
        """Fixed steps, or enough to draw each untrained sample ``target_sample_reuse`` times."""
        optimizer = self.config.optimizer
        if optimizer.target_sample_reuse is None:
            return optimizer.training_steps_per_iteration
        wanted = math.ceil(self.untrained_samples * optimizer.target_sample_reuse / optimizer.batch_size)
        return max(1, min(optimizer.training_steps_per_iteration, wanted))

    def train_updates(self) -> dict:
        config = self.config
        count = self.training_step_count()
        auxiliary = self.model.auxiliary
        self.model.train()
        initial = next(self.model.parameters()).detach().clone()
        started = time.monotonic()
        # Loss sums stay on the device so a step needs no host round trip just
        # to log; float64 matches summing the per-step Python floats exactly.
        device = self.runtime.device
        totals = torch.zeros(5, dtype=torch.float32 if device.type == "mps" else torch.float64,
                             device=device)
        updates = 0
        pin = config.hardware.pin_memory and self.runtime.device.type == "cuda"
        for step in range(1, count + 1):
            self.control()
            batch = self.replay.sample(config.optimizer.batch_size)
            features, policies, values, ownership, scores, aux_weights = (
                (item.pin_memory() if pin else item).to(self.runtime.device, non_blocking=True)
                for item in batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self.runtime.autocast():
                outputs = self.training_model(features, auxiliary=auxiliary)
                predicted_policy, predicted_value = outputs[:2]
                policy_loss = -(policies * F.log_softmax(predicted_policy.float(), dim=1)).sum(dim=1).mean()
                value_loss = F.mse_loss(predicted_value.float(), values)
                loss = policy_loss + value_loss
                ownership_loss = score_loss = torch.zeros((), device=self.runtime.device)
                if auxiliary:
                    # Only played-out games carry ownership and score targets.
                    weight_sum = aux_weights.sum().clamp(min=1.0)
                    ownership_loss = (F.binary_cross_entropy_with_logits(
                        outputs[2].float(), (ownership + 1) / 2, reduction="none",
                    ).mean(dim=1) * aux_weights).sum() / weight_sum
                    score_loss = (F.smooth_l1_loss(
                        outputs[3].float(), scores, reduction="none") * aux_weights).sum() / weight_sum
                    loss = (loss + config.optimizer.ownership_loss_weight * ownership_loss
                            + config.optimizer.score_loss_weight * score_loss)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  config.optimizer.gradient_clip_norm)
            # The step's single host sync: never apply a non-finite loss, nor (without
            # AMP, whose scaler skips overflowing steps itself) a non-finite gradient.
            finite = torch.isfinite(loss)
            if not self.scaler.is_enabled():
                finite = finite & torch.isfinite(norm)
            if not finite.item():
                raise RuntimeError("Non-finite training loss or gradient norm")
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < previous_scale:
                self.progress({"event": "amp_overflow", "attempt": step,
                               "scale": self.scaler.get_scale()})
                continue
            updates += 1
            self.training_steps += 1
            step_losses = torch.stack([loss, policy_loss, value_loss,
                                       ownership_loss, score_loss]).detach()
            totals += step_losses
            if step % config.runtime.log_every_training_steps == 0 or step == count:
                losses = step_losses.tolist()
                event = {"event": "training_step", "step": step, "total": count,
                         "training_steps": self.training_steps,
                         "loss": losses[0], "policy_loss": losses[1], "value_loss": losses[2]}
                if auxiliary:
                    event.update(ownership_loss=losses[3], score_loss=losses[4])
                self.progress(event)
        if torch.equal(initial, next(self.model.parameters()).detach()):
            raise RuntimeError("Training completed without changing model weights")
        new_samples, self.untrained_samples = self.untrained_samples, 0
        totals = totals.tolist()
        result = {"steps": updates, "attempts": count,
                  "loss": totals[0] / updates, "policy_loss": totals[1] / updates,
                  "value_loss": totals[2] / updates, "seconds": time.monotonic() - started,
                  "untrained_samples": new_samples,
                  "sample_reuse": (count * config.optimizer.batch_size / new_samples
                                   if new_samples else None)}
        if auxiliary:
            result.update(ownership_loss=totals[3] / updates, score_loss=totals[4] / updates)
        return {**result,
                "weights_updated": True}

    def evaluation_jobs(self, games: int | None = None) -> list[GameJob]:
        games = self.config.evaluation.games if games is None else games
        opponent = {"opponent_name": f"accepted_{self.best_iteration:06d}",
                    "opponent_sha256": fingerprint(cpu_state(self.best))}
        jobs = []
        for pair in range(games // 2):
            seed = int(self.rng.integers(0, 2 ** 32))
            jobs.extend((GameJob(pair * 2, seed, BLACK, **opponent),
                         GameJob(pair * 2 + 1, seed, WHITE, **opponent)))
        return jobs

    def selfplay_jobs(self, iteration: int):
        """Pair colors against accepted champions and distinct frozen candidates."""
        config = self.config
        count = config.self_play.games_per_iteration
        possible_pairs = count // 2
        champion_pairs = min(possible_pairs, int(
            possible_pairs * config.self_play.champion_fraction + 0.5))
        milestone_pairs = min(possible_pairs - champion_pairs, int(
            possible_pairs * config.self_play.milestone_fraction + 0.5))
        learner_sha = fingerprint(cpu_state(self.model))
        champions = (select_anchors(self.anchors, config,
                                    candidate_sha256=learner_sha,
                                    kinds=("accepted",)) if champion_pairs else [])
        if not champions:
            champion_pairs = 0
        milestones = (select_anchors(self.anchors, config,
                                     candidate_sha256=learner_sha,
                                     kinds=("milestone",), recent=True) if milestone_pairs else [])
        seen = {champion["sha256"] for champion in champions}
        milestones = [milestone for milestone in milestones if milestone["sha256"] not in seen]
        if not milestones:
            milestone_pairs = 0

        evaluators = {0: self.runtime.evaluator(self.model)}
        best_sha = fingerprint(cpu_state(self.best))
        for model_id, opponent in enumerate([*champions, *milestones], 1):
            if opponent["sha256"] == best_sha:
                frozen = self.best
            else:
                payload, old_config = load_checkpoint(opponent["path"])
                frozen = PolicyValueNet(old_config).to(self.runtime.device)
                frozen.load_state_dict(payload["model"])
            evaluators[model_id] = self.runtime.evaluator(frozen)
            opponent["model_id"] = model_id

        jobs = []
        for pool, pairs in ((champions, champion_pairs), (milestones, milestone_pairs)):
            for pair in range(pairs):
                opponent = pool[((iteration - 1) * pairs + pair) % len(pool)]
                seed = int(self.rng.integers(0, 2 ** 32))
                for color in (BLACK, WHITE):
                    jobs.append(GameJob(len(jobs), seed, color,
                                        opponent_id=opponent["model_id"],
                                        opponent_name=opponent["name"],
                                        opponent_sha256=opponent["sha256"]))
        for _ in range(count - len(jobs)):
            jobs.append(GameJob(len(jobs), int(self.rng.integers(0, 2 ** 32)),
                                opponent_name="candidate_self",
                                opponent_sha256=learner_sha))
        return jobs, evaluators, champions

    def sprt(self, results) -> dict:
        evaluation = self.config.evaluation
        return paired_sprt(results, alpha=evaluation.sprt_alpha, beta=evaluation.sprt_beta,
                           pair_win_rate=evaluation.sprt_pair_win_rate)

    def confirmation_games(self, config: RLTrainingConfig, metadata: dict) -> list:
        """Play the promotion match; under SPRT stop at the first decisive batch.

        All seeds are drawn up front, so a stopped match is a prefix of the
        fixed-length one and stays reproducible.
        """
        jobs = self.evaluation_jobs()
        evaluators = {0: self.runtime.evaluator(self.best), 1: self.runtime.evaluator(self.model)}
        if config.evaluation.promotion_test != "paired_sprt":
            return run_games(config, jobs, evaluators, training=False, check=self.control,
                             progress=self.progress, metadata=metadata)
        results = []
        step = 2 * config.evaluation.sprt_batch_pairs
        for start in range(0, len(jobs), step):
            results += run_games(config, jobs[start:start + step], evaluators, training=False,
                                 check=self.control, progress=self.progress, metadata=metadata)
            test = self.sprt(results)
            # A truncated pair already fails promotion; more games cannot fix it.
            if test["decision"] != "continue" or test["truncated_pairs"]:
                break
        return results

    def run_iteration(self) -> dict:
        """Self-play, then (once the replay is warm) learning and evaluation, then checkpoint."""
        config, number = self.config, self.iteration + 1
        directory = self.output / "iterations" / f"{number:06d}"
        started = time.monotonic()
        summary = self._selfplay_phase(number, directory, started)
        if len(self.replay) >= config.optimizer.minimum_replay_size:
            self.progress.phase = "training"
            summary["training"] = self.train_updates()
            summary["position_quality"] = self._position_quality(directory)
            summary["opponent_pool"] = self._opponent_pool(number, directory)
            match_metadata, summary["screening"] = self._screening(directory)
            summary["evaluation"], summary["promoted"] = self._promotion(
                number, directory, match_metadata, summary["screening"])
        else:
            summary["training"] = {"steps": 0, "reason": "replay_warmup",
                                   "required_samples": config.optimizer.minimum_replay_size}
            summary["promoted"] = False
        return self._commit_iteration(number, directory, summary, started)

    @staticmethod
    def _next_due(number: int, every: int) -> int:
        return number + (every - number % every)

    def _selfplay_phase(self, number: int, directory: Path, started: float) -> dict:
        config = self.config
        self.progress.phase = "selfplay"
        self.progress({"event": "iteration_started", "iteration": number})
        jobs, evaluators, champions = self.selfplay_jobs(number)
        self.progress({"event": "selfplay_schedule", "iteration": number,
                       "champion_games": sum(job.opponent_name.startswith("accepted_") for job in jobs),
                       "milestone_games": sum(job.opponent_name.startswith("milestone_") for job in jobs),
                       "candidate_self_games": sum(job.opponent_id == 0 for job in jobs),
                       "champions": [{"name": champion["name"],
                                      "sha256": champion["sha256"]}
                                     for champion in champions],
                       "milestones": list({job.opponent_name: {
                           "name": job.opponent_name, "sha256": job.opponent_sha256}
                           for job in jobs if job.opponent_name.startswith("milestone_")}.values())})
        games = run_games(config, jobs, evaluators, training=True,
                          check=self.control, progress=self.progress,
                          metadata={"candidate_sha256": fingerprint(cpu_state(self.model)),
                                    "best_iteration": self.best_iteration})
        write_games(games, config, directory / "selfplay")
        for game in games:
            self.replay.extend(game.examples)
            self.untrained_samples += len(game.examples)
        opponent_rows = {}
        for game in games:
            name = game.opponent_name or "candidate_self"
            row = opponent_rows.setdefault(name, {"name": name,
                                                  "sha256": game.opponent_sha256,
                                                  "games": 0, "finished": 0,
                                                  "samples": 0})
            row["games"] += 1
            row["finished"] += int(game.reason != "length_limit")
            row["samples"] += len(game.examples)
        return {
            "iteration": number, "selfplay_games": len(games),
            "selfplay_finished": sum(game.reason != "length_limit" for game in games),
            "selfplay_truncated": sum(game.reason == "length_limit" for game in games),
            "new_samples": sum(len(game.examples) for game in games), "replay_samples": len(self.replay),
            "selfplay_seconds": time.monotonic() - started,
            "selfplay_opponents": list(opponent_rows.values()),
        }

    def _position_quality(self, directory: Path) -> dict:
        """Score the candidate on the frozen KataGo-labelled positions, when configured."""
        config = self.config
        self.progress.phase = "position_diagnostics"
        suite_name = config.evaluation.position_suite_path
        teacher_name = config.evaluation.teacher_labels_path
        if not suite_name:
            return {"status": "not_configured"}
        root = Path(__file__).resolve().parents[2]
        suite_path, teacher_path = (root / suite_name, root / teacher_name)
        if not (suite_path.is_file() and teacher_path.is_file()):
            return {"status": "unavailable",
                    "reason": "fixed positions or KataGo labels are missing"}
        quality = evaluate_positions(
            self.model, self.runtime, config, suite_path, teacher_path,
            simulations=config.evaluation.position_simulations_per_move,
            check=self.control, progress=self.progress,
        )
        atomic_json(quality, directory / "position_quality.json")
        return {
            "status": "measured", "suite_sha256": quality["suite_sha256"],
            "teacher_labels_sha256": quality["teacher_labels_sha256"],
            "teacher_model_sha256": quality["teacher_model_sha256"],
            "teacher_engine_sha256": quality["teacher_engine_sha256"],
            "teacher_config_sha256": quality["teacher_config_sha256"],
            "teacher_visits": quality["teacher_visits"],
            "candidate_visits": quality["candidate_visits"],
            "groups": quality["groups"],
        }

    def _opponent_pool(self, number: int, directory: Path) -> dict:
        """Matches against frozen references on every pool_every_iterations-th iteration."""
        config = self.config
        every = config.evaluation.pool_every_iterations
        if number % every:
            return {"status": "scheduled", "next_due_iteration": self._next_due(number, every)}
        self.progress.phase = "opponent_pool"
        anchors = select_anchors(self.anchors, config,
                                 candidate_sha256=fingerprint(cpu_state(self.model)))
        report = evaluate_pool(
            self.model, self.runtime, config, anchors, directory / "pool",
            games=config.evaluation.pool_games,
            simulations=config.evaluation.pool_simulations_per_move,
            check=self.control, progress=self.progress,
        )
        return {"status": "measured", **report}

    def _screening(self, directory: Path) -> tuple[dict, dict]:
        """Cheap candidate-vs-accepted match; returns (match metadata, summary)."""
        config = self.config
        self.progress.phase = "screening"
        match_metadata = {"candidate_sha256": fingerprint(cpu_state(self.model)),
                          "opponent_name": f"accepted_{self.best_iteration:06d}",
                          "opponent_sha256": fingerprint(cpu_state(self.best)),
                          "best_iteration": self.best_iteration}
        screen_config = replace(config, evaluation=replace(
            config.evaluation, games=config.evaluation.screen_games,
            simulations_per_move=config.evaluation.screen_simulations_per_move))
        screening = run_games(
            screen_config, self.evaluation_jobs(config.evaluation.screen_games),
            {0: self.runtime.evaluator(self.best), 1: self.runtime.evaluator(self.model)},
            training=False, check=self.control, progress=self.progress,
            metadata=match_metadata)
        write_games(screening, screen_config, directory / "screening")
        return match_metadata, {
            **match_metadata,
            **evaluation_summary(screening),
            "paired": paired_confidence(screening, confidence=config.evaluation.confidence_level),
            "simulations_per_move": config.evaluation.screen_simulations_per_move,
        }

    def _promotion(self, number: int, directory: Path, match_metadata: dict,
                   screening: dict) -> tuple[dict, bool]:
        """Run the due confirmation match after a passing screen; promote on success."""
        config = self.config
        every = config.evaluation.full_every_iterations
        due = number % every == 0
        eligible = (screening["truncated"] == 0 and
                    screening["score_rate"] >= config.evaluation.screen_min_score_rate)
        if not (due and eligible):
            return {"status": "screen_rejected" if due else "scheduled",
                    "next_due_iteration": self._next_due(number, every)}, False
        self.progress.phase = "evaluation"
        confirmation_config = replace(
            config, self_play=replace(
                config.self_play,
                max_game_length_factor=config.evaluation.confirmation_max_game_length_factor))
        results = self.confirmation_games(confirmation_config, match_metadata)
        write_games(results, confirmation_config, directory / "evaluation")
        evaluation = {
            **match_metadata,
            **evaluation_summary(results),
            "paired": paired_confidence(results, confidence=config.evaluation.confidence_level),
            "paired_sign": paired_sign_test(results, confidence=config.evaluation.confidence_level),
            "paired_sprt": self.sprt(results),
            "planned_games": config.evaluation.games,
            "promotion_test": config.evaluation.promotion_test,
            "max_game_length_factor": config.evaluation.confirmation_max_game_length_factor,
            "simulations_per_move": config.evaluation.simulations_per_move,
        }
        promoted = confirmed_improvement(evaluation,
                                         threshold=config.evaluation.promotion_win_rate,
                                         method=config.evaluation.promotion_test)
        if promoted:
            self.best.load_state_dict(self.model.state_dict())
            self.best_iteration = number
        return evaluation, promoted

    def _commit_iteration(self, number: int, directory: Path, summary: dict,
                          started: float) -> dict:
        config = self.config
        self.iteration = number
        summary.update({"training_steps": self.training_steps, "best_iteration": self.best_iteration,
                        "model_sha256": fingerprint(cpu_state(self.model)),
                        "seconds": time.monotonic() - started})
        self.progress.phase = "checkpoint"
        self.save()
        # The full checkpoint is the commit point. If interrupted immediately
        # afterward, the next resume can recreate either missing frozen copy.
        if summary["promoted"]:
            save_anchor(self.anchors, self.best, config, kind="accepted", iteration=number)
        if (self.training_steps > 0 and
                number % config.evaluation.milestone_every_iterations == 0):
            save_anchor(self.anchors, self.model, config, kind="milestone", iteration=number)
        atomic_json(summary, directory / "summary.json")
        atomic_json(summary, self.output / "summary.json")
        self.progress({"event": "iteration_completed", **summary})
        return summary

def train(config, output, iterations, resume=None, *, resume_path=None, resume_checksum=None):
    with RunLock(output):
        existing = [path for path in output.iterdir() if path.name != "run.lock"]
        if resume is None:
            if existing:
                raise ValueError("Output already contains a run; use --resume or a new --output directory")
        elif existing:
            latest = output / "latest.pt"
            if (not latest.is_file() or resume_path is None or resume_checksum is None
                    or latest.resolve() != Path(resume_path).resolve()
                    or sha256_file(latest) != resume_checksum):
                raise ValueError("Occupied output can only resume its own unchanged latest.pt; choose a new --output for another checkpoint")
        with Progress(output, operation="train") as progress:
            trainer = Trainer(config, output, resume, progress=progress)
            if resume is None:
                trainer.save(archive=False)
            for _ in range(iterations):
                trainer.run_iteration()
            return trainer
