"""CPU self-play processes with one centrally batched neural-net evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from queue import Empty
import time
import traceback
from typing import Callable

import numpy as np

from ..engine import BLACK, WHITE
from ..rl_config import RLTrainingConfig
from .search import choose_action, search
from .state import Position


@dataclass(frozen=True)
class GameJob:
    index: int
    seed: int
    candidate_color: int = BLACK
    opponent_id: int = 0
    opponent_name: str = ""
    opponent_sha256: str | None = None


@dataclass
class GameResult:
    index: int
    seed: int
    candidate_color: int
    winner: int | None
    reason: str
    black_score: float
    white_score: float
    moves: list[tuple[int, int]]
    seconds: float
    examples: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    opponent_name: str = ""
    opponent_sha256: str | None = None

    def record(self, config: RLTrainingConfig) -> dict:
        record = {
            "index": self.index, "seed": self.seed, "candidate_color": self.candidate_color,
            "winner": self.winner, "reason": self.reason,
            "black_score": self.black_score, "white_score": self.white_score,
            "moves": self.moves, "seconds": self.seconds, "samples": len(self.examples),
            "board_size": config.game.board_size, "komi": config.game.komi,
        }
        if self.opponent_name:
            record["training_opponent"] = {"name": self.opponent_name,
                                           "sha256": self.opponent_sha256}
        return record


def play_game(config, job, evaluate, *, training: bool) -> GameResult:
    rng = np.random.default_rng(job.seed)
    state = Position.new(config.game.board_size, config.game.komi)
    started = time.monotonic()
    samples, moves = [], []
    reason = "length_limit"
    # Use identical two-stone openings for color-swapped evaluation pairs.
    if not training:
        for _ in range(2):
            action = int(rng.choice(np.flatnonzero(state.legal[:-1])))
            moves.append((state.game.current_player, action))
            state = state.play(action)
    limit = int(config.self_play.max_game_length_factor * config.game.board_size ** 2)
    while len(moves) < limit and not state.game.game_over:
        color = state.game.current_player
        if training:
            model_id = 0 if color == job.candidate_color else job.opponent_id
        else:
            model_id = int(color == job.candidate_color)
        evaluator = lambda features: evaluate(model_id, features)
        simulations = (
            config.search.simulations_per_move if training
            else config.evaluation.simulations_per_move
        )
        policy, value = search(
            state, evaluator, config.search, rng,
            simulations=simulations, add_noise=training,
        )
        threshold = config.self_play.resign_threshold if training else None
        if threshold is not None and len(moves) >= config.self_play.resign_min_move and value < threshold:
            state.game.resign()
            reason = "resign"
            break
        if training:
            samples.append((state.features(), policy, color))
        temperature = (
            config.search.root_temperature
            if training and len(moves) < config.search.temperature_moves else 0.0
        )
        action = choose_action(policy, temperature, rng)
        moves.append((color, action))
        state = state.play(action)
    if state.game.game_over and reason != "resign":
        reason = "two_passes"
    score = state.game.calculate_score()
    # A move cap is truncation, never a fabricated terminal win/loss target.
    winner = state.game.winner if state.game.game_over else None
    examples = []
    if state.game.game_over:
        for features, policy, color in samples:
            target = 0.0 if winner is None else (1.0 if color == winner else -1.0)
            examples.append((features, policy, target))
    return GameResult(
        job.index, job.seed, job.candidate_color, winner, reason,
        score.black_total, score.white_total, moves, time.monotonic() - started, examples,
        opponent_name=job.opponent_name, opponent_sha256=job.opponent_sha256,
    )


def _worker(config, jobs, training, actor, requests, responses, completed, stop):
    """Spawn target: imports NumPy and the rules, but never creates a CUDA context."""
    try:
        def evaluate(model_id, features):
            if stop.is_set():
                raise InterruptedError("Training stopped")
            requests.put((actor, model_id, features))
            while not stop.is_set():
                try:
                    result = responses.get(timeout=0.25)
                    return result
                except Empty:
                    pass
            raise InterruptedError("Training stopped")

        for job in jobs:
            if stop.is_set():
                break
            result = play_game(config, job, evaluate, training=training)
            completed.put(("game", result))
        completed.put(("done", actor))
    except BaseException:
        completed.put(("error", (actor, traceback.format_exc())))


def run_games(
    config: RLTrainingConfig,
    jobs: list[GameJob],
    evaluators: dict[int, Callable],
    *,
    training: bool,
    check: Callable = lambda: None,
    progress: Callable = lambda event: None,
) -> list[GameResult]:
    """Run actors and serve inference batches in this process. Always reap children."""
    if not jobs:
        return []
    context = mp.get_context("spawn")
    workers = min(config.self_play.workers, len(jobs))
    requests, completed = context.Queue(), context.Queue()
    responses = [context.Queue() for _ in range(workers)]
    stop = context.Event()
    processes, results, done = [], [], set()
    last_progress = time.monotonic()
    batches, positions = 0, 0
    try:
        for actor in range(workers):
            process = context.Process(
                target=_worker,
                args=(config, jobs[actor::workers], training, actor, requests,
                      responses[actor], completed, stop),
                name=f"go-selfplay-{actor}",
            )
            process.start()
            processes.append(process)
        while len(done) < workers:
            check()
            while True:
                try:
                    kind, payload = completed.get_nowait()
                except Empty:
                    break
                if kind == "error":
                    raise RuntimeError(f"Self-play worker {payload[0]} failed:\n{payload[1]}")
                if kind == "done":
                    done.add(payload)
                else:
                    results.append(payload)
                    progress({"event": "game", "index": payload.index,
                              "moves": len(payload.moves), "reason": payload.reason,
                              "samples": len(payload.examples), "seconds": round(payload.seconds, 2),
                              "completed": len(results), "total": len(jobs)})
            for actor, process in enumerate(processes):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"Self-play worker {actor} exited with {process.exitcode}")
            if len(done) == workers:
                break
            try:
                batch = [requests.get(timeout=0.02)]
            except Empty:
                continue
            deadline = time.monotonic() + 0.002
            while len(batch) < config.self_play.inference_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(requests.get(timeout=remaining))
                except Empty:
                    break
            for model_id in {request[1] for request in batch}:
                subset = [request for request in batch if request[1] == model_id]
                policies, values = evaluators[model_id](np.stack([item[2] for item in subset]))
                for item, policy, value in zip(subset, policies, values):
                    responses[item[0]].put((policy, float(value)))
                batches += 1
                positions += len(subset)
            if time.monotonic() - last_progress >= 10:
                progress({"event": "search_progress", "completed": len(results),
                          "total": len(jobs), "inference_positions": positions,
                          "inference_batches": batches})
                last_progress = time.monotonic()
        if len(results) != len(jobs):
            raise RuntimeError("Self-play workers exited without returning every game")
        return sorted(results, key=lambda game: game.index)
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for queue in [requests, completed, *responses]:
            queue.cancel_join_thread()
            queue.close()


def evaluation_summary(results: list[GameResult]) -> dict:
    truncated = sum(game.reason == "length_limit" for game in results)
    wins = sum(game.winner == game.candidate_color for game in results)
    losses = sum(game.winner is not None and game.winner != game.candidate_color for game in results)
    draws = len(results) - truncated - wins - losses
    # Truncations conservatively contribute no points and block promotion.
    rate = (wins + 0.5 * draws) / len(results) if results else 0.0
    return {"games": len(results), "wins": wins, "losses": losses,
            "draws": draws, "truncated": truncated, "score_rate": rate}
