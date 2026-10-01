"""CPU self-play processes with one centrally batched neural-net evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing.connection import wait
from queue import Empty
import time
import traceback
from typing import Callable
import uuid

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
    # (features, policy, value, ownership, score, aux_weight); see ReplayBuffer.
    examples: list[tuple] = field(default_factory=list)
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


def play_game(config, job, evaluate, *, training: bool, evaluate_batch=None) -> GameResult:
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
        batch_evaluator = (None if evaluate_batch is None
                           else lambda features: evaluate_batch(model_id, features))
        simulations = (
            config.search.simulations_per_move if training
            else config.evaluation.simulations_per_move
        )
        # Playout cap randomization: cheap moves advance the game without noise
        # and never become policy targets. Draw only when enabled so that the
        # default keeps the exact random stream of the uncapped search.
        full_search = (not training or config.search.full_search_probability >= 1
                       or rng.random() < config.search.full_search_probability)
        if not full_search:
            simulations = min(simulations, config.search.fast_simulations_per_move)
        policy, value = search(
            state, evaluator, config.search, rng,
            simulations=simulations, add_noise=training and full_search,
            evaluate_batch=batch_evaluator,
        )
        threshold = config.self_play.resign_threshold if training else None
        if threshold is not None and len(moves) >= config.self_play.resign_min_move and value < threshold:
            state.game.resign()
            reason = "resign"
            break
        if training and full_search:
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
        # A resigned board was never played out, so it has no ownership/score target.
        scored = reason == "two_passes"
        area = config.game.board_size ** 2
        owners = {color: (state.ownership_target(color) if scored else np.zeros(area, dtype=np.float32))
                  for color in (BLACK, WHITE)}
        black_margin = (score.black_total - score.white_total) / area
        for features, policy, color in samples:
            target = 0.0 if winner is None else (1.0 if color == winner else -1.0)
            margin = black_margin if color == BLACK else -black_margin
            examples.append((features, policy, target, owners[color],
                             margin if scored else 0.0, 1.0 if scored else 0.0))
    return GameResult(
        job.index, job.seed, job.candidate_color, winner, reason,
        score.black_total, score.white_total, moves, time.monotonic() - started, examples,
        opponent_name=job.opponent_name, opponent_sha256=job.opponent_sha256,
    )


def _worker(config, jobs, training, actor, conn, completed, stop):
    """Spawn target: imports NumPy and the rules, but never creates a CUDA context."""
    try:
        def evaluate_batch(model_id, features):
            if stop.is_set():
                raise InterruptedError("Training stopped")
            try:
                conn.send((model_id, features))
                while not stop.is_set():
                    # Returns the moment the reply arrives; the timeout only bounds stop latency.
                    if conn.poll(0.25):
                        return conn.recv()
            except (EOFError, OSError):
                pass
            raise InterruptedError("Training stopped")

        def evaluate(model_id, features):
            policies, values = evaluate_batch(model_id, features[None, ...])
            return policies[0], float(values[0])

        for job in jobs:
            if stop.is_set():
                break
            completed.put(("game_started", {"index": job.index, "actor": actor}))
            result = play_game(config, job, evaluate, training=training,
                               evaluate_batch=evaluate_batch)
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
    metadata: dict | None = None,
) -> list[GameResult]:
    """Run actors and serve inference batches in this process. Always reap children."""
    if not jobs:
        return []
    batch_id = uuid.uuid4().hex
    job_metadata = {job.index: {
        "index": job.index, "seed": job.seed, "candidate_color": job.candidate_color,
        "opponent_name": job.opponent_name or ("candidate_self" if training and job.opponent_id == 0
                                                else "unknown"),
        "opponent_sha256": job.opponent_sha256,
    } for job in jobs}
    progress({**(metadata or {}), "event": "games_started", "batch_id": batch_id,
              "total": len(jobs), "training": training,
              "simulations_per_move": (config.search.simulations_per_move if training
                                        else config.evaluation.simulations_per_move),
              "jobs": list(job_metadata.values())})
    context = mp.get_context("spawn")
    workers = min(config.self_play.workers, len(jobs))
    completed = context.Queue()
    # One duplex pipe per actor: a blocking Queue.get(timeout) costs ~16 ms per
    # round trip on Windows, while a pipe plus connection.wait is ~0.1 ms.
    pipes = [context.Pipe() for _ in range(workers)]
    stop = context.Event()
    processes, results, done = [], [], set()
    connections = {}
    last_progress = time.monotonic()
    batches, positions = 0, 0
    try:
        for actor in range(workers):
            process = context.Process(
                target=_worker,
                args=(config, jobs[actor::workers], training, actor, pipes[actor][1],
                      completed, stop),
                name=f"go-selfplay-{actor}",
            )
            process.start()
            pipes[actor][1].close()
            connections[pipes[actor][0]] = actor
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
                elif kind == "game_started":
                    progress({"event": "game_started", "batch_id": batch_id,
                              **job_metadata[payload["index"]], "actor": payload["actor"]})
                else:
                    results.append(payload)
                    progress({"event": "game", "batch_id": batch_id,
                              **job_metadata[payload.index],
                              "winner": payload.winner,
                              "result": game_outcome(payload, training=training,
                                                     opponent_name=job_metadata[payload.index]["opponent_name"]),
                              "moves": len(payload.moves), "reason": payload.reason,
                              "samples": len(payload.examples), "seconds": round(payload.seconds, 2),
                              "completed": len(results), "total": len(jobs)})
            for actor, process in enumerate(processes):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"Self-play worker {actor} exited with {process.exitcode}")
            if len(done) == workers:
                break
            batch = []
            ready = wait(list(connections), timeout=0.02)
            while ready:
                for connection in ready:
                    try:
                        model_id, features = connection.recv()
                    except (EOFError, OSError):
                        del connections[connection]  # actor finished or died
                        continue
                    batch.append((connections[connection], model_id, features))
                if sum(len(item[2]) for item in batch) >= config.self_play.inference_batch_size:
                    break
                # Requests already sitting in other pipes join this batch for free.
                waiting = [c for c in connections if connections[c] not in
                           {item[0] for item in batch}]
                ready = wait(waiting, timeout=0) if waiting else []
            if not batch:
                continue
            for model_id in {request[1] for request in batch}:
                subset = [request for request in batch if request[1] == model_id]
                policies, values = evaluators[model_id](
                    np.concatenate([item[2] for item in subset]))
                start = 0
                for item in subset:
                    stop_row = start + len(item[2])
                    pipes[item[0]][0].send((policies[start:stop_row], values[start:stop_row]))
                    start = stop_row
                batches += 1
                positions += start
            if time.monotonic() - last_progress >= 10:
                progress({"event": "search_progress", "completed": len(results),
                          "batch_id": batch_id,
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
        for connection, _ in pipes:
            connection.close()
        completed.cancel_join_thread()
        completed.close()


def game_outcome(game: GameResult, *, training: bool, opponent_name: str) -> str:
    """Describe completed games without treating self-play as a strength test."""
    if game.reason == "length_limit":
        return "truncated"
    if game.winner is None:
        return "draw"
    if training and opponent_name == "candidate_self":
        return "black_win" if game.winner == BLACK else "white_win"
    return "win" if game.winner == game.candidate_color else "loss"


def evaluation_summary(results: list[GameResult]) -> dict:
    truncated = sum(game.reason == "length_limit" for game in results)
    wins = sum(game.winner == game.candidate_color for game in results)
    losses = sum(game.winner is not None and game.winner != game.candidate_color for game in results)
    draws = len(results) - truncated - wins - losses
    # Truncations conservatively contribute no points and block promotion.
    rate = (wins + 0.5 * draws) / len(results) if results else 0.0
    return {"games": len(results), "wins": wins, "losses": losses,
            "draws": draws, "truncated": truncated, "score_rate": rate}
