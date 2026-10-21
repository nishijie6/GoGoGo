"""Desktop adapter for native policy/value checkpoints; dependencies load on demand."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError
from dataclasses import replace
from threading import Event

from .ai import AIMove
from .engine import BLACK, GoGame
from .rl_opponents import NativeTrainingOpponent


class NativeModelError(ValueError):
    """A local checkpoint cannot be used for this game."""


def position_from_game(game: GoGame):
    """Rebuild the trainer's eight-board history, including passes and superko."""
    import numpy as np
    from .rl.state import Position

    replay = GoGame(game.size, game.komi, record_undo=False)
    history = deque([np.array(replay.board, dtype=np.int8)], maxlen=8)
    for move in game.moves:
        if move.color != replay.current_player:
            raise NativeModelError("棋局与行棋记录不一致，无法加载本地模型局面")
        if move.kind == "pass":
            legal = replay.pass_turn()
        elif move.kind == "play":
            legal = replay.play(move.row, move.col).legal
        else:
            raise NativeModelError("无法从已结束的棋局创建本地模型局面")
        if not legal:
            raise NativeModelError("行棋记录中存在非法落子，无法创建本地模型局面")
        history.appendleft(np.array(replay.board, dtype=np.int8))
    if (replay.board_hash() != game.board_hash()
            or replay.current_player != game.current_player):
        raise NativeModelError("棋局与行棋记录不一致，无法加载本地模型局面")
    return Position(replay, tuple(history))


class NativeTrainingAI:
    def __init__(self, opponent: NativeTrainingOpponent, *,
                 simulations: int | None = None, device: str | None = None):
        try:
            import numpy as np
            import torch
            from .rl.network import PolicyValueNet, Runtime
            from .rl.storage import checkpoint_config
        except ImportError as error:
            raise NativeModelError(
                "本地模型需要 PyTorch 和 NumPy，请在启动程序使用的 Python 环境中安装训练依赖"
            ) from error

        try:
            payload = torch.load(opponent.checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("无效的模型检查点")
            raw = payload["config"]
            # Opening policies, optimizers and training hardware do not affect
            # inference. New trainer options must not prevent playing its model.
            inference_config = {key: raw[key] for key in
                                ("schema_version", "preset", "game", "network", "search")}
            inference_config["evaluation"] = {
                "simulations_per_move": raw["evaluation"]["simulations_per_move"],
            }
            inference_config["runtime"] = {"random_seed": raw["runtime"]["random_seed"]}
            config = checkpoint_config({**payload, "config": inference_config})
            if config.game.board_size != opponent.board_size:
                raise ValueError(f"此模型仅支持 {config.game.board_size}×{config.game.board_size} 棋盘")
            if payload.get("kind") not in ("training", "model"):
                raise ValueError("不支持的模型检查点类型")
            if device is None:
                device = ("cuda:0" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
            # Training hardware may belong to another computer. Inference uses
            # this machine, full precision and the saved network architecture.
            config = replace(config, hardware=replace(
                config.hardware, device=device, precision="float32", compile_model=False,
            ))
            self.runtime = Runtime(config)
            self.model = PolicyValueNet(config).to(self.runtime.device)
            weights = (payload["best_model"]
                       if opponent.role == "best" and payload["kind"] == "training"
                       else payload["model"])
            self.model.load_state_dict(weights)
            self.model.eval()
            self.iteration = payload.get(
                "best_iteration" if opponent.role == "best" else "iteration", 0,
            )
        except Exception as error:
            raise NativeModelError(f"无法加载本地模型 {opponent.checkpoint_path.name}：{error}") from error
        self.opponent = opponent
        self.config = config
        self.simulations = config.evaluation.simulations_per_move if simulations is None else simulations
        if self.simulations < 1:
            raise NativeModelError("本地模型搜索次数必须大于零")
        self.predict = self.runtime.evaluator(self.model)
        self.rng = np.random.default_rng(config.runtime.random_seed)

    def choose_move(self, game: GoGame, cancel_event: Event | None = None) -> AIMove:
        from .rl.search import choose_action, search

        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("本地模型计算已取消")

        check_cancelled()
        if game.size != self.config.game.board_size:
            raise NativeModelError(
                f"此本地模型仅支持 {self.config.game.board_size}×{self.config.game.board_size} 棋盘"
            )
        if game.game_over:
            return AIMove(None, "对局已经结束")
        position = position_from_game(game)

        def predict(batch):
            check_cancelled()
            return self.predict(batch)

        def evaluate(features):
            policies, values = predict(features[None])
            return policies[0], float(values[0])

        policy, value = search(
            position, evaluate, self.config.search, self.rng,
            simulations=self.simulations, add_noise=False, evaluate_batch=predict,
        )
        check_cancelled()
        action = choose_action(policy, 0, self.rng)
        point = None if action == game.size ** 2 else divmod(action, game.size)
        black_value = value if game.current_player == BLACK else -value
        return AIMove(
            point, f"{self.opponent.label} · 第 {self.iteration} 轮 · {self.simulations} 次搜索",
            black_win_probability=max(0.0, min(1.0, (black_value + 1) / 2)),
            analysis_visits=self.simulations,
        )
