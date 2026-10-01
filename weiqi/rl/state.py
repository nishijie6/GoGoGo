"""Training positions and features backed by the application's Go rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..engine import BLACK, WHITE, GoGame


FEATURE_VERSION = 1
INPUT_PLANES = 20


@dataclass(frozen=True)
class Position:
    game: GoGame
    # This board and up to seven before it, newest first, as int8 arrays.
    history: tuple[np.ndarray, ...]

    @classmethod
    def new(cls, size: int, komi: float) -> "Position":
        game = GoGame(size, komi, record_undo=False)
        return cls(game, (np.array(game.board, dtype=np.int8),))

    @cached_property
    def legal(self) -> np.ndarray:
        mask = np.zeros(self.game.size ** 2 + 1, dtype=np.bool_)
        if not self.game.game_over:
            mask[self.game.legal_points()] = True
            mask[-1] = True
        return mask

    def play(self, action: int) -> "Position":
        if not 0 <= action < len(self.legal) or not self.legal[action]:
            raise ValueError(f"Illegal action: {action}")
        game = self.game.clone()
        if action == game.size ** 2:
            game.pass_turn()
        else:
            result = game.play(*divmod(action, game.size))
            if not result.legal:
                raise ValueError(result.reason)
        return Position(game, (np.array(game.board, dtype=np.int8),) + self.history[:7])

    def features(self) -> np.ndarray:
        """Eight boards relative to the player to move, plus rule context.

        Planes 16..19 are black-to-play, signed komi / 20, preceding pass,
        and legal board moves. Full superko history remains in GoGame.
        """
        size = self.game.size
        own = self.game.current_player
        other = WHITE if own == BLACK else BLACK
        result = np.zeros((INPUT_PLANES, size, size), dtype=np.float32)
        boards = np.stack(self.history)
        count = len(boards)
        result[0:2 * count:2] = boards == own
        result[1:2 * count:2] = boards == other
        result[16].fill(own == BLACK)
        result[17].fill(self.game.komi / 20 * (1 if own == WHITE else -1))
        result[18].fill(min(self.game.consecutive_passes, 1))
        result[19] = self.legal[:-1].reshape(size, size)
        return result

    def ownership_target(self, perspective: int) -> np.ndarray:
        """Final area owner per intersection: +1 ``perspective``, -1 opponent, 0 neutral."""
        owners = np.array(self.game.area_ownership(), dtype=np.int8).ravel()
        other = WHITE if perspective == BLACK else BLACK
        return ((owners == perspective).astype(np.float32)
                - (owners == other).astype(np.float32))

    def terminal_value(self) -> float:
        if not self.game.game_over:
            raise ValueError("A value target requires a finished game")
        if self.game.winner is None:
            return 0.0
        return 1.0 if self.game.winner == self.game.current_player else -1.0


def augment(features: np.ndarray, policy: np.ndarray, symmetry: int):
    """Apply the same square symmetry to features and board policy; keep pass."""
    size = features.shape[-1]
    board_policy = policy[:-1].reshape(size, size)
    if symmetry >= 4:
        features = np.flip(features, axis=-1)
        board_policy = np.flip(board_policy, axis=-1)
    features = np.rot90(features, symmetry % 4, axes=(-2, -1))
    board_policy = np.rot90(board_policy, symmetry % 4)
    policy = np.concatenate((board_policy.ravel(), policy[-1:]))
    return np.ascontiguousarray(features), np.ascontiguousarray(policy)


def augment_batch(features: np.ndarray, policies: np.ndarray, symmetries: np.ndarray,
                  ownership: np.ndarray | None = None):
    """Apply a per-row square symmetry to a batch; identical to row-wise ``augment``.

    ``ownership`` rows (one value per intersection) follow the same symmetry and
    are returned as a third array when given.
    """
    features, policies = features.copy(), policies.copy()
    if ownership is not None:
        ownership = ownership.copy()
    for symmetry in np.unique(symmetries):
        rows = np.flatnonzero(symmetries == symmetry)
        features[rows], policies[rows] = _augment_rows(features[rows], policies[rows], int(symmetry))
        if ownership is not None:
            ownership[rows] = _transform_boards(ownership[rows], int(symmetry))
    return (features, policies) if ownership is None else (features, policies, ownership)


def _transform_boards(flat: np.ndarray, symmetry: int) -> np.ndarray:
    size = int(round(flat.shape[-1] ** 0.5))
    boards = flat.reshape(len(flat), size, size)
    if symmetry >= 4:
        boards = np.flip(boards, axis=-1)
    return np.rot90(boards, symmetry % 4, axes=(-2, -1)).reshape(len(flat), -1)


def _augment_rows(features: np.ndarray, policies: np.ndarray, symmetry: int):
    if symmetry >= 4:
        features = np.flip(features, axis=-1)
    features = np.rot90(features, symmetry % 4, axes=(-2, -1))
    board = _transform_boards(policies[:, :-1], symmetry)
    return features, np.concatenate((board, policies[:, -1:]), axis=1)
