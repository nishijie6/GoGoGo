"""Mine replayable surprise positions from training SGFs, never match results.

The per-game maximum change in the annotated black win probability is only a
weakness heuristic. Null hints do not assert a correct move. Whole-game hash
partitions keep mining and validation separate.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .sgf import MalformedSGF, parse_sgf_game

MOVE = re.compile(r';([BW])\[([a-s]{0,2})\](?:C\[([^\]]*)\])?')
VALUE = re.compile(r'^([0-9.]+) ([0-9.]+) ([0-9.]+) (-?[0-9.]+) v=(\d+)')
LETTERS = 'ABCDEFGHJKLMNOPQRST'


def mine_game(game: str, *, board_size: int = 19, minimum_swing: float = .04):
    parsed = parse_sgf_game(game)
    header = game.split(';B[', 1)[0].split(';W[', 1)[0]
    if (parsed['board_size'] != board_size or 'KM[6.5]' not in header
            or 'koPOSITIONALscoreAREAtaxNONEsui0' not in header
            or any(prop in header for prop in ('AB[', 'AW[', 'AE['))
            or game.count('(') != 1):
        return None
    matches = list(MOVE.finditer(game))
    if [(m[1], m[2]) for m in matches] != parsed['moves']:
        raise MalformedSGF('Unsupported move properties in training record')
    candidates = []
    previous = None
    for turn, node in enumerate(matches):
        value = VALUE.match(node[3] or '')
        if value is None:
            previous = None
            continue
        probability, visits = float(value[1]), int(value[5])
        if not 0 <= probability <= 1:
            raise MalformedSGF('Invalid annotated probability')
        if previous is not None and 30 <= turn <= 220 and visits >= 100:
            swing = abs(probability - previous)
            if swing >= minimum_swing and node[2]:
                candidates.append((swing, turn))
        previous = probability
    if not candidates:
        return None
    swing, turn = max(candidates)
    history = matches[:turn]
    locations = []
    for node in history:
        xy = node[2]
        locations.append('pass' if not xy else f'{LETTERS[ord(xy[0])-97]}{board_size-(ord(xy[1])-97)}')
    game_hash = hashlib.sha256(game.encode()).hexdigest()
    return {'xSize': board_size, 'ySize': board_size,
            'board': '/'.join(['.' * board_size] * board_size), 'nextPla': 'B',
            'moveLocs': locations, 'movePlas': [node[1] for node in history],
            'initialTurnNumber': 0, 'hintLoc': 'null', 'weight': 1.0,
            'metadata': f'training-game:{game_hash};turn:{turn};swing:{swing:.4f}',
            'game_sha256': game_hash, 'swing': swing, 'turn': turn}


def mine_training_files(files: list[Path], limit: int = 256):
    selected, seen, rejected = [], set(), 0
    for path in files:
        for game in path.read_text(encoding='utf-8').splitlines():
            if not game.strip():
                continue
            try:
                position = mine_game(game)
            except (MalformedSGF, ValueError, IndexError):
                rejected += 1
                continue
            if position and position['game_sha256'] not in seen:
                seen.add(position['game_sha256'])
                selected.append(position)
        if len(selected) >= limit * 4:
            break
    selected.sort(key=lambda p: (-p['swing'], p['game_sha256']))
    # Reserve whole games before selecting the hardest positions.
    training = [p for p in selected if int(p['game_sha256'][:8], 16) % 5 != 0][:limit]
    heldout = [p for p in selected if int(p['game_sha256'][:8], 16) % 5 == 0][:max(32, limit // 4)]
    return training, heldout, rejected
