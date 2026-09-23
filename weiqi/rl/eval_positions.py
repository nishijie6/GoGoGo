"""Frozen 9x9 evaluation positions with full rule-relevant move histories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from ..engine import BLACK, GoGame


SUITE_VERSION = 1
TEACHER_VERSION = 1


def canonical_hash(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_json(value: object) -> str:
    """Keep large generated suites readable without a line per board cell."""
    if (not isinstance(value, dict) or value.get("version") != 1
            or not isinstance(value.get("positions"), (list, dict))):
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fields = []
    for key, item in value.items():
        name = json.dumps(key, ensure_ascii=False)
        if key != "positions":
            encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        elif isinstance(item, list):
            rows = ["    " + json.dumps(row, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False) for row in item]
            encoded = "[\n" + ",\n".join(rows) + "\n  ]" if rows else "[]"
        else:
            rows = ["    " + json.dumps(row_key, ensure_ascii=False) + ":" +
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    for row_key, row in item.items()]
            encoded = "{\n" + ",\n".join(rows) + "\n  }" if rows else "{}"
        fields.append(f"  {name}: {encoded}")
    return "{\n" + ",\n".join(fields) + "\n}\n"


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_render_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def replay_position(record: dict, *, size: int, komi: float) -> GoGame:
    game = GoGame(size, komi, record_undo=False)
    for color, action in record["moves"]:
        if type(color) is not int or color != game.current_player:
            raise ValueError(f"Position {record.get('id')} has an invalid player history")
        if type(action) is not int or not 0 <= action <= size * size:
            raise ValueError(f"Position {record.get('id')} has an invalid action")
        if action == size * size:
            if not game.pass_turn():
                raise ValueError("Position history continues after a finished game")
        else:
            result = game.play(*divmod(action, size))
            if not result.legal:
                raise ValueError(f"Position {record.get('id')} contains an illegal move: {result.reason}")
    if game.game_over or game.current_player != record["to_play"]:
        raise ValueError(f"Position {record.get('id')} does not end in the expected turn")
    if "board" in record and record["board"] != [list(row) for row in game.board_hash()]:
        raise ValueError(f"Position {record.get('id')} has a mismatched board")
    return game


def generate_suite(*, size: int = 9, komi: float = 6.5, seed: int = 20260923,
                   count: int = 96) -> dict:
    if size != 9 or count < 4 or count % 4:
        raise ValueError("The fixed position suite requires 9x9 and a multiple of four positions")
    turns = (9, 22, 43, 60)
    phases = ("opening", "middle", "middle", "endgame")
    positions = []
    game_index = 0
    while len(positions) < count:
        if game_index >= count * 10:
            raise RuntimeError("Could not generate enough independent legal positions")
        game = GoGame(size, komi, record_undo=False)
        rng = random.Random(seed + game_index)
        selected = []
        for turn in range(turns[-1] + 1):
            if turn in turns and not game.game_over:
                phase = phases[turns.index(turn)]
                selected.append({"id": f"p{len(positions) + len(selected):04d}",
                                 "phase": phase, "source_game": game_index,
                                 "moves": [[move.color, size * size if move.kind == "pass"
                                            else move.row * size + move.col] for move in game.moves],
                                 "to_play": game.current_player,
                                 "board": [list(row) for row in game.board_hash()]})
            if turn == turns[-1] or game.game_over:
                break
            legal = list(game.legal_moves())
            if legal:
                played = game.play(*rng.choice(legal))
                assert played.legal
            else:
                game.pass_turn()
        if len(selected) == 4:
            positions.extend(selected)
        game_index += 1
    suite = {"version": SUITE_VERSION, "board_size": size, "komi": komi,
             "scoring": "area", "ko": "positional_superko", "seed": seed,
             "positions": positions}
    validate_suite(suite)
    return suite


def validate_suite(suite: dict, *, size: int | None = None, komi: float | None = None) -> None:
    if (not isinstance(suite, dict) or suite.get("version") != SUITE_VERSION
            or suite.get("board_size") != 9 or suite.get("scoring") != "area"
            or suite.get("ko") != "positional_superko"):
        raise ValueError("Unsupported fixed position suite or rules")
    if size is not None and suite["board_size"] != size:
        raise ValueError("Position suite board size differs from the candidate")
    if komi is not None and suite["komi"] != komi:
        raise ValueError("Position suite komi differs from the candidate")
    positions = suite.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("Position suite is empty")
    seen = set()
    for record in positions:
        if not isinstance(record, dict) or record.get("id") in seen:
            raise ValueError("Position suite contains a duplicate or malformed id")
        seen.add(record["id"])
        replay_position(record, size=9, komi=suite["komi"])


def read_suite(path: Path, *, size: int | None = None, komi: float | None = None) -> dict:
    suite = json.loads(path.read_text(encoding="utf-8"))
    validate_suite(suite, size=size, komi=komi)
    return suite


def read_teacher(path: Path, suite: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("version") != TEACHER_VERSION or not payload.get("complete")
            or payload.get("suite_sha256") != canonical_hash(suite)
            or len(payload.get("positions", {})) != len(suite["positions"])):
        raise ValueError("KataGo labels do not match the complete frozen position suite")
    for record in suite["positions"]:
        info = payload["positions"].get(record["id"])
        if not isinstance(info, dict) or len(info.get("policy", [])) != 82:
            raise ValueError(f"KataGo label is missing for {record['id']}")
    return payload
