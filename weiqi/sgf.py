"""Minimal, strict SGF reading for KataGo self-play and match records."""

from __future__ import annotations

import math
import re
from typing import Optional


class MalformedSGF(ValueError):
    """A game tree that cannot be trusted."""


def split_sgf_collection(content: str) -> tuple[list[str], int]:
    """Split an SGF collection without being fooled by comments or escapes.

    KataGo's ``.sgfs`` files normally contain one complete tree per line, but
    parsing balanced game trees makes this also work for wrapped SGFs and SGF
    collections.  The second return value counts an unterminated trailing tree.
    """

    games: list[str] = []
    start: Optional[int] = None
    depth = 0
    in_value = False
    escaped = False
    for index, character in enumerate(content):
        if in_value:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "]":
                in_value = False
            continue
        if character == "[" and depth > 0:
            in_value = True
        elif character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                games.append(content[start : index + 1])
                start = None
    return games, int(depth != 0 or in_value)


def _read_value(game: str, offset: int) -> tuple[str, int]:
    if offset >= len(game) or game[offset] != "[":
        raise MalformedSGF("SGF property is missing its value")
    offset += 1
    value: list[str] = []
    while offset < len(game):
        character = game[offset]
        if character == "]":
            return "".join(value), offset + 1
        if character == "\\":
            offset += 1
            if offset >= len(game):
                raise MalformedSGF("unterminated SGF escape")
            escaped = game[offset]
            # SGF line continuation removes the escaped line break.
            if escaped == "\r" and offset + 1 < len(game) and game[offset + 1] == "\n":
                offset += 1
            elif escaped not in "\r\n":
                value.append(escaped)
        else:
            value.append(character)
        offset += 1
    raise MalformedSGF("unterminated SGF property value")


def parse_sgf_game(game: str) -> dict[str, object]:
    """Parse the root result and main-line moves of one SGF game tree."""

    if "�" in game:
        raise MalformedSGF("invalid UTF-8 replacement character")
    depth = 0
    node_index = -1
    root: dict[str, list[str]] = {}
    moves: list[tuple[str, str]] = []
    index = 0
    while index < len(game):
        character = game[index]
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth -= 1
            if depth < 0:
                raise MalformedSGF("unbalanced SGF game tree")
            index += 1
            continue
        if character == ";":
            if depth == 1:
                node_index += 1
            index += 1
            continue
        if not character.isalpha():
            index += 1
            continue

        property_start = index
        while index < len(game) and game[index].isalpha():
            index += 1
        identifier = game[property_start:index].upper()
        while index < len(game) and game[index].isspace():
            index += 1
        values: list[str] = []
        while index < len(game) and game[index] == "[":
            value, index = _read_value(game, index)
            values.append(value)
            while index < len(game) and game[index].isspace():
                index += 1
        if not values:
            # Alphabetic text is only legal as a property identifier followed
            # by one or more values.  Treat it as corruption, rather than
            # accidentally mining moves embedded in free-form text.
            raise MalformedSGF(f"property {identifier!r} has no value")
        if depth == 1 and node_index == 0:
            root.setdefault(identifier, []).extend(values)
        if depth == 1 and node_index >= 1 and identifier in {"B", "W"}:
            moves.append((identifier, values[0]))
    if depth != 0 or node_index < 0:
        raise MalformedSGF("incomplete SGF game tree")

    size_value = (root.get("SZ") or [""])[0]
    try:
        dimensions = [int(part) for part in size_value.split(":")]
    except ValueError as error:
        raise MalformedSGF("invalid or missing board size") from error
    if not dimensions or dimensions[0] <= 0 or any(part != dimensions[0] for part in dimensions):
        raise MalformedSGF("only square positive boards are supported")
    result = (root.get("RE") or [""])[0].strip()
    return {
        "board_size": dimensions[0],
        "result": result,
        "moves": moves,
    }


def classify_result(result: str) -> tuple[str, Optional[float]]:
    """Return ``(black|white|draw|no_result, absolute margin or None)``."""

    normalized = result.strip()
    folded = normalized.casefold()
    if folded in {"0", "draw", "jigo"}:
        return "draw", 0.0
    match = re.fullmatch(r"([BWbw])\+(.+)", normalized)
    if match is None:
        return "no_result", None
    winner = "black" if match.group(1).upper() == "B" else "white"
    try:
        margin: Optional[float] = float(match.group(2))
    except ValueError:
        margin = None
    if margin is not None and not math.isfinite(margin):
        margin = None
    return winner, abs(margin) if margin is not None else None
