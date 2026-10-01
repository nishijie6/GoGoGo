"""Colors and value-formatting helpers shared by the AI analysis window modules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from .engine import Point


BOARD_COLOR = "#d9a85f"
PANEL_COLOR = "#172019"
PANEL_DARK = "#101713"
TEXT_COLOR = "#edf2eb"
MUTED_COLOR = "#a9b5aa"
ACCENT_COLOR = "#e5a447"
SUCCESS_COLOR = "#69a979"
ERROR_COLOR = "#e07a6d"


def field(value: object, *names: str, default: Any = None) -> Any:
    """Read the first matching attribute/key from a workbench view object."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def sequence(value: object) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def point_from(value: object) -> Optional[Point]:
    """Best-effort extraction used only for rendering coordinates."""

    if value is None:
        return None
    nested = field(value, "point", "move", "vertex", default=value)
    if nested is not value:
        return point_from(nested)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
    row = field(value, "row")
    col = field(value, "col", "column")
    if row is None or col is None:
        return None
    try:
        return int(row), int(col)
    except (TypeError, ValueError):
        return None


def finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_probability(value: object) -> str:
    number = finite_float(value)
    if number is None:
        return "—"
    if -1.0 <= number <= 1.0:
        number *= 100.0
    return f"{number:.1f}%"


def format_lead(value: object) -> str:
    number = finite_float(value)
    if number is None:
        return "—"
    if abs(number) < 0.05:
        return "均势"
    return f"黑+{number:.1f}" if number > 0 else f"白+{-number:.1f}"


def blend(color: str, background: str, strength: float) -> str:
    """Blend two ``#rrggbb`` colors without relying on Canvas alpha support."""

    strength = max(0.0, min(1.0, strength))
    foreground_rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    background_rgb = tuple(
        int(background[index : index + 2], 16) for index in (1, 3, 5)
    )
    mixed = tuple(
        round(base + (front - base) * strength)
        for front, base in zip(foreground_rgb, background_rgb)
    )
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def ownership_color(value: object) -> Optional[str]:
    number = finite_float(value)
    if number is None:
        return None
    number = max(-1.0, min(1.0, number))
    if abs(number) < 0.04:
        return None
    foreground = "#315f95" if number > 0 else "#c95e55"
    return blend(foreground, BOARD_COLOR, 0.18 + abs(number) * 0.52)
