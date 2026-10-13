"""Prevent an exported training run from restarting before handoff cancellation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex


HANDOFF_MARKER = "HANDOFF.json"
IMPORT_PENDING_MARKER = "HANDOFF_IMPORTING.json"


class HandoffLockedError(ValueError):
    """An export hold or incomplete import prevents local training writes."""


def ensure_training_available(directory: Path) -> None:
    """Reject held runs and descendants of an incomplete imported snapshot.

    Callers must hold the relevant OS lock so export and training cannot race.
    Marker existence is authoritative; metadata only improves the error message.
    """

    for parent in (directory, *directory.parents):
        pending = parent / IMPORT_PENDING_MARKER
        if pending.exists() or pending.is_symlink():
            raise HandoffLockedError(
                f"接力导入尚未完成或已经失败，不能启动训练：{pending}。"
                "请保留现场并重新导入到新的目录。"
            )

    marker = directory / HANDOFF_MARKER
    if not marker.exists() and not marker.is_symlink():
        return
    details: list[str] = []
    trainer = "katago"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            if value.get("trainer") == "native":
                trainer = "native"
            for key, label in (
                ("bundle_id", "编号"),
                ("bundle", "接力包"),
                ("created_at", "时间"),
            ):
                item = value.get(key)
                if isinstance(item, str) and item:
                    details.append(f"{label}：{item}")
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    metadata = f"（{'，'.join(details)}）" if details else ""
    quoted = ("'" + str(directory).replace("'", "''") + "'"
              if os.name == "nt" else shlex.quote(str(directory)))
    cancel = (f"py train.py handoff-cancel --run-dir {quoted}"
              if trainer == "native" else "py train_rl.py handoff-cancel")
    raise HandoffLockedError(
        f"训练目录已导出接力包{metadata}，本机训练已锁定：{directory}。"
        "如需在本机恢复，请先确认其他电脑已停止同一次训练，"
        f"再运行 {cancel} 取消交接锁定。"
    )


__all__ = [
    "HANDOFF_MARKER",
    "IMPORT_PENDING_MARKER",
    "HandoffLockedError",
    "ensure_training_available",
]
