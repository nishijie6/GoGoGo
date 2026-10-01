"""Crash-safe file writes, file hashing, and timestamps shared by training code.

Every atomic write goes to a hidden ``.<name>.<pid>.<uuid>.tmp`` sibling, is
fsynced, and then replaces the target.  Stale-temp sweeps match the ``.tmp``
suffix, so the naming scheme must stay stable.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator


class FileLockHeld(OSError):
    """Another process already holds the lock."""


def local_timestamp() -> str:
    """Local wall-clock time with offset, to the second, for persisted records."""

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def temporary_sibling(path: Path, suffix: str = ".tmp") -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}{suffix}")


@contextlib.contextmanager
def atomic_open(path: Path, mode: str = "w") -> Iterator[IO]:
    """Yield a stream whose contents replace ``path`` only if the block succeeds."""

    if mode not in ("w", "wb"):
        raise ValueError("atomic_open only supports 'w' and 'wb'")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(path)
    text = {"encoding": "utf-8", "newline": "\n"} if mode == "w" else {}
    try:
        with temporary.open(mode, **text) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_write_text(path: Path, content: str) -> None:
    with atomic_open(path) as stream:
        stream.write(content)


def atomic_write_json(
    path: Path, value: object, *, sort_keys: bool = True, allow_nan: bool = True
) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys,
                   allow_nan=allow_nan) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire_file_lock(path: Path) -> IO[bytes]:
    """Take a nonblocking OS lock on ``path`` and record this process ID in it.

    The operating system releases the lock if the process dies. Raises
    :class:`FileLockHeld` when another process holds it; release with
    :func:`release_file_lock`.
    """

    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise FileLockHeld(f"{path} is locked by another process") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
    except BaseException:
        handle.close()
        raise
    return handle


def release_file_lock(handle: IO[bytes]) -> None:
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
