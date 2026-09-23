"""Run ownership, progress logging, and cooperative pause/stop checks."""

import json
import os
from pathlib import Path
import time

from ..rl_activity import game_is_active


class TrainingStopped(Exception):
    pass


class RunLock:
    """OS-owned lock, released automatically if a training process crashes."""
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "run.lock"

    def __enter__(self):
        self.stream = self.path.open("a+b")
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise ValueError(f"Another trainer owns {self.path.parent}") from error
        return self

    def __exit__(self, *_):
        self.stream.close()


class Progress:
    def __init__(self, directory: Path):
        self.path = directory / "events.jsonl"
        self.phase = "startup"

    def __call__(self, event: dict):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": self.phase, **event}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


class TrainingControl:
    def __init__(self, directory: Path, pause_for_game: bool, progress):
        self.directory, self.pause_for_game, self.progress = directory, pause_for_game, progress
        self.next_check = 0.0

    def __call__(self):
        if time.monotonic() < self.next_check:
            return
        paused = False
        while True:
            if (self.directory / "STOP").exists():
                raise TrainingStopped("STOP file detected")
            manual = (self.directory / "PAUSE").exists()
            active = self.pause_for_game and game_is_active()
            if not manual and not active:
                break
            if not paused:
                self.progress({"event": "paused", "reason": "PAUSE" if manual else "active_game"})
                paused = True
            time.sleep(0.25)
        if paused:
            self.progress({"event": "resumed"})
        self.next_check = time.monotonic() + 0.25
