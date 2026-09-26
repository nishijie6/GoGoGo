"""Run ownership, progress logging, and cooperative pause/stop checks."""

import json
import os
from pathlib import Path
import socket
import tempfile
import time
import uuid

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
    def __init__(self, directory: Path, *, operation: str = "train"):
        self.path = directory / "events.jsonl"
        self.phase = "startup"
        self.monitor_path = directory / "monitor.json"
        self.session_id = uuid.uuid4().hex
        self.operation = operation
        self.state = {"schema_version": 1, "session_id": self.session_id,
                      "pid": os.getpid(), "hostname": socket.gethostname(),
                      "operation": operation, "status": "running", "last_event": None,
                      "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "started_at_epoch": time.time()}
        self._last_heartbeat = float("-inf")

    def __enter__(self):
        self({"event": "run_started", "operation": self.operation})
        return self

    def __exit__(self, error_type, error, _traceback):
        if error_type is None:
            self({"event": "run_completed"})
        elif isinstance(error, (TrainingStopped, KeyboardInterrupt)):
            self({"event": "run_stopped", "reason": str(error) or "KeyboardInterrupt"})
        else:
            self({"event": "run_failed", "error": f"{error_type.__name__}: {error}"})
        return False

    def heartbeat(self, *, force: bool = False):
        """Write a small sidecar without importing the optional training packages.

        A heartbeat proves the trainer is reaching cooperative checks, rather
        than guessing liveness from a lock file left behind by an old run.
        """
        now = time.monotonic()
        if not force and now - self._last_heartbeat < 2.0:
            return
        record = {**self.state, "phase": self.phase,
                  "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "updated_at_epoch": time.time()}
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix="monitor.", suffix=".tmp",
                                                      dir=self.monitor_path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.monitor_path)
            self._last_heartbeat = now
        except OSError:
            # A transient sidecar write failure must not interrupt training.
            pass
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def __call__(self, event: dict):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": self.phase,
                  "session_id": self.session_id, **event}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        kind = event.get("event")
        self.state["last_event"] = kind
        self.state["last_event_at"] = record["time"]
        if "iteration" in event:
            self.state["iteration"] = event["iteration"]
        if kind in ("paused", "run_stopped"):
            self.state["reason"] = event.get("reason")
        if kind == "run_failed":
            self.state["error"] = event.get("error")
        states = {"paused": "paused", "resumed": "running", "run_started": "running",
                  "run_completed": "completed", "run_stopped": "stopped", "run_failed": "failed"}
        if kind in states:
            self.state["status"] = states[kind]
        if kind == "resumed":
            self.state.pop("reason", None)
        self.heartbeat(force=True)


class TrainingControl:
    def __init__(self, directory: Path, pause_for_game: bool, progress):
        self.directory, self.pause_for_game, self.progress = directory, pause_for_game, progress
        self.next_check = 0.0

    def __call__(self):
        if time.monotonic() < self.next_check:
            return
        paused = False
        while True:
            heartbeat = getattr(self.progress, "heartbeat", None)
            if heartbeat is not None:
                heartbeat()
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
