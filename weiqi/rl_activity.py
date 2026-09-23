"""Dependency-free Windows/WSL game-activity heartbeat for training pauses."""

from pathlib import Path
import time
import uuid


ACTIVITY_DIRECTORY = Path(__file__).resolve().parents[1] / "training_runs" / ".activity"
HEARTBEAT_TTL = 8.0


class GameActivity:
    def __init__(self, directory: Path = ACTIVITY_DIRECTORY):
        self.path = directory / f"{uuid.uuid4().hex}.active"

    def update(self, active: bool) -> None:
        try:
            if active:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.touch()
            else:
                self.close()
        except OSError:
            # A read-only application folder must not prevent playing Go.
            pass

    def close(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def game_is_active(directory: Path = ACTIVITY_DIRECTORY) -> bool:
    now = time.time()
    for marker in directory.glob("*.active"):
        try:
            if now - marker.stat().st_mtime < HEARTBEAT_TTL:
                return True
        except FileNotFoundError:
            pass
    return False
