"""Cached, read-only NVIDIA telemetry without importing training dependencies."""

from __future__ import annotations

import csv
from collections import deque
from datetime import datetime, timezone
import math
import subprocess
import threading
import time


def _number(value, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        return None
    return number


class GPUMonitor:
    """Share one bounded query across clients for each two-second interval."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cached = None
        self.expires = 0
        self.history = deque(maxlen=900)
        self.stop_event = threading.Event()
        self.worker = None

    def start(self):
        """Keep collecting while the browser is closed; history stays in memory."""
        if self.worker is None:
            self.worker = threading.Thread(target=self._collect, daemon=True, name="gpu-monitor")
            self.worker.start()

    def stop(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=3)

    def _collect(self):
        while not self.stop_event.is_set():
            self.snapshot()
            self.stop_event.wait(2)

    def snapshot(self):
        with self.lock:
            if self.cached is not None and time.monotonic() < self.expires:
                return {**self.cached, "history": list(self.history)}
            result = {"available": False, "gpus": [], "sampled_at": None,
                      "reason": "未检测到可读取的 NVIDIA GPU"}
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=2, check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
                    if len(row) != 6 or not row[0].strip().isdigit():
                        continue
                    index, name, utilization, used, total, temperature = (value.strip() for value in row)
                    result["gpus"].append({
                        "index": int(index), "name": name,
                        "utilization_percent": _number(utilization, 100),
                        "memory_used_mib": _number(used), "memory_total_mib": _number(total),
                        "temperature_c": _number(temperature),
                    })
                result["available"] = bool(result["gpus"])
                if result["available"]:
                    result["reason"] = None
                    result["sampled_at"] = datetime.now(timezone.utc).isoformat()
            except subprocess.TimeoutExpired:
                result["reason"] = "GPU 状态读取超时"
            except (OSError, subprocess.SubprocessError):
                result["reason"] = "无法读取 GPU 状态，请确认 NVIDIA 驱动可用"
            self.cached = result
            self.expires = time.monotonic() + 2
            now = datetime.now(timezone.utc)
            self.history.append({
                "sampled_at": now.isoformat(),
                "gpus": [{key: gpu[key] for key in ("index", "name", "utilization_percent")}
                         for gpu in result["gpus"]],
            })
            while self.history and (now - datetime.fromisoformat(self.history[0]["sampled_at"])).total_seconds() > 1800:
                self.history.popleft()
            return {**result, "history": list(self.history)}
