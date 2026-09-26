"""Read-only, dependency-free views of training artifacts for the local dashboard."""

from __future__ import annotations

from collections import OrderedDict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time


MATCH_PHASES = {"selfplay", "screening", "evaluation", "opponent_pool"}
TERMINAL = {"completed", "stopped", "failed"}
MAX_JSON = 8 * 1024 * 1024


def iso_time(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat()


def timestamp(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def strict_json(value):
    def reject_constant(token):
        raise ValueError(f"Invalid JSON number: {token}")
    return json.loads(value, parse_constant=reject_constant)


def metric(row, **extra):
    """Win rate uses every scheduled result, as does the trainer's score rate."""
    result = {key: row.get(key) for key in (
        "games", "wins", "losses", "draws", "truncated", "paired", "simulations_per_move",
        "opponent_name", "opponent_sha256", "candidate_sha256", "best_iteration")}
    games, wins = row.get("games"), row.get("wins")
    result["win_rate"] = wins / games if isinstance(games, (int, float)) and games > 0 and isinstance(wins, (int, float)) else None
    result["score_rate"] = row.get("score_rate") if games else None
    result.update(extra)
    return result


class EventTail:
    """Incremental JSONL reader; incomplete trailing records are retried next poll."""

    def __init__(self):
        self.offset = 0
        self.identity = None
        self.modified = None
        self.events = deque(maxlen=40)
        self.ready = {}
        self.last = {}
        self.iteration = None
        self.schedule = {}
        self.batch = None
        self.results = {}
        self.active_games = {}
        self.progress = None
        self.step = {}
        self.session = None
        self.lifecycle = {}
        self.skipped = False
        self.malformed = False

    def update(self, path):
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if (self.identity is not None and self.identity != identity) or stat.st_size < self.offset or (
                    stat.st_size == self.offset and self.modified is not None and stat.st_mtime_ns != self.modified):
                self.__init__()
            self.identity, self.modified = identity, stat.st_mtime_ns
            with path.open("rb") as stream:
                if stat.st_size - self.offset > MAX_JSON:
                    stream.seek(stat.st_size - MAX_JSON)
                    stream.readline()
                    self.offset = stream.tell()
                    self.skipped = True
                stream.seek(self.offset)
                data = stream.read(MAX_JSON)
            end = data.rfind(b"\n")
            if end < 0:
                return
            self.offset += end + 1
            for line in data[:end].splitlines():
                try:
                    event = strict_json(line.decode("utf-8-sig"))
                    if isinstance(event, dict) and isinstance(event.get("event"), str):
                        self.accept(event)
                except (ValueError, UnicodeError):
                    self.malformed = True
        except OSError:
            pass

    def accept(self, event):
        name, phase = event.get("event"), event.get("phase")
        session = event.get("session_id")
        if session and session != self.session:
            self.session, self.lifecycle = session, {}
            self.batch, self.results, self.active_games = None, {}, {}
            self.schedule, self.step, self.progress = {}, {}, None
        if name in {"run_started", "run_completed", "run_stopped", "run_failed"}:
            self.lifecycle = event
        if name in {"run_started", "ready", "iteration_started"}:
            self.batch, self.results, self.progress = None, {}, None
            self.active_games = {}
            self.schedule, self.step = {}, {}
        if name == "run_started":
            self.session = event.get("session_id")
            self.ready = {}
        if name == "ready":
            self.ready = event
        if name in {"ready", "iteration_started", "iteration_completed"}:
            self.iteration = event.get("iteration", self.iteration)
        if name == "iteration_completed":
            self.step = {}
        if phase != self.last.get("phase"):
            self.progress = None
            self.active_games = {}
        if name == "selfplay_schedule":
            self.schedule = event
        if name == "games_started":
            self.batch, self.results = dict(event), {}
            self.active_games = {}
            self.batch["iteration"] = self.iteration
            self.progress = {"completed": 0, "total": event.get("total"), "unit": "games"}
        if name == "game_started" and self.batch and event.get("batch_id") == self.batch.get("batch_id"):
            self.active_games[event.get("index")] = event
        if name == "game":
            # Legacy logs lack batch boundaries and winners. Retain progress only.
            if self.batch is not None and phase == self.batch.get("phase"):
                self.results[event.get("index")] = event
                self.active_games.pop(event.get("index"), None)
        if name in {"run_completed", "run_stopped", "run_failed"}:
            self.active_games = {}
        if name in {"game", "search_progress"}:
            self.progress = {"completed": event.get("completed"), "total": event.get("total"), "unit": "games"}
        if name == "training_step":
            self.step = event
            self.progress = {"completed": event.get("step"), "total": event.get("total"), "unit": "steps"}
        if name in {"position_progress", "position_diagnostic"} and "total" in event:
            self.progress = {"completed": event.get("completed"), "total": event.get("total"), "unit": "positions"}
        self.last = event
        # Iteration completion contains large reports; keep event feed compact.
        self.events.append({key: value for key, value in event.items() if key not in {
            "jobs", "position_quality", "opponent_pool", "selfplay_opponents", "config"}})

    def current_match(self, phase):
        if not self.batch or phase != self.batch.get("phase"):
            return None
        rows = list(self.results.values())
        jobs = self.batch.get("jobs", [])
        opponents = list(dict.fromkeys(job.get("opponent_name") or "unknown" for job in jobs))
        opponents = [name for name in opponents if name != "candidate_self"]
        measured = [row for row in rows if row.get("opponent_name") != "candidate_self"]
        counts = {name: sum(row.get("result") == value for row in measured)
                  for name, value in (("wins", "win"), ("losses", "loss"), ("draws", "draw"), ("truncated", "truncated"))}
        known = sum(counts.values())
        unknown = len(measured) - known
        count = len(measured)
        hashes = list(dict.fromkeys(job.get("opponent_sha256") for job in jobs if job.get("opponent_name") != "candidate_self"))
        return {"phase": phase, "iteration": self.batch.get("iteration"),
                "opponent": " / ".join(opponents) if opponents else "candidate_self",
                "opponent_sha256": hashes[0] if len(hashes) == 1 else None,
                "games": count, **counts, "unknown": unknown,
                "total": sum(job.get("opponent_name") != "candidate_self" for job in jobs),
                "batch_total": self.batch.get("total"),
                "win_rate": counts["wins"] / count if count and not unknown else None,
                "score_rate": (counts["wins"] + counts["draws"] / 2) / count if count and not unknown else None}


class MonitorStore:
    def __init__(self, root: Path, preferred_run=None):
        self.root = root.resolve()
        self.preferred_run = preferred_run
        self.lock = threading.RLock()
        self.tails = OrderedDict()
        self.json_cache = OrderedDict()

    def safe_path(self, path):
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("路径超出训练目录")
        return resolved

    def read_json(self, path):
        path = self.safe_path(path)
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = self.json_cache.get(path)
            if cached and cached[0] == signature:
                self.json_cache.move_to_end(path)
                return cached[1]
            if stat.st_size > MAX_JSON:
                return {}
            value = strict_json(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                return {}
            self.json_cache[path] = (signature, value)
            if len(self.json_cache) > 2048:
                self.json_cache.popitem(last=False)
            return value
        except (OSError, ValueError):
            return {}

    def run_path(self, run_id):
        if not run_id or run_id in {".", ".."} or any(char in run_id for char in "/\\\0"):
            raise ValueError("无效的训练记录")
        path = self.safe_path(self.root / run_id)
        if not path.is_dir() or run_id.startswith("."):
            raise FileNotFoundError("找不到训练记录")
        return path

    def run_status(self, path, tail=None):
        monitor = self.read_json(path / "monitor.json")
        last = tail.last if tail else {}
        lifecycle = tail.lifecycle if tail else {}
        terminal = {"run_completed": "completed", "run_stopped": "stopped", "run_failed": "failed"}.get(lifecycle.get("event"))
        if terminal and (not monitor or not tail.session or monitor.get("session_id") == tail.session):
            when = timestamp(lifecycle.get("time"))
            return {"state": terminal, "phase": lifecycle.get("phase"),
                    "updated_at": iso_time(when) if when else None,
                    "age_seconds": round(max(0, time.time() - when), 1) if when else None,
                    "reason": lifecycle.get("error") or lifecycle.get("reason")}
        if tail and tail.session and monitor and monitor.get("session_id") != tail.session:
            when = timestamp(last.get("time"))
            return {"state": "stale", "phase": last.get("phase"),
                    "updated_at": iso_time(when) if when else None,
                    "age_seconds": round(max(0, time.time() - when), 1) if when else None,
                    "reason": "日志与心跳来自不同会话，等待新的训练心跳"}
        when = monitor.get("updated_at_epoch")
        if not isinstance(when, (float, int)):
            when = timestamp(monitor.get("updated_at"))
        if monitor and when is not None:
            state = monitor.get("status", "stale")
            if state not in TERMINAL | {"running", "paused"}:
                state = "stale"
            age = max(0, time.time() - when)
            if state in {"running", "paused"} and age > 15:
                state = "stale"
            return {"state": state, "phase": monitor.get("phase", last.get("phase")),
                    "updated_at": iso_time(when), "age_seconds": round(age, 1),
                    "reason": monitor.get("error") or monitor.get("reason")}
        when = timestamp(last.get("time"))
        if when is None:
            for name in ("events.jsonl", "summary.json", "config.resolved.json"):
                try:
                    value = self.safe_path(path / name).stat().st_mtime
                    when = max(when or 0, value)
                except OSError:
                    pass
        # A recent log is evidence of activity, never proof a legacy process is alive.
        return {"state": "historical" if when else "empty", "phase": last.get("phase"),
                "updated_at": iso_time(when) if when else None,
                "age_seconds": round(max(0, time.time() - when), 1) if when else None,
                "reason": "此记录没有训练心跳，无法确认进程状态" if when else "等待训练数据"}

    def runs(self):
        with self.lock:
            rows = []
            if self.root.is_dir():
                for path in self.root.iterdir():
                    if path.name.startswith(".") or not path.is_dir():
                        continue
                    try:
                        self.safe_path(path)
                        if not any((path / name).is_file() for name in ("events.jsonl", "summary.json", "monitor.json", "config.resolved.json")):
                            continue
                        status = self.run_status(path)
                        summary = self.read_json(path / "summary.json")
                        rows.append({"id": path.name, "name": path.name, "status": status["state"],
                                     "updated_at": status["updated_at"], "iteration": summary.get("iteration")})
                    except (ValueError, OSError):
                        continue
            rows.sort(key=lambda row: (row["status"] in {"running", "paused"}, row["updated_at"] or ""), reverse=True)
            preferred = next((row["id"] for row in rows if row["id"] == self.preferred_run), None)
            return {"runs": rows, "default_run": preferred or (rows[0]["id"] if rows else None)}

    def historical_opponents(self, path, summary):
        rows = []
        iteration = summary.get("iteration")
        # Legacy per-game artifacts provide exact results even when old JSONL did not.
        records = {}
        if isinstance(iteration, int):
            for game_path in sorted((path / "iterations" / f"{iteration:06d}" / "selfplay").glob("game_*.json")):
                game = self.read_json(game_path)
                name = game.get("training_opponent", {}).get("name")
                if name:
                    records.setdefault(name, []).append(game)
        for source in summary.get("selfplay_opponents", []):
            data = dict(source)
            name = source.get("name")
            games = records.get(name, [])
            if name != "candidate_self" and len(games) == source.get("games") and games:
                wins = sum(game.get("reason") != "length_limit" and game.get("winner") is not None
                           and game.get("winner") == game.get("candidate_color") for game in games)
                truncated = sum(game.get("reason") == "length_limit" for game in games)
                losses = sum(game.get("reason") != "length_limit" and game.get("winner") is not None
                             and game.get("winner") != game.get("candidate_color") for game in games)
                draws = len(games) - wins - truncated - losses
                data.update(wins=wins, losses=losses, draws=draws, truncated=truncated,
                            score_rate=(wins + draws / 2) / len(games))
            if name == "candidate_self":
                for key in ("wins", "losses", "draws", "score_rate"):
                    data.pop(key, None)
                data["truncated"] = source.get("games", 0) - source.get("finished", 0)
            rows.append(metric(data, name=name, sha256=source.get("sha256"), role="training", iteration=iteration))
        pool = summary.get("opponent_pool", {})
        for source in pool.get("opponents", []):
            rows.append(metric(source, name=source.get("opponent"), sha256=source.get("opponent_sha256"),
                               role="pool", iteration=iteration, simulations_per_move=pool.get("simulations_per_move")))
        for role in ("screening", "evaluation"):
            source = summary.get(role, {})
            if source.get("games"):
                rows.append(metric(source, name=source.get("opponent_name") or source.get("opponent", "best（当轮基准）"),
                                   sha256=source.get("opponent_sha256"), role=role, iteration=iteration))
        # Standalone evaluate and benchmark outputs.
        if not isinstance(iteration, int) and summary.get("games"):
            rows.append(metric(summary, name=Path(summary.get("opponent", "unknown").replace("\\", "/")).name,
                               sha256=summary.get("opponent_sha256"), role="evaluation", iteration=None))
        return rows

    def snapshot(self, run_id):
        with self.lock:
            path = self.run_path(run_id)
            tail = self.tails.setdefault(run_id, EventTail())
            self.tails.move_to_end(run_id)
            if len(self.tails) > 32:
                self.tails.popitem(last=False)
            tail.update(self.safe_path(path / "events.jsonl"))
            latest = self.read_json(path / "summary.json")
            summaries = []
            for summary_path in sorted((path / "iterations").glob("*/summary.json")):
                row = self.read_json(summary_path)
                if isinstance(row.get("iteration"), int):
                    summaries.append(row)
            if latest.get("iteration") and not any(row["iteration"] == latest["iteration"] for row in summaries):
                summaries.append(latest)
            summaries.sort(key=lambda row: row["iteration"])
            history = [{"iteration": row["iteration"], "training_steps": row.get("training_steps"),
                        "replay_samples": row.get("replay_samples"), "loss": row.get("training", {}).get("loss"),
                        "screening": metric(row.get("screening", {})), "evaluation": metric(row.get("evaluation", {})),
                        "pool": [metric(p, name=p.get("opponent"), sha256=p.get("opponent_sha256")) for p in row.get("opponent_pool", {}).get("opponents", [])],
                        "promoted": row.get("promoted", False)} for row in summaries]
            status = self.run_status(path, tail)
            phase = status["phase"] or tail.last.get("phase")
            config = self.read_json(path / "config.resolved.json")
            match = tail.current_match(phase)
            jobs = (tail.batch or {}).get("jobs", []) if (tail.batch or {}).get("phase") == phase else []
            opponents = {}
            for job in jobs:
                name, checksum = job.get("opponent_name", "unknown"), job.get("opponent_sha256")
                row = opponents.setdefault((name, checksum), {"name": name, "sha256": checksum, "games": 0})
                row["games"] += 1
            if not opponents and phase == "selfplay":
                for row in tail.schedule.get("champions", []):
                    opponents[(row["name"], row.get("sha256"))] = {**row, "games": None}
                if tail.schedule.get("candidate_self_games"):
                    opponents[("candidate_self", None)] = {"name": "candidate_self", "sha256": None, "games": tail.schedule["candidate_self_games"]}
            warnings = []
            if status["state"] in {"historical", "stale"}:
                warnings.append(status["reason"] or "训练心跳已超过 15 秒未更新；可能已退出、失联或卡住。")
            if phase in MATCH_PHASES and not tail.batch:
                warnings.append("旧版日志没有逐局胜负或对手记录；实时胜率留空，历史成绩使用已保存的汇总与棋谱。")
            if tail.skipped:
                warnings.append("日志较大，实时视图只载入最近 8 MB；历史趋势来自完整轮次汇总。")
            if tail.malformed:
                warnings.append("已跳过无法解析的日志行；后续有效记录仍会更新。")
            last_completed = summaries[-1] if summaries else latest
            current = {
                "iteration": tail.iteration if tail.iteration is not None else latest.get("iteration"),
                "completed_iteration": last_completed.get("iteration"),
                "best_iteration": latest.get("best_iteration", tail.ready.get("best_iteration")),
                "training_steps": tail.step.get("training_steps", latest.get("training_steps", tail.ready.get("training_steps"))),
                "replay_samples": latest.get("replay_samples", tail.ready.get("replay_samples")),
                "device": tail.ready.get("device", config.get("hardware", {}).get("device")),
                "gpu": tail.ready.get("gpu"),
                "loss": tail.step.get("loss", latest.get("training", {}).get("loss")),
                "progress": tail.progress if phase == tail.last.get("phase") else None,
                "opponents": list(opponents.values()), "match": match,
                "active_games": list(tail.active_games.values()) if status["state"] in {"running", "paused"} else [],
            }
            return {"run": {"id": run_id, "name": run_id}, "server_time": iso_time(), "status": status,
                    "current": current, "latest": latest, "history": history,
                    "opponents": self.historical_opponents(path, last_completed),
                    "events": list(reversed(tail.events)), "warnings": warnings}
