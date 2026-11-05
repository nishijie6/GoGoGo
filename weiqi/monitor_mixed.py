"""Read mixed KataGo metrics and promotion evidence for the shared dashboard."""
from __future__ import annotations

import time

from .monitor_match import match_totals, read_paired_match
from .rl_metrics import parse_model_name
from .training_monitor import iso_time, timestamp
from .training_standard import get_training_standard, sample_evaluation_schedule

STAGES = ("confirmation", "screen")
OUTCOMES = ("promoted", "health_rejected", "confirmation_rejected", "screen_rejected", "evaluation_failed")
COUNTS = ("wins", "losses", "draws")
BOARD_SIZES = (7, 8, 9, 13, 19)
HEALTH_FIELDS = ("window_games", "black_score_rate", "short_game_rate", "early_double_pass_rate",
                 "pass_move_rate", "extreme_margin_rate", "warnings", "estimable", "health_reasons", "status")


def _evaluation_root(path, cycle, candidate):
    if type(cycle) is not int or cycle < 1 or not isinstance(candidate, str) or parse_model_name(candidate) is None:
        return None
    return path / "promotion" / "evaluations" / f"cycle-{cycle}-{candidate}"


def _evaluation_boards(record, training_boards):
    boards = record.get("evaluation_boards", training_boards)
    if (not isinstance(boards, list) or not boards
            or any(type(board) is not int or board not in training_boards for board in boards)
            or len(set(boards)) != len(boards)):
        return []
    return boards


def _protocol(store, stage_path, boards, identity):
    if not boards:
        return None
    manifest = store.read_json(stage_path / "evaluation_manifest.json")
    protocol = manifest.get("protocol") or {}
    games = protocol.get("games_per_board")
    if (protocol.get("board_sizes") != boards or type(games) is not int
            or not 2 <= games <= 1000 or games % 2):
        return None
    for role, model, checksum in (("candidate", identity["candidate"], identity["candidate_sha256"]),
                                  ("baseline", identity["baseline"], identity["baseline_sha256"])):
        spec = manifest.get(role) or {}
        if not checksum or spec.get("model_name") != model or spec.get("sha256") != checksum:
            return None
    return games


def _match(store, stage_path, boards, identity, *, live):
    games = _protocol(store, stage_path, boards, identity)
    if games is None:
        return None
    candidate, champion = identity["candidate"], identity["baseline"]
    rows, outcomes = [], []
    for board in boards:
        row = read_paired_match(store, stage_path / "boards" / f"{board}x{board}" / "pairs", games,
                                candidate, champion)
        previous = dict.fromkeys(COUNTS, 0)
        for point in row.pop("points"):
            outcomes.append(next(key for key in COUNTS if point[key] != previous[key]))
            previous = point
        rows.append({"board": board, **row})
    progress = store.read_json(stage_path / "progress.json")
    updated = max((time for row in rows if (time := timestamp(row["updated_at"])) is not None), default=None)
    counts = {key: sum(row[key] for row in rows) for key in COUNTS}
    aggregate = match_totals(counts, games * len(boards), candidate, champion,
                             sum(row["completed_pairs"] for row in rows), outcomes,
                             iso_time(updated) if updated is not None else progress.get("updated_at"))
    return {**aggregate, "live": live, "by_board": rows, "evaluation_stage": stage_path.name,
            "current_board": progress.get("current_board")}


def _rounds(store, path, boards):
    directory = store.safe_path(path / "promotion/evaluations")
    rounds = []
    for folder in directory.glob("cycle-*"):
        decision = store.read_json(folder / "decision.json")
        cycle, candidate = decision.get("cycle"), decision.get("candidate")
        if _evaluation_root(path, cycle, candidate) != folder or decision.get("outcome") not in OUTCOMES:
            continue
        rounds.append(decision)
    rows = []
    for decision in sorted(rounds, key=lambda row: row["cycle"])[-100:]:
        cycle, candidate = decision["cycle"], decision["candidate"]
        directory = _evaluation_root(path, cycle, candidate)
        selected = _evaluation_boards(decision, boards)
        row = {"champion_round": cycle, "iteration": cycle, "candidate": candidate,
               "name": decision.get("baseline"), "outcome": decision["outcome"], "role": "champion",
               "promoted": decision["outcome"] == "promoted", "timestamp": decision.get("at"),
               "evaluation_boards": selected, "require_confidence": decision.get("require_confidence") is not False,
               "health_reasons": decision.get("health_reasons") or [],
               "candidate_sha256": decision.get("candidate_sha256"), "baseline_sha256": decision.get("baseline_sha256")}
        for stage in STAGES:
            stage_path = directory / stage
            games = _protocol(store, stage_path, selected, {**row, "baseline": row["name"]})
            summary = store.read_json(stage_path / "summary.json")
            by_board = summary.get("results_by_board") or {}
            if not games or summary.get("status") != "complete" or set(by_board) != set(map(str, selected)):
                continue
            boards_results = []
            for board in selected:
                entry = by_board[str(board)]
                counts = (entry.get("candidate_wins"), entry.get("baseline_wins"), entry.get("draws"))
                if (entry.get("games") != games or any(type(n) is not int or n < 0 for n in counts)
                        or sum(counts) != games):
                    break
                wins, losses, draws = counts
                boards_results.append({"board": board, "games": games, "total": games,
                                       "wins": wins, "losses": losses, "draws": draws, "win_rate": wins/games,
                                       "score_rate": (wins + draws / 2)/games})
            if len(boards_results) != len(selected):
                continue
            counts = {key: sum(item[key] for item in boards_results) for key in COUNTS}
            total = games * len(selected)
            row.update(**counts, games=total, total=total, win_rate=counts["wins"]/total,
                       score_rate=(counts["wins"] + counts["draws"]/2)/total,
                       by_board=boards_results, evaluation_stage=stage)
            if stage == "confirmation" and isinstance(decision.get("confirmation_lower"), (int, float)):
                row["interval"] = {"method": "配对得分单侧下界", "lower": decision["confirmation_lower"]}
            break
        rows.append(row)
    return rows


def historical_curve(store, path, championship, row):
    stage = row.get("evaluation_stage")
    if stage not in STAGES:
        return {"games": 0, "total": 0, **row, "live": False, "points": []}
    root = _evaluation_root(path, row["iteration"], row["candidate"])
    boards = _evaluation_boards(row, championship["boards"])
    curve = _match(store, root / stage, boards, {**row, "baseline": row["name"]}, live=False)
    return {**row, "live": False, "points": (curve or {}).get("points", [])}


def runtime_status(store, path, latest, promotion, activity=None, stage="mixed_training"):
    when = activity or latest.get("timestamp")
    epoch = timestamp(when)
    status = {"state": "historical", "phase": stage, "updated_at": when,
              "age_seconds": round(max(0, time.time() - epoch), 1) if epoch is not None else None,
              "reason": "显示最近记录，进程未核验"}
    if promotion.get("halted"):
        halted = promotion["halted"]
        when = halted.get("at") or when
        epoch = timestamp(when)
        return {**status, "state": "failed", "updated_at": when,
                "age_seconds": round(max(0, time.time() - epoch), 1) if epoch is not None else None,
                "reason": "健康停机 · " + "、".join(halted.get("reasons") or [])}
    if store.read_json(path / "evaluation_pause.json").get("active"):
        return {**status, "state": "paused", "reason": "训练暂缓，正在执行独立评测或对照"}
    process = store.read_json(path / "process.json")
    if type(process.get("pid")) is not int or not isinstance(process.get("create_time"), (int, float)):
        return status
    try:
        import psutil
    except ImportError:
        return status
    try:
        owner = psutil.Process(process["pid"])
        if (abs(owner.create_time() - process["create_time"]) >= .01
                or owner.cmdline() != process.get("command") or owner.status() == psutil.STATUS_ZOMBIE):
            return {**status, "state": "stopped", "reason": "记录中的训练进程已退出"}
        return {**status, "state": "running", "reason": "训练进程已核验"}
    except psutil.NoSuchProcess:
        return {**status, "state": "stopped", "reason": "记录中的训练进程已退出"}
    except (psutil.Error, OSError, ValueError):
        return status


def snapshot(store, path, run_id):
    manifest = store.read_json(path / "mixed_manifest.json")
    boards = [size for size in (manifest.get("identity") or {}).get("board_sizes", []) if size in BOARD_SIZES]
    history = store.read_history(path / "metrics/history.jsonl")
    latest = store.read_json(path / "metrics/latest.json") or (history[-1] if history else {})
    health = store.read_json(path / "health_by_board.json")
    promotion = store.read_json(path / "promotion/state.json")
    if latest and (not history or history[-1].get("cycle") != latest.get("cycle")):
        history = [*history, latest]
    diagnostics = [
        {"iteration": row.get("cycle"), "samples": row.get("trained_samples"),
         "data_rows_per_hour": row.get("new_data_rows_per_hour"),
         **{key: row.get(key) for key in ("loss", "policy_loss", "value_loss",
                                         "selfplay_stage_seconds", "shuffle_seconds", "train_seconds")}}
        for row in history[-300:]
    ]
    rounds = _rounds(store, path, boards)
    champion, count = promotion.get("champion"), promotion.get("promotions", 0)
    pending = promotion.get("pending") or {}
    root = _evaluation_root(path, pending.get("cycle"), pending.get("candidate"))
    phase, live = ("promotion_waiting" if not champion else "mixed_training"), None
    if root and not promotion.get("halted"):
        identity = {"candidate": pending["candidate"], "candidate_sha256": pending.get("sha256"),
                    "baseline": champion, "baseline_sha256": promotion.get("champion_sha256")}
        selected = _evaluation_boards(pending, boards)
        phase = "promotion_pending"
        for stage in STAGES:
            if (live := _match(store, root / stage, selected, identity, live=True)) is not None:
                phase = f"promotion_{stage}"
                break
        if any(store.safe_path(root).glob("health-*")):
            phase = "promotion_health"
            if live:
                live["live"] = False
    current = {"model": champion, "label": f"T{count}", "tier": count} if champion else None
    health_rows = health.get("by_board") or {}
    standard = get_training_standard(promotion.get("training_standard"))
    return {"source": "mixed", "run": {"id": run_id, "name": run_id}, "server_time": iso_time(),
            "training_standard": standard.to_dict() if standard else None,
            "evaluation_schedule": sample_evaluation_schedule(latest.get("trained_samples"), promotion.get("next_samples")),
            "status": runtime_status(store, path, latest, promotion, live and live["updated_at"], phase),
            "current": {"iteration": latest.get("cycle"), "completed_iteration": latest.get("cycle"),
                        "loss": latest.get("loss"), "replay_samples": latest.get("data_rows")},
            "mixed": {"boards": boards, "samples": latest.get("trained_samples"), "model": latest.get("model"),
                      "health_by_board": {str(board): {key: health_rows.get(str(board), {}).get(key)
                                                       for key in HEALTH_FIELDS} for board in boards},
                      "device": (manifest.get("initial_config", {}).get("hardware") or {}).get("device")},
            "diagnostics": diagnostics, "evaluations": rounds,
            "championship": {"stage": "mixed", "boards": boards, "current": current, "rounds": rounds,
                             "promotions": count, "next_cycle": promotion.get("next_cycle"),
                             "latest_model": latest.get("model"), "pending": bool(pending), "phase": phase,
                             "live_match": live, "match": rounds[-1] if rounds else None}}
