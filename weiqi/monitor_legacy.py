"""Read older KataGo dashboards without importing their training runtime."""

from __future__ import annotations

import time

from .training_monitor import iso_time, timestamp


def champion_lineage(stage):
    """Derive per-board T numbers from actual recorded champion replacements."""
    records = stage.get("evaluations", [])
    initial = next((record.get("champion_before") for record in records if record.get("champion_before")), None)
    if initial is None:
        initial = next((opponent.get("model") for record in records
                        for opponent in (record.get("opponents") or [])
                        if "champion" in opponent.get("roles", [])), None)
    initial = initial or stage.get("entry_model") or stage.get("champion_model")
    champions = [{"tier": 0, "label": "T0", "model": initial}] if initial else []
    versions = {initial: 0} if initial else {}
    tiers = []
    seen = set()
    for record in records:
        tiers.append(dict(versions))
        before = record.get("champion_before")
        after = record.get("champion_after") or record.get("candidate_model")
        if record.get("champion_promoted") and after and after != before and after not in seen and after not in versions:
            tier = len(champions)
            versions[after] = tier
            champions.append({"tier": tier, "label": f"T{tier}", "model": after})
            seen.add(after)
    return champions, tiers


def snapshot(store, path, run_id):
    state = store.read_json(path / "state.json")
    stages = state.get("stages", {})
    active = str(state.get("active_board_size", ""))
    stage = stages.get(f"{active}x{active}", {})
    records = store.read_history(path / "metrics" / "history.jsonl")
    if stage and not records:
        records = store.read_history(store.root / f"{active}x{active}" / "metrics" / "history.jsonl")
    latest = records[-1] if records else {}
    sample_cycles = {row.get("trained_samples"): row.get("cycle") for row in records
                     if row.get("trained_samples") is not None}
    diagnostics = [{"iteration": row.get("cycle"), "samples": row.get("trained_samples"),
                    "loss": row.get("loss"), "policy_loss": row.get("policy_loss"),
                    "value_loss": row.get("value_loss"),
                    "samples_per_second": row["sample_delta"] / row["train_seconds"]
                    if row.get("train_seconds", 0) > 0 and row.get("sample_delta") is not None else None}
                   for row in records]
    evaluations = []
    lineages = {}
    for stage_name, item in stages.items():
        champions, versions = champion_lineage(item)
        lineages[stage_name] = champions
        for index, record in enumerate(item.get("evaluations", [])):
            opponents = record.get("opponents")
            if opponents is None and record.get("match"):
                opponents = [{"model": record.get("baseline_model", "未知旧基准"),
                              "roles": ["fixed"], "match": record["match"]}]
            for opponent in opponents or []:
                match = opponent.get("match", {})
                games, wins, draws = match.get("games"), match.get("candidate_wins"), match.get("draws")
                champion_role = "champion" in opponent.get("roles", [])
                tier = versions[index].get(opponent.get("model")) if champion_role else None
                evaluations.append({"name": opponent.get("model"), "stage": stage_name,
                                    "role": "champion" if champion_role else "pool",
                                    "champion_tier": tier, "champion_label": f"T{tier}" if tier is not None else None,
                                    "evaluation_round": index + 1,
                                    "iteration": record.get("training_cycle") or (
                                        sample_cycles.get(record.get("stage_samples")) if stage_name == f"{active}x{active}" else None),
                                    "samples": record.get("stage_samples"), "timestamp": record.get("timestamp"),
                                    "candidate": record.get("candidate_model"), "games": games,
                                    "wins": wins, "losses": match.get("baseline_wins"), "draws": draws,
                                    "win_rate": wins / games if games and wins is not None else None,
                                    "score_rate": (wins + draws / 2) / games if games and wins is not None and draws is not None else None,
                                    "interval": {"method": "Wilson 得分区间（旧版）", "mean_score": match.get("win_rate"), "lower": match.get("wilson_lower"),
                                                 "upper": match.get("wilson_upper")},
                                    "elo": match.get("elo"), "promoted": champion_role and bool(record.get("champion_promoted"))})
    when = state.get("updated_at") or latest.get("timestamp")
    epoch = timestamp(when)
    health = {"black_score_rate": latest.get("health_black_win_rate"),
              "extreme_rate": latest.get("health_extreme_result_rate"),
              "invalid_rate": latest.get("health_invalid_game_rate"),
              "double_pass_rate": latest.get("health_opening_double_pass_rate"),
              "games": latest.get("health_window_games"),
              "passed": latest.get("health_passed"), "complete": latest.get("health_window_complete")}
    active_results = [row for row in evaluations if row["stage"] == f"{active}x{active}"]
    last_sample = max((row.get("samples") or 0 for row in active_results), default=0)
    recent = [row for row in active_results if (row.get("samples") or 0) == last_sample]
    recent_match = next((row for row in recent if row["role"] == "champion"), recent[0] if recent else None)
    champions = lineages.get(f"{active}x{active}", [])
    current_champion = next((row for row in champions if row["model"] == stage.get("champion_model")), None)
    champion_rounds = [row for row in active_results if row["role"] == "champion"]
    for index, row in enumerate(champion_rounds, 1):
        row["champion_round"] = index
    current_match = next((row for row in reversed(champion_rounds)
                          if row["candidate"] == stage.get("latest_model")
                          and row["name"] == stage.get("champion_model")), None)
    from .monitor_match import live_champion_match
    live_match = live_champion_match(store, path, state, stage) if stage else None
    return {"run": {"id": run_id, "name": run_id}, "server_time": iso_time(),
            "source": "legacy", "status": {"state": "historical", "phase": "unknown", "updated_at": when,
                "age_seconds": max(0, time.time() - epoch) if epoch else None,
                "reason": "旧版 KataGo 记录没有训练心跳，无法确认进程是否运行"},
            "current": {"iteration": latest.get("cycle"), "completed_iteration": latest.get("cycle"),
                        "loss": latest.get("loss"), "replay_samples": latest.get("data_rows")},
            "latest": {}, "history": [], "opponents": [], "events": [],
            "recent_evaluation": {**recent_match, "phase": "evaluation", "historical": True} if recent_match else None,
            "diagnostics": diagnostics, "evaluations": evaluations, "health": health,
            "championship": {"stage": f"{active}x{active}", "champions": champions,
                             "current": current_champion, "latest_model": stage.get("latest_model"),
                             "pending": stage.get("pending_champion_evaluation", False),
                             "match": current_match, "live_match": live_match,
                             "rounds": champion_rounds} if stage else None,
            "curriculum": {"board_size": state.get("active_board_size"), "samples": stage.get("samples"),
                           "champion": stage.get("champion_model"), "passes": stage.get("consecutive_passes"),
                           "next_evaluation_sample": stage.get("next_evaluation_sample"),
                           "disk_free_gib": state.get("disk", {}).get("free_gib")},
            "warnings": ["正在读取旧版历史记录；训练曲线最多显示最近 300 轮，文件较大时读取末尾 8 MB。页面刷新不代表训练进程存活。"]}
