"""Prepare fixed Go positions and label them with the local Windows KataGo."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from weiqi.katago import (
    KataGoEngine, KataGoError, KataGoSettings, DEFAULT_ANALYSIS_CONFIG, vertex_to_point,
)
from weiqi.rl.eval_positions import (
    TEACHER_VERSION, atomic_json, canonical_hash, file_hash, generate_suite,
    read_suite, read_teacher, replay_position,
)


def _action(vertex: str, size: int) -> int:
    point = vertex_to_point(vertex, size)
    return size * size if point is None else point[0] * size + point[1]


def _label_position(game, response: dict) -> dict:
    if not isinstance(response.get("rootInfo"), dict):
        raise ValueError("KataGo response lacks rootInfo")
    expected = "B" if game.current_player == 1 else "W"
    if response["rootInfo"].get("currentPlayer") != expected:
        raise ValueError("KataGo analyzed the wrong player")
    raw = response.get("policy")
    if not isinstance(raw, list) or len(raw) != game.size * game.size + 1:
        raise ValueError("KataGo did not return a complete policy")
    legal = {row * game.size + col for row, col in game.legal_moves()}
    legal.add(game.size * game.size)
    policy = []
    for action, value in enumerate(raw):
        number = float(value)
        if not math.isfinite(number) or (action in legal and number < 0):
            raise ValueError(f"KataGo policy conflicts with the local rules at {action}")
        if action not in legal and number >= 0:
            raise ValueError(f"KataGo permitted an illegal local action at {action}")
        policy.append(number)
    if not math.isclose(sum(value for value in policy if value >= 0), 1.0, abs_tol=1e-4):
        raise ValueError("KataGo policy does not sum to one")
    moves = {}
    top_action = None
    for info in response.get("moveInfos", []):
        action = _action(str(info["move"]), game.size)
        if action not in legal:
            raise ValueError("KataGo returned a locally illegal move")
        score = info.get("scoreLead")
        if score is not None and math.isfinite(float(score)):
            moves[str(action)] = float(score)  # analysis config reports BLACK's lead
        if info.get("order") == 0:
            top_action = action
    if top_action is None:
        raise ValueError("KataGo did not return a preferred move")
    return {"policy": policy, "top_action": top_action,
            "score_lead_black_by_action": moves,
            "visits": int(response["rootInfo"].get("visits", 0))}


def label_suite(suite_path: Path, output: Path, visits: int) -> dict:
    if visits < 1:
        raise ValueError("--visits must be positive")
    suite = read_suite(suite_path)
    settings = replace(KataGoSettings.load(), human_model="")
    settings.require_valid()
    executable, model = settings.resolved_executable(), settings.resolved_model()
    assert executable is not None and model is not None
    config = DEFAULT_ANALYSIS_CONFIG.read_text(encoding="utf-8")
    if "reportAnalysisWinratesAs = BLACK" not in config:
        raise ValueError("KataGo scoreLead needs a fixed BLACK perspective")
    expected = {"version": TEACHER_VERSION, "suite_sha256": canonical_hash(suite),
                "engine_sha256": file_hash(executable), "model_sha256": file_hash(model),
                "analysis_config_sha256": file_hash(DEFAULT_ANALYSIS_CONFIG),
                "visits": visits}
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing KataGo labels use another suite, engine, model or visit budget")
        if payload.get("complete"):
            return payload
    else:
        payload = {**expected, "complete": False, "positions": {}}
        atomic_json(payload, output)
    engine = KataGoEngine(settings)
    try:
        for index, record in enumerate(suite["positions"], 1):
            if record["id"] in payload["positions"]:
                continue
            game = replay_position(record, size=9, komi=suite["komi"])
            response = engine.analyze_position(game, max_visits=visits, pv_length=1,
                                               include_ownership=False, include_policy=True,
                                               preserve_history=True)
            payload["positions"][record["id"]] = _label_position(game, response)
            atomic_json(payload, output)
            if index % 8 == 0 or index == len(suite["positions"]):
                print(json.dumps({"labeled": index, "total": len(suite["positions"])}), flush=True)
    finally:
        engine.close()
    payload["complete"] = True
    atomic_json(payload, output)
    return payload


def score_candidate_actions(suite_path: Path, teacher_path: Path, quality_path: Path,
                            cache_path: Path, output: Path, visits: int) -> dict:
    """Score actions omitted from moveInfos by analyzing their actual child positions.

    This is intentionally an on-demand, cached operation. The cheap position
    report remains available without running KataGo after every training turn.
    """
    if visits < 1:
        raise ValueError("--visits must be positive")
    suite = read_suite(suite_path)
    teacher = read_teacher(teacher_path, suite)
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("suite_sha256") != canonical_hash(suite):
        raise ValueError("Candidate position report belongs to another fixed suite")
    if len(quality.get("positions", [])) != len(suite["positions"]):
        raise ValueError("Candidate report has an incomplete fixed position set")
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (existing.get("candidate_sha256") != quality.get("candidate_sha256")
                or existing.get("suite_sha256") != canonical_hash(suite)):
            raise ValueError("Existing score report belongs to another candidate or suite")
    actions = {record["id"]: record["action"] for record in quality["positions"]}
    if set(actions) != {record["id"] for record in suite["positions"]}:
        raise ValueError("Candidate report has missing or duplicated positions")
    settings = replace(KataGoSettings.load(), human_model="")
    settings.require_valid()
    executable, model = settings.resolved_executable(), settings.resolved_model()
    assert executable is not None and model is not None
    if (file_hash(model) != teacher["model_sha256"]
            or file_hash(executable) != teacher["engine_sha256"]
            or file_hash(DEFAULT_ANALYSIS_CONFIG) != teacher["analysis_config_sha256"]):
        raise ValueError("KataGo installation differs from the frozen teacher labels")
    specification = {"version": 1, "teacher_sha256": canonical_hash(teacher),
                     "suite_sha256": canonical_hash(suite), "visits": visits}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if any(cache.get(key) != value for key, value in specification.items()):
            raise ValueError("Score cache belongs to another teacher or visit budget")
    else:
        cache = {**specification, "actions": {}}
        atomic_json(cache, cache_path)
    engine = None
    try:
        for index, record in enumerate(suite["positions"], 1):
            action = actions[record["id"]]
            if type(action) is not int or not 0 <= action <= 81:
                raise ValueError("Candidate report contains an invalid action")
            key = f"{record['id']}:{action}"
            if str(action) in teacher["positions"][record["id"]]["score_lead_black_by_action"]:
                continue
            if key in cache["actions"]:
                continue
            game = replay_position(record, size=9, komi=suite["komi"])
            if action == 81:
                if not game.pass_turn():
                    raise ValueError("Candidate attempted to pass after terminal play")
            else:
                outcome = game.play(*divmod(action, 9))
                if not outcome.legal:
                    raise ValueError("Candidate report contains an illegal move")
            if game.game_over:
                score = game.calculate_score()
                estimated = float(score.black_total - score.white_total)
                source = "terminal_area_score"
            else:
                if engine is None:
                    engine = KataGoEngine(settings)
                response = engine.analyze_position(game, max_visits=visits, pv_length=1,
                                                   include_ownership=False, preserve_history=True)
                info = response.get("rootInfo")
                lead = info.get("scoreLead") if isinstance(info, dict) else None
                if not isinstance(lead, (int, float)) or not math.isfinite(float(lead)):
                    raise ValueError("KataGo did not evaluate the candidate's resulting position")
                estimated = float(lead)
                source = "post_move_root"
            cache["actions"][key] = {"score_lead_black": estimated, "source": source}
            atomic_json(cache, cache_path)
            if index % 8 == 0 or index == len(suite["positions"]):
                print(json.dumps({"scored": index, "total": len(suite["positions"]),
                                  "cache_entries": len(cache["actions"])}), flush=True)
    finally:
        if engine is not None:
            engine.close()
    rows = []
    for record in suite["positions"]:
        action = actions[record["id"]]
        label = teacher["positions"][record["id"]]
        reference = label["score_lead_black_by_action"].get(str(label["top_action"]))
        if reference is None:
            raise ValueError("KataGo's preferred move lacks an individual score")
        direct = label["score_lead_black_by_action"].get(str(action))
        if direct is not None:
            chosen, source = direct, "same_root_move_info"
        else:
            measured = cache["actions"][f"{record['id']}:{action}"]
            chosen, source = measured["score_lead_black"], measured["source"]
        delta = (reference - chosen) * (1 if record["to_play"] == 1 else -1)
        rows.append({"id": record["id"], "phase": record["phase"], "action": action,
                     "reference_delta_points": delta, "candidate_score_source": source})
    groups = {}
    for phase in sorted({row["phase"] for row in rows} | {"all"}):
        subset = rows if phase == "all" else [row for row in rows if row["phase"] == phase]
        groups[phase] = {"positions": len(subset),
                         "mean_reference_delta_points": sum(row["reference_delta_points"] for row in subset) / len(subset),
                         "post_move_queries": sum(row["candidate_score_source"] == "post_move_root" for row in subset)}
    report = {**specification, "candidate_sha256": quality["candidate_sha256"],
              "positions": rows, "groups": groups,
              "interpretation": "Signed point difference from KataGo's preferred move. Post-move queries use the full history and the same BLACK score perspective; search estimates can be noisy."}
    atomic_json(report, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Local fixed 9x9 evaluation positions and KataGo labels")
    commands = parser.add_subparsers(dest="action", required=True)
    positions = commands.add_parser("positions")
    positions.add_argument("--output", type=Path, default=Path("config/rl_eval_positions_9x9.json"))
    positions.add_argument("--count", type=int, default=96)
    positions.add_argument("--seed", type=int, default=20260923)
    labels = commands.add_parser("label")
    labels.add_argument("--positions", type=Path, default=Path("config/rl_eval_positions_9x9.json"))
    labels.add_argument("--output", type=Path, default=Path("config/rl_eval_teacher_9x9.json"))
    labels.add_argument("--visits", type=int, default=64)
    scores = commands.add_parser("score")
    scores.add_argument("--positions", type=Path, default=Path("config/rl_eval_positions_9x9.json"))
    scores.add_argument("--teacher", type=Path, default=Path("config/rl_eval_teacher_9x9.json"))
    scores.add_argument("--quality", type=Path, required=True)
    scores.add_argument("--cache", type=Path, default=Path("training_runs/eval_katago_score_cache.json"))
    scores.add_argument("--output", type=Path, required=True)
    scores.add_argument("--visits", type=int, default=64)
    arguments = parser.parse_args()
    try:
        if arguments.action == "positions":
            if arguments.output.exists():
                raise ValueError("Position suite exists; choose a new output to preserve fixed comparisons")
            suite = generate_suite(count=arguments.count, seed=arguments.seed)
            atomic_json(suite, arguments.output)
            print(json.dumps({"positions": len(suite["positions"]),
                              "suite_sha256": canonical_hash(suite), "path": str(arguments.output)}))
        elif arguments.action == "label":
            payload = label_suite(arguments.positions, arguments.output, arguments.visits)
            print(json.dumps({"complete": payload["complete"], "labeled": len(payload["positions"]),
                              "path": str(arguments.output)}))
        else:
            report = score_candidate_actions(
                arguments.positions, arguments.teacher, arguments.quality,
                arguments.cache, arguments.output, arguments.visits)
            print(json.dumps({"scored": len(report["positions"]), "groups": report["groups"],
                              "path": str(arguments.output)}))
    except (ValueError, OSError, KeyError, KataGoError) as error:
        parser.exit(1, f"Benchmark error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Rerun the same command to reuse completed KataGo labels.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
