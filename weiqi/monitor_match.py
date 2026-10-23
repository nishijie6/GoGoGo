"""Read provisional champion results from completed SGFs without touching training."""

from .rl_match import parse_match_game
from .sgf import split_sgf_collection
from .training_monitor import MAX_JSON, iso_time


def _text(store, path):
    path = store.safe_path(path)
    if path.stat().st_size > MAX_JSON:
        raise ValueError("评测文件过大")
    return path.read_text(encoding="utf-8")


def _games(store, path):
    path = store.safe_path(path)
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = store.match_cache.get(path)
    if cached and cached[0] == signature:
        store.match_cache.move_to_end(path)
        return cached[1]
    trees, trailing = split_sgf_collection(_text(store, path))
    games, invalid = [], 0
    for tree in trees:
        game = parse_match_game(tree)
        black, white = (game.black or "").casefold(), (game.white or "").casefold()
        if game.damaged or game.winner is None or {black, white} != {"candidate", "baseline"}:
            invalid += 1
            continue
        color = "B" if black == "candidate" else "W"
        outcome = "draws" if game.winner == "D" else "wins" if game.winner == color else "losses"
        games.append((color, outcome))
    value = (games, invalid, trailing)
    store.match_cache[path] = (signature, value)
    if len(store.match_cache) > 1024:
        store.match_cache.popitem(last=False)
    return value


def _attempt(store, path, expected_config):
    config = store.safe_path(path / "match.cfg")
    if _text(store, config) != expected_config:
        return None
    games, invalid, trailing = [], 0, 0
    events = []
    modified = config.stat().st_mtime
    for sgf in sorted((path / "sgfs").glob("*.sgfs")):
        results, damaged, partial = _games(store, sgf)
        games.extend(results)
        events.extend((sgf.stat().st_mtime_ns, str(sgf), index, result)
                      for index, (_, result) in enumerate(results))
        invalid += damaged
        trailing += partial
        modified = max(modified, sgf.stat().st_mtime)
    # A pair cannot contain repeated colors or more than two games. Invalid
    # attempts remain uncounted rather than inflating the provisional rate.
    if invalid or len(games) > 2 or len({color for color, _ in games}) != len(games):
        return None
    return {"games": games, "events": events, "complete": len(games) == 2 and not trailing,
            "modified": modified, "started": config.stat().st_mtime_ns}


def _pair(store, pair):
    expected = _text(store, pair / "match.cfg")
    attempts = [pair, *sorted(pair.glob("retry-*"))]
    partials = []
    for path in attempts:
        try:
            attempt = _attempt(store, path, expected)
            if attempt is None:
                continue
            if attempt["complete"]:
                # Same ordering as the official runner: reuse the first valid
                # complete attempt, never sum multiple retries of one opening.
                return attempt
            partials.append(attempt)
        except (OSError, ValueError, UnicodeError):
            continue
    return max(partials, key=lambda item: item["started"]) if partials else None


def live_champion_match(store, path, state, stage):
    candidate, champion = stage.get("latest_model"), stage.get("champion_model")
    if state.get("phase") != "evaluating" or not candidate or not champion or candidate == champion:
        return None
    return read_champion_match(store, path, candidate=candidate, champion=champion,
                               board=state.get("active_board_size"), samples=stage.get("samples"))


def read_champion_match(store, path, *, candidate, champion, board, samples):
    """Rebuild cumulative rates in SGF save order, including a finished round."""
    parent = store.safe_path(path / "evaluations" / f"{board}x{board}")
    for evaluation in sorted(parent.glob(f"s{samples}-{candidate}-*"), reverse=True):
        for opponent in sorted(evaluation.glob("opponent-*")):
            try:
                manifest = store.read_json(opponent / "evaluation_manifest.json")
            except (OSError, ValueError):
                continue
            roles = manifest.get("roles")
            if (manifest.get("candidate_model") != candidate or manifest.get("baseline_model") != champion
                    or manifest.get("board_size") != board or not isinstance(roles, list) or "champion" not in roles):
                continue
            total = manifest.get("games")
            if isinstance(total, bool) or not isinstance(total, int) or total < 2 or total > 10000 or total % 2:
                continue
            counts = {"wins": 0, "losses": 0, "draws": 0}
            events = []
            completed_pairs = 0
            modified = None
            for index in range(total // 2):
                try:
                    pair = _pair(store, opponent / "pairs" / f"opening-{index:03d}")
                except (OSError, ValueError, UnicodeError):
                    continue
                if pair is not None:
                    for _, result in pair["games"]:
                        counts[result] += 1
                    events.extend(pair["events"])
                    completed_pairs += int(pair["complete"])
                    modified = max(modified or 0, pair["modified"])
            games = sum(counts.values())
            cumulative = {"wins": 0, "losses": 0, "draws": 0}
            points = []
            for index, (_, _, _, result) in enumerate(sorted(events), 1):
                cumulative[result] += 1
                points.append({"completed": index, **cumulative,
                               "win_rate": cumulative["wins"] / index,
                               "score_rate": (cumulative["wins"] + cumulative["draws"] / 2) / index})
            return {"live": True, "candidate": candidate, "name": champion,
                    "games": games, "total": total, **counts, "completed_pairs": completed_pairs,
                    "win_rate": counts["wins"] / games if games else None,
                    "score_rate": (counts["wins"] + counts["draws"] / 2) / games if games else None,
                    "points": points,
                    "updated_at": iso_time(modified) if modified else None}
    return None
