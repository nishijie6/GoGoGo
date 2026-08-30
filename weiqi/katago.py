"""KataGo analysis-engine integration for professional opponent tiers.

KataGo itself and its neural-network files are intentionally not bundled with
this project.  This module discovers user-supplied files, persists their paths,
starts the official JSON analysis engine lazily, and translates between the
local rules engine and KataGo's protocol.
"""

from __future__ import annotations

import json
import os
import queue
import random
import shutil
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .ai import AIMove, KATAGO_DIFFICULTIES
from .engine import BLACK, WHITE, GoGame, MoveRecord, Point


GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KATAGO_FOLDER = PROJECT_ROOT / "katago"
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "config" / "katago_analysis.cfg"


class KataGoError(RuntimeError):
    """Base exception for user-facing KataGo failures."""


class KataGoConfigurationError(KataGoError):
    """Raised when required KataGo files have not been configured."""


class KataGoEngineError(KataGoError):
    """Raised when the KataGo process cannot answer a request."""


@dataclass(frozen=True)
class KataGoProfile:
    """One simulated professional tier.

    The human-style profile provides the rank imitation when the optional
    human SL model is installed.  ``max_visits`` also rises monotonically so
    that the normal network fallback becomes stronger from tier to tier.
    """

    label: str
    dan: int
    max_visits: int
    human_sl_profile: str


_VISITS_BY_DAN = (24, 36, 54, 80, 120, 180, 270, 400, 600)

KATAGO_PROFILES = tuple(
    KataGoProfile(
        label=label,
        dan=index + 1,
        max_visits=_VISITS_BY_DAN[index],
        human_sl_profile=f"rank_{index + 1}d",
    )
    for index, label in enumerate(KATAGO_DIFFICULTIES)
)


def profile_for_difficulty(label: str) -> KataGoProfile:
    """Return the KataGo search profile for a UI difficulty label."""

    for profile in KATAGO_PROFILES:
        if profile.label == label:
            return profile
    raise ValueError(f"未知 KataGo 难度：{label}")


def settings_file_path() -> Path:
    """Return the per-user settings path without placing it in the repository."""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    else:
        base = Path(
            os.environ.get(
                "XDG_CONFIG_HOME",
                str(Path.home() / ".config"),
            )
        )
    return base / "YijingGo" / "katago.json"


@dataclass(frozen=True)
class KataGoSettings:
    """Paths required to start KataGo's analysis engine."""

    executable: str = ""
    model: str = ""
    human_model: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "KataGoSettings":
        """Load saved settings, environment overrides, and local discovery."""

        settings_path = path or settings_file_path()
        saved: dict[str, str] = {}
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                saved = {
                    key: str(data.get(key, "")).strip()
                    for key in ("executable", "model", "human_model")
                }
        except (OSError, ValueError, TypeError):
            saved = {}

        discovered = cls._discover_local_files()
        return cls(
            executable=(
                os.environ.get("KATAGO_EXE", "").strip()
                or saved.get("executable", "")
                or discovered.executable
            ),
            model=(
                os.environ.get("KATAGO_MODEL", "").strip()
                or saved.get("model", "")
                or discovered.model
            ),
            human_model=(
                os.environ.get("KATAGO_HUMAN_MODEL", "").strip()
                or saved.get("human_model", "")
                or discovered.human_model
            ),
        )

    @classmethod
    def _discover_local_files(cls) -> "KataGoSettings":
        executable = ""
        for candidate in (
            KATAGO_FOLDER / "katago.exe",
            KATAGO_FOLDER / "katago",
        ):
            if candidate.is_file():
                executable = str(candidate)
                break
        if not executable:
            executable = shutil.which("katago") or ""

        human_candidates = sorted(
            path
            for path in KATAGO_FOLDER.glob("*.bin.gz")
            if "human" in path.name.lower()
        )
        main_candidates = sorted(
            path
            for path in KATAGO_FOLDER.glob("*.bin.gz")
            if "human" not in path.name.lower()
        )
        return cls(
            executable=executable,
            model=str(main_candidates[-1]) if main_candidates else "",
            human_model=str(human_candidates[-1]) if human_candidates else "",
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist paths in the current user's application-data directory."""

        settings_path = path or settings_file_path()
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = settings_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "executable": self.executable.strip(),
                    "model": self.model.strip(),
                    "human_model": self.human_model.strip(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(settings_path)
        return settings_path

    def resolved_executable(self) -> Optional[Path]:
        value = self.executable.strip()
        if not value:
            return None
        direct = Path(value).expanduser()
        if direct.is_file():
            return direct.resolve()
        located = shutil.which(value)
        return Path(located).resolve() if located else None

    def resolved_model(self) -> Optional[Path]:
        return self._resolved_file(self.model)

    def resolved_human_model(self) -> Optional[Path]:
        if not self.human_model.strip():
            return None
        return self._resolved_file(self.human_model)

    @staticmethod
    def _resolved_file(value: str) -> Optional[Path]:
        if not value.strip():
            return None
        path = Path(value.strip()).expanduser()
        return path.resolve() if path.is_file() else None

    def validation_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.resolved_executable() is None:
            errors.append("没有找到 KataGo 可执行文件")
        if self.resolved_model() is None:
            errors.append("没有找到 KataGo 主神经网络模型（*.bin.gz）")
        if self.human_model.strip() and self.resolved_human_model() is None:
            errors.append("已填写的人类风格模型路径无效")
        if not DEFAULT_ANALYSIS_CONFIG.is_file():
            errors.append("程序自带的 KataGo 分析配置文件缺失")
        return tuple(errors)

    def require_valid(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise KataGoConfigurationError("；".join(errors))

    @property
    def human_style_enabled(self) -> bool:
        return self.resolved_human_model() is not None

    @property
    def fingerprint(self) -> tuple[str, str, str]:
        executable = self.resolved_executable()
        model = self.resolved_model()
        human_model = self.resolved_human_model()
        return (
            str(executable or ""),
            str(model or ""),
            str(human_model or ""),
        )


def point_to_vertex(point: Point, board_size: int) -> str:
    """Convert the local top-left-based point to a GTP vertex."""

    row, col = point
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError("落点超出棋盘")
    return f"{GTP_COLUMNS[col]}{board_size - row}"


def vertex_to_point(vertex: str, board_size: int) -> Optional[Point]:
    """Convert a GTP vertex to a local point; ``pass`` becomes ``None``."""

    normalized = vertex.strip().upper()
    if normalized == "PASS":
        return None
    if len(normalized) < 2 or normalized[0] not in GTP_COLUMNS:
        raise ValueError(f"KataGo 返回了无效坐标：{vertex}")
    try:
        number = int(normalized[1:])
    except ValueError as error:
        raise ValueError(f"KataGo 返回了无效坐标：{vertex}") from error
    col = GTP_COLUMNS.index(normalized[0])
    row = board_size - number
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"KataGo 返回了棋盘外坐标：{vertex}")
    return row, col


def _move_to_protocol(move: MoveRecord, board_size: int) -> list[str]:
    color = "B" if move.color == BLACK else "W"
    if move.kind == "pass":
        return [color, "pass"]
    if move.kind != "play" or move.row is None or move.col is None:
        raise ValueError("认输记录不能发送给 KataGo 继续分析")
    return [color, point_to_vertex((move.row, move.col), board_size)]


def build_analysis_query(
    game: GoGame,
    profile: KataGoProfile,
    request_id: str,
    human_style: bool,
) -> dict[str, Any]:
    """Build one official KataGo JSON analysis request from a game snapshot."""

    query: dict[str, Any] = {
        "id": request_id,
        "moves": [_move_to_protocol(move, game.size) for move in game.moves],
        "rules": {
            "ko": "POSITIONAL",
            "scoring": "AREA",
            "tax": "NONE",
            "suicide": False,
            "hasButton": False,
            "whiteHandicapBonus": "N",
            "friendlyPassOk": False,
        },
        "komi": game.komi,
        "boardXSize": game.size,
        "boardYSize": game.size,
        "maxVisits": profile.max_visits,
        "analysisPVLen": 8,
    }
    if not game.moves:
        query["initialPlayer"] = "B" if game.current_player == BLACK else "W"
    if human_style:
        query["includePolicy"] = True
        query["overrideSettings"] = {
            "humanSLProfile": profile.human_sl_profile,
            "ignorePreRootHistory": False,
        }
    return query


class KataGoEngine:
    """A lazy, persistent KataGo JSON analysis process."""

    def __init__(
        self,
        settings: KataGoSettings,
        timeout_seconds: float = 300.0,
        seed: Optional[int] = None,
    ) -> None:
        settings.require_valid()
        self.settings = settings
        self.timeout_seconds = timeout_seconds
        self._random = random.Random(seed)
        self._process: Optional[subprocess.Popen[str]] = None
        self._responses: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=30)
        self._query_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_started(self) -> subprocess.Popen[str]:
        with self._state_lock:
            if self._closed:
                raise KataGoEngineError("KataGo 引擎已经关闭")
            if self.running:
                assert self._process is not None
                return self._process

            executable = self.settings.resolved_executable()
            model = self.settings.resolved_model()
            if executable is None or model is None:
                raise KataGoConfigurationError("KataGo 路径配置已经失效")

            command = [
                str(executable),
                "analysis",
                "-config",
                str(DEFAULT_ANALYSIS_CONFIG),
                "-model",
                str(model),
            ]
            human_model = self.settings.resolved_human_model()
            if human_model is not None:
                command.extend(["-human-model", str(human_model)])

            creation_flags = 0
            if os.name == "nt":
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            runtime_folder = settings_file_path().parent / "runtime"
            try:
                runtime_folder.mkdir(parents=True, exist_ok=True)
                process = subprocess.Popen(
                    command,
                    cwd=str(runtime_folder),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creation_flags,
                )
            except OSError as error:
                raise KataGoEngineError(f"无法启动 KataGo：{error}") from error

            response_queue: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue()
            self._responses = response_queue
            self._stderr_lines.clear()
            self._process = process
            threading.Thread(
                target=self._read_stdout,
                args=(process, response_queue),
                name="katago-stdout",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_stderr,
                args=(process,),
                name="katago-stderr",
                daemon=True,
            ).start()
            return process

    def _read_stdout(
        self,
        process: subprocess.Popen[str],
        response_queue: "queue.Queue[Optional[dict[str, Any]]]",
    ) -> None:
        stream = process.stdout
        if stream is None:
            response_queue.put(None)
            return
        try:
            for line in stream:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    response_queue.put(payload)
        finally:
            response_queue.put(None)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        stream = process.stderr
        if stream is None:
            return
        for line in stream:
            stripped = line.strip()
            if stripped:
                self._stderr_lines.append(stripped)

    def choose_move(self, game: GoGame, profile: KataGoProfile) -> AIMove:
        """Analyze ``game`` and return a legal KataGo move without mutating it."""

        if game.game_over:
            return AIMove(None, "对局已经结束")
        with self._query_lock:
            process = self._ensure_started()
            request_id = uuid.uuid4().hex
            human_style = self.settings.human_style_enabled
            query_data = build_analysis_query(
                game,
                profile,
                request_id,
                human_style,
            )
            if process.stdin is None:
                raise KataGoEngineError("KataGo 标准输入不可用")
            try:
                process.stdin.write(
                    json.dumps(query_data, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise KataGoEngineError(self._failure_detail("KataGo 连接已中断")) from error

            response = self._wait_for_response(request_id)
            return self._decision_from_response(
                game,
                profile,
                response,
                human_style,
            )

    def _wait_for_response(self, request_id: str) -> dict[str, Any]:
        warnings: list[str] = []
        while True:
            try:
                response = self._responses.get(timeout=self.timeout_seconds)
            except queue.Empty as error:
                self.close()
                raise KataGoEngineError(
                    "KataGo 计算超时；可先选择较低职业段位，或检查显卡后端配置"
                ) from error
            if response is None:
                raise KataGoEngineError(self._failure_detail("KataGo 进程意外退出"))
            if str(response.get("id", "")) != request_id:
                continue
            if "warning" in response:
                warnings.append(str(response["warning"]))
                continue
            if "error" in response:
                field = response.get("field")
                detail = str(response["error"])
                if field:
                    detail = f"{field}: {detail}"
                raise KataGoEngineError(f"KataGo 拒绝分析请求：{detail}")
            if response.get("isDuringSearch"):
                continue
            if response.get("noResults"):
                raise KataGoEngineError("KataGo 未返回分析结果")
            if warnings:
                response = dict(response)
                response["_warnings"] = warnings
            return response

    def _decision_from_response(
        self,
        game: GoGame,
        profile: KataGoProfile,
        response: dict[str, Any],
        human_style_requested: bool,
    ) -> AIMove:
        move_infos = response.get("moveInfos")
        if not isinstance(move_infos, list):
            move_infos = []
        move_infos = [info for info in move_infos if isinstance(info, dict)]
        move_infos.sort(key=lambda info: int(info.get("order", 999999)))

        top_vertex = str(move_infos[0].get("move", "pass")) if move_infos else "pass"
        human_policy = response.get("humanPolicy")
        used_human_style = (
            human_style_requested
            and isinstance(human_policy, list)
            and len(human_policy) == game.size * game.size + 1
        )

        if top_vertex.strip().upper() == "PASS":
            point: Optional[Point] = None
        elif used_human_style:
            point = self._sample_human_policy(game, human_policy)
            if point is None:
                point = self._first_legal_candidate(game, move_infos)
        else:
            point = self._first_legal_candidate(game, move_infos)

        if point is not None and not game.analyze_move(*point).legal:
            point = self._first_legal_candidate(game, move_infos)
        if point is None and top_vertex.strip().upper() != "PASS":
            # If every returned move conflicts with the local rules, do not
            # disguise the protocol mismatch as an intentional pass.
            raise KataGoEngineError("KataGo 没有返回与本程序规则一致的合法落点")

        root_info = response.get("rootInfo")
        if not isinstance(root_info, dict):
            root_info = {}
        selected_vertex = "pass" if point is None else point_to_vertex(point, game.size)
        selected_info = next(
            (
                info
                for info in move_infos
                if str(info.get("move", "")).upper() == selected_vertex.upper()
            ),
            None,
        )
        evaluation = selected_info if selected_info is not None else root_info
        black_win_probability = self._optional_probability(evaluation.get("winrate"))
        black_lead = self._optional_float(evaluation.get("scoreLead"))
        visits = int(root_info.get("visits", profile.max_visits) or 0)

        if point is None:
            reason = f"KataGo 判断当前应当虚手，完成约 {visits} 次搜索"
        elif used_human_style:
            reason = (
                f"KataGo 人类风格模型按 {profile.dan} 段棋谱分布选点，"
                f"主网络完成约 {visits} 次局面核验"
            )
        else:
            reason = (
                f"KataGo 主网络完成约 {visits} 次搜索并选择最高评价点；"
                "未配置人类风格模型"
            )
        return AIMove(
            point=point,
            explanation=reason,
            black_win_probability=black_win_probability,
            black_lead=black_lead,
            analysis_visits=visits,
        )

    def _sample_human_policy(
        self,
        game: GoGame,
        policy: list[Any],
    ) -> Optional[Point]:
        weighted_points: list[tuple[Point, float]] = []
        for index, raw_weight in enumerate(policy[:-1]):
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            point = (index // game.size, index % game.size)
            if game.analyze_move(*point).legal:
                weighted_points.append((point, weight))
        total = sum(weight for _, weight in weighted_points)
        if total <= 0:
            return None
        threshold = self._random.random() * total
        cumulative = 0.0
        for point, weight in weighted_points:
            cumulative += weight
            if cumulative >= threshold:
                return point
        return weighted_points[-1][0]

    @staticmethod
    def _first_legal_candidate(
        game: GoGame,
        move_infos: list[dict[str, Any]],
    ) -> Optional[Point]:
        for info in move_infos:
            vertex = str(info.get("move", ""))
            try:
                point = vertex_to_point(vertex, game.size)
            except ValueError:
                continue
            if point is None:
                return None
            if game.analyze_move(*point).legal:
                return point
        return None

    @staticmethod
    def _optional_probability(value: Any) -> Optional[float]:
        number = KataGoEngine._optional_float(value)
        if number is None:
            return None
        return max(0.0, min(1.0, number))

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _failure_detail(self, prefix: str) -> str:
        if not self._stderr_lines:
            return prefix
        return f"{prefix}：{self._stderr_lines[-1]}"

    def close(self) -> None:
        """Terminate the child process, including an in-flight query."""

        self._stop_process(mark_closed=True)

    def stop(self) -> None:
        """Cancel current work while allowing a lazy restart later."""

        self._stop_process(mark_closed=False)

    def _stop_process(self, mark_closed: bool) -> None:
        with self._state_lock:
            if mark_closed:
                self._closed = True
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass


class KataGoAI:
    """AI adapter exposing the same ``choose_move`` contract as ``GoAI``."""

    def __init__(self, engine: KataGoEngine, difficulty: str) -> None:
        self.engine = engine
        self.difficulty = difficulty
        self.profile = profile_for_difficulty(difficulty)

    def choose_move(self, game: GoGame) -> AIMove:
        return self.engine.choose_move(game, self.profile)
