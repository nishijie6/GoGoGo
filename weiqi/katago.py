"""KataGo analysis-engine integration for professional opponent tiers.

KataGo itself and its neural-network files are intentionally not bundled with
this project.  This module discovers user-supplied files, persists their paths,
starts the official JSON analysis engine lazily, and translates between the
local rules engine and KataGo's protocol.

This module runs the engine process; settings and play profiles live in
``katago_settings``, protocol translation in ``katago_protocol`` and per-tier
move choice in ``katago_moves``. All their public names are re-exported here.
"""

from __future__ import annotations

import json
import os
import queue
import random
import subprocess
import threading
import uuid
from collections import deque
from typing import Any, Optional

from .ai import AIMove
from .engine import GoGame
from .katago_moves import MoveSelection
from .katago_protocol import (
    GTP_COLUMNS,
    build_analysis_query,
    point_to_vertex,
    vertex_to_point,
)
from .katago_settings import (
    ALL_KATAGO_PROFILES,
    DEFAULT_ANALYSIS_CONFIG,
    HUMANSL_PASS_VISITS,
    HUMANSL_PROFILES,
    HUMANSL_PROFILE_NAMES,
    KATAGO_FOLDER,
    KATAGO_PROFILES,
    KataGoConfigurationError,
    KataGoEngineError,
    KataGoError,
    KataGoProfile,
    KataGoSettings,
    LEGACY_KATAGO_FOLDER,
    PROJECT_ROOT,
    katago_subprocess_environment,
    profile_for_difficulty,
    settings_file_path,
)


class KataGoEngine(MoveSelection):
    """A lazy, persistent KataGo JSON analysis process."""

    def __init__(
        self,
        settings: KataGoSettings,
        timeout_seconds: float = 600.0,
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

    def _ensure_started(
        self,
        cancel_event: Optional[threading.Event] = None,
    ) -> subprocess.Popen[str]:
        with self._state_lock:
            if cancel_event is not None and cancel_event.is_set():
                raise KataGoEngineError("KataGo 分析已经取消")
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
                    env=katago_subprocess_environment(),
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

    def choose_move(
        self,
        game: GoGame,
        profile: KataGoProfile,
        cancel_event: Optional[threading.Event] = None,
    ) -> AIMove:
        """Analyze ``game`` and return a legal KataGo move without mutating it."""

        if game.game_over:
            return AIMove(None, "对局已经结束")
        with self._query_lock:
            request_id = uuid.uuid4().hex
            human_style = self.settings.human_style_enabled
            query_data = build_analysis_query(
                game,
                profile,
                request_id,
                human_style,
            )
            response = self._send_analysis_query(
                request_id,
                query_data,
                cancel_event=cancel_event,
            )
            return self._decision_from_response(
                game,
                profile,
                response,
                human_style,
            )

    def analyze_position(
        self,
        game: GoGame,
        max_visits: int = 400,
        pv_length: int = 12,
        include_ownership: bool = True,
        cancel_event: Optional[threading.Event] = None,
        include_policy: bool = False,
        preserve_history: bool = False,
    ) -> dict[str, Any]:
        """Return KataGo's raw analysis response for an unchanged game snapshot."""

        with self._query_lock:
            request_id = uuid.uuid4().hex
            query_data = build_analysis_query(
                game,
                KATAGO_PROFILES[-1],
                request_id,
                False,
                max_visits=max_visits,
                pv_length=pv_length,
                include_ownership=include_ownership,
            )
            if include_policy:
                query_data["includePolicy"] = True
            if preserve_history:
                query_data.setdefault("overrideSettings", {})["ignorePreRootHistory"] = False
            return self._send_analysis_query(
                request_id,
                query_data,
                cancel_event=cancel_event,
            )

    def _send_analysis_query(
        self,
        request_id: str,
        query_data: dict[str, Any],
        cancel_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        """Send one serialized query and wait for its matching final response."""

        process = self._ensure_started(cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            self.stop()
            raise KataGoEngineError("KataGo 分析已经取消")
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
        return self._wait_for_response(request_id)

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

    def _failure_detail(self, prefix: str) -> str:
        if not self._stderr_lines:
            return prefix
        return f"{prefix}：{self._stderr_lines[-1]}"

    def close(self) -> None:
        """Terminate the child process, including an in-flight query."""

        self._stop_process(mark_closed=True, wait=True)

    def stop(self) -> None:
        """Cancel current work without blocking Tk while allowing a restart."""

        self._stop_process(mark_closed=False, wait=False)

    def _stop_process(self, mark_closed: bool, wait: bool) -> None:
        with self._state_lock:
            if mark_closed:
                self._closed = True
            process = self._process
            self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            # The process may already be exiting, or terminate may have been
            # denied transiently.  The reaper still owns the detached handle
            # and will fall back to kill instead of losing track of it.
            pass
        if wait:
            self._reap_process(process)
        else:
            threading.Thread(
                target=self._reap_process,
                args=(process,),
                name="katago-reaper",
                daemon=True,
            ).start()

    @staticmethod
    def _reap_process(process: subprocess.Popen[str]) -> None:
        """Wait for a terminated engine off the Tk thread and kill if needed."""

        try:
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=2.0)
            except OSError:
                pass
            except subprocess.TimeoutExpired:
                pass


class KataGoAI:
    """AI adapter exposing the same ``choose_move`` contract as ``GoAI``."""

    def __init__(self, engine: KataGoEngine, difficulty: str) -> None:
        self.engine = engine
        self.difficulty = difficulty
        self.profile = profile_for_difficulty(difficulty)
        if (
            self.profile.selection_mode == "human_rank"
            and not engine.settings.human_style_enabled
        ):
            raise KataGoConfigurationError(
                "HumanSL 人类段位需要先配置人类风格模型"
            )

    def choose_move(
        self,
        game: GoGame,
        cancel_event: Optional[threading.Event] = None,
    ) -> AIMove:
        return self.engine.choose_move(
            game,
            self.profile,
            cancel_event=cancel_event,
        )


__all__ = [
    "ALL_KATAGO_PROFILES",
    "DEFAULT_ANALYSIS_CONFIG",
    "GTP_COLUMNS",
    "HUMANSL_PASS_VISITS",
    "HUMANSL_PROFILES",
    "HUMANSL_PROFILE_NAMES",
    "KATAGO_FOLDER",
    "KATAGO_PROFILES",
    "KataGoAI",
    "KataGoConfigurationError",
    "KataGoEngine",
    "KataGoEngineError",
    "KataGoError",
    "KataGoProfile",
    "KataGoSettings",
    "LEGACY_KATAGO_FOLDER",
    "PROJECT_ROOT",
    "build_analysis_query",
    "katago_subprocess_environment",
    "point_to_vertex",
    "profile_for_difficulty",
    "settings_file_path",
    "vertex_to_point",
]
