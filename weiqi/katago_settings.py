"""Locating and configuring KataGo: paths, environment, errors, play profiles, settings.

KataGo itself and its neural-network files are not bundled with this project;
this module discovers user-supplied files and persists their paths.
"""

from __future__ import annotations

import json
import os
import shutil
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .ai import HUMANSL_DIFFICULTIES, KATAGO_DIFFICULTIES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KATAGO_FOLDER = PROJECT_ROOT / "katago"
LEGACY_KATAGO_FOLDER = PROJECT_ROOT / "vendor" / "katago"
DEFAULT_ANALYSIS_CONFIG = PROJECT_ROOT / "config" / "katago_analysis.cfg"


def katago_subprocess_environment(
    purelib: Optional[Path] = None,
    platform_name: Optional[str] = None,
) -> dict[str, str]:
    """Expose PyTorch's bundled CUDA DLLs to KataGo on Windows."""

    environment = os.environ.copy()
    if (platform_name or os.name) != "nt":
        return environment
    package_folder = purelib or Path(sysconfig.get_path("purelib"))
    torch_libraries = package_folder / "torch" / "lib"
    if torch_libraries.is_dir():
        current_path = environment.get("PATH", "")
        environment["PATH"] = (
            str(torch_libraries)
            if not current_path
            else str(torch_libraries) + os.pathsep + current_path
        )
    return environment


class KataGoError(RuntimeError):
    """Base exception for user-facing KataGo failures."""


class KataGoConfigurationError(KataGoError):
    """Raised when required KataGo files have not been configured."""


class KataGoEngineError(KataGoError):
    """Raised when the KataGo process cannot answer a request."""


@dataclass(frozen=True)
class KataGoProfile:
    """One professional-strength or HumanSL rank simulation profile.

    Professional tiers blend modern professional-game priors with searched
    utility. Human-rank tiers sample the selected HumanSL policy directly for
    non-pass moves while retaining a small normal search to decide when to pass.
    """

    label: str
    dan: Optional[int]
    max_visits: int
    human_sl_profile: str
    move_temperature: float
    utility_scale: float
    selection_mode: Literal["professional", "human_rank", "training"] = "professional"


_VISITS_BY_DAN = (24, 36, 54, 80, 120, 180, 270, 400, 600)
_TEMPERATURE_BY_DAN = (1.00, 0.90, 0.80, 0.71, 0.63, 0.55, 0.48, 0.41, 0.35)
_UTILITY_SCALE_BY_DAN = (0.75, 0.66, 0.58, 0.50, 0.43, 0.36, 0.30, 0.25, 0.20)

KATAGO_PROFILES = tuple(
    KataGoProfile(
        label=label,
        dan=index + 1,
        max_visits=_VISITS_BY_DAN[index],
        human_sl_profile="proyear_2023",
        move_temperature=_TEMPERATURE_BY_DAN[index],
        utility_scale=_UTILITY_SCALE_BY_DAN[index],
    )
    for index, label in enumerate(KATAGO_DIFFICULTIES)
)

HUMANSL_PROFILE_NAMES = (
    *(f"rank_{kyu}k" for kyu in range(20, 0, -1)),
    *(f"rank_{dan}d" for dan in range(1, 10)),
)
HUMANSL_PASS_VISITS = 64
HUMANSL_PROFILES = tuple(
    KataGoProfile(
        label=label,
        dan=None,
        max_visits=HUMANSL_PASS_VISITS,
        human_sl_profile=profile_name,
        move_temperature=1.0,
        utility_scale=1.0,
        selection_mode="human_rank",
    )
    for label, profile_name in zip(
        HUMANSL_DIFFICULTIES,
        HUMANSL_PROFILE_NAMES,
    )
)
ALL_KATAGO_PROFILES = KATAGO_PROFILES + HUMANSL_PROFILES


def profile_for_difficulty(label: str) -> KataGoProfile:
    """Return the KataGo search profile for a UI difficulty label."""

    if label.startswith("训练·"):
        return KataGoProfile(label, None, 200, "", 0.0, 0.0, "training")
    for profile in ALL_KATAGO_PROFILES:
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
                or os.environ.get("WEIQI_KATAGO_EXE", "").strip()
                or saved.get("executable", "")
                or discovered.executable
            ),
            model=(
                os.environ.get("KATAGO_MODEL", "").strip()
                or os.environ.get("WEIQI_KATAGO_MODEL", "").strip()
                or saved.get("model", "")
                or discovered.model
            ),
            human_model=(
                os.environ.get("KATAGO_HUMAN_MODEL", "").strip()
                or os.environ.get("WEIQI_KATAGO_HUMAN_MODEL", "").strip()
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
            LEGACY_KATAGO_FOLDER / "runtime" / "katago.exe",
            LEGACY_KATAGO_FOLDER / "runtime" / "katago",
        ):
            if candidate.is_file():
                executable = str(candidate)
                break
        if not executable:
            executable = shutil.which("katago") or ""

        model_folders = (
            KATAGO_FOLDER,
            LEGACY_KATAGO_FOLDER / "models",
        )

        def newest_model(*, human: bool) -> str:
            for folder in model_folders:
                candidates = sorted(
                    path
                    for path in folder.glob("*.bin.gz")
                    if ("human" in path.name.lower()) is human
                )
                if candidates:
                    return str(candidates[-1])
            return ""

        return cls(
            executable=executable,
            model=newest_model(human=False),
            human_model=newest_model(human=True),
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


__all__ = [
    "ALL_KATAGO_PROFILES",
    "DEFAULT_ANALYSIS_CONFIG",
    "HUMANSL_PASS_VISITS",
    "HUMANSL_PROFILES",
    "HUMANSL_PROFILE_NAMES",
    "KATAGO_FOLDER",
    "KATAGO_PROFILES",
    "KataGoConfigurationError",
    "KataGoEngineError",
    "KataGoError",
    "KataGoProfile",
    "KataGoSettings",
    "LEGACY_KATAGO_FOLDER",
    "PROJECT_ROOT",
    "katago_subprocess_environment",
    "profile_for_difficulty",
    "settings_file_path",
]
