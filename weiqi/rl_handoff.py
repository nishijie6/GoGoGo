"""Offline, verified transfer of stopped native or KataGo training runs.

Packages contain data, never executable setup scripts. Import publishes only a
new directory, and neither operation starts training or uploads any file.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from .fileio import acquire_file_lock, atomic_write_json, local_timestamp, release_file_lock, sha256_file
from .rl_config import load_rl_training_config, resolve_rl_training_config
from .rl_curriculum_config import load_curriculum_config, validate_training_profiles
from .rl_curriculum_state import CurriculumStateStore
from .rl_handoff_guard import HANDOFF_MARKER, IMPORT_PENDING_MARKER, HandoffLockedError, ensure_training_available


FORMAT_VERSION = 1
RECEIPT = "handoff_receipt.json"
_CHUNK = 1024 * 1024
_MANIFEST_LIMIT = 16 * 1024 * 1024
_SKIP_NAMES = {HANDOFF_MARKER, IMPORT_PENDING_MARKER, "training.lock", "curriculum.lock", "run.lock", "PAUSE", "STOP"}
_SKIP_DIRS = {"shufflescratch", "scratch", "__pycache__"}
_HARDWARE_FIELDS = {
    "hardware": None,
    "self_play": {"workers", "inference_batch_size"},
    "optimizer": {"batch_size"},
}


class HandoffError(ValueError):
    """An incomplete, busy, incompatible, or damaged handoff was rejected."""


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise HandoffError(f"无法读取有效 JSON：{path}") from error
    if not isinstance(value, dict):
        raise HandoffError(f"JSON 根节点必须是对象：{path}")
    return value


def _safe_name(name: object) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise HandoffError(f"接力包包含无效路径：{name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or pure.as_posix() != name:
        raise HandoffError(f"接力包包含非规范路径：{name}")
    for part in pure.parts:
        if part in (".", "..") or part.endswith((".", " ")) or any(c in part for c in '<>"|?*'):
            raise HandoffError(f"接力包包含越界或无效路径：{name}")
        if re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".")[0]):
            raise HandoffError(f"接力包包含系统保留路径：{name}")
    return name


def _resolve(project_root: Path, path: Path) -> Path:
    return (path if path.is_absolute() else project_root / path).resolve()


def _git_revision(root: Path) -> str | None:
    if not root.is_dir():
        return None
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if Path(top).resolve() != root.resolve():
            return None
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _code_metadata(project_root: Path) -> dict:
    paths = sorted((project_root / "weiqi").rglob("*.py"))
    paths += [project_root / "train.py", project_root / "train_rl.py", project_root / "requirements-training.txt"]
    # Keep the version-1 source hash unchanged. New archives additionally
    # protect the fixed evaluation assets; machine-specific profiles are
    # restored from the archive and deliberately excluded from this signature.
    assets = [project_root / "config" / "rl_eval_positions_9x9.json",
              project_root / "config" / "rl_eval_teacher_9x9.json"]
    return {"revision": _git_revision(project_root), "source_sha256": _content_sha256(project_root, paths),
            "behavior_assets_sha256": _content_sha256(project_root, assets),
            "katago_revision": _git_revision(project_root / "katago" / "source")}


def _content_sha256(project_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(project_root).as_posix().encode("utf-8") + b"\0")
            # Git autocrlf must not make the same source incompatible across OSes.
            digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def _profile_document(path: Path) -> dict:
    config = load_rl_training_config(path).to_dict()
    return {"schema_version": config["schema_version"], "preset": config["preset"],
            "overrides": {key: value for key, value in config.items()
                          if key not in {"schema_version", "preset"}}}


def _check_bundle_git_ignore(project_root: Path, destination: Path) -> None:
    """A model-bearing archive inside this checkout must stay unaddable by default."""
    if not destination.is_relative_to(project_root) or not (project_root / ".git").exists():
        return
    result = subprocess.run(["git", "-C", str(project_root), "check-ignore", "--quiet", "--",
                             destination.relative_to(project_root).as_posix()],
                            capture_output=True, timeout=10)
    if result.returncode != 0:
        raise HandoffError("接力包包含模型，必须保存在 Git 忽略路径：handoff_bundles/ 或 *.gogogo-handoff.zip")


def _source_plan(project_root: Path, curriculum_config: Path | None,
                 training_config: Path | None, native_run: Path | None = None) -> dict:
    if sum(path is not None for path in (curriculum_config, training_config, native_run)) != 1:
        raise HandoffError("请指定普通版运行目录，或 KataGo 的课程/单棋盘配置")
    if native_run is not None:
        from .rl_native_handoff import native_source_plan
        return native_source_plan(project_root, native_run)
    if training_config is not None:
        path = _resolve(project_root, training_config)
        config = load_rl_training_config(path)
        root = _resolve(project_root, Path(config.runtime.output_directory))
        return {"kind": "single", "trainer": "katago", "sources": [{"root": root, "prefix": "run",
                "board_size": config.game.board_size, "lock": "training.lock"}],
                "configs": {"config/training.json": _profile_document(path)},
                "active_prefix": "run"}
    path = _resolve(project_root, curriculum_config)
    config = load_curriculum_config(path)
    validate_training_profiles(config, project_root)
    state_root = config.state_root(project_root)
    state = CurriculumStateStore(state_root).load()
    if state is None:
        raise HandoffError("课程尚无 state.json；单个已有运行请使用 --single-run --config")
    if state.phase != "training" and not (state.phase == "paused_disk" and state.resume_phase == "training"):
        raise HandoffError(f"课程当前处于 {state.phase}；请在评测或阶段迁移完成后导出")
    sources = [{"root": state_root, "prefix": "curriculum", "lock": "curriculum.lock"}]
    documents = {"config/curriculum.json": _read_json(path)}
    for stage in config.stages:
        profile_path = stage.config_path(project_root)
        profile = load_rl_training_config(profile_path)
        root = _resolve(project_root, Path(profile.runtime.output_directory))
        documents[f"config/{stage.key}.json"] = _profile_document(profile_path)
        if root.is_dir():
            sources.append({"root": root, "prefix": f"stages/{stage.key}",
                            "board_size": stage.board_size, "lock": "training.lock"})
    return {"kind": "curriculum", "trainer": "katago", "sources": sources, "configs": documents,
            "state": state, "active_prefix": f"stages/{state.active.board_size}x{state.active.board_size}"}


def _check_sources(plan: dict) -> None:
    roots = [source["root"] for source in plan["sources"]]
    for index, root in enumerate(roots):
        if not root.is_dir():
            raise HandoffError(f"训练目录不存在：{root}")
        for other in roots[:index]:
            if root == other or root.is_relative_to(other) or other.is_relative_to(root):
                raise HandoffError("课程状态与各阶段目录必须相互独立，不能嵌套")
    active = next((source for source in plan["sources"] if source["prefix"] == plan["active_prefix"]), None)
    checkpoint = "latest.pt" if plan.get("trainer", "katago") == "native" else "train/gogogo/checkpoint.ckpt"
    if active is None or not (active["root"] / checkpoint).is_file():
        raise HandoffError(f"缺少完整续训检查点 {checkpoint}，不能只导出下棋模型")
    if plan["kind"] == "curriculum":
        for stage in plan["state"].stages.values():
            source = next((s for s in plan["sources"] if s.get("board_size") == stage.board_size), None)
            if source is None:
                raise HandoffError(f"课程已有 {stage.board_size}x{stage.board_size} 进度但对应目录缺失")
            references = [stage.entry_model, stage.baseline_model, stage.latest_model,
                          stage.champion_model, *stage.fixed_baseline_models]
            for name in filter(None, references):
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name):
                    raise HandoffError("课程模型引用包含无效名称")
                for relative in (f"models/{name}/model.bin.gz", f"torchmodels_toexport/{name}/model.ckpt"):
                    if not (source["root"] / relative).is_file():
                        raise HandoffError(f"受保护模型或其检查点缺失：{source['root'] / relative}")
        for name in plan["state"].protected_models:
            if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name) or not any(
                    (s["root"] / f"models/{name}/model.bin.gz").is_file()
                    and (s["root"] / f"torchmodels_toexport/{name}/model.ckpt").is_file()
                    for s in plan["sources"] if "board_size" in s):
                raise HandoffError(f"全局受保护模型或其检查点缺失：{name}")


@contextlib.contextmanager
def _source_locks(plan: dict, *, allow_suspended: bool = False):
    handles = []
    try:
        for source in plan["sources"]:
            root = source["root"]
            if not root.is_dir():
                raise HandoffError(f"训练目录不存在：{root}")
            handle = acquire_file_lock(root / source["lock"])
            handles.append(handle)
            if not allow_suspended:
                ensure_training_available(root)
        yield
    except HandoffLockedError as error:
        raise HandoffError(str(error)) from error
    except OSError as error:
        raise HandoffError(f"训练目录正在使用或无法锁定；请先停止训练：{error}") from error
    finally:
        for handle in reversed(handles):
            release_file_lock(handle)


def _inventory(plan: dict) -> list[tuple[str, Path, os.stat_result]]:
    files = []
    for source in plan["sources"]:
        root = source["root"]
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part.endswith(".tmp") or part in _SKIP_DIRS for part in relative.parts) or path.name in _SKIP_NAMES:
                continue
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise HandoffError(f"接力包不跟随符号链接或目录联接：{path}")
            if path.is_file():
                name = _safe_name(f"{source['prefix']}/{relative.as_posix()}")
                files.append((name, path, path.stat()))
    return files


def _record(name: str, size: int, digest: str, mtime_ns: int) -> dict:
    return {"path": name, "size": size, "sha256": digest, "mtime_ns": mtime_ns}


def export_handoff(project_root: Path, destination: Path, *, curriculum_config: Path | None = None,
                   training_config: Path | None = None, native_run: Path | None = None) -> dict:
    """Freeze a stopped run into an immutable ZIP, then suspend its source."""
    if destination.is_symlink():
        raise HandoffError("接力包目标不能是符号链接")
    project_root, destination = project_root.resolve(), destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise HandoffError(f"接力包已存在，拒绝覆盖：{destination}")
    _check_bundle_git_ignore(project_root, destination)
    plan = _source_plan(project_root, curriculum_config, training_config, native_run)
    _check_sources(plan)
    if any(destination.is_relative_to(source["root"]) for source in plan["sources"]):
        raise HandoffError("接力包必须写到训练目录之外，例如 handoff_bundles/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle_id = uuid.uuid4().hex
    temp = destination.with_name(f".{destination.name}.{bundle_id}.tmp")
    markers = []
    with _source_locks(plan):
        # State may have changed while the exporter was waiting for its locks.
        locked_plan = _source_plan(project_root, curriculum_config, training_config, native_run)
        if locked_plan["sources"] != plan["sources"]:
            raise HandoffError("课程目录在导出前发生变化，请重新导出")
        plan = locked_plan
        _check_sources(plan)
        code = _code_metadata(project_root)
        if plan["trainer"] == "native":
            code["katago_revision"] = None
        inventory = _inventory(plan)
        manifest = {"format_version": FORMAT_VERSION, "bundle_id": bundle_id,
                    "kind": plan["kind"], "trainer": plan["trainer"], "created_at": local_timestamp(),
                    "source_host": socket.gethostname(), "source_project": str(project_root),
                    "code": code, "active_prefix": plan["active_prefix"],
                    "sources": [{"prefix": s["prefix"], "source_root": str(s["root"]),
                                 "board_size": s.get("board_size")} for s in plan["sources"]],
                    "configs": list(plan["configs"]), "files": []}
        try:
            with zipfile.ZipFile(temp, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for name, path, before in inventory:
                    digest = hashlib.sha256()
                    size = 0
                    with path.open("rb") as original, archive.open(name, "w", force_zip64=True) as target:
                        for block in iter(lambda: original.read(_CHUNK), b""):
                            digest.update(block)
                            size += len(block)
                            target.write(block)
                    after = path.stat()
                    if (size, after.st_size, after.st_mtime_ns, after.st_ino) != (
                            before.st_size, before.st_size, before.st_mtime_ns, before.st_ino):
                        raise HandoffError(f"导出期间文件发生变化：{path}")
                    manifest["files"].append(_record(name, size, digest.hexdigest(), before.st_mtime_ns))
                for name, document in plan["configs"].items():
                    data = _json_bytes(document)
                    archive.writestr(name, data)
                    manifest["files"].append(_record(name, len(data), hashlib.sha256(data).hexdigest(), 0))
                archive.writestr("manifest.json", _json_bytes(manifest))
            current = _inventory(plan)
            if [(n, s.st_size, s.st_mtime_ns, s.st_ino) for n, _, s in current] != [
                    (n, s.st_size, s.st_mtime_ns, s.st_ino) for n, _, s in inventory]:
                raise HandoffError("导出期间训练目录发生变化，请确认所有训练进程已停止")
            current_code = _code_metadata(project_root)
            if plan["trainer"] == "native":
                current_code["katago_revision"] = None
            if current_code != code:
                raise HandoffError("导出期间程序代码发生变化，请重新导出")
            verify_handoff(temp)
            marker = {"format_version": FORMAT_VERSION, "bundle_id": bundle_id, "trainer": plan["trainer"],
                      "bundle": str(destination), "created_at": manifest["created_at"]}
            for source in plan["sources"]:
                path = source["root"] / HANDOFF_MARKER
                atomic_write_json(path, marker)
                markers.append(path)
            # Publish exclusively, using a checked copy on removable disks
            # that do not support hard links. Never replace another bundle.
            _publish_file(temp, destination)
        except BaseException:
            for path in markers:
                path.unlink(missing_ok=True)
            raise
        finally:
            temp.unlink(missing_ok=True)
    return {"bundle_id": bundle_id, "bundle": str(destination), "kind": plan["kind"], "trainer": plan["trainer"],
            "files": len(manifest["files"]), "bytes": sum(f["size"] for f in manifest["files"]),
            "source_suspended": True}


def _manifest(archive: zipfile.ZipFile) -> dict:
    infos = archive.infolist()
    names = set()
    for info in infos:
        name = _safe_name(info.filename)
        folded = name.casefold()
        mode = stat.S_IFMT(info.external_attr >> 16)
        if folded in names or info.is_dir() or mode not in (0, stat.S_IFREG) or info.flag_bits & 1:
            raise HandoffError(f"接力包包含重复路径、链接或不支持的条目：{name}")
        names.add(folded)
    try:
        entry = archive.getinfo("manifest.json")
        if entry.file_size > _MANIFEST_LIMIT:
            raise HandoffError("接力包清单过大")
        value = json.loads(archive.read(entry))
    except (KeyError, ValueError, UnicodeError) as error:
        raise HandoffError("接力包缺少有效 manifest.json") from error
    if not isinstance(value, dict) or type(value.get("format_version")) is not int or value["format_version"] != FORMAT_VERSION:
        raise HandoffError("接力包版本不受支持")
    if not isinstance(value.get("kind"), str) or value["kind"] not in {"single", "curriculum"} or not isinstance(value.get("bundle_id"), str) or not re.fullmatch(r"[a-f0-9]{32}", value["bundle_id"]):
        raise HandoffError("接力包类型或身份无效")
    trainer = value.get("trainer", "katago")
    if not isinstance(trainer, str) or trainer not in {"native", "katago"} or (trainer == "native" and value["kind"] != "single"):
        raise HandoffError("接力包训练类型无效")
    # Version-1 packages created before native handoff support were KataGo-only.
    value["trainer"] = trainer
    records = value.get("files")
    if not isinstance(records, list) or not records:
        raise HandoffError("接力包文件清单为空或无效")
    expected = {"manifest.json"}
    expected_folded = {"manifest.json"}
    for record in records:
        if not isinstance(record, dict):
            raise HandoffError("接力包文件记录无效")
        name = _safe_name(record.get("path"))
        parts = PurePosixPath(name).parts
        if parts[-1] in _SKIP_NAMES or any(part.endswith(".tmp") or part in _SKIP_DIRS for part in parts):
            raise HandoffError(f"接力包包含运行锁或未完成的临时文件：{name}")
        if name.casefold() in expected_folded:
            raise HandoffError(f"清单包含重复路径：{name}")
        if type(record.get("size")) is not int or record["size"] < 0 or type(record.get("mtime_ns")) is not int or record["mtime_ns"] < 0:
            raise HandoffError(f"清单文件大小或时间无效：{name}")
        if not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", ""))):
            raise HandoffError(f"清单 SHA-256 无效：{name}")
        expected.add(name)
        expected_folded.add(name.casefold())
    if expected != {info.filename for info in infos}:
        raise HandoffError("接力包实际文件与清单不一致")
    for name in expected:
        if any(parent.as_posix() in expected for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise HandoffError("接力包中文件与目录路径冲突")
    configs = value.get("configs")
    if not isinstance(configs, list) or not configs or any(not isinstance(c, str) or c not in expected or not c.startswith("config/") for c in configs):
        raise HandoffError("接力包配置清单无效")
    canonical_configs = ({"config/training.json"} if value["kind"] == "single" else
                         {"config/curriculum.json", "config/9x9.json", "config/13x13.json", "config/19x19.json"})
    if len(configs) != len(canonical_configs) or set(configs) != canonical_configs:
        raise HandoffError("接力包配置路径或数量不符合运行类型")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise HandoffError("接力包训练目录清单无效")
    prefixes = []
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_root"), str) or not source["source_root"]:
            raise HandoffError("接力包训练目录记录无效")
        prefix = _safe_name(source.get("prefix"))
        if prefix not in {"curriculum", "run", "stages/9x9", "stages/13x13", "stages/19x19"} or prefix in prefixes:
            raise HandoffError("接力包训练目录重复或无效")
        if prefix != "curriculum" and (type(source.get("board_size")) is not int or
                                      source["board_size"] not in (9, 13, 19)):
            raise HandoffError("接力包棋盘尺寸无效")
        if prefix.startswith("stages/") and prefix != f"stages/{source['board_size']}x{source['board_size']}":
            raise HandoffError("接力包棋盘尺寸与阶段目录不符")
        prefixes.append(prefix)
    active = value.get("active_prefix")
    checkpoint = "latest.pt" if trainer == "native" else "train/gogogo/checkpoint.ckpt"
    if not isinstance(active, str) or active not in prefixes or f"{active}/{checkpoint}" not in expected:
        raise HandoffError("接力包缺少完整续训检查点")
    if (value["kind"] == "single" and (prefixes != ["run"] or active != "run")) or (
            value["kind"] == "curriculum" and ("curriculum" not in prefixes or not active.startswith("stages/"))):
        raise HandoffError("接力包训练目录与运行类型不符")
    if any(name != "manifest.json" and name not in canonical_configs and
           not any(name.startswith(prefix + "/") for prefix in prefixes) for name in expected):
        raise HandoffError("接力包包含未声明的训练目录")
    required_config = "config/curriculum.json" if value["kind"] == "curriculum" else "config/training.json"
    if required_config not in configs or (value["kind"] == "curriculum" and "curriculum/state.json" not in expected):
        raise HandoffError("接力包缺少课程状态或恢复配置")
    if not isinstance(value.get("code"), dict):
        raise HandoffError("接力包缺少代码版本信息")
    if not re.fullmatch(r"[a-f0-9]{64}", str(value["code"].get("source_sha256", ""))):
        raise HandoffError("接力包源码内容校验值无效")
    if "behavior_assets_sha256" in value["code"] and not re.fullmatch(
            r"[a-f0-9]{64}", str(value["code"]["behavior_assets_sha256"])):
        raise HandoffError("接力包固定评测文件内容校验值无效")
    return value


def _expect_trainer(manifest: dict, expected_trainer: str | None) -> None:
    if expected_trainer is not None and manifest["trainer"] != expected_trainer:
        entry = "train.py" if manifest["trainer"] == "native" else "train_rl.py"
        raise HandoffError(f"接力包训练类型不匹配：{manifest['trainer']}，请使用 {entry} 处理")


def _verify_archive(archive: zipfile.ZipFile, expected_trainer: str | None = None) -> dict:
    manifest = _manifest(archive)
    _expect_trainer(manifest, expected_trainer)
    for record in manifest["files"]:
        entry = archive.getinfo(record["path"])
        if entry.file_size != record["size"]:
            raise HandoffError(f"接力包文件大小不匹配：{entry.filename}")
        digest = hashlib.sha256()
        with archive.open(entry) as stream:
            for block in iter(lambda: stream.read(_CHUNK), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise HandoffError(f"接力包 SHA-256 校验失败：{entry.filename}")
    return manifest


def verify_handoff(bundle: Path, *, expected_trainer: str | None = None) -> dict:
    """Verify every member without extracting or loading a pickle checkpoint."""
    try:
        with zipfile.ZipFile(bundle) as archive:
            manifest = _verify_archive(archive, expected_trainer)
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        raise HandoffError(f"无法验证接力包：{error}") from error
    return {**manifest, "verified": True, "bytes": sum(f["size"] for f in manifest["files"])}


def _hardware_overrides(path: Path | None) -> dict:
    if path is None:
        return {}
    raw = _read_json(path)
    if set(raw) == {"overrides"}:
        raw = raw["overrides"]
    if not isinstance(raw, dict) or not raw or set(raw) - _HARDWARE_FIELDS.keys():
        raise HandoffError("硬件配置只允许 hardware、self_play.workers/inference_batch_size、optimizer.batch_size")
    for section, values in raw.items():
        if not isinstance(values, dict) or not values:
            raise HandoffError(f"硬件配置 {section} 必须是非空对象")
        allowed = _HARDWARE_FIELDS[section]
        if allowed is not None and set(values) - allowed:
            raise HandoffError(f"硬件配置不能修改 {section} 的训练算法参数")
    return raw


def _remap_paths(value: object, mappings: list[tuple[str, Path]]) -> object:
    if isinstance(value, dict):
        return {key: _remap_paths(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_paths(item, mappings) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for original, target in mappings:
            prefix = original.replace("\\", "/").rstrip("/")
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return str(target / Path(*PurePosixPath(normalized[len(prefix):].lstrip("/")).parts))
    return value


def _prepare_import(temp: Path, destination: Path, manifest: dict, overrides: dict) -> dict:
    mappings = sorted([(source["source_root"], destination / source["prefix"])
                       for source in manifest["sources"]], key=lambda item: len(item[0]), reverse=True)
    report = {}
    for name in manifest["configs"]:
        if name == "config/curriculum.json":
            continue
        document = _read_json(temp / name)
        key = Path(name).stem
        prefix = "run" if manifest["kind"] == "single" else f"stages/{key}"
        document["overrides"]["runtime"]["output_directory"] = str(destination / prefix)
        for section, changes in overrides.items():
            document["overrides"][section].update(changes)
        resolve_rl_training_config(document["preset"], document["overrides"])
        atomic_write_json(temp / name, document)
    if manifest["kind"] == "curriculum":
        path = temp / "config/curriculum.json"
        document = _read_json(path)
        document["state_directory"] = str(destination / "curriculum")
        for stage in document["stages"]:
            stage["training_config"] = str(destination / f"config/{stage['board_size']}x{stage['board_size']}.json")
            if stage["board_size"] != 9 and "batch_size" in overrides.get("optimizer", {}):
                stage["autotune_batches"] = [overrides["optimizer"]["batch_size"]]
        atomic_write_json(path, document)
        load_curriculum_config(path)
        state_path = temp / "curriculum/state.json"
        state = _read_json(state_path)
        original_state = _json_bytes(state)
        state["migration"] = _remap_paths(state.get("migration", {}), mappings)
        if "batch_size" in overrides.get("optimizer", {}):
            for stage in state["stages"].values():
                stage["autotune_batch"] = None
        if _json_bytes(state) != original_state:
            atomic_write_json(state_path, state)
        CurriculumStateStore(temp / "curriculum").load()
        report["curriculum_config"] = str(destination / "config/curriculum.json")
    else:
        report["training_config"] = str(destination / "config/training.json")
        if manifest["trainer"] == "native":
            report["resume_checkpoint"] = str(destination / "run/latest.pt")
            resolved = load_rl_training_config(temp / "config/training.json").to_dict()
            atomic_write_json(temp / "run/config.resolved.json", resolved)
            monitor_path = temp / "run/monitor.json"
            if monitor_path.is_file():
                monitor = _read_json(monitor_path)
                monitor.update(status="stopped", phase="handoff_imported", pid=None,
                               hostname=socket.gethostname(), updated_at=local_timestamp(),
                               updated_at_epoch=time.time())
                atomic_write_json(monitor_path, monitor)
            if overrides:
                atomic_write_json(temp / "config/hardware.json", overrides)
                report["hardware_profile"] = str(destination / "config/hardware.json")
    for source in manifest["sources"]:
        path = temp / source["prefix"] / "migration_manifest.json"
        if path.is_file():
            atomic_write_json(path, _remap_paths(_read_json(path), mappings))
    return report


def _validate_restored_payload(root: Path, manifest: dict) -> None:
    sources = [{"root": root / source["prefix"], "prefix": source["prefix"],
                **({"board_size": source["board_size"]} if source["prefix"] != "curriculum" else {})}
               for source in manifest["sources"]]
    plan = {"kind": manifest["kind"], "trainer": manifest["trainer"],
            "sources": sources, "active_prefix": manifest["active_prefix"]}
    if manifest["kind"] == "curriculum":
        config = load_curriculum_config(root / "config/curriculum.json")
        # The published configs point to the final destination. Validate the
        # same profiles in staging without starting or opening the final run.
        staged_config = replace(config, stages=tuple(
            replace(stage, training_config=str(root / f"config/{stage.key}.json"))
            for stage in config.stages))
        validate_training_profiles(staged_config, root)
        state = CurriculumStateStore(root / "curriculum").load()
        if state is None or f"stages/{state.active.board_size}x{state.active.board_size}" != manifest["active_prefix"]:
            raise HandoffError("课程当前棋盘与接力包检查点目录不一致")
        if state.phase != "training" and not (state.phase == "paused_disk" and state.resume_phase == "training"):
            raise HandoffError("接力包课程不处于可接管的稳定训练阶段")
        plan["state"] = state
    else:
        config = load_rl_training_config(root / "config/training.json")
        if config.game.board_size != sources[0]["board_size"]:
            raise HandoffError("接力包训练配置与棋盘尺寸不一致")
    _check_sources(plan)


def _native_publish_directory(source: Path, destination: Path) -> None:
    """Atomically rename without replacing even a concurrently created empty dir."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise HandoffError("系统不支持安全的原子目录发布 renameat2")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise HandoffError("系统不支持安全的原子目录发布 renamex_np")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 4)
    else:
        raise HandoffError("当前系统暂不支持安全的原子目录发布")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _publish_file(source: Path, destination: Path) -> None:
    """Publish a private staged file exclusively, including on non-link filesystems."""
    try:
        os.link(source, destination)
        return
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            raise
    owned_identity = None
    try:
        with destination.open("xb") as target, source.open("rb") as original:
            created = os.fstat(target.fileno())
            owned_identity = (created.st_dev, created.st_ino)
            shutil.copyfileobj(original, target, _CHUNK)
            target.flush()
            os.fsync(target.fileno())
        metadata = source.stat()
        os.utime(destination, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    except BaseException:
        if owned_identity is not None and destination.is_file() and not destination.is_symlink():
            current = destination.stat()
            if (current.st_dev, current.st_ino) == owned_identity:
                destination.unlink()
        raise


def _publish_directory(source: Path, destination: Path) -> None:
    try:
        _native_publish_directory(source, destination)
        return
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise
    # WSL's Windows-mounted filesystem may reject RENAME_NOREPLACE. Reserve a
    # new directory exclusively and keep all training entries blocked until
    # the complete validated tree has been published. No existing path is
    # replaced, and failed cleanup never removes another writer's files.
    destination.mkdir()
    root_stat = destination.stat()
    pending = destination / IMPORT_PENDING_MARKER
    owned = []
    pending_identity = None
    try:
        atomic_write_json(pending, {"operation_id": uuid.uuid4().hex,
                                    "status": "importing", "created_at": local_timestamp()})
        pending_stat = pending.stat()
        pending_identity = (pending_stat.st_dev, pending_stat.st_ino)
        for path in sorted(source.rglob("*"), key=lambda p: (len(p.relative_to(source).parts), str(p))):
            target = destination / path.relative_to(source)
            if path.is_dir():
                target.mkdir()
            else:
                _publish_file(path, target)
            metadata = target.lstat()
            owned.append((target, (metadata.st_dev, metadata.st_ino)))
        current = pending.lstat()
        if (current.st_dev, current.st_ino) != pending_identity or pending.is_symlink():
            raise HandoffError("导入完成标记被修改，保留现场")
        pending.unlink()
    except BaseException:
        current_root = destination.lstat()
        if not destination.is_symlink() and (current_root.st_dev, current_root.st_ino) == (root_stat.st_dev, root_stat.st_ino):
            for path, identity in reversed(owned):
                if not path.resolve().is_relative_to(destination):
                    continue
                with contextlib.suppress(OSError):
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == identity and not path.is_symlink():
                        path.rmdir() if path.is_dir() else path.unlink()
            if pending_identity is not None and list(destination.iterdir()) == [pending]:
                current = pending.lstat()
                if (current.st_dev, current.st_ino) == pending_identity and not pending.is_symlink():
                    pending.unlink()
                    destination.rmdir()
        raise


def _installed_inventory(root: Path) -> list[dict]:
    return [_record(path.relative_to(root).as_posix(), path.stat().st_size,
                    sha256_file(path), path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != RECEIPT]


def import_handoff(project_root: Path, bundle: Path, destination: Path, *,
                   allow_version_mismatch: bool = False, hardware_profile: Path | None = None,
                   expected_trainer: str | None = None) -> dict:
    """Verify and atomically publish a new local run; never overwrite a run."""
    if destination.is_symlink():
        raise HandoffError("导入目标不能是符号链接")
    project_root, destination = project_root.resolve(), destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise HandoffError(f"导入目录必须不存在，拒绝覆盖已有成果：{destination}")
    overrides = _hardware_overrides(hardware_profile)
    warnings = []
    temp = None
    try:
        with zipfile.ZipFile(bundle) as archive:
            manifest = _verify_archive(archive, expected_trainer)
            current = _code_metadata(project_root)
            # Publishing a snapshot changes HEAD even if no training code
            # changes. Content, rather than the enclosing Git commit, defines
            # compatibility; declared KataGo revisions still require equality.
            mismatches = [key for key in ("source_sha256", "behavior_assets_sha256", "katago_revision")
                          if manifest["code"].get(key) is not None and current.get(key) != manifest["code"][key]]
            code_revision_differs = (manifest["code"].get("revision") is not None
                                     and current.get("revision") != manifest["code"]["revision"])
            code_content_matched = not any(key in mismatches for key in ("source_sha256", "behavior_assets_sha256"))
            if mismatches:
                message = "代码或 KataGo 版本与接力包不一致：" + ", ".join(mismatches)
                if not allow_version_mismatch:
                    raise HandoffError(message + "；请同步版本，必要时显式使用 --allow-version-mismatch")
                warnings.append(message)
            if code_revision_differs and code_content_matched:
                warnings.append("Git 提交不同，但训练源码及已声明的固定评测文件内容校验一致")
            total = sum(record["size"] for record in manifest["files"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(destination.parent).free < total + 16 * 1024 * 1024:
                raise HandoffError("目标磁盘剩余空间不足以导入接力包")
            temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.handoff-", suffix=".tmp", dir=destination.parent))
            for record in manifest["files"]:
                path = temp / record["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(record["path"]) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target, _CHUNK)
                if sha256_file(path) != record["sha256"]:
                    raise HandoffError(f"导入落盘校验失败：{record['path']}")
                if record["mtime_ns"]:
                    os.utime(path, ns=(record["mtime_ns"], record["mtime_ns"]))
        report = _prepare_import(temp, destination, manifest, overrides)
        _validate_restored_payload(temp, manifest)
        receipt = {"format_version": FORMAT_VERSION, "bundle_id": manifest["bundle_id"],
                   "kind": manifest["kind"], "trainer": manifest["trainer"], "imported_at": local_timestamp(),
                   "destination": str(destination), "project_root": str(project_root),
                   "manifest": manifest, "resume": report, "warnings": warnings,
                   "code_revision_differs": code_revision_differs, "code_content_matched": code_content_matched,
                   "hardware_overrides": overrides, "files": _installed_inventory(temp)}
        atomic_write_json(temp / RECEIPT, receipt)
        if destination.exists():
            raise HandoffError("导入期间目标目录被创建，拒绝覆盖")
        # Serializes cooperating importers; the final syscall also refuses a
        # directory concurrently created by another program.
        lock = acquire_file_lock(destination.parent / f".{destination.name}.handoff-import.lock")
        try:
            if destination.exists():
                raise HandoffError("导入期间目标目录被创建，拒绝覆盖")
            _publish_directory(temp, destination)
            if temp.exists():
                if temp.resolve().parent != destination.parent or not temp.name.startswith(f".{destination.name}.handoff-"):
                    raise HandoffError("临时导入目录身份发生变化，保留现场")
                shutil.rmtree(temp)
            temp = None
        finally:
            release_file_lock(lock)
        return {"root": str(destination), "bundle_id": manifest["bundle_id"], "trainer": manifest["trainer"],
                "files": len(manifest["files"]), "bytes": total, "warnings": warnings,
                "code_revision_differs": code_revision_differs, "code_content_matched": code_content_matched, **report}
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
        raise HandoffError(f"导入接力包失败：{error}") from error
    finally:
        if temp is not None:
            # This path was created by this invocation under the verified parent.
            if temp.resolve().parent != destination.parent or not temp.name.startswith(f".{destination.name}.handoff-"):
                raise HandoffError("临时导入目录身份发生变化，保留现场")
            shutil.rmtree(temp)


def _load_full_checkpoint(path: Path) -> dict:
    try:
        import torch
    except ImportError as error:
        raise HandoffError("检查点加载校验需要当前 Python 环境安装 PyTorch") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise HandoffError(f"检查点无法以受限 CPU 加载方式读取：{path}：{error}") from error
    if not isinstance(payload, dict) or any(not isinstance(payload.get(key), dict)
                                           for key in ("model", "optimizer", "train_state", "config")):
        raise HandoffError(f"检查点缺少完整 model/optimizer/train_state/config：{path}")
    model = payload["model"]
    if not model or not payload["optimizer"].get("param_groups"):
        raise HandoffError(f"检查点网络或优化器参数为空：{path}")
    for name, tensor in model.items():
        if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all().item():
            raise HandoffError(f"检查点权重包含非有限值：{name}")
    return {"checkpoint": str(path), "loaded": True,
            "global_step_samples": payload["train_state"].get("global_step_samples"),
            "has_optimizer": True, "has_swa": isinstance(payload.get("swa_model"), dict)}


def check_import(directory: Path, *, load_checkpoint: bool = False,
                 expected_trainer: str | None = None) -> dict:
    """Check an imported snapshot before training changes its file inventory."""
    directory = directory.resolve()
    receipt = _read_json(directory / RECEIPT)
    if receipt.get("format_version") != FORMAT_VERSION or receipt.get("destination") != str(directory):
        raise HandoffError("导入回执无效或目录已移动，请重新导入到目标位置")
    for record in receipt["files"]:
        name = _safe_name(record["path"])
        path = directory / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise HandoffError(f"导入后的文件完整性校验失败：{name}")
    manifest = receipt["manifest"]
    # Receipts from original KataGo-only version-1 imports remain readable.
    manifest.setdefault("trainer", "katago")
    _expect_trainer(manifest, expected_trainer)
    resume = receipt["resume"]
    _validate_restored_payload(directory, manifest)
    if manifest["kind"] == "curriculum":
        config = load_curriculum_config(Path(resume["curriculum_config"]))
        validate_training_profiles(config, Path(receipt["project_root"]))
        CurriculumStateStore(directory / "curriculum").load()
    else:
        load_rl_training_config(Path(resume["training_config"]))
    report = {"bundle_id": receipt["bundle_id"], "trainer": manifest["trainer"], "verified": True,
              "files": len(receipt["files"]), "checkpoint_loaded": False,
              "validation_scope": "文件完整性与配置；尚未运行 GPU 训练更新", **resume}
    if load_checkpoint:
        if manifest["trainer"] == "native":
            from .rl_native_handoff import check_native_checkpoint
            path = directory / manifest["active_prefix"] / "latest.pt"
            report["checkpoint"] = check_native_checkpoint(path, Path(resume["training_config"]))
        else:
            path = directory / manifest["active_prefix"] / "train/gogogo/checkpoint.ckpt"
            report["checkpoint"] = _load_full_checkpoint(path)
        report["checkpoint_loaded"] = True
    return report


def cancel_handoff(project_root: Path, *, curriculum_config: Path | None = None,
                   training_config: Path | None = None, native_run: Path | None = None) -> dict:
    """Release a source hold after the operator confirms no remote trainer runs."""
    plan = _source_plan(project_root.resolve(), curriculum_config, training_config, native_run)
    removed = []
    with _source_locks(plan, allow_suspended=True):
        for source in plan["sources"]:
            marker = source["root"] / HANDOFF_MARKER
            if marker.exists() or marker.is_symlink():
                if not marker.is_file() or marker.is_symlink():
                    raise HandoffError(f"接力标记无效，请先检查：{marker}")
                removed.append(marker)
        for marker in removed:
            marker.unlink()
    return {"source_resumed": True, "removed_markers": [str(path) for path in removed],
            "warning": "请确认其他电脑已停止同一次训练后再启动本机"}
