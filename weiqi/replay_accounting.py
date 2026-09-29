"""Crash-safe accounting for KataGo replay rows removed by retention.

KataGo's shuffler reports the row coordinates that are consumed by
``train.py``.  Those coordinates must stay monotonic even when old raw NPZ
files are pruned from a bounded replay window.  ``ReplayRowLedger`` keeps an
atomic inventory of immutable raw files and turns files that disappear from
that inventory into a persistent ``deleted_rows_offset`` for
``shuffle.py -add-to-data-rows``.

The full inventory is also the write-ahead record for retention: callers sync
before deleting files and again afterwards.  If the process stops anywhere in
between, the next sync observes the missing inventory entries and accounts for
each of them exactly once.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional

from .rl_curriculum import npz_row_count


LEDGER_SCHEMA_VERSION = 1
MODEL_NAME_RE = re.compile(r"^gogogo-s(?P<samples>\d+)-d(?P<rows>\d+)$")


class ReplayAccountingError(RuntimeError):
    """Raised when replay history cannot be accounted for safely."""


@dataclass(frozen=True)
class ReplayRowSnapshot:
    """A durable view of one board size's raw replay coordinate space."""

    path: Path
    deleted_rows_offset: int
    physical_rows: int
    logical_rows: int
    files: Mapping[str, int]
    initialized_at: str
    updated_at: str


@dataclass(frozen=True)
class _FileRecord:
    rows: int
    size: int
    mtime_ns: int


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_plain_nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


class ReplayRowLedger:
    """Maintain a stage-local, atomically persisted deleted-row offset."""

    def __init__(self, run_root: Path):
        self.run_root = run_root.resolve()
        self.path = self.run_root / "replay_rows.json"

    def _validate_key(self, key: object) -> str:
        if not isinstance(key, str) or not key:
            raise ReplayAccountingError("replay_rows.json 包含无效文件路径")
        pure = PurePosixPath(key)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or len(pure.parts) < 4
            or pure.parts[0] != "selfplay"
            or pure.parts[-2] != "tdata"
            or not pure.name.endswith(".npz")
        ):
            raise ReplayAccountingError(
                f"replay_rows.json 包含越界或非原始 NPZ 路径：{key}"
            )
        resolved = (self.run_root / Path(*pure.parts)).resolve()
        try:
            resolved.relative_to(self.run_root)
        except ValueError as error:
            raise ReplayAccountingError(
                f"replay_rows.json 路径越出阶段目录：{key}"
            ) from error
        return pure.as_posix()

    def _load(
        self,
    ) -> Optional[tuple[int, dict[str, _FileRecord], str]]:
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReplayAccountingError(
                f"无法读取有效回放行账本：{self.path}"
            ) from error
        if not isinstance(value, dict) or value.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ReplayAccountingError("replay_rows.json 版本或根节点无效")
        offset = value.get("deleted_rows_offset")
        raw_files = value.get("files")
        initialized_at = value.get("initialized_at")
        if (
            not _is_plain_nonnegative_integer(offset)
            or not isinstance(raw_files, dict)
            or not isinstance(initialized_at, str)
            or not initialized_at
        ):
            raise ReplayAccountingError("replay_rows.json 缺少有效的累计字段")
        records: dict[str, _FileRecord] = {}
        for raw_key, raw_record in raw_files.items():
            key = self._validate_key(raw_key)
            if key in records or not isinstance(raw_record, dict):
                raise ReplayAccountingError("replay_rows.json 文件清单无效")
            rows = raw_record.get("rows")
            size = raw_record.get("size")
            mtime_ns = raw_record.get("mtime_ns")
            if not all(
                _is_plain_nonnegative_integer(item)
                for item in (rows, size, mtime_ns)
            ) or rows == 0:
                raise ReplayAccountingError(
                    f"replay_rows.json 文件记录无效：{key}"
                )
            records[key] = _FileRecord(rows=rows, size=size, mtime_ns=mtime_ns)
        return int(offset), records, initialized_at

    def _scan(
        self, previous: Mapping[str, _FileRecord]
    ) -> tuple[dict[str, _FileRecord], dict[str, int]]:
        records: dict[str, _FileRecord] = {}
        mtimes: dict[str, int] = {}
        for path in sorted(self.run_root.glob("selfplay/**/tdata/*.npz")):
            try:
                resolved = path.resolve(strict=True)
                relative = resolved.relative_to(self.run_root).as_posix()
            except (OSError, ValueError) as error:
                raise ReplayAccountingError(
                    f"无法安全定位原始回放文件：{path}"
                ) from error
            key = self._validate_key(relative)
            if key in records:
                raise ReplayAccountingError(f"原始回放文件路径重复：{key}")
            try:
                stat = resolved.stat()
            except OSError as error:
                raise ReplayAccountingError(f"无法读取回放文件状态：{resolved}") from error
            prior = previous.get(key)
            if (
                prior is not None
                and prior.size == stat.st_size
                and prior.mtime_ns == stat.st_mtime_ns
            ):
                rows = prior.rows
            else:
                try:
                    rows = npz_row_count(resolved)
                except Exception as error:
                    raise ReplayAccountingError(
                        f"无法统计原始回放文件行数：{resolved}"
                    ) from error
                if rows <= 0:
                    raise ReplayAccountingError(f"原始回放文件没有有效行：{resolved}")
                if prior is not None and prior.rows != rows:
                    raise ReplayAccountingError(
                        f"已登记的原始回放文件被改写，拒绝继续：{resolved}"
                    )
            records[key] = _FileRecord(
                rows=int(rows), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns)
            )
            mtimes[key] = int(stat.st_mtime_ns)
        return records, mtimes

    def _latest_model_watermark(self) -> tuple[int, Optional[str]]:
        candidates: list[tuple[int, int, str]] = []
        for parent_name in ("models", "torchmodels_toexport"):
            parent = self.run_root / parent_name
            if not parent.is_dir():
                continue
            for path in parent.iterdir():
                if not path.is_dir() or ".tmp" in path.name:
                    continue
                match = MODEL_NAME_RE.fullmatch(path.name)
                if match:
                    candidates.append(
                        (
                            int(match.group("samples")),
                            int(match.group("rows")),
                            path.name,
                        )
                    )
        if not candidates:
            return 0, None
        # A normal run increases both coordinates.  Prefer the historical
        # maximum ``d`` explicitly so even a previously malformed higher-s
        # export cannot make first-time recovery move the data watermark back.
        _, rows, name = max(candidates, key=lambda item: (item[1], item[0], item[2]))
        return rows, name

    def _checkpoint_watermark(self) -> tuple[int, Optional[int]]:
        checkpoint = self.run_root / "train" / "gogogo" / "checkpoint.ckpt"
        if not checkpoint.is_file():
            return 0, None
        try:
            import torch

            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ReplayAccountingError(
                f"无法读取训练检查点的数据水位：{checkpoint}"
            ) from error
        train_state = value.get("train_state") if isinstance(value, dict) else None
        if train_state is None:
            # A curriculum SWA seed deliberately has no counters.
            return 0, checkpoint.stat().st_mtime_ns
        if not isinstance(train_state, dict):
            raise ReplayAccountingError("训练检查点的 train_state 无效")
        raw_values = [
            train_state.get("total_num_data_rows", 0),
            train_state.get("train_bucket_level_at_row", 0),
        ]
        if not all(_is_plain_nonnegative_integer(item) for item in raw_values):
            raise ReplayAccountingError("训练检查点的数据行水位无效")
        try:
            cutoff = checkpoint.stat().st_mtime_ns
        except OSError as error:
            raise ReplayAccountingError(f"无法读取训练检查点时间：{checkpoint}") from error
        return max(int(item) for item in raw_values), cutoff

    def _infer_initial_offset(
        self, records: Mapping[str, _FileRecord], mtimes: Mapping[str, int]
    ) -> int:
        model_rows, model_name = self._latest_model_watermark()
        checkpoint_rows, checkpoint_mtime_ns = self._checkpoint_watermark()
        watermark = max(model_rows, checkpoint_rows)
        physical_rows = sum(record.rows for record in records.values())
        if watermark <= 0:
            return 0

        if checkpoint_mtime_ns is not None and checkpoint_rows >= model_rows:
            accounted_rows = sum(
                record.rows
                for key, record in records.items()
                if mtimes[key] <= checkpoint_mtime_ns
            )
            return max(0, watermark - accounted_rows)

        # Fallback for an imported/exported model without a main checkpoint:
        # files produced under that latest model are newer, untrained rows.
        new_rows = 0
        if model_name is not None:
            prefix = f"selfplay/{model_name}/tdata/"
            new_rows = sum(
                record.rows for key, record in records.items() if key.startswith(prefix)
            )
        return max(0, watermark + new_rows - physical_rows)

    def _save(
        self,
        *,
        offset: int,
        records: Mapping[str, _FileRecord],
        initialized_at: str,
    ) -> ReplayRowSnapshot:
        physical_rows = sum(record.rows for record in records.values())
        updated_at = _now()
        document = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "deleted_rows_offset": offset,
            "physical_rows": physical_rows,
            "logical_rows": offset + physical_rows,
            "files": {
                key: {
                    "rows": record.rows,
                    "size": record.size,
                    "mtime_ns": record.mtime_ns,
                }
                for key, record in sorted(records.items())
            },
            "initialized_at": initialized_at,
            "updated_at": updated_at,
        }
        self.run_root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as error:
            raise ReplayAccountingError(f"无法原子保存回放行账本：{self.path}") from error
        finally:
            if temporary.exists():
                temporary.unlink()
        return ReplayRowSnapshot(
            path=self.path,
            deleted_rows_offset=offset,
            physical_rows=physical_rows,
            logical_rows=offset + physical_rows,
            files={key: record.rows for key, record in records.items()},
            initialized_at=initialized_at,
            updated_at=updated_at,
        )

    def sync(self) -> ReplayRowSnapshot:
        """Reconcile the immutable NPZ inventory and atomically save it."""

        loaded = self._load()
        if loaded is None:
            previous: dict[str, _FileRecord] = {}
            offset = 0
            initialized_at = _now()
        else:
            offset, previous, initialized_at = loaded
        records, mtimes = self._scan(previous)
        if loaded is None:
            offset = self._infer_initial_offset(records, mtimes)
        else:
            missing = set(previous).difference(records)
            offset += sum(previous[key].rows for key in missing)
        return self._save(
            offset=offset,
            records=records,
            initialized_at=initialized_at,
        )


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "ReplayAccountingError",
    "ReplayRowLedger",
    "ReplayRowSnapshot",
]
