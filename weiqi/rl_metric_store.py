"""Persistent per-run metric history (JSONL, CSV, latest JSON and dashboard)."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

from .fileio import atomic_write_json, atomic_write_text
from .rl_metrics import (
    CSV_FIELDS,
    checkpoint_metrics,
    parse_model_name,
    parse_sgfs,
    rolling_health_metrics,
)
from .rl_metrics_dashboard import render_dashboard


def _timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds"
    )


class RLMetricStore:
    """Synchronize accepted checkpoints into durable metrics and charts."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.models_dir = self.run_root / "models"
        self.exports_dir = self.run_root / "torchmodels_toexport"
        self.selfplay_dir = self.run_root / "selfplay"
        self.metrics_dir = self.run_root / "metrics"
        self.jsonl_path = self.metrics_dir / "history.jsonl"
        self.csv_path = self.metrics_dir / "history.csv"
        self.latest_path = self.metrics_dir / "latest.json"
        self.dashboard_path = self.run_root / "dashboard.html"
        size_match = re.fullmatch(r"(?P<size>\d+)x(?P=size)", self.run_root.name)
        self.board_size = int(size_match.group("size")) if size_match else None

    def paths(self) -> dict[str, str]:
        return {
            "metrics_jsonl": str(self.jsonl_path),
            "metrics_csv": str(self.csv_path),
            "metrics_latest": str(self.latest_path),
            "dashboard": str(self.dashboard_path),
        }

    def _read_existing(self) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        if not self.jsonl_path.is_file():
            return records
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            model = record.get("model")
            if isinstance(model, str):
                records[model] = record
        return records

    def _model_record(self, model_dir: Path) -> dict[str, object]:
        parsed = parse_model_name(model_dir.name)
        if parsed is None:
            raise ValueError(f"无法解析 KataGo 模型名：{model_dir.name}")
        trained_samples, data_rows = parsed
        model_file = model_dir / "model.bin.gz"
        record: dict[str, object] = {
            "schema_version": 1,
            "timestamp": _timestamp(model_dir),
            "model": model_dir.name,
            "trained_samples": trained_samples,
            "data_rows": data_rows,
            "model_bytes": model_file.stat().st_size if model_file.is_file() else None,
            "source": "checkpoint-bootstrap",
        }
        checkpoint = self.exports_dir / model_dir.name / "model.ckpt"
        try:
            record.update(checkpoint_metrics(checkpoint))
        except (ImportError, OSError, RuntimeError, ValueError):
            # The name itself still provides valid monotonic progress if an old
            # checkpoint has been pruned or is temporarily unavailable.
            pass
        return record

    def sync(
        self,
        *,
        live_model: Optional[str] = None,
        live_metrics: Optional[Mapping[str, object]] = None,
    ) -> list[dict[str, object]]:
        """Import new accepted models, then atomically refresh every artifact."""

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_existing()
        model_dirs = {
            path.name: path
            for path in self.models_dir.iterdir()
            if path.is_dir()
            and ".tmp-" not in path.name
            and (path / "model.bin.gz").is_file()
            and parse_model_name(path.name) is not None
        } if self.models_dir.is_dir() else {}
        # Model/checkpoint retention is intentionally much shorter than metric
        # retention.  Build from the union so pruning a model never truncates
        # the durable JSONL/CSV history on the next refresh.
        model_names = sorted(
            {
                name
                for name in (*existing.keys(), *model_dirs.keys())
                if parse_model_name(name) is not None
            },
            key=lambda name: parse_model_name(name) or (0, 0),
        )
        records: list[dict[str, object]] = []
        for model_name in model_names:
            model_dir = model_dirs.get(model_name)
            record = existing.get(model_name)
            if record is None:
                if model_dir is None:  # Defensive; union guarantees this cannot happen.
                    continue
                record = self._model_record(model_dir)
            else:
                record = dict(record)
            if model_name == live_model and live_metrics:
                record.update(live_metrics)
                record["source"] = "completed-cycle"
            sgf_paths = sorted(
                (self.selfplay_dir / model_name / "sgfs").glob("*.sgfs")
            )
            if sgf_paths:
                sgf_bytes = sum(path.stat().st_size for path in sgf_paths)
                if record.get("sgf_file_bytes") != sgf_bytes:
                    record.update(parse_sgfs(sgf_paths))
            records.append(record)

        previous_samples = 0
        previous_rows = 0
        previous_time: Optional[datetime] = None
        for index, record in enumerate(records, start=1):
            record["schema_version"] = 2
            record["cycle"] = index
            record["accepted_models"] = index
            samples = int(record.get("trained_samples", 0))
            rows = int(record.get("data_rows", 0))
            record["sample_delta"] = samples - previous_samples
            record["data_rows_delta"] = rows - previous_rows
            current_time: Optional[datetime]
            try:
                current_time = datetime.fromisoformat(str(record["timestamp"]))
            except (KeyError, TypeError, ValueError):
                current_time = None
            if "cycle_seconds" not in record and previous_time and current_time:
                record["cycle_seconds"] = max(
                    0.0, (current_time - previous_time).total_seconds()
                )
            previous_samples = samples
            previous_rows = rows
            previous_time = current_time or previous_time

        health_records: list[dict[str, object]] = []
        for record in records:
            if "sgf_entries" in record:
                health_records.append(record)
            record.update(rolling_health_metrics(health_records))

        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        jsonl = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        atomic_write_text(self.jsonl_path, jsonl)
        table = io.StringIO()
        writer = csv.DictWriter(table, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        # The BOM lets Excel detect UTF-8 for the Chinese column values.
        atomic_write_text(self.csv_path, "﻿" + table.getvalue())
        atomic_write_json(self.latest_path, records[-1] if records else {})
        atomic_write_text(
            self.dashboard_path,
            render_dashboard(records, generated_at, board_size=self.board_size),
        )
        return records


__all__ = ["RLMetricStore"]
