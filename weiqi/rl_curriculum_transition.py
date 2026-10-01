"""Curriculum board transitions: batch autotuning and crash-safe stage migration."""

from __future__ import annotations

import contextlib
import json
import math
import re
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

from .fileio import atomic_write_json, local_timestamp, sha256_file
from .katago_rl import KataGoRLRunner, KataGoRLRunnerError
from .rl_benchmark import (
    BenchmarkMeasurement,
    build_benchmark_command,
    choose_autotune_batch,
    parse_benchmark_output,
    prepare_benchmark_npz,
)
from .rl_curriculum import (
    CurriculumMigrationError,
    CurriculumStage,
    CurriculumState,
    CurriculumStateError,
)
from .rl_migration import prepare_stage_migration, verify_checkpoint_loads


@dataclass
class _MigrationAttempt:
    """How far one migration run got; decides retry, failure or fail-closed on error."""

    target_preexisting: bool
    source_stage_index: int
    target_identity_confirmed: bool = False
    target_published: bool = False
    committed: bool = False


class TransitionSteps:
    """Stage-transition half of :class:`~weiqi.rl_curriculum_runtime.CurriculumRuntime`.

    Mixed into the runtime; uses its ``config``, ``store``, ``controller`` and
    its ``_make_runner``, ``_stage_runner`` and ``_run_external`` methods.
    """

    def _benchmark_measurement(
        self,
        runner: KataGoRLRunner,
        data_file: Path,
        batch_size: int,
        log_root: Path,
    ) -> BenchmarkMeasurement:
        command = build_benchmark_command(
            python_executable=Path(sys.executable),
            benchmark_script=runner.python_source / "benchmark_fresh_model.py",
            data_file=data_file,
            board_size=runner.config.game.board_size,
            batch_size=batch_size,
            gpu_index=0,
            model_kind=runner.model_kind,
        )
        environment = runner._environment()
        execution = self._run_external(
            command,
            cwd=runner.python_source,
            env=environment,
            log_path=log_root / f"batch-{batch_size}.log",
        )
        return parse_benchmark_output(
            execution.stdout, batch_size=batch_size, returncode=execution.returncode
        )

    def _autotune(
        self,
        stage: CurriculumStage,
        runner: KataGoRLRunner,
        source_npz: Path,
    ) -> tuple[int, Optional[str]]:
        if not stage.autotune_batches:
            return runner.config.optimizer.batch_size, None
        tune_root = runner.run_root / "autotune"
        data_file = prepare_benchmark_npz(
            source_npz,
            tune_root / "benchmark.npz",
            max(stage.autotune_batches),
        )
        persist_path = runner.run_root / "autotune.json"
        measurements: dict[int, BenchmarkMeasurement] = {}
        for batch in stage.autotune_batches:
            try:
                measurements[batch] = self._benchmark_measurement(
                    runner, data_file, batch, tune_root
                )
            except Exception as error:
                message = str(error)
                measurements[batch] = BenchmarkMeasurement(
                    batch_size=batch,
                    success=False,
                    oom=(
                        "out of memory" in message.casefold()
                        or re.search(r"\boom\b", message, re.IGNORECASE) is not None
                    ),
                    error=message,
                )
        try:
            result = choose_autotune_batch(
                stage.autotune_batches,
                lambda batch: measurements[batch],
                max_gpu_memory_gib=self.config.max_gpu_memory_gib,
                persist_path=None,
            )
            atomic_write_json(
                persist_path,
                {
                    "selected_batch_size": result.selected_batch_size,
                    "measurements": [
                        {
                            **asdict(measurement),
                            "peak_gpu_gib": (
                                measurement.peak_gpu_gib
                                if math.isfinite(measurement.peak_gpu_gib)
                                else None
                            ),
                        }
                        for measurement in result.measurements
                    ],
                    "completed_at": local_timestamp(),
                },
            )
            return result.selected_batch_size, None
        except Exception as error:
            warning = f"自动批次测试无合格结果，拒绝迁移：{error}"
            atomic_write_json(
                persist_path,
                {
                    "selected_batch_size": None,
                    "failed": True,
                    "warning": warning,
                    "measurements": [
                        {
                            **asdict(measurements[batch]),
                            "peak_gpu_gib": (
                                measurements[batch].peak_gpu_gib
                                if math.isfinite(measurements[batch].peak_gpu_gib)
                                else None
                            ),
                        }
                        for batch in stage.autotune_batches
                    ],
                    "completed_at": local_timestamp(),
                },
            )
            raise CurriculumMigrationError(warning) from error
        finally:
            with contextlib.suppress(FileNotFoundError):
                data_file.unlink()

    def _training_smoke(
        self,
        stage: CurriculumStage,
        seed_checkpoint: Path,
        source_npz: Path,
        batch_size: int,
        parent: Path,
    ) -> None:
        smoke_root = parent / f".training-smoke-{uuid.uuid4().hex}.tmp"
        try:
            checkpoint = smoke_root / "train" / "gogogo" / "checkpoint.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed_checkpoint, checkpoint)
            data_file = smoke_root / "selfplay" / "seed" / "tdata" / "smoke.npz"
            prepare_benchmark_npz(source_npz, data_file, max(2, batch_size * 2))
            smoke_runner = self._make_runner(stage, run_root_override=smoke_root)
            smoke_runner.shuffle(min_rows=1, smoke=True)
            smoke_runner.train(smoke=True, batch_size=batch_size)
            if not checkpoint.is_file():
                raise CurriculumMigrationError("单批训练冒烟未生成检查点")
        finally:
            if smoke_root.exists():
                shutil.rmtree(smoke_root, ignore_errors=True)

    def _migration_checks(
        self,
        stage: CurriculumStage,
        temporary_root: Path,
        seed_checkpoint: Path,
    ) -> tuple[int, Optional[str]]:
        runner = self._make_runner(stage, run_root_override=temporary_root)
        exported = runner.export_new_models()
        seed_model = temporary_root / "models" / "gogogo-s0-d0" / "model.bin.gz"
        if not seed_model.is_file() and not exported:
            raise CurriculumMigrationError("迁移种子未能导出为 KataGo 模型")
        runner.selfplay(games=4, visits=8, smoke=True)
        npz_files = sorted(
            (temporary_root / "selfplay").glob("**/tdata/*.npz"),
            key=lambda path: path.stat().st_mtime,
        )
        if not npz_files:
            raise CurriculumMigrationError("4 局低 visits 冒烟未生成目标棋盘 NPZ")
        selected, warning = self._autotune(stage, runner, npz_files[-1])
        self._training_smoke(
            stage, seed_checkpoint, npz_files[-1], selected, temporary_root
        )
        # Smoke games validate the target but must never enter its replay pool.
        for folder in (
            temporary_root / "selfplay",
            temporary_root / "shuffleddata",
            temporary_root / "shufflescratch",
            temporary_root / "autotune",
        ):
            if folder.exists():
                shutil.rmtree(folder)
        # The four validation games are intentionally discarded and must not
        # become the new stage's deleted-row offset.  Its real ledger starts at
        # zero after the temporary tree is published.
        with contextlib.suppress(FileNotFoundError):
            (temporary_root / "replay_rows.json").unlink()
        return selected, warning

    @staticmethod
    def _read_selected_batch(path: Path) -> Optional[int]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            selected = int(value["selected_batch_size"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return selected if selected > 0 else None

    def _record_migration_retry(self, state: CurriculumState, message: str) -> None:
        state.phase = "migrating"
        state.warnings.append(
            {"timestamp": local_timestamp(), "kind": "migration_retry", "message": message}
        )
        self.store.save(state)
        print(f"迁移暂时无法完成，将保留身份后重试：{message}", file=sys.stderr, flush=True)

    def _migrate(self, state: CurriculumState) -> None:
        """Seed the next board size from the latest accepted model, crash-safely.

        A migration has a durable identity (source model, checkpoint hash and
        target board). The target tree is built in a temporary directory and
        renamed into place with a manifest carrying that identity, so a run
        that crashes anywhere can either adopt the exact published tree or
        retry without touching a directory it cannot prove it owns.
        """
        if state.active_stage_index >= len(self.config.stages) - 1:
            state.phase = "training"
            self.store.save(state)
            return
        current_runner = self._stage_runner(state)
        source_model = self._protected_source_model(state)
        if source_model is None:
            return
        target_stage = self.config.stages[state.active_stage_index + 1]
        target_runner = self._make_runner(target_stage)
        target_root = target_runner.run_root
        source_checkpoint = (
            current_runner.run_root
            / "torchmodels_toexport"
            / source_model
            / "model.ckpt"
        )
        if not source_checkpoint.is_file():
            self.controller.migration_failed(
                state, f"找不到最新接纳模型的 SWA 检查点：{source_checkpoint}"
            )
            return
        source_sha256 = self._migration_source_hash(state, source_checkpoint)
        if source_sha256 is None:
            return
        identity = {
            "source_model": source_model,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_sha256": source_sha256,
            "target_board_size": target_stage.board_size,
            "seed_model": "gogogo-s0-d0",
        }
        if not self._claim_migration_identity(state, identity, target_root):
            return

        attempt = _MigrationAttempt(target_preexisting=target_root.exists(),
                                    source_stage_index=state.active_stage_index)
        try:
            target_lock = (
                target_runner.lock() if attempt.target_preexisting else contextlib.nullcontext()
            )
            with target_lock:
                if attempt.target_preexisting:
                    selected_batch, warning = self._adopt_published_target(
                        state, target_root, target_runner, attempt), None
                else:
                    selected_batch, warning = self._publish_migration_target(
                        state, target_stage, target_root, target_runner,
                        source_checkpoint, attempt)
                self.controller.complete_migration(
                    state,
                    seed_model="gogogo-s0-d0",
                    autotune_batch=selected_batch,
                )
                attempt.committed = True
            if warning:
                state.warnings.append(
                    {"timestamp": local_timestamp(), "kind": "autotune", "message": warning}
                )
                self.store.save(state)
            print(
                f"课程已切换到 {target_stage.board_size}x{target_stage.board_size}，训练批次 {selected_batch}",
                flush=True,
            )
        except Exception as error:
            self._recover_failed_migration(state, error, attempt)

    def _protected_source_model(self, state: CurriculumState) -> Optional[str]:
        """The latest accepted model, pinned against cleanup; None after recording why not."""
        source_model = state.active.latest_model
        if not source_model:
            self.controller.migration_failed(state, "迁移前缺少最新模型")
            return None
        if source_model not in state.protected_models:
            state.protected_models.append(source_model)
            self.store.save(state)
        return source_model

    def _migration_source_hash(
        self, state: CurriculumState, source_checkpoint: Path
    ) -> Optional[str]:
        """Hash the source; an unreadable file postpones rather than fails the migration."""
        try:
            return sha256_file(source_checkpoint)
        except OSError as error:
            if state.phase == "migrating" and state.migration:
                self._record_migration_retry(state, str(error))
            else:
                state.warnings.append(
                    {
                        "timestamp": local_timestamp(),
                        "kind": "migration_retry",
                        "message": f"暂时无法读取迁移源检查点：{error}",
                    }
                )
                self.store.save(state)
                print(
                    f"迁移源暂时不可读，将保持待迁移状态后重试：{error}",
                    file=sys.stderr,
                    flush=True,
                )
            return None

    def _claim_migration_identity(
        self, state: CurriculumState, identity: Mapping[str, object], target_root: Path
    ) -> bool:
        """Start a new migration or confirm the persisted one is the same transition."""
        if state.phase == "transition_ready":
            state.migration = {
                "id": uuid.uuid4().hex,
                **identity,
                "started_at": local_timestamp(),
            }
            self.store.save(state)
            self.controller.begin_migration(state)
            return True
        if state.phase != "migrating":
            raise CurriculumStateError(f"当前 phase={state.phase}，不能迁移")
        if not state.migration:
            # Backward-compatible recovery is safe only before a target
            # directory has become visible.  A visible directory without
            # an identity manifest could belong to another transition.
            if target_root.exists():
                self.controller.migration_failed(
                    state,
                    f"目标目录缺少迁移身份，拒绝触碰现有目录：{target_root}",
                )
                return False
            state.migration = {
                "id": uuid.uuid4().hex,
                **identity,
                "started_at": local_timestamp(),
            }
            self.store.save(state)
            return True
        if (
            not isinstance(state.migration.get("id"), str)
            or any(state.migration.get(key) != value for key, value in identity.items())
        ):
            suffix = ""
            if target_root.exists():
                suffix = f"；未触碰现有目标目录 {target_root}"
            self.controller.migration_failed(
                state, f"迁移源检查点或目标阶段已变化，拒绝继续旧迁移{suffix}"
            )
            return False
        return True

    def _adopt_published_target(
        self,
        state: CurriculumState,
        target_root: Path,
        target_runner: KataGoRLRunner,
        attempt: "_MigrationAttempt",
    ) -> int:
        """Reuse a tree published before a crash; its manifest must prove it is ours."""
        if not target_root.exists():
            raise CurriculumMigrationError("目标阶段目录在加锁期间消失")
        # Recovery for a crash after the atomic directory rename but
        # before complete_migration() persisted the new active stage.
        checkpoint = target_root / "train" / "gogogo" / "checkpoint.ckpt"
        model = target_root / "models" / "gogogo-s0-d0" / "model.bin.gz"
        selected_batch = self._read_selected_batch(target_root / "autotune.json")
        manifest_path = target_root / "migration_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = None
        manifest_matches = isinstance(manifest, dict) and all(
            manifest.get(key) == value
            for key, value in state.migration.items()
        )
        if isinstance(manifest, dict):
            manifest_matches = manifest_matches and (
                manifest.get("selected_batch_size") == selected_batch
            )
        if (
            not checkpoint.is_file()
            or not model.is_file()
            or selected_batch is None
            or not manifest_matches
            or manifest.get("seed_checkpoint_sha256")
            != sha256_file(checkpoint)
        ):
            raise CurriculumMigrationError(
                f"已存在目标目录不完整或不属于当前迁移；拒绝触碰 {target_root}"
            )
        attempt.target_identity_confirmed = True
        verify_checkpoint_loads(
            checkpoint,
            boards=(9, 13, 19),
            python_source=target_runner.python_source,
        )
        return selected_batch

    def _publish_migration_target(
        self,
        state: CurriculumState,
        target_stage: CurriculumStage,
        target_root: Path,
        target_runner: KataGoRLRunner,
        source_checkpoint: Path,
        attempt: "_MigrationAttempt",
    ) -> tuple[int, Optional[str]]:
        """Build, verify and atomically publish the target tree; return (batch, warning)."""
        if target_root.exists():
            raise CurriculumMigrationError(
                f"目标目录在迁移准备期间被其他进程创建：{target_root}"
            )
        check_result: dict[str, object] = {}

        def verify(checkpoint: Path, _board: int) -> None:
            verify_checkpoint_loads(
                checkpoint,
                boards=(9, 13, 19),
                python_source=target_runner.python_source,
            )

        def smoke(root: Path, checkpoint: Path, _board: int) -> None:
            batch, tune_warning = self._migration_checks(
                target_stage, root, checkpoint
            )
            atomic_write_json(
                root / "migration_manifest.json",
                {
                    **state.migration,
                    "selected_batch_size": batch,
                    "seed_checkpoint_sha256": sha256_file(checkpoint),
                    "completed_at": local_timestamp(),
                },
            )
            check_result["batch"] = batch
            check_result["warning"] = tune_warning

        prepare_stage_migration(
            source_checkpoint=source_checkpoint,
            target_stage_root=target_root,
            target_board_size=target_stage.board_size,
            load_verifier=verify,
            smoke_verifier=smoke,
            seed_name="gogogo-s0-d0",
        )
        # From this point the verified tree and its identity
        # manifest are atomically visible.  A later state-save
        # failure must retain this migration identity so the next
        # run can adopt that exact tree instead of orphaning it.
        attempt.target_published = True
        warning_value = check_result.get("warning")
        return int(check_result["batch"]), str(warning_value) if warning_value else None

    def _recover_failed_migration(
        self, state: CurriculumState, error: Exception, attempt: "_MigrationAttempt"
    ) -> None:
        """Record a failure without regressing a commit or orphaning a published tree."""
        # If complete_migration's atomic save returned an error after the
        # replacement, trust the durable state instead of the mutated
        # in-memory object.
        try:
            persisted = self.store.load()
            if persisted is not None:
                state.__dict__.update(persisted.__dict__)
                attempt.committed = (
                    persisted.active_stage_index > attempt.source_stage_index
                )
        except (CurriculumStateError, OSError) as reload_error:
            if attempt.target_published or attempt.target_preexisting:
                # The target is visible but we cannot tell whether the
                # atomic state replacement committed.  Any write here
                # could regress a successfully committed migration or
                # persist a half-mutated object, so fail closed and let a
                # fresh process read the durable state.
                raise CurriculumStateError(
                    "迁移目标已发布，但无法确认课程状态提交结果；拒绝继续写状态"
                ) from reload_error
        if attempt.committed:
            state.warnings.append(
                {
                    "timestamp": local_timestamp(),
                    "kind": "migration_artifact",
                    "message": f"迁移状态已提交，但后续记录失败：{error}",
                }
            )
            self.store.save(state)
            print(
                f"迁移已提交；后续记录失败：{error}", file=sys.stderr, flush=True
            )
        elif attempt.target_published or (
            attempt.target_preexisting
            and (
                isinstance(error, (KataGoRLRunnerError, OSError))
                or attempt.target_identity_confirmed
            )
        ):
            self._record_migration_retry(state, str(error))
        else:
            self.controller.migration_failed(state, str(error))
            print(f"迁移失败，继续旧棋盘：{error}", file=sys.stderr, flush=True)
