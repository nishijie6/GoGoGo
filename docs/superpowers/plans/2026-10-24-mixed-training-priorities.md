# Mixed KataGo training priorities implementation plan

> **For agentic workers:** Execute the independent health and evaluation modules with focused TDD; integrate and review before changing the running training process. The user approved the priority order and continuation.

**Goal:** Make progress measurable per board, diagnose degeneration, and run isolated, reproducible opening/compute experiments before selecting new training settings.

**Architecture:** Keep training, diagnostics, frozen-model evaluation, and branch preparation in separate modules. Reuse the existing KataGo paired-match runner and full checkpoint/replay mechanisms. A fixed-budget experiment creates new directories and never overwrites the baseline training state.

**Tech Stack:** Python unittest, official KataGo v1.18.2, PyTorch MPS, Metal engine, atomic JSON/SGF/NPZ artifacts.

**Spec:** The accepted priority table in this chat and `docs/research/KATAGO_TRAINING_LESSONS_20261024.md`.

## Global constraints

- Supported mixed boards: 7, 8, 9, 13, 19; common training tensor length 19.
- Preserve existing models, optimizer, replay watermarks, and rule/komi identity.
- Freeze model bytes and protocol before evaluation; never silently reselect a moving model on resume.
- Pair each opening with swapped colors; fix game budget before starting; no significance-based early stopping.
- Use M4-safe serial evaluation, small inference batches, FP32 training, no silent CPU fallback.
- Health metrics are diagnostic; they do not prove playing strength and do not automatically discard samples.
- Preserve the current training configuration unless a controlled experiment supplies sufficient evidence for a change.

## Task 1: Per-board health

Files: create `weiqi/katago_mixed_health.py`, `tests/test_katago_mixed_health.py`.

Interface:

`build_mixed_health(run_root, board_sizes, *, window_games=256, minimum_games=64) -> dict`
returns JSON-safe per-board diagnostics and atomically persists `health_by_board.json`.

- [x] Test valid 7/8/19 SGFs, sliding windows, changing files, malformed complete trees, and pending live tails.
- [x] Observe failures, implement strict parsing and incremental inventory, run focused tests.
- [x] Integrate `mixed-health` and health reporting after training cycles.
- [x] Run against the existing live run and retain a timestamped baseline report.

## Task 2: Frozen paired evaluation

Files: create `weiqi/katago_mixed_eval.py`, `tests/test_katago_mixed_eval.py`; extend KataGo-only match/opening validators for 7/8.

Interface:

`evaluate_mixed(runner, output_dir, *, candidate=None, baseline, boards=None,
games_per_board=12, visits=32) -> dict` freezes inputs, verifies immutable manifests,
and reuses only completed matching pairs.

- [x] Test frozen-file hashes, incompatible resumes, deleted source models, five-board protocols, and complete color pairs.
- [x] Observe failures, implement using `run_opening_pairs`, run focused tests.
- [x] Integrate `mixed-evaluate` with explicit baseline, candidate, output, games and visits arguments.
- [x] Gracefully stop the baseline process for GPU-isolated evaluation, preserve checkpoints, and run the initial fixed-budget comparison.
- [x] Restore continued baseline training after finite experiments, with updated PID/provenance metadata.

## Task 3: Controlled opening experiment

Files: create `weiqi/katago_mixed_experiments.py`, its tests, and a small bounded trial controller; add optional validated exploration settings to mixed configuration.

- [x] Freeze one full checkpoint with optimizer and one retained replay snapshot after the source run is stopped.
- [x] Materialize identical control and treatment branches; treatment uses 10% deterministic legal SGF starts and distinct output paths.
- [x] Separate training starts from held-out evaluation openings, record input hashes, and reject mutations on resume.
- [x] Run equal wall-clock-budget branches; count actual completed updates and valid data per board.
- [x] Evaluate frozen outputs on the same held-out protocol. Label small pilots as inconclusive when appropriate; do not infer causal gains from one random run.

## Task 4: Sampling/search/loop budget accounting

- [x] Include actual retained rows, new rows, per-board game lengths, training samples and stage seconds in diagnostic/experiment reports.
- [x] Prepare reproducible alternative search/replay/cycle-size profiles as independent experiments, retaining the baseline.
- [x] Compare useful data and fixed evaluation at equal resource budget; do not switch to large-model/Muon/full asynchronous operation without local evidence.

## Verification and delivery

- [x] Focused tests for each module, then full unittest suite with canonical temporary paths and local-server permissions.
- [x] Read-only independent review of manifest safety, pairing, resource controls and training restart.
- [x] Verify frozen models, real Metal matches, actual MPS updates, health artifacts, and an alive resumed training process.
- [x] Document commands, evaluation limitations, experiment findings and current training status.

## Measured completion

Completed SGF control/treatment and all five resource profiles at 180 seconds each, with frozen per-board comparisons. Protocol v2 completed 480 games. Full suite: 647 run, 644 passed, 3 skipped. Source training resumed from 908,724 samples; verified at 911,820 with live MPS PID 54874. Original configuration retained because the pilots do not establish a stable advantage. Results: `docs/research/KATAGO_MIXED_PILOT_20261024.md`.
