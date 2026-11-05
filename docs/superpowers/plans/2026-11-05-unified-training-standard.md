# Unified training standard implementation plan

> **For agentic workers:** Use subagent-driven-development to execute the independent adapters and review the integrated result.

**Goal:** Apply one versioned promotion, health and sample-progress standard to native training, KataGo curriculum and Mac mixed training.

**Architecture:** A dependency-free `weiqi/training_standard.py` owns the rules. Each existing trainer adapts its real paired-match and self-play evidence to that interface. Hardware settings remain in hardware profiles; old configurations and pending evaluation records retain their recorded rules.

**Tech stack:** Python dataclasses, unittest, existing JSON checkpoints and monitoring adapters.

**Approved design:** The user approved unifying training standards after remote-code integration. Champion admission requires a complete color-swapped match, at least 55% score overall, at least 50% on each evaluated board, and healthy self-play. Confidence intervals and extreme margins remain diagnostics. Evaluation is due every 51,200 actual gradient-sample draws, preserving the Mac's approximate previous 50-cycle cadence. Curriculum board transitions retain their explicit stage-sample and consecutive-pass requirements.

## Constraints

- Preserve current model, optimizer, replay, frozen champion and historical decisions.
- Keep all-black/all-white, early-pass, short-game, invalid-game and excessive-pass protection.
- Keep Mac batch 8, inference batch 4, eight games and Metal/ANE hybrid settings.
- Freeze full new evaluation protocol and policy before games start; legacy pending records use legacy semantics.
- Activate checked-in production profiles only after tests pass; resume current training and verify a completed cycle.

## Tasks

- [x] Implement and test shared versioned rules and SGF health adapter.
- [x] Integrate native trainer with persisted actual-sample counters and healthy champion admission.
- [x] Integrate curriculum controller and both champion-promotion paths.
- [x] Integrate mixed promotion and migrate cycle schedule without losing remaining progress.
- [x] Select the same standard in production profiles and expose policy/sample progress in the monitor.
- [x] Run focused and full tests, independent review, then restart and verify production.

## Decisions

The standard name is explicit (`unified_v1`) so archived configurations without it remain reproducible. Network formats and native board-size support remain backend capabilities. Search experiments and board curricula remain explicit strategy choices rather than implicit GPU-tier changes.

Review ruling: legacy curriculum manifests did not freeze admission thresholds. On standard adoption, explicitly defer incomplete legacy curriculum work with an audit event and preserve its files; start a separate complete suite under the new policy. Inferring an old threshold would risk reinterpreting evidence. Mixed pending rounds already contain their old gate and continue with that gate.

## Verification

Full regression: 778 tests run, 775 passed, 3 skipped. Both independent scoped reviews cleared their findings. JavaScript syntax and diff checks passed.

The Mac run resumed its unchanged full checkpoint at 9,181,972 trained samples. The new version completed cycle 5385 with 9,183,020 samples, all five board health checks passed, champion T1 was preserved, and the HTTP monitor reported the same 9,220,884 next-evaluation sample boundary. Runtime evidence is stored in `training_runs/katago_existing_model_gated_20261030/run/unified_standard_activation_20261105.json`.
