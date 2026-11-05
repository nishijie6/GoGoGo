# Cold bootstrap readiness implementation plan

**Goal:** Prevent untrained or immature neural networks from supplying the first reinforcement-learning data.

**Architecture:** Fresh mixed runs collect with official native random selfplay. The learner seed is private. Trained candidates remain outside the actor role until an isolated probe meets explicit data-quality conditions. Existing resumed runs retain their actor policy; original failed runs are preserved.

**Spec:** The user authorized tracing the original cause then fixing it. Controlled 200-game evidence is recorded in `training_runs/katago_coldstart_rootcause_20261030/causal_comparison.json`. Native-first alone reproduced short games after the first1024sample learner, so readiness must be checked after learning, not inferred from export success.

**Tech stack:** Existing KataGo v1.18.2 native selfplay, MPS learner, atomic JSON state, unittest.

## Constraints

- No original model, checkpoint, or replay deletion or overwrite.
- Keep original 7/8/9/13/19 mix,6.5komi,32/8visits,FP32MPS, batch8.
- Quality readiness is not a playing-strength or promotion claim.
- Do not gate on Black/White wins alone; small-board komi may produce legitimate bias.
- Bootstrap actor remains native random while candidates fail/inconclusive; candidate probe data is never added to learner replay.
- Probe100games at most once per4096new learner samples; require at least4 completed games per board, no damaged/no-result/pending records, short-game rate<=25% on every board, actual weighted policy pass probability<=20%.
- Preserve default existing-run behavior when no bootstrap state exists; corrupt recorded state fails closed.

## Tasks

- [x] Reproduce original first-data bias and compare native-random generator with same settings.
- [x] Exclude seed tensor,CPU/Metal parity, score-sign and first-checkpoint mismatches.
- [x] Move fresh learner seed out of inference/export queues; remove eager CLI publication.
- [x] Implement persisted bootstrap state and isolated candidate quality assessment with tests first.
- [x] Integrate actor directory selection and post-cycle readiness probes.
- [x] Test failure, interruption/resume, successful readiness and unchanged legacy resumes.
- [x] Independently review the fix, run full suite, and run real closed-loop validation across original collapse stage.
- [x] Record honest runtime outcome and keep repaired training on validated bootstrap path.

## Completed evidence

Cold loop:48 cycles,48384samples,1152games; all replay remained native random while immature candidates failed probes. Recovery preserved full model+optimizer, passed100gameprobe at3653756samples, and120actual neural-actor games passed independent data-quality reassessment.8completed recovery cycles then normal128batch target resumed.Continuous MPS PID93100 verified at3679876samples; original3641260checkpoint unchanged.Tests675run,672pass,3skip; post-config focused45pass. See docs/research/KATAGO_COLD_START_ROOT_CAUSE_20261030.md.
