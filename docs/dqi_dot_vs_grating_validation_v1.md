# DQI / Q_dec Cross-Task Confirmatory Validation v1

The target is clean4 dot-versus-grating decoding only: dot is class 0 and grating is class 1. Historical outer FCNN checkpoints and their cycle folds are reused exactly. No outer model is trained.

`Q_DG` is generated with three deterministic cycle-grouped inner folds inside each historical outer-training set. Each inner model uses only its inner-training frames for arcsinh/pixel z-score normalization and runs the historical 40-epoch FCNN protocol. Its raw logits are converted to probabilities and the four frame probabilities are averaged per block. The formal task quality is BA on all concatenated inner-OOF blocks, not a mean of inner-fold BAs.

Formal execution freezes all nine session `Q_DG` values before reading or reconstructing any outer-test target. It then reconstructs every historical outer prediction from the validated checkpoint and saved outer-train normalization; a mismatch against the provenance-validated historical aggregate aborts the run.

The pre-freeze phase reads checkpoint metadata and may hash the opaque historical prediction file, but it cannot parse that prediction table. The Phase-2 loader requires `QUALITY_FROZEN.json` plus an exact hash match to the frozen session-Q table. Historical checkpoint reconstruction is explicitly CPU-only even when the 738 inner models are trained with CUDA, so model and input devices cannot diverge.

Before any Phase-1 inner training, formal full performs a CPU-only 246/246 checkpoint preflight covering existence, manifest SHA256, loadability, task/session/seed/fold and cycle membership, classes, and the frozen FCNN architecture. A single failure writes a failed `historical_checkpoint_preflight.json` and aborts before the first inner model. `RUN_COMPLETE.json` is written only when every required aggregate exists and is hashed; status also revalidates the frozen session-Q SHA256.

The only confirmatory relationship is session-level `Q_DG` versus historical session-level DG BA. The gate requires Spearman rho >= .75, exhaustive two-sided 9! permutation p <= .05, LOSO median rho >= .65, and LOSO minimum rho > .30. Presence/DG cross-task relationships, fold-level associations, and within-session correlations are descriptive only and never enter the gate.

Run plan, then CPU sanity:

```bash
PYTHONPATH=src python scripts/baselines/run_dqi_dot_vs_grating_validation.py --stage plan --project-root /data2/yuq1ngr/6.30project --output-dir outputs/dqi_dot_vs_grating_validation_v1
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python scripts/baselines/run_dqi_dot_vs_grating_validation.py --stage sanity --project-root /data2/yuq1ngr/6.30project --output-dir outputs/dqi_dot_vs_grating_validation_v1 --device cpu --sanity-epochs 1
```

Formal full execution is intentionally guarded and was not run during code review:

```bash
PYTHONPATH=src python scripts/baselines/run_dqi_dot_vs_grating_validation.py --stage full --project-root /data2/yuq1ngr/6.30project --output-dir outputs/dqi_dot_vs_grating_validation_v1 --device cuda --review-approved
```
