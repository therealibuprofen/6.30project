# Cycle-Calibrated Late Fusion v1

CCLF v1 is a pre-registered test of training-only scalar temperature calibration. It is not uncertainty weighting. Each formal outer task reuses the exact validated historical binary FCNN checkpoint and that checkpoint's saved outer-training normalization.

For each session/fold/seed, the runner partitions only the outer-training cycles into three deterministic cycle-grouped inner folds. It fits normalization on each inner-training fold, trains the historical FCNN protocol for 40 fixed epochs, and predicts raw logits for the held-out inner cycles. Concatenating those predictions gives one cross-fitted logit vector for every outer-training frame. One positive scalar temperature is then fitted by minimizing frame-level cross-entropy NLL.

The fitted temperature is applied to raw logits from the unchanged historical outer checkpoint. Both baseline and calibrated block predictions are the arithmetic mean of the four frame probability vectors. Every frame therefore has weight 0.25. Confidence, entropy, margin, timestamp, frame position, and block type cannot affect fusion.

Formal evaluation concatenates outer-held-out block predictions before computing balanced accuracy. The frozen gate is:

- overall BA delta at least 0.005;
- Strong-3 BA delta at least -0.01;
- Weak-6 BA delta at least 0.01;
- calibrated overall frame ECE no more than 0.80 times baseline ECE.

All four conditions are required. The gate does not control temperature fitting; it is evaluated only after all formal OOF predictions exist.

## Commands

Plan:

```bash
PYTHONPATH=src python scripts/baselines/run_fcnn_cycle_calibrated_late_fusion.py --stage plan --project-root /data2/yuq1ngr/6.30project --output-dir /data2/yuq1ngr/6.30project/outputs/fcnn_cycle_calibrated_late_fusion_v1
```

CPU sanity:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/baselines/run_fcnn_cycle_calibrated_late_fusion.py --stage sanity --project-root /data2/yuq1ngr/6.30project --output-dir /data2/yuq1ngr/6.30project/outputs/fcnn_cycle_calibrated_late_fusion_v1 --device cpu --sanity-epochs 1
```

Formal full, only after review approval:

```bash
PYTHONPATH=src python scripts/baselines/run_fcnn_cycle_calibrated_late_fusion.py --stage full --project-root /data2/yuq1ngr/6.30project --output-dir /data2/yuq1ngr/6.30project/outputs/fcnn_cycle_calibrated_late_fusion_v1 --device cuda --review-approved
```

Status:

```bash
PYTHONPATH=src python scripts/baselines/run_fcnn_cycle_calibrated_late_fusion.py --stage status --project-root /data2/yuq1ngr/6.30project --output-dir /data2/yuq1ngr/6.30project/outputs/fcnn_cycle_calibrated_late_fusion_v1
```

`full` trains 738 inner calibration models and zero outer final models. It is intentionally blocked unless `--review-approved` is supplied. Strict resume accepts a task only when its parent fingerprint, cache identities, coverage, protocol fields, and artifact hashes validate. `status` also validates final aggregate hashes and reports `integrity-failed` if a completed artifact changed.
