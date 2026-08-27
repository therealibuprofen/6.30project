# Cross-Cycle Same-Block Mixup Temporal1D v1

This experiment reruns paired `raw` and `cyclemix` Temporal1D controls. The only experimental difference is a training-only input augmentation. Model, classifier, folds, normalization, initialization procedure, shuffle seed, optimizer, learning rate, batch size, fixed 40 epochs, and evaluation are unchanged.

## Exact augmentation

For each real anchor `i` in a training batch, the partner pool contains only samples from the current outer-training fold satisfying:

```text
block_id_j == block_id_i AND cycle_id_j != cycle_id_i AND partner_idx != anchor_idx
```

Thus only `G-G`, `D-D`, `S-S`, and `T-T` pairs are legal. Grating is never mixed with dot, and stop-after-grating is never mixed with static. At most one partner is drawn per anchor using a deterministic RNG keyed by `(seed, fold, epoch, batch_index)`. An anchor without a legal partner produces no synthetic counterpart and its real sample remains in training.

The fixed mix and label are:

```text
X_mix = 0.5 * X_anchor_norm + 0.5 * X_partner_norm
y_mix = y_anchor = y_partner
```

No soft labels, Beta distribution, ratio search, auxiliary loss, consistency loss, hard mining, prototype, memory bank, or special sampler is used.

## Preprocessing and loss order

Normalization is fitted using real outer-training samples only:

```text
clean4 -> arcsinh(real train and test) -> fit pixel z-score on real outer-train only -> transform real train/test -> construct CycleMix from normalized real train samples
```

Synthetic samples never affect normalization statistics. Test samples are transformed normally but never mixed. Test block identity is not passed to training or inference; test cycle IDs are used only for the outer train/test disjointness assertion and provenance audit.

For `cyclemix`, every original batch `B` is preserved and concatenated with its available mixed counterparts (normally another `B`). One ordinary mean cross-entropy is computed over the concatenated logits and hard labels. The configured real DataLoader batch size remains 16; a CUDA preflight explicitly checks the usual `2B=32` forward. `raw` uses the historical real-only training path, and a regression test requires identical histories and model state tensors against `_train_epochs`.

## Train accuracy, coverage, and statistics

Synthetic predictions do not enter the formal train accuracy. Every epoch records real-only train accuracy separately from the optional mixed accuracy. Task `final_train_accuracy_real` is the real-sample accuracy at fixed epoch 40. Seed `final_train_accuracy` is the mean of those fold-level fixed-epoch values, and:

```text
train_test_gap = final_train_accuracy - OOF_test_BA
```

`best_epoch` remains descriptive and never enters the gap. Every task records real/mixed counts, mix coverage, anchors without partners, examples, and same-cycle/different-block/self-pair violation counts; all violation counts must be zero. Session deltas use the existing exact two-sided paired sign-flip test over nine session-level BA differences and the preregistered continue/stop rule.

The formal plan is dynamically rebuilt from the explicit completed clean4 fold source and must equal 9 sessions, 2 variants, 3 seeds, 82 folds, and 492 tasks. Raw is rerun, not loaded from historical results.

## Local review commands

Every command is one line.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_cyclemix_temporal1d.py
```

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/run_cyclemix_temporal1d.py --stage plan --device cpu --workers 0 --output-dir outputs/cyclemix_temporal1d_v1_local_review
```

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/run_cyclemix_temporal1d.py --stage sanity --device cpu --workers 0 --sanity-epochs 1 --output-dir outputs/cyclemix_temporal1d_v1_local_review
```

## Formal server commands

Run the full command only after review approval. GNU screen is required; tmux is not supported. Resume uses the identical screen command after the previous screen session has ended. Valid tasks are skipped; corrupt/incomplete tasks are rerun. `RUN_COMPLETE.json` is written only after all 492 tasks and final aggregates validate.

```bash
cd /data2/yuq1ngr/6.30project && git pull --ff-only
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python -m pytest -q tests/test_cyclemix_temporal1d.py
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cyclemix_temporal1d.py --stage plan --device cpu --workers 0 --output-dir outputs/cyclemix_temporal1d_v1
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cyclemix_temporal1d.py --stage sanity --device cpu --workers 0 --sanity-epochs 1 --output-dir outputs/cyclemix_temporal1d_v1
```

```bash
cd /data2/yuq1ngr/6.30project && nvidia-smi
```

```bash
screen -dmS cyclemix_temporal1d_v1 bash -lc 'cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cyclemix_temporal1d.py --stage full --device cuda --workers 0 --review-approved --output-dir outputs/cyclemix_temporal1d_v1 2>&1 | tee outputs/cyclemix_temporal1d_v1/run_log_server.txt'
```

```bash
screen -r cyclemix_temporal1d_v1
```

```bash
cd /data2/yuq1ngr/6.30project && tail -n 100 outputs/cyclemix_temporal1d_v1/run_log_server.txt
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cyclemix_temporal1d.py --stage status --device cuda --workers 0 --output-dir outputs/cyclemix_temporal1d_v1
```

```bash
cd /data2/yuq1ngr/6.30project && test -f outputs/cyclemix_temporal1d_v1/RUN_COMPLETE.json && cat outputs/cyclemix_temporal1d_v1/RUN_COMPLETE.json
```

## Formal outputs

Finalization writes `task_level_results.csv`, `seed_summary.csv`, `session_summary.csv`, `overall_summary.csv`, `paired_sign_flip.csv`, `overfitting_audit.csv`, `predictions.csv`, `confusion_matrices.csv`, `training_history.csv`, `mix_coverage_audit.csv`, `decision_rule_audit.json`, `cyclemix_temporal1d_report.md`, and `RUN_COMPLETE.json`, plus strict per-task provenance and completion artifacts.
