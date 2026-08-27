# Cycle-Consistent Temporal1D v1

This experiment reruns paired `raw` and `cycle_consistent` Temporal1D controls. The only optimization difference is `lambda_consistency=0` versus the frozen `lambda_consistency=0.1`; input, architecture, classifier, initialization, mini-batch shuffle, optimizer, training duration, and evaluation remain fixed.

## Classifier-input embedding

`CNN2DTemporal1D` encodes the four clean frames, transposes the resulting sequence into channels-by-time, and passes it through `temporal_conv`. Its `[B,64]` output is the last latent tensor immediately before the unchanged `Dropout(0.25) -> Linear(64,2)` classifier. `forward_with_embedding(x)` exposes that tensor and its logits. Default `model(x)` still accepts only x and returns the same logits as the historical expression `classifier(temporal_conv(encode_sequence(x).transpose(1,2)))`.

The model architecture and parameter count remain unchanged at 115,890. No projection head, new layer, hidden-size change, or four-class classifier is added.

## Positive pairs and loss

Training metadata maps the four block identities to integer IDs solely inside the experiment-specific training DataLoader. For batch samples i and j, a pair is valid exactly when:

```text
block_i == block_j AND cycle_i != cycle_j AND i < j
```

The upper triangle prevents double counting. For classifier-input embeddings z, the implementation computes:

```text
z_norm = L2_normalize(z)
L_cons = mean_valid_pairs(1 - cosine(z_norm_i, z_norm_j))
L_total = L_cls + 0.1 * L_cons
```

No negatives, temperature, margin, memory bank, queue, prototype, special sampler, or cross-batch state is used. A batch with no valid pairs uses `L_cons=0` and still performs its ordinary classification update.

## Frozen input and leakage boundary

Both variants use the unchanged pipeline:

```text
clean4 -> arcsinh -> outer-train-fold pixel z-score -> CNN2DTemporal1D
```

Only `X_train`, binary training labels, training block identities, and training cycle IDs enter the training DataLoader. The runner passes no test block identity into `train_fold`; test cycle IDs are used only for a disjointness assertion and provenance audit. Prediction calls the existing X-only `predict_probabilities(model, X_test)` path. Test embeddings are not computed, and the optional representation diagnostic uses normalized outer-training samples only after training; it is never used for selection.

Each mini-batch pair mask is asserted to contain only current outer-training cycle IDs. Every task records epoch-level and aggregate batch count, batches with pairs, valid-pair batch fraction, total valid positive pairs, mean pairs per batch, classification/consistency/total losses, and a training-only same-block cross-cycle cosine diagnostic.

## Raw equivalence and summaries

The raw branch uses the same experiment-specific DataLoader but takes the historical `model(x)`, CE loss, and optimizer path without adding a zero-valued term. A regression test compares one multi-batch epoch against the existing `_train_epochs` implementation and requires identical history and every final parameter/buffer tensor.

Both variants rerun the exact formal clean4 fold manifests, seeds 0/1/2, 115,890-parameter model, AdamW, learning rate 0.001, weight decay 0.001, batch size 16, fixed 40 epochs, binary classifier, and OOF Balanced Accuracy. `best_epoch` is descriptive only. Seed-level `final_train_accuracy` is the mean of each fold's fixed epoch-40 accuracy, and `train_test_gap = final_train_accuracy - OOF_test_BA`.

The dynamically derived plan must contain 9 sessions, 2 variants, 3 seeds, 82 folds, and 492 tasks. Raw historical results are not loaded. The paired test is the existing exact two-sided nine-session sign-flip implementation.

## Local review commands

Every command is intentionally one line.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_cycle_consistent_temporal1d.py
```

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/run_cycle_consistent_temporal1d.py --stage plan --device cpu --workers 0 --output-dir outputs/cycle_consistent_temporal1d_v1_local_review
```

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/run_cycle_consistent_temporal1d.py --stage sanity --device cpu --workers 0 --sanity-epochs 1 --output-dir outputs/cycle_consistent_temporal1d_v1_local_review
```

## Formal server commands

Run these only after code-review approval. GNU screen is required; tmux is not supported. Each command is a single line with no continuation character.

```bash
cd /data2/yuq1ngr/6.30project && git pull --ff-only
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python -m pytest -q tests/test_cycle_consistent_temporal1d.py
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cycle_consistent_temporal1d.py --stage plan --device cpu --workers 0 --output-dir outputs/cycle_consistent_temporal1d_v1
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cycle_consistent_temporal1d.py --stage sanity --device cpu --workers 0 --sanity-epochs 1 --output-dir outputs/cycle_consistent_temporal1d_v1
```

```bash
screen -dmS cycle_consistent_temporal1d_v1 bash -lc 'cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_cycle_consistent_temporal1d.py --stage full --device cuda --workers 0 --review-approved --output-dir outputs/cycle_consistent_temporal1d_v1 2>&1 | tee outputs/cycle_consistent_temporal1d_v1/run_log_server.txt'
```

```bash
screen -r cycle_consistent_temporal1d_v1
```

Resume uses the identical `screen -dmS ...` command after the prior screen session has ended. Valid completed tasks are skipped; corrupt or incomplete tasks rerun. `RUN_COMPLETE.json` is written only after every expected task and aggregate validates.

## Formal outputs

Finalization writes `task_level_results.csv`, `seed_summary.csv`, `session_summary.csv`, `overall_summary.csv`, `paired_sign_flip.csv`, `overfitting_audit.csv`, `predictions.csv`, `confusion_matrices.csv`, `training_history.csv`, `pair_coverage_audit.csv`, `representation_audit.csv`, `decision_rule_audit.json`, `cycle_consistent_temporal1d_report.md`, and `RUN_COMPLETE.json`, plus strict per-task provenance and completion artifacts.
