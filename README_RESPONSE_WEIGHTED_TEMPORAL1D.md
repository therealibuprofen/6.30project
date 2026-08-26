# Response-Weighted Temporal1D v1

This experiment reruns a paired `raw` control and changes exactly one input operation in `response_weighted`: every outer fold derives one fixed soft spatial weight map from that fold's training cycles, then multiplies the same map into normalized train and test blocks. It does not launch a downstream experiment automatically.

## Frozen response map

For every training cycle, the existing clean4 blocks are transformed with `arcsinh` and averaged across their four frames to obtain G (grating), S (stop-after-grating), D (dot), and T (static). The cycle contrast is

```text
C_c = 0.5 * (G_c + D_c) - 0.5 * (S_c + T_c)
```

Across training cycles only,

```text
R = abs(mean_c(C_c)) / (std_c(C_c, ddof=0) + 1e-6)
```

`scipy.stats.rankdata(..., method="average")` ranks R across all spatial pixels. The percentile is `P=(rank-1)/(n_pixels-1)` and the fixed soft map is `W=0.5+P`. Thus W is in `[0.5,1.5]`; average ranks make its spatial mean 1.0, including tied scores.

## Actual preprocessing order

```text
clean4 extraction -> arcsinh -> outer-train-fold pixel z-score -> multiply by W -> Temporal1D
```

The raw branch ends after the existing z-score and is bitwise identical to the existing formal normalization function. The weighted branch applies W after normalization, so pixel-wise z-scoring cannot cancel the intended scaling. Normalization statistics and W are independent training-fold-only quantities. Test blocks use the training normalization and the exact same W object/hash; they never update either one.

## Leakage controls

The response-map API accepts only `X_train`, training block names, and training cycle IDs. It has no full-session, test, held-out-index, GLM-map, label, threshold, or hyperparameter input. Runtime assertions require disjoint train/test cycle IDs and require the map's recorded cycles to equal the outer training fold. The fold cache key binds session, fold, sorted training cycles, clean4/arcsinh protocol, response-map implementation version, and relevant source SHA-256. Seeds 0/1/2 may share a map only inside that identical fold. No full-session GLM file is loaded.

Every weighted task records response/weight hashes, cache key, train/test cycle provenance, map shape and distribution, weight range and distribution, and the absence of a full-session GLM. Each session's fold 1 additionally stores a small compressed response-score/weight artifact. Completed tasks are accepted only after artifact, identity, probability, metric, 40-epoch history, normalization, response-map, architecture, source/config, and runtime provenance validation.

## Pairing and summaries

Both variants are rerun with the same exact formal clean4 split manifests, fold indices, seeds, initialization logic, 115,890-parameter `CNN2DTemporal1D`, AdamW settings, batch size 16, fixed 40 epochs, classifier, evaluation, and OOF aggregation. There is no validation split, early stopping, threshold selection, or model selection. `best_epoch` remains descriptive only. `final_train_accuracy` is each fold's epoch-40 train accuracy averaged within seed, and `train_test_gap = final_train_accuracy - OOF_test_BA`.

The formal plan is derived from the actual fold manifests and must validate as 9 sessions, 2 variants, 3 seeds, 82 folds, and 492 tasks. Raw historical results are not reused. The paired inferential output is the existing exact two-sided sign-flip test across nine session deltas.

## Local review commands

Each command below is deliberately a single shell line.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_response_weighted_temporal1d.py
```

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/run_response_weighted_temporal1d.py --stage plan --device cpu --workers 0 --output-dir outputs/response_weighted_temporal1d_v1_local_review
```

```bash
PYTHONPATH=src .venv/bin/python scripts/baselines/run_response_weighted_temporal1d.py --stage sanity --device cpu --workers 0 --sanity-epochs 1 --output-dir outputs/response_weighted_temporal1d_v1_local_review
```

## Formal server commands (run by the user after review)

Only GNU screen is supported for the formal run. Do not use tmux. These are single-line commands with no continuation characters.

```bash
cd /data2/yuq1ngr/6.30project && git pull --ff-only
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python -m pytest -q tests/test_response_weighted_temporal1d.py
```

```bash
cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_response_weighted_temporal1d.py --stage plan --device cpu --workers 0 --output-dir outputs/response_weighted_temporal1d_v1
```

```bash
screen -dmS response_weighted_temporal1d_v1 bash -lc 'cd /data2/yuq1ngr/6.30project && PYTHONPATH=src /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_response_weighted_temporal1d.py --stage full --device cuda --workers 0 --review-approved --output-dir outputs/response_weighted_temporal1d_v1 2>&1 | tee outputs/response_weighted_temporal1d_v1/run_log_server.txt'
```

```bash
screen -r response_weighted_temporal1d_v1
```

Resume uses the identical `screen -dmS ...` command after confirming the old screen session has ended. Valid tasks are skipped; incomplete or corrupt tasks are rerun. `RUN_COMPLETE.json` is written only after all task-plan entries validate and every required aggregate exists.

## Formal outputs

The finalized run writes `task_level_results.csv`, `seed_summary.csv`, `session_summary.csv`, `overall_summary.csv`, `paired_sign_flip.csv`, `overfitting_audit.csv`, `predictions.csv`, `confusion_matrices.csv`, `training_history.csv`, `response_map_audit.csv`, `decision_rule_audit.json`, `response_weighted_temporal1d_report.md`, and `RUN_COMPLETE.json`, plus per-task completion/provenance artifacts and fold response-map audits.
