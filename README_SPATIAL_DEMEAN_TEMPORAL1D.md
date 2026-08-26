# Spatial-Demean Temporal1D v1

This experiment asks one controlled question: does subtracting each clean4
frame's own global spatial mean improve within-session binary stimulus decoding?

## Frozen protocol

- Sessions: 626, 628, 708, 709, 710, 807, 813, 817, 822.
- Input: formal clean4 blocks with shape `[4, 128, 501]`.
- Target: grating + dot versus stop-after-grating + static.
- CV: the existing cycle-grouped formal folds, capped at 10 folds per session.
- Seeds: 0, 1, 2.
- Model: the unchanged `CNN2DTemporal1D` and unchanged classifier.
- Training: AdamW, learning rate `1e-3`, weight decay `1e-3`, batch size 16,
  dropout 0.25, cross-entropy, exactly 40 epochs.
- Epoch selection: none. The formal baseline has no validation fold, patience,
  early stopping, or checkpoint selection. `best_epoch` in task results is only
  the descriptive epoch of maximum training accuracy; every test prediction is
  made after the fixed epoch 40 model.
- Raw control: rerun by this experiment. Historical raw results are not reused.
- Fold provenance: one explicit completed
  `outputs/multiscale_temporal1d_v1` run, requiring version
  `multiscale_temporal1d_v1.0.0`, `RUN_COMPLETE.json`, 492/492 completed tasks,
  and exact per-session manifest equality. The runner does not search a list of
  historical result candidates.

The paired formal plan contains 9 sessions, 2 variants, 3 seeds, 82 total
session-folds, and 492 fold-level training tasks.

## Preprocessing order and leakage boundary

Raw:

```text
clean4 extraction
-> arcsinh
-> pixel-wise z-score fit on outer-train blocks/all four frames only
-> unchanged Temporal1D
```

Spatial demean:

```text
clean4 extraction
-> arcsinh
-> for every sample and time frame: frame - mean_H,W(frame)
-> pixel-wise z-score fit on outer-train blocks/all four frames only
-> unchanged Temporal1D
```

The demean reduction axes are only `H,W` on one sample and one frame. It never
reduces across samples, time, cycles, folds, or labels. Each test frame is
demeaned using only its own pixels. Only the subsequent z-score has fitted
statistics, and its mean/std use the outer training fold exclusively. The test
fold is not used for normalization fit, epoch choice, model selection, response
maps, or hyperparameter selection.

## Local checks

```bash
.venv/bin/python -m pytest -q tests/test_spatial_demean_temporal1d.py
.venv/bin/python scripts/baselines/run_spatial_demean_temporal1d.py \
  --stage plan \
  --device cpu \
  --output-dir /tmp/spatial_demean_temporal1d_plan
.venv/bin/python scripts/baselines/run_spatial_demean_temporal1d.py \
  --stage sanity \
  --device cpu \
  --output-dir /tmp/spatial_demean_temporal1d_sanity
```

Sanity output is debug-only and is never accepted as a formal result.

## Formal server run (user-operated; GNU screen)

Do not start this run until code review approves the commit. On the lab server:

```bash
cd /data2/yuq1ngr/6.30project
git pull
git rev-parse HEAD
nvidia-smi
screen -S spatial_demean_temporal1d_v1
```

Inside GNU screen, first run sanity and inspect its completion marker:

```bash
/data2/yuq1ngr/conda_envs/fus/bin/python \
  scripts/baselines/run_spatial_demean_temporal1d.py \
  --stage sanity \
  --device cuda:0 \
  --output-dir outputs/spatial_demean_temporal1d_v1_sanity
```

Then generate and inspect the exact formal task plan:

```bash
/data2/yuq1ngr/conda_envs/fus/bin/python \
  scripts/baselines/run_spatial_demean_temporal1d.py \
  --stage plan \
  --device cuda:0 \
  --output-dir outputs/spatial_demean_temporal1d_v1
```

After confirming that the runner prints 9 sessions, 2 variants, 3 seeds, 82
folds, and 492 expected tasks, start the reviewed run:

```bash
/data2/yuq1ngr/conda_envs/fus/bin/python \
  scripts/baselines/run_spatial_demean_temporal1d.py \
  --stage full \
  --device cuda:0 \
  --review-approved \
  --output-dir outputs/spatial_demean_temporal1d_v1 \
  2>&1 | tee outputs/spatial_demean_temporal1d_v1/full_run.log
```

Detach with `Ctrl-a d`; resume the screen with
`screen -r spatial_demean_temporal1d_v1`. If the process stops, rerun the exact
same full command. A task is skipped only after all of its artifacts, identity
fingerprints, normalization audit, predictions, metrics, confusion matrix, and
40-epoch history validate. Missing or corrupt tasks remain pending.

## Formal outputs

- `task_plan.csv`, `task_plan_metadata.json`, `dataset_and_fold_audit.csv`
- `run_status.csv`
- `task_level_results.csv`
- `seed_summary.csv`
- `session_summary.csv`
- `overall_summary.csv`
- `paired_sign_flip.csv`
- `overfitting_audit.csv`
- `predictions.csv`, `confusion_matrices.csv`, `training_history.csv`
- `decision_rule_audit.json`
- `spatial_demean_temporal1d_report.md`
- `config.json`, `environment.json`, `git_state.json`, command records
- `RUN_COMPLETE.json`

`RUN_COMPLETE.json` is written only after all 492 tasks revalidate and all
required aggregate outputs exist. It records expected/completed task counts and
the frozen coverage dimensions.
