# FCNN Mean+Std Temporal Statistics v1

This experiment tests only whether population temporal standard deviation in the
formal FCNN bottleneck improves clean4 within-session presence decoding.

## Frozen implementation

The historical `FCNNMeanPool` does not average images. Its actual path is:

```text
clean4 [B,4,H,W]
-> per-frame arcsinh
-> pixel z-score fitted on outer-training blocks and all four training frames
-> [B,4,1,H,W]
-> unchanged shared FCNNFrameEncoder per frame
-> bottleneck Z [B,4,3]
-> temporal reduction
-> classifier
```

`mean_only` directly instantiates the existing `FCNNMeanPool` and applies
`mean_t(Z) -> Linear(3,2)`. `mean_std` reuses the identical encoder and applies
`concat(mean_t(Z), std_t(Z, correction=0)) -> Linear(6,2)`. There is no
secondary bottleneck normalization. The only parameter increase is six weights
from the classifier input expansion; its bias and every encoder parameter are
unchanged.

The test fold is transformed with statistics fitted solely from the outer
training fold. Test data do not enter normalization, feature scaling, early
stopping, or model selection. Both variants are rerun and paired by fold and
seed; historical FCNN results are only a sanity reference.

Final train accuracy is the real-training-sample accuracy recorded at fixed
epoch 40, averaged across folds. Best epoch remains descriptive and never enters
the train-test gap. OOF balanced accuracy is reconstructed from all held-out
predictions for a session/seed.

## Local review workflow

All commands are intentionally single-line:

```bash
.venv/bin/python -m pytest -q tests/test_fcnn_temporal_statistics.py
```

```bash
.venv/bin/python scripts/baselines/run_fcnn_mean_std_temporal_statistics.py --stage sanity --device cpu --workers 0 --sanity-epochs 1 --output-dir outputs/fcnn_mean_std_temporal_statistics_v1_sanity
```

```bash
.venv/bin/python scripts/baselines/run_fcnn_mean_std_temporal_statistics.py --stage plan --device cpu --workers 0 --output-dir outputs/fcnn_mean_std_temporal_statistics_v1_plan
```

## Formal server workflow

Codex does not run the formal experiment. After review, use GNU screen:

```bash
cd /data2/yuq1ngr/6.30project && nvidia-smi
```

```bash
cd /data2/yuq1ngr/6.30project && /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_mean_std_temporal_statistics.py --stage plan --device cuda --workers 2 --output-dir outputs/fcnn_mean_std_temporal_statistics_v1
```

```bash
cd /data2/yuq1ngr/6.30project && /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_mean_std_temporal_statistics.py --stage sanity --device cuda --workers 0 --sanity-epochs 1 --output-dir outputs/fcnn_mean_std_temporal_statistics_v1
```

```bash
screen -dmS fcnn_mean_std_v1 bash -lc 'cd /data2/yuq1ngr/6.30project && /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_mean_std_temporal_statistics.py --stage full --device cuda --workers 2 --review-approved --output-dir outputs/fcnn_mean_std_temporal_statistics_v1 2>&1 | tee outputs/fcnn_mean_std_temporal_statistics_v1/full_run.log'
```

```bash
screen -ls
```

```bash
screen -r fcnn_mean_std_v1
```

```bash
tail -n 100 /data2/yuq1ngr/6.30project/outputs/fcnn_mean_std_temporal_statistics_v1/full_run.log
```

```bash
cd /data2/yuq1ngr/6.30project && /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_mean_std_temporal_statistics.py --stage status --device cuda --workers 2 --output-dir outputs/fcnn_mean_std_temporal_statistics_v1
```

Rerunning the identical `screen` full command resumes only tasks whose complete
artifact set, fingerprints, identities, metrics, and SHA-256 hashes validate.
Corrupted or partial tasks are recomputed. `RUN_COMPLETE.json` is written only
after all 492 tasks validate and every required aggregate output exists.
