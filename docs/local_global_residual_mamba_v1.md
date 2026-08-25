# Local + Global Residual Spatial Mamba proposed v1

## Scope

This is the first proposed-method validation candidate plus its two required same-backbone mechanistic controls. It does not add multi-scale features, ROI, GLM, vascular masks, Mamba2/Mamba3, or any data/fold/training change.

After external review approval, the runner trains exactly three models defined by one shared class and a frozen `fusion_mode`:

- `same_stem_local_temporal1d` (`local_only`): select `F_local`; no spatial position, Mamba, or trainable gate.
- `same_stem_global_mamba_temporal1d` (`global_only`): select the exact `F_global`; fixed effective alpha 1; no trainable gate.
- `local_global_residual_mamba` (`gated_local_global`): select `F_local + sigmoid(g) * (F_global - F_local)` with `g=-2` initially.

The existing formal `cnn2d_temporal1d`, Spatial Mamba, Factorized Transformer, and FCNN mean-pool results are read-only external baselines. In particular, the old Temporal 1D-CNN is not described as a same-backbone ablation.

## Frozen architecture

The shared per-frame stem is exactly the reviewed Transformer/Spatial-Mamba stem: `Conv2d-BatchNorm-GELU` with channels 1→16→32→64 and stride 2, followed by `AdaptiveAvgPool2d((8,32))`. It returns `F_local [B,4,64,8,32]`.

The global branch uses the exact Spatial Mamba v1.1 path: two-dimensional row `[1,8,1,64]` plus column `[1,1,32,64]` positions, row-major 256-token flattening, two bidirectional shared-weight official Mamba blocks, `d_model=64`, `d_state=16`, `d_conv=4`, `expand=2`.

The three modes share the same stem, spatial reduction, Temporal 1D-CNN, and classifier definitions. Only the global branch and fusion behavior differ. The gated candidate remains exactly `F_fused = F_local + sigmoid(g) * (F_global - F_local)`. It has one trainable scalar `g`, initialized to `-2.0`, so initial alpha is approximately `0.11920292`; there are no session/channel/pixel/label-conditioned gates.

After spatial mean pooling, every selected frame sequence is `[B,4,64]`. The old formal `SmallCNNFrameEncoder.feature_dim` is `16 * 4 * 8 = 512`, so `cnn2d_temporal1d` begins with `Conv1d(512,64,kernel_size=3,padding=1)`. The shared new skeleton makes only the required 512→64 input-width adaptation: `Conv1d(64,64,kernel_size=3,padding=1)`. BatchNorm, ReLU, the second `Conv1d(64,64,3,padding=1)`, temporal pooling, dropout, and `Linear(64,2)` classifier are directly transferred from the reviewed formal implementation. The old baseline source and old results are not modified.

The frozen formal parameter audits are:

- local-only: 48,482 = CNN stem 23,520 + Temporal 1D head 24,832 + classifier 130;
- global-only: 116,578 = stem 23,520 + 2D positions 2,560 + Mamba 65,536 + Temporal head 24,832 + classifier 130;
- gated: 116,579 = global-only plus one scalar gate.

Parameter matching is intentionally not attempted with meaningless layers. The batch-16 CUDA preflight and each completed-task validator require the exact total for its model.

## Frozen protocol and decision

Data are the exact formal clean4 binary samples and formal cycle-grouped folds for sessions 626, 628, 708, 709, 710, 807, 813, 817, and 822. Normalization is fitted only on the outer training fold. Training is fixed at seeds 0/1/2, 40 epochs, AdamW, learning rate `1e-3`, weight decay `1e-3`, batch size 16, with no test/validation selection.

The primary mechanistic comparisons are gated vs same-stem local-only, gated vs same-stem global-only, and same-stem global-only vs same-stem local-only. Each reports the nine-session mean and median delta, improved/tied/worsened count, exact two-sided sign-flip p-value, and pre-defined strong/weak means. External comparisons against old Temporal 1D-CNN, pure Spatial Mamba, Factorized Transformer, and FCNN mean-pool are reported separately.

Strong sessions are fixed as 708/709/710; the other six are weak sessions. “Not notably down” is pre-fixed as delta BA ≥ -0.02. The gate interpretation is exploratory and never feeds back into training. Continuing the route requires the pre-registered mechanistic, strong-session, external Temporal1D, pure-Mamba recovery, overfit, and gate checks. The runner writes the rule audit and stops without starting a next model.

## Integrity and execution boundary

Every task is one `session × model × seed × fold`. It writes predictions, confusion matrix, training history, normalization audit, architecture/parameter audit, control audit, and `COMPLETE.json` atomically per artifact. Only the gated model stores a learned alpha for each epoch plus initial/final/last-five summaries. Local-only records fixed `effective_alpha=0`, global-only records fixed `effective_alpha=1`, and neither fixed control contains a fabricated learned-alpha curve. Resume accepts a task only after strict fingerprint, environment, identity, metric, OOF artifact, 40-epoch history, and mode-specific control validation.

Formal execution is locked unless `--review-approved` is passed after external code review. The formal interpreter must be under `/data2/yuq1ngr/conda_envs/fus_mamba`, CUDA must be available, and the fixed batch-16 forward/backward preflight must pass independently for all three models before the task plan is created. With the current fold counts the expected plan is 738 tasks; the runner computes the exact total from the audited folds.

The proposed run identity retains all Spatial Mamba v1.1 runtime signature fields and additionally fingerprints `h5py`, because it is a direct clean4 HDF5 reader dependency. The audited server environment currently reports `h5py 3.11.0`. This is documentation and identity validation only; the runner never installs packages.

GNU screen is the only documented background runner. After review approval, the single-line server command is:

```bash
cd /data2/yuq1ngr/6.30project && mkdir -p outputs/local_global_residual_mamba_v1/logs && screen -dmS residual_mamba_v1 bash -lc 'CUDA_VISIBLE_DEVICES=1 /data2/yuq1ngr/conda_envs/fus_mamba/bin/python scripts/baselines/run_local_global_residual_mamba.py --stage full --review-approved --device cuda --workers 0 >> outputs/local_global_residual_mamba_v1/logs/server_full.log 2>&1'
```

No formal command may be run before review greenlight. No dependency is installed automatically.
