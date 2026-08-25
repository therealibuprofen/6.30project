# Local + Global Residual Spatial Mamba proposed v1

## Scope

This is the first proposed-method validation candidate only: `CNN + Gated Residual Spatial Mamba + Temporal 1D-CNN`. It does not add multi-scale features, ROI, GLM, vascular masks, Mamba2/Mamba3, or any data/fold/training change.

The runner reads the existing formal `cnn2d_temporal1d`, Spatial Mamba, Factorized Transformer, and FCNN mean-pool results. It trains only `local_global_residual_mamba` after external review approval.

## Frozen architecture

The shared per-frame stem is exactly the reviewed Transformer/Spatial-Mamba stem: `Conv2d-BatchNorm-GELU` with channels 1→16→32→64 and stride 2, followed by `AdaptiveAvgPool2d((8,32))`. It returns `F_local [B,4,64,8,32]`.

The global branch uses the exact Spatial Mamba v1.1 path: two-dimensional row `[1,8,1,64]` plus column `[1,1,32,64]` positions, row-major 256-token flattening, two bidirectional shared-weight official Mamba blocks, `d_model=64`, `d_state=16`, `d_conv=4`, `expand=2`.

Fusion is fixed as `F_fused = F_local + sigmoid(g) * (F_global - F_local)`. There is exactly one trainable scalar `g`, initialized to `-2.0`, so initial alpha is approximately `0.11920292`. There are no session/channel/pixel/label-conditioned gates. Every task stores initial alpha, post-epoch alpha, final alpha, and the last-five-epoch mean.

After spatial mean pooling, the fused frame sequence is `[B,4,64]`. The formal `cnn2d_temporal1d` receives 2048-D SmallCNN frame features, so its first `Conv1d(2048,64,...)` cannot accept the proposed 64-D features. Proposed v1 makes only the necessary input-width adaptation to `Conv1d(64,64,kernel_size=3,padding=1)`. BatchNorm, ReLU, the second `Conv1d(64,64,3,padding=1)`, temporal pooling, dropout, and `Linear(64,2)` classifier are directly transferred from the reviewed formal implementation. No projection layer or new TCN is added.

The frozen formal parameter audit is 116,579 trainable parameters: CNN stem 23,520; two-dimensional positions 2,560; Spatial Mamba blocks 65,536; scalar gate 1; Temporal 1D head 24,832; classifier 130. The batch-16 CUDA preflight and every completed-task validator require this exact total.

## Frozen protocol and decision

Data are the exact formal clean4 binary samples and formal cycle-grouped folds for sessions 626, 628, 708, 709, 710, 807, 813, 817, and 822. Normalization is fitted only on the outer training fold. Training is fixed at seeds 0/1/2, 40 epochs, AdamW, learning rate `1e-3`, weight decay `1e-3`, batch size 16, with no test/validation selection.

Strong sessions are fixed as 708/709/710; the other six are weak sessions. “Not notably down” is pre-fixed as delta BA ≥ -0.02. Strong-session recovery over pure Spatial Mamba requires mean delta ≥ +0.02 and at least two of three sessions improving. Continuing the Mamba route also requires positive nine-session mean delta over Temporal 1D-CNN, at least 6/9 sessions non-decreasing, at least 2/3 strong sessions not notably down, and fewer severe-overfit sessions than pure Spatial Mamba. The runner writes the rule audit and stops without starting a next model.

## Integrity and execution boundary

Every proposed task is one `session × seed × fold`. It writes predictions, confusion matrix, training history including alpha, normalization audit, architecture/parameter audit, gate audit, and `COMPLETE.json` atomically per artifact. Resume accepts a task only after strict fingerprint, environment, identity, metric, OOF artifact, 40-epoch history, and gate-history validation.

Formal execution is locked unless `--review-approved` is passed after external code review. The formal interpreter must be under `/data2/yuq1ngr/conda_envs/fus_mamba`, CUDA must be available, and the fixed batch-16 forward/backward preflight must pass before the task plan is created.

The proposed run identity retains all Spatial Mamba v1.1 runtime signature fields and additionally fingerprints `h5py`, because it is a direct clean4 HDF5 reader dependency. The audited server environment currently reports `h5py 3.11.0`. This is documentation and identity validation only; the runner never installs packages.

GNU screen is the only documented background runner. After review approval, the single-line server command is:

```bash
cd /data2/yuq1ngr/6.30project && mkdir -p outputs/local_global_residual_mamba_v1/logs && screen -dmS residual_mamba_v1 bash -lc 'CUDA_VISIBLE_DEVICES=1 /data2/yuq1ngr/conda_envs/fus_mamba/bin/python scripts/baselines/run_local_global_residual_mamba.py --stage full --review-approved --device cuda --workers 0 >> outputs/local_global_residual_mamba_v1/logs/server_full.log 2>&1'
```

No formal command may be run before review greenlight. No dependency is installed automatically.
