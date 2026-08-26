# Lightweight Multi-scale Spatial CNN + Temporal 1D-CNN v1

## Scope and frozen question

This candidate asks only whether two spatial receptive fields improve clean4 within-session decoding over one receptive field. It contains no Mamba, Transformer, attention, ROI, GLM, branch gate, residual path, multi-scale temporal module, or hyperparameter search.

The mechanistic comparison trains exactly `same_backbone_single_scale` and `multiscale_temporal1d`. Existing formal Temporal 1D-CNN, FCNN mean-pool, Factorized Transformer, Spatial Mamba, and Gated Mamba results are read only and are never retrained by this runner.

The Gated Mamba external baseline is provenance-locked to the completed three-model run at `outputs/local_global_residual_mamba_v1_1`. Both `proposed_summary.csv` and `overfitting_comparison.csv` are read from that directory only, and the loader requires its `RUN_COMPLETE.json` to have `status: complete`. The gated summary must contain exactly one `local_global_residual_mamba` row for each of the nine frozen sessions. The earlier `outputs/local_global_residual_mamba_v1` run is not an automatic fallback.

## Formal Temporal 1D-CNN source audit

The formal source is `SmallCNNFrameEncoder` plus `CNN2DTemporal1D` in `src/ultrasound_decoding/multiframe/models.py`.

Per frame, the formal encoder is:

1. `Conv2d(1,8,kernel_size=(5,9),stride=1,padding=(2,4))`
2. `BatchNorm2d(8)`, `ReLU`
3. `MaxPool2d((2,4))`
4. `Conv2d(8,16,kernel_size=(5,7),stride=1,padding=(2,3))`
5. `BatchNorm2d(16)`, `ReLU`
6. `AdaptiveAvgPool2d((4,8))`, `Flatten`

The spatial output is `[B*T,16,4,8]`, and `SmallCNNFrameEncoder.feature_dim = 16*4*8 = 512`.

The formal temporal head is `Conv1d(512,64,kernel_size=3,padding=1)`, `BatchNorm1d(64)`, `ReLU`, `Conv1d(64,64,kernel_size=3,padding=1)`, `ReLU`, `AdaptiveAvgPool1d(1)`, and `Flatten`. The classifier is `Dropout(0.25)` followed by `Linear(64,2)`.

The formal training configuration is AdamW, learning rate `1e-3`, weight decay `1e-3`, batch size 16, fixed 40 epochs, seeds 0/1/2, cross-entropy, and no validation/test-fold model selection.

## Controlled spatial encoders

Both new models use one unified class and preserve the exact formal first stage through `MaxPool2d((2,4))`.

`same_backbone_single_scale` replaces the formal second `(5×7)` convolution with `Conv2d(8,16,3,padding=1,dilation=1)`.

`multiscale_temporal1d` uses two parallel branches receiving the same first-stage tensor: local `Conv2d(8,8,3,padding=1,dilation=1)` and context `Conv2d(8,8,3,padding=2,dilation=2)`. Their outputs are concatenated directly to 16 channels. There is no projection or learnable fusion.

Both then use the same `BatchNorm2d(16)`, `ReLU`, and `AdaptiveAvgPool2d((4,8))`. Both produce exactly 512 frame features. Their temporal head and classifier are direct module instances from the formal `CNN2DTemporal1D` definition and are not reconstructed or changed.

The two controlled models naturally have the same parameter count, 112,562: frame encoder 1,584, Temporal 1D head 110,848, classifier 130. The existing formal Temporal 1D-CNN has 115,890 parameters because its second spatial convolution is `(5×7)`. No artificial parameter-matching layer is added.

## Frozen evaluation and stopping rule

Data, labels, complete-cycle restriction, clean4 frames, formal cycle folds, outer-train-only normalization, seeds, optimizer, batch size, and epoch count are unchanged. The primary metric is OOF Balanced Accuracy.

The primary mechanistic comparison is multi-scale versus same-backbone single-scale. The primary practical comparison is multi-scale versus existing formal Temporal 1D-CNN. Both report nine-session mean and median delta, improved/tied/worsened counts, exact two-sided sign-flip p-value, and pre-defined strong/weak means. Strong sessions are 708/709/710; weak sessions are 626/628/807/813/817/822.

Continuation requires positive mean delta over the single-scale control, at least 6/9 non-decreasing sessions, at least 2/3 strong sessions no worse than -0.02, no mean decline versus formal Temporal1D, and no worse severe-overfit count than the complex Mamba route. The runner stops after reporting and never launches another architecture automatically.

## Execution boundary

Local work is limited to static checks and tiny CPU sanity. Formal execution requires CUDA and the existing `/data2/yuq1ngr/conda_envs/fus` interpreter. The full stage is locked behind `--review-approved`; no dependency is installed.

GNU screen is the only documented background runner. After explicit review approval, the single-line server command is:

```bash
cd /data2/yuq1ngr/6.30project && mkdir -p outputs/multiscale_temporal1d_v1/logs && screen -dmS multiscale_t1d_v1 bash -lc 'CUDA_VISIBLE_DEVICES=1 /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_multiscale_temporal1d.py --stage full --review-approved --device cuda --workers 0 >> outputs/multiscale_temporal1d_v1/logs/server_full.log 2>&1'
```

No formal command may be run before the review greenlight.
