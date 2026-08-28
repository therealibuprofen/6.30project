# Canonical-midpoint single-frame FCNN v1

This experiment reconstructs a block-level single-frame FCNN comparator from the frozen historical `fcnn_late_fusion` checkpoints. It never trains, fine-tunes, or updates a model. The formal runner is CPU-only.

## Scientific definition

Historical training is preserved: all four clean4 frames from each training block were treated as independent frame-wise supervised samples. At evaluation, each held-out block contributes exactly one canonical frame:

```text
k* = argmin_k |clean4_relative_time_s[k] - 15 s|
```

An exact tie is resolved in favor of the earlier timestamp. Selection is computed dynamically from saved clean4 timestamps and never reads a label, prediction, or performance metric. No position search is performed.

The selected frame is transformed as follows:

```text
float32 canonical frame
→ arcsinh
→ z-score with checkpoint-saved outer-training-fold pixel mean/std
→ historical 48,011-parameter FCNN in eval/no-grad mode
→ one two-class probability vector
```

The canonical path contains no probability/logit averaging, vote, temporal mean/std, or other fusion. One held-out block produces one prediction. Session/seed OOF Balanced Accuracy is calculated only after concatenating all outer-held-out block predictions; fold BAs are diagnostic and are never averaged into the primary metric.

As a provenance gate, the same loaded checkpoint and saved normalization also reconstruct historical late fusion for every held-out block: all four normalized clean4 frames are forwarded independently, softmax is applied per frame, and the four probability vectors are averaged. The resulting 1,368 block probabilities, predictions, identities, truths and session/seed OOF BAs must match the formal historical aggregate (`atol=2e-6`, `rtol=1e-6`). This tight tolerance only permits float32 CPU/GPU matrix-reduction noise; predictions and metadata must match exactly. The real-checkpoint CPU sanity comparison had maximum probability difference `0.0`. The audit is saved before validation is enforced; any failure blocks all formal summaries. Final A-versus-B results use the reconstructed predictions, while the aggregate remains an external reference.

## Comparisons and interpretation

- Canonical single-frame versus historical late fusion uses the same checkpoint and isolates canonical one-frame inference versus four-frame probability averaging.
- Late fusion versus MeanPool compares frame-wise CE plus probability averaging with block-wise end-to-end CE plus latent mean aggregation.
- Canonical single-frame versus MeanPool is the primary overall baseline comparison, but the training units/objectives also differ. It must not be described as a pure temporal-mean ablation.

Recommended manuscript name: **Canonical-midpoint single-frame FCNN with frame-wise clean4 training**. It may be described as a **Berthon-style single-frame FCNN adapted to the same clean4 frame-wise training pool**, but not as a complete reproduction of the Berthon et al. training protocol.

## Safety gates

The formal stage requires `--review-approved`, rejects every device except CPU, and validates all 246 checkpoint hashes and payloads before the first formal forward. It also validates the frozen MeanPool output/model protocol, clean4 binary labels, seed/fold/task coverage, epoch 40, train-fold-only normalization, implementation version and exact task membership. MeanPool `RUN_COMPLETE.json`, `config.json`, `task_plan.csv` and `predictions.csv` hashes are recorded. Canonical and MeanPool OOF identities/truths must match exactly. Any missing/mismatched task or comparator stops the entire run; there is no retraining fallback.

The formal pass contains 1,368 held-out canonical-frame predictions. The registered train-side diagnostic evaluates 11,640 outer-training block/checkpoint combinations, for 13,008 canonical/diagnostic single-frame forwards. The same-checkpoint late-fusion provenance gate adds 5,472 held-out frame forwards, producing 1,368 reconstructed block predictions. Total model frame forwards are therefore 18,480. These are batched CPU inference operations, not optimizer steps.

## Commands

All commands are single-line commands.

Plan:

```text
CUDA_VISIBLE_DEVICES="" /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_canonical_single_frame.py --stage plan --device cpu
```

One-checkpoint CPU sanity after code review preparation:

```text
CUDA_VISIBLE_DEVICES="" /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_canonical_single_frame.py --stage sanity --device cpu
```

Formal CPU reconstruction after ChatGPT code review approval:

```text
CUDA_VISIBLE_DEVICES="" /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_canonical_single_frame.py --stage full --device cpu --review-approved
```

Status:

```text
CUDA_VISIBLE_DEVICES="" /data2/yuq1ngr/conda_envs/fus/bin/python scripts/baselines/run_fcnn_canonical_single_frame.py --stage status --device cpu
```
