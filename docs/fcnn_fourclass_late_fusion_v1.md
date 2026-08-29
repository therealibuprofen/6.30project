# Four-Class FCNN Late-Fusion v1

This experiment answers one frozen question: whether the four physical block identities contain stable cross-cycle decoding information that is strong enough to justify a later hierarchical multi-task experiment. It introduces no model or hyperparameter search.

## Frozen task and data

The formal task is `block_identity_4class`, with `0=grating`, `1=stop_after_grating`, `2=dot`, and `3=static`. Labels are derived directly from the exporter `block_name`; every complete cycle contributes one block of every class. No block is filtered. The input is the existing `clean4` tensor, four frames per block.

The nine session cycle counts are 626:8, 628:8, 708:6, 709:22, 710:18, 807:12, 813:10, 817:20, and 822:10. Chance is 0.25. Outer train/test cycle membership is asserted fold by fold against the completed `outputs/fcnn_mean_std_temporal_statistics_v1` `mean_only` task plan before any training. The resulting frozen schedule is 82 folds × seeds 0,1,2 = 246 tasks.

## Model, training, and inference

The only model is the established single-frame FCNN: `MaxPool2d(2,2) -> Flatten(16000) -> Linear(16000,3) -> ReLU -> Linear(3,4)`. Its 48,019 parameters differ from the 48,011-parameter binary model only by the eight extra weights/biases in the final head.

Each training block becomes four independent frame samples with the same block label. Splitting happens first at cycle/block level, so frames from one cycle cannot cross the outer boundary. The approved normalization implementation applies `arcsinh`, fits pixel-wise mean/std from outer-training frames only, and applies those fixed statistics to held-out frames. AdamW uses lr 0.001, weight decay 0.001, batch size 16, cross entropy, exactly 40 epochs, and no early stopping.

At inference, each held-out frame produces a four-class softmax vector. The arithmetic mean of exactly four vectors is the block probability; argmax produces one block prediction. Frame probabilities are retained for audit.

## Evaluation and completion

The primary session/seed score concatenates every outer held-out block before calculating balanced accuracy. Fold-BA averaging is not primary. The run also records accuracy, macro-F1, recalls, raw and row-normalized confusion matrices, within-coarse errors (`grating↔dot`, `stop_after_grating↔static`), and cross-coarse errors.

The secondary binary diagnostic collapses probabilities without retraining: stimulus is `P(grating)+P(dot)` and nonstimulus is `P(stop_after_grating)+P(static)`. Existing binary FCNN late-fusion results are read only during final aggregation for descriptive comparison; they cannot affect training or selection. Strong sessions remain 708/709/710 and weak sessions remain the other six.

The pre-registered gate is conjunction of four-class nine-session mean BA ≥0.35, at least six session BAs strictly >0.30, and collapsed-binary nine-session mean BA ≥0.55. Its only possible decisions are `four_class_signal_sufficient_for_multitask_experiment` and `four_class_signal_insufficient_for_multitask_experiment`.

Every formal task saves an epoch-40 loadable checkpoint, exact identities and hashes, normalization arrays/config, training diagnostics, block predictions, and frame predictions. Resume skips only a task whose complete marker, hashes, checkpoint, metadata, output head, class mapping, fold membership, probability coverage, and fusion calculation all validate. `RUN_COMPLETE.json` is written only after all 246 tasks and all aggregate audits pass.

## Commands

Run a plan first:

```bash
python scripts/baselines/run_fcnn_fourclass_late_fusion.py --stage plan --project-root /data2/yuq1ngr/6.30project --output-dir outputs/fcnn_fourclass_late_fusion_v1
```

Run a one-epoch, one-fold sanity task (it never creates a formal completion marker):

```bash
python scripts/baselines/run_fcnn_fourclass_late_fusion.py --stage sanity --project-root /data2/yuq1ngr/6.30project --output-dir outputs/fcnn_fourclass_late_fusion_v1 --device cuda --sanity-epochs 1
```

After external review approval only, run formal work:

```bash
python scripts/baselines/run_fcnn_fourclass_late_fusion.py --stage full --project-root /data2/yuq1ngr/6.30project --output-dir outputs/fcnn_fourclass_late_fusion_v1 --device cuda --review-approved
```

Inspect strict resumable status:

```bash
python scripts/baselines/run_fcnn_fourclass_late_fusion.py --stage status --project-root /data2/yuq1ngr/6.30project --output-dir outputs/fcnn_fourclass_late_fusion_v1
```
