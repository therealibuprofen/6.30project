# Adaptive Mean/Std Nested-CV v1

## Scope

This experiment is training-only nested model selection between the two frozen
FCNN bottleneck temporal-statistics candidates. It is not a new architecture,
learned gate, ensemble, or mixture of experts.

The design was motivated by already observed session-dependent outer-test
behavior in the presence-binary Mean+Std development experiment. It must
therefore be reported as exploratory method development, not as independent
confirmatory evidence. No stimulus-type/four-class decoder is included.

## Frozen candidates

Both candidates reuse `fcnn_mean_std_temporal_statistics_v1.0.0` without model
or training changes:

- Mean-only: clean4 → per-frame arcsinh → training-fold-only all-frame
  pixel-wise z-score → unchanged shared FCNN encoder → `[B,4,3]` bottleneck →
  temporal mean → `Linear(3,2)`.
- Mean+Std: the same pipeline, followed by
  `concat(temporal_mean, temporal_std(correction=0))` → `Linear(6,2)`.

Optimizer, learning rate, weight decay, batch size 16, fixed epoch 40,
cross-entropy, initialization, and DataLoader RNG are unchanged.

## Scheme A selection

For every `session / outer_fold / seed`, the same seed is used for both inner
candidates. The existing formal outer split is frozen. Only its outer-training
cycles are passed to `grouped_cv_splits(..., max_folds=10)`.

Every inner fold independently fits pixel normalization on its inner-training
samples. Inner validation and outer test samples are excluded. Candidate score
is Balanced Accuracy computed once after concatenating the complete inner OOF
predictions; it is not the mean of fold BAs.

The only selection rule is:

```text
if inner_OOF_BA(mean_std) > inner_OOF_BA(mean_only):
    selected_variant = mean_std
else:
    selected_variant = mean_only
```

Thus an exact tie selects Mean-only.

## Stage separation

- `plan` validates the fixed provenance and writes immutable split/cache plans.
- `inner` has no fixed-result argument or outer-reader dependency. It trains or
  validates 2,550 strict unique caches and produces 4,332 parent-scoped inner
  evaluations.
- `select` accepts only inner artifacts and writes 246 hash-locked selections.
  It has no fixed-result argument or outer-reader dependency.
- `outer` locally imports the separate outer-reuse module after all selections
  exist. A capability object can read only the selected variant.
- `summarize` may then read both fixed controls for comparison.
- `full` runs the same stages in order and resumes only validated caches.
- `status` validates and reports current coverage without training.

The fixed run is pinned to commit
`96d26bfac562f934e74e49e059d9ea74e78617af`, run fingerprint
`34387397a0e9712caddb70f06f9199b272fd89fa94bedcf2782b30da771075ca`,
and config fingerprint
`f5b7f4175baebdefec5250586db2811cee33f54dfd2e0fc06b05c12e43a67a74`.

## Strict cache and resume

Training and evaluation keys are distinct canonical SHA-256 identities.
Training cache identity contains exact sample/cycle membership, dataset and
manifest hashes, candidate/model hashes, normalization-fit membership, full
training configuration, seed/RNG policy, runtime, and protocol fingerprint.

Evaluation identity additionally contains the exact validation samples/cycles
and parent outer fold. Its reader rejects any validation cycle outside the
current outer-training allowlist or inside the current outer-test denylist.

A training cache is complete only when checkpoint, float64 normalization
statistics, epoch history, metadata, completion marker, hashes, and strict
state-dict loading all validate. Partial or corrupt cache entries are not
skipped. A locked selection with changed content or hash is rejected rather
than overwritten.

## Expected plan

```text
82 outer folds
722 inner split definitions
425 unique inner training cycle sets
4,332 logical candidate/seed inner jobs
2,550 unique cached inner trainings
246 locked per-seed selections
246 selected formal outer outputs reused after lock
```

Any mismatch stops before formal training.

## Preregistered decision

Continuation requires all four conditions:

1. Adaptive overall BA is greater than fixed Mean-only overall BA.
2. Adaptive overall BA is at least fixed Mean+Std overall BA.
3. Adaptive strong-3 BA is no more than 0.010 below fixed Mean-only strong-3.
4. Adaptive weak-6 gain over Mean-only retains at least 75% of the fixed
   Mean+Std weak-6 gain.

No result-dependent threshold changes or automatic follow-up experiment are
allowed.

## Server commands

All commands below are single-line commands. Formal background execution uses
GNU screen only.

Plan:

```bash
python scripts/baselines/run_adaptive_mean_std_nestedcv.py --stage plan
```

Sanity:

```bash
python scripts/baselines/run_adaptive_mean_std_nestedcv.py --stage sanity
```

Formal resumable run after code-review approval:

```bash
screen -dmS adaptive_mean_std_nestedcv_v1 bash -lc 'cd /data2/yuq1ngr/6.30project && python scripts/baselines/run_adaptive_mean_std_nestedcv.py --stage full --review-approved > outputs/adaptive_mean_std_nestedcv_v1/full_run.log 2>&1'
```

Status:

```bash
python scripts/baselines/run_adaptive_mean_std_nestedcv.py --stage status
```
