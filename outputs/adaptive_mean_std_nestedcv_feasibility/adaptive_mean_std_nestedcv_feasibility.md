# Adaptive Mean/Std Nested-CV Feasibility Plan

## Scope and source provenance

This is a read-only protocol and task-enumeration audit. No model was trained, no
GPU job was started, and the FCNN or temporal-statistics implementation was not
changed.

The enumeration uses the completed formal run
`outputs/fcnn_mean_std_temporal_statistics_v1`:

- `RUN_COMPLETE.json`: `status=complete`, `492/492` tasks, 82 folds, 3 seeds,
  2 variants.
- Git commit recorded by the run:
  `96d26bfac562f934e74e49e059d9ea74e78617af`.
- Run fingerprint:
  `34387397a0e9712caddb70f06f9199b272fd89fa94bedcf2782b30da771075ca`.
- Configuration fingerprint shared by all 492 tasks:
  `f5b7f4175baebdefec5250586db2811cee33f54dfd2e0fc06b05c12e43a67a74`.
- Model implementation version:
  `fcnn_mean_std_temporal_statistics_v1.0.0`.
- Formal protocol: clean4, binary presence, arcsinh, train-fold-only all-frame
  pixel-wise z-score, batch size 16, 40 fixed epochs, seeds 0/1/2, no early
  stopping.

The completed run contains 492 valid result/COMPLETE pairs: 246 `mean_only`
and 246 `mean_std`. It stores predictions, metrics, training history, and
normalization audits, but no `.pt`, `.pth`, or `.ckpt` model weights.

## Proposed nested protocol

For each `session / outer_fold / outer_seed`:

1. Freeze the outer test cycles and make them inaccessible to the selector.
2. Apply the existing `grouped_cv_splits(..., max_folds=10)` logic only to the
   outer-training cycle IDs. With at most 10 cycles this is leave-one-cycle-out;
   above 10 cycles, sorted unique cycle IDs are partitioned into 10 validation
   groups by `numpy.array_split`.
3. For each fixed candidate (`mean_only`, `mean_std`) and each inner fold, fit
   arcsinh plus pixel-wise z-score statistics on inner-training samples only,
   train for the same fixed 40 epochs, and evaluate only the corresponding
   inner-validation cycles.
4. Concatenate all inner-validation predictions for a candidate and compute one
   inner OOF balanced accuracy, matching the current formal OOF aggregation.
   Do not average best epochs and do not inspect outer-test performance.
5. Select `mean_std` only when its inner OOF BA is strictly greater than that of
   `mean_only`; ties select `mean_only`.
6. Lock the selection, then use a from-scratch model trained on all outer-training
   cycles with normalization refit on those cycles. Evaluate the held-out outer
   test cycles once.

The existing outer-task result for the selected candidate is mathematically the
same as step 6 when all fingerprints match. It may therefore supply the final
outer predictions and metrics after selection is locked. It cannot supply a
deployable checkpoint because the formal run did not save weights.

## Actual outer-fold composition and dynamic inner folds

Cycle IDs below are zero-based. Each entry in `outer test groups` corresponds to
one outer fold in order; the outer-training set is its complement within that
session. Full train/test memberships for every fold are in
`adaptive_mean_std_task_plan.csv`.

| session | outer folds | outer test groups, F1 onward | outer-training cycle counts | inner-fold counts | total inner folds |
|---|---:|---|---|---|---:|
| 626 | 8 | 0; 1; 2; 3; 4; 5; 6; 7 | 7 each | 7 each | 56 |
| 628 | 8 | 0; 1; 2; 3; 4; 5; 6; 7 | 7 each | 7 each | 56 |
| 708 | 6 | 0; 1; 2; 3; 4; 5 | 5 each | 5 each | 30 |
| 709 | 10 | 0,1,2; 3,4,5; 6,7; 8,9; 10,11; 12,13; 14,15; 16,17; 18,19; 20,21 | 19,19,20,20,20,20,20,20,20,20 | 10 each | 100 |
| 710 | 10 | 0,1; 2,3; 4,5; 6,7; 8,9; 10,11; 12,13; 14,15; 16; 17 | 16,16,16,16,16,16,16,16,17,17 | 10 each | 100 |
| 807 | 10 | 0,1; 2,3; 4; 5; 6; 7; 8; 9; 10; 11 | 10,10,11,11,11,11,11,11,11,11 | 10 each | 100 |
| 813 | 10 | 0; 1; 2; 3; 4; 5; 6; 7; 8; 9 | 9 each | 9 each | 90 |
| 817 | 10 | 0,1; 2,3; 4,5; 6,7; 8,9; 10,11; 12,13; 14,15; 16,17; 18,19 | 18 each | 10 each | 100 |
| 822 | 10 | 0; 1; 2; 3; 4; 5; 6; 7; 8; 9 | 9 each | 9 each | 90 |
| **Total** | **82** | | | | **722** |

The 722 explicit inner split memberships and all disjointness checks are in
`adaptive_mean_std_split_audit.csv`.

## Exact training counts

Let `K_o` be the dynamic number of inner folds for outer fold `o`. Here:

`sum_o K_o = 722`, and there are 82 outer folds.

### Scheme A: full per-seed selection

For every outer fold and each of three seeds:

`2 candidates * K_o inner trainings + 1 selected outer refit`.

Therefore:

- Inner trainings: `3 * 2 * 722 = 4,332`.
- Outer refits: `3 * 82 = 246`.
- Total: **4,578 model trainings**.
- Relative to 492: **9.3049x** by logical task count.

### Scheme B: shared selector seed 0

For every outer fold, selection is run once, followed by three outer refits:

- Inner trainings: `2 * 722 = 1,444`.
- Outer refits: `3 * 82 = 246`.
- Total: **1,690 model trainings**.
- Relative to 492: **3.4350x** by logical task count.

Session totals are:

| session | inner-fold definitions | Scheme A total | Scheme B total |
|---|---:|---:|---:|
| 626 | 56 | 360 | 136 |
| 628 | 56 | 360 | 136 |
| 708 | 30 | 198 | 78 |
| 709 | 100 | 630 | 230 |
| 710 | 100 | 630 | 230 |
| 807 | 100 | 630 | 230 |
| 813 | 90 | 570 | 210 |
| 817 | 100 | 630 | 230 |
| 822 | 90 | 570 | 210 |
| **Total** | **722** | **4,578** | **1,690** |

## Runtime estimate

The completed 492-task run on the recorded RTX 3080 took 73.08 minutes from
the first to the last task completion. No benchmark was run for this plan.

- Linear task-count estimate: Scheme A about 11.3 hours; Scheme B about 4.2
  hours.
- Accounting for smaller inner training sets, the aggregate 40-epoch optimizer
  step ratios are approximately 8.594x and 3.198x, corresponding to about 10.5
  and 3.9 hours on the same otherwise-idle machine.
- Practical uncached expectation: approximately **10.5-11.3 hours for A** and
  **3.9-4.2 hours for B**. Scheduling, I/O, and GPU contention can dominate this
  estimate.

## Strict cache analysis

Across the 722 inner-fold definitions there are 425 unique
`(session, inner_train_cycle_ids)` sets. The remaining 297 definitions repeat a
training set in another outer fold; the maximum multiplicity is two.

| session | inner definitions | unique inner training sets | repeated definitions |
|---|---:|---:|---:|
| 626 | 56 | 28 | 28 |
| 628 | 56 | 28 | 28 |
| 708 | 30 | 15 | 15 |
| 709 | 100 | 72 | 28 |
| 710 | 100 | 64 | 36 |
| 807 | 100 | 64 | 36 |
| 813 | 90 | 45 | 45 |
| 817 | 100 | 64 | 36 |
| 822 | 90 | 45 | 45 |
| **Total** | **722** | **425** | **297** |

A strict training cache can therefore reduce new inner trainings to:

- Scheme A: `425 * 2 candidates * 3 seeds = 2,550`.
- Scheme B: `425 * 2 candidates * 1 selector seed = 850`.

This saves 41.14% of inner model trainings and is a meaningful reduction. It is
safe only if a repeated job has identical sample membership, normalization fit,
candidate, seed/RNG behavior, code/config, and runtime-relevant protocol. The
trained model can be shared, but validation prediction/metric records must be
separately keyed by the requested inner-validation cycle IDs. For example, the
same training set can arise with cycles `a` and `b` omitted, while `a` is the
outer test cycle in one parent fold and `b` is the outer test cycle in another.
Neither parent's selector may read the other parent's outer-test metric.

The safest training-cache key is a SHA-256 of canonical metadata containing at
least:

- session and exact sorted training sample IDs plus cycle IDs;
- dataset/source content hash, fold-manifest hash, clean4 and binary-label/block
  mapping;
- candidate (`mean_only` or `mean_std`) and complete model/source/version hash;
- arcsinh and pixel-z-score implementation/config, epsilon, and the exact IDs
  used to fit normalization;
- optimizer, learning rate, weight decay, batch size, epochs, loss, no-early-
  stopping flag, and initialization/DataLoader seed policy;
- seed and relevant deterministic/runtime fingerprint.

An evaluation-cache key must additionally contain the training-cache key, exact
validation sample/cycle IDs, metric implementation hash, and the parent
outer-fold identity. Split membership and deterministic preprocessing metadata
may be cached independently, but normalization arrays must remain keyed to the
exact inner-training sample set.

## Reuse of the completed 492-task run

- **Inner selection:** zero existing tasks are reusable. Exact enumeration found
  no nested inner-training cycle set equal to any completed formal outer-training
  cycle set. Thus reuse is `0/4,332` inner tasks for A and `0/1,444` for B.
- **Adaptive outer evaluation:** after selection is irrevocably locked, exactly
  one of the two completed candidate tasks exists for every
  `session / outer_fold / seed`. Therefore **246/246 adaptive final outer task
  outputs are reusable**: 50% of the 492 existing candidate tasks and 100% of
  the adaptive final outer-evaluation requirement. This is 5.37% of A's 4,578
  logical tasks or 14.56% of B's 1,690 logical tasks.
- **Fixed controls:** all 492 existing candidate outputs are exact fixed-control
  comparators, provided the future implementation pins the recorded fingerprints.
- **Weights:** zero checkpoints are reusable because the formal output contains
  no model-weight files. If a persisted selected outer checkpoint is required,
  all 246 selected outer models must be refit; this does not invalidate reuse of
  their existing predictions for the adaptive BA analysis.

Combining strict duplicate-training cache with existing outer-result reuse leaves
2,550 new unique inner trainings for A or 850 for B. Estimated same-GPU
wall-clock becomes roughly 6.1-6.3 hours for A or 2.0-2.1 hours for B, excluding
implementation and audit overhead.

## Leakage audit

All 722 enumerated splits pass:

- `outer_test_cycles intersect inner_train_cycles = empty`;
- `outer_test_cycles intersect inner_validation_cycles = empty`;
- `inner_train_cycles intersect inner_validation_cycles = empty`;
- `inner_train_cycles union inner_validation_cycles = outer_train_cycles`;
- inner-validation cycles are absent from normalization-fit cycle IDs.

No leakage is present in the enumerated split plan. A future implementation must
still prevent these concrete leakage paths:

1. fitting inner normalization on all outer-training cycles;
2. reading existing outer BA/predictions before `selected_variant` is locked;
3. using known session-level Mean+Std effects, an outer-test-informed threshold,
   or outer-test-informed tie breaking;
4. sharing validation metrics across parent folds merely because their model
   training sets match;
5. allowing current outer-test predictions generated for another parent fold to
   enter selection;
6. selecting epochs, checkpoints, or any tuning parameter with outer-test data.

The result reader should enforce a state transition: first write a signed
selection artifact containing both inner OOF BAs and the selected variant, then
allow lookup of only the selected existing outer result. Outer predictions and
metrics must not be loaded into the selector process before that artifact exists.

## Recommended scheme

For a formal experiment, use **Scheme A (full per-seed nested selection)**. It
keeps each adaptive outer result paired to its own training seed, quantifies the
selector's seed stability directly, and avoids making the scientific conclusion
depend on seed 0 alone. The strict cache makes its cost material but feasible on
the observed hardware. Scheme B is acceptable only as an explicitly labeled
compute-saving pilot; it cannot answer whether different seeds frequently choose
different variants.

## Scientific naming

The restrained name is **nested-CV temporal-statistics model selection** or,
when “adaptive” is operationally useful, **training-only adaptive temporal-
representation selection**. It is not a new neural architecture, learned gate,
or mixture-of-experts decoder. “Adaptive decoder” should be qualified as model
selection between two fixed decoders.

## Minimum future output schema

1. Split/task manifest: session, outer fold/seed, selector seed, candidate, inner
   fold, all four cycle-ID sets, sample counts, normalization fingerprint,
   training/evaluation cache keys, protocol/task fingerprints, and status.
2. Inner predictions and metrics: sample ID, true label, probability/prediction,
   candidate, inner fold, validation BA, concatenated inner OOF BA, and complete
   inner-fold coverage assertion.
3. Locked selection: `inner_BA_mean_only`, `inner_BA_mean_std`, delta, tie flag,
   selected variant, rule version, selection fingerprint/timestamp, and an
   assertion that no outer result was read.
4. Outer result: selected variant, new-or-reused provenance, selected task
   fingerprint, outer predictions, OOF BA, fixed-epoch-40 real-train accuracy,
   and train-test gap.
5. Summary: fixed Mean-only, fixed Mean+Std, adaptive BA; per-session candidate
   selection proportions; cross-seed agreement; overall/strong-3/weak-6 results;
   and a small paired statistical audit.
6. Resume/provenance: strict artifact hashes, cache-hit provenance, fold/seed
   coverage, environment and source hashes, and `RUN_COMPLETE` only after all
   selection and outer-evaluation artifacts validate.

## Feasibility decision

This is worth implementing as a tightly scoped nested model-selection experiment:
the observed candidate effects are large and session-dependent, the selector uses
training cycles only, fixed controls and all outer predictions already exist, and
strict caching reduces Scheme A to 2,550 new unique inner trainings. The main risk
is selection variance with only 5-20 outer-training cycles, which is precisely why
Scheme A and selection-stability reporting are preferable.

Before any future run, keep the simple rule and preregister the requested compact
comparison: adaptive must beat fixed Mean-only overall, match or exceed fixed
Mean+Std overall, avoid clear strong-3 harm, and retain most weak-6 Mean+Std
benefit. Do not derive thresholds from the existing outer-test results.

**No training was started as part of this feasibility audit.**
