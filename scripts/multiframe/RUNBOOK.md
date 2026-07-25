# Multiframe Benchmark Runbook

## Fresh Clone Setup

Clone the code repository:

```bash
git clone git@github.com:therealibuprofen/6.30project.git
cd 6.30project
```

Create a Python environment. On a GPU server, install the PyTorch build that matches the CUDA driver first if needed; otherwise `requirements.txt` is enough for CPU/basic CUDA-enabled installs already provided by the environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Set a writable matplotlib cache directory:

```bash
export MPLCONFIGDIR=/tmp/ultrasound_decoding_mpl_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$MPLCONFIGDIR"
```

## Data Placement

The Git repo intentionally does not include raw or processed data.

Preferred: copy the already exported clean4 block data to:

```text
processed_data/block_sequences_v1/
```

The directory should contain:

```text
session_708_blocks.h5
session_708_block_metadata.csv
session_709_blocks.h5
session_709_block_metadata.csv
session_710_blocks.h5
session_710_block_metadata.csv
session_807_blocks.h5
session_807_block_metadata.csv
session_813_blocks.h5
session_813_block_metadata.csv
session_817_blocks.h5
session_817_block_metadata.csv
session_822_blocks.h5
session_822_block_metadata.csv
label_mapping.json
```

If only raw `.mat` files are available, copy them to:

```text
data/{session}/*.mat
```

Then regenerate and inspect `block_sequences_v1`:

```bash
.venv/bin/python scripts/data/export_block_sequences.py \
  --output-dir processed_data/block_sequences_v1

.venv/bin/python scripts/data/inspect_block_sequences.py \
  --output-dir processed_data/block_sequences_v1
```

## Preflight

```bash
.venv/bin/python -m unittest discover -s tests

.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage dry-run \
  --sessions 708 709 710 807 813 817 822 \
  --tasks binary stimulus_type

.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage smoke \
  --overwrite
```

## Stage 2

Strong-session binary benchmark.

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage benchmark \
  --tasks binary \
  --sessions 708 709 710 \
  --methods pca_lda_flat4 cpca_lda_flat4 cnn2d_meanpool cnn2d_lstm cnn2d_temporal1d single_frame_late_fusion \
  --seeds 0 1 2 \
  --max-epochs 40 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --weight-decay 1e-3 \
  --device auto \
  --reuse-compatible-results
```

## Stage 3

All-session binary benchmark. If Stage 2 completed in the default output directory, this skips completed sessions and runs the remaining sessions.

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage benchmark \
  --tasks binary \
  --sessions 708 709 710 807 813 817 822 \
  --methods pca_lda_flat4 cpca_lda_flat4 cnn2d_meanpool cnn2d_lstm cnn2d_temporal1d single_frame_late_fusion \
  --seeds 0 1 2 \
  --max-epochs 40 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --weight-decay 1e-3 \
  --device auto \
  --reuse-compatible-results
```

## Stage 4

All-session stimulus_type benchmark.

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage benchmark \
  --tasks stimulus_type \
  --sessions 708 709 710 807 813 817 822 \
  --methods pca_lda_flat4 cpca_lda_flat4 cnn2d_meanpool cnn2d_lstm cnn2d_temporal1d single_frame_late_fusion \
  --seeds 0 1 2 \
  --max-epochs 40 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --weight-decay 1e-3 \
  --device auto \
  --reuse-compatible-results
```

## Rebuild Aggregate Outputs

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage aggregate-only \
  --tasks binary \
  --sessions 708 709 710 807 813 817 822

.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage aggregate-only \
  --tasks stimulus_type \
  --sessions 708 709 710 807 813 817 822
```

`aggregate-only` does not train models. It rebuilds aggregate CSVs/plots plus the audit-only outputs:

- `aggregate/sampling_time_audit.csv`
- `aggregate/sampling_time_audit_summary.json`
- `aggregate/overfitting_audit.csv`
- `aggregate/overfitting_method_summary.csv`
- `aggregate/checkpoint_manifest.csv`
- `aggregate/order_sensitivity_predictions.csv` when future runs saved per-block order predictions
- `aggregate/order_sensitivity_oof_summary.csv` when per-block order predictions are available

The sampling-time audit reports the nominal clean4 grids observed in the exported block metadata. For example, stimulus blocks may use `[10,14,18,22]` while no-stimulus blocks may use `[8,12,16,20]`. This is reported as the phase relationship between the original 4-second sampling grid and 30-second block boundaries; the audit does not automatically label it as leakage and does not modify or interpolate data.

Legacy `block_clean4_binary_v1` neural runs did not save checkpoints or per-block order-sensitivity predictions. Their checkpoint manifest rows are therefore marked `not_available_for_legacy_run`, and no per-block order predictions are reconstructed.

## FCNN Multiframe Binary Benchmark

Static implementation/data audit:

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage dry-run \
  --tasks binary \
  --sessions 708 709 710 807 813 817 822 \
  --methods fcnn_late_fusion fcnn_meanpool fcnn_lstm
```

Smoke test, isolated from the legacy CNN run:

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage smoke \
  --tasks binary \
  --sessions 710 \
  --methods fcnn_late_fusion fcnn_meanpool fcnn_lstm \
  --seeds 0 \
  --max-epochs 2 \
  --run-name block_clean4_binary_fcnn_smoke_v1 \
  --overwrite
```

Formal FCNN binary benchmark:

```bash
.venv/bin/python scripts/multiframe/run_multiframe_benchmark.py \
  --stage benchmark \
  --tasks binary \
  --sessions 708 709 710 807 813 817 822 \
  --methods fcnn_late_fusion fcnn_meanpool fcnn_lstm \
  --seeds 0 1 2 \
  --max-epochs 40 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --weight-decay 1e-3 \
  --device auto \
  --run-name block_clean4_binary_fcnn_v1
```

Each neural fold saves:

- `session_{session}/checkpoints/{method}/seed_{seed}/fold_{fold}/checkpoint.pt`
- `session_{session}/checkpoint_manifest.csv`

Order-sensitive methods additionally save:

- `session_{session}/order_sensitivity_predictions.csv`
- `session_{session}/order_sensitivity_oof_summary.csv`

Use `--reuse-compatible-results` only when you want to skip an existing session after validating that the split, data version, input shape, optimizer settings, seeds, normalization, model architecture, and config version match exactly.

## Merge Legacy CNN and FCNN Runs

```bash
.venv/bin/python scripts/multiframe/merge_multiframe_runs.py \
  --base-run results/runs/multiframe/block_clean4_binary_v1 \
  --additional-run results/runs/multiframe/block_clean4_binary_fcnn_v1 \
  --output-run results/runs/multiframe/block_clean4_binary_all_models_v1
```

The merge stops if task, sessions, data version, cycle splits, class mapping, normalization protocol, max epochs, optimizer settings, or seed lists differ. It does not invent legacy checkpoints or legacy per-block order predictions.

## Independent Sensitivity Experiments

Epoch sensitivity, separate from the formal 40-epoch benchmark:

```bash
.venv/bin/python scripts/multiframe/run_epoch_sensitivity.py \
  --sessions 710 807 813 817 822 \
  --methods cnn2d_lstm cnn2d_temporal1d fcnn_lstm \
  --epochs 10 20 40 \
  --seeds 0 1 2 \
  --device auto
```

Frame-count/window ablation, separate from the formal benchmark:

```bash
.venv/bin/python scripts/multiframe/run_frame_count_ablation.py \
  --sessions 708 709 710 \
  --methods cnn2d_meanpool cnn2d_lstm cnn2d_temporal1d fcnn_meanpool fcnn_lstm \
  --seeds 0 1 2 \
  --max-epochs 40 \
  --device auto
```

## Outputs

- `results/runs/multiframe/block_clean4_binary_v1/`
- `results/runs/multiframe/block_clean4_stimulus_type_v1/`

The script does not overwrite completed session outputs unless `--overwrite` is passed. Use `--overwrite` only for the specific interrupted or intentionally repeated session command.

Main aggregate files:

- `aggregate/multiframe_master_summary.csv`
- `aggregate/multiframe_method_summary.csv`
- `aggregate/multiframe_fold_summary.csv`
- `aggregate/multiframe_completeness_report.csv`
- `aggregate/multiframe_order_sensitivity.csv`
- `aggregate/multiframe_vs_singleframe_reference.csv`
- `aggregate/*.png`
- `aggregate/*.pdf`
