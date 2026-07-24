# Multiframe Benchmark Runbook

Run from the project root:

```bash
cd /path/to/6.30project
export MPLCONFIGDIR=/tmp/ultrasound_decoding_mpl_cache
mkdir -p "$MPLCONFIGDIR"
```

Optional preflight:

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

Stage 2: strong-session binary benchmark.

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
  --device auto
```

Stage 3: all-session binary benchmark. If Stage 2 completed in the default output directory, this skips completed sessions and runs the remaining sessions.

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
  --device auto
```

Stage 4: all-session stimulus_type benchmark.

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
  --device auto
```

Rebuild aggregate outputs only:

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

Default output directories:

- `results/runs/multiframe/block_clean4_binary_v1/`
- `results/runs/multiframe/block_clean4_stimulus_type_v1/`

The script does not overwrite completed session outputs unless `--overwrite` is passed. Use `--overwrite` only for the specific interrupted or intentionally repeated session command.
