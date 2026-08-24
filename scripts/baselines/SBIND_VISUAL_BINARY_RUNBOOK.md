# SBIND-adapted visual binary baseline v1 runbook

This runbook is for the laboratory server. Do not run `--stage full` on the local Mac.

## Completion semantics

A fold/seed task is reusable only when its `COMPLETE.json` and all result, prediction,
confusion, training-history, normalization, and model-config files pass validation. The runner
writes one task at a time and writes `COMPLETE.json` last.

The whole experiment is complete only when:

1. `run_status.csv` reports every planned task as `complete`;
2. all six aggregate CSV files exist; and
3. `RUN_COMPLETE.json` exists and reports equal `completed_tasks` and `total_tasks`.

Check without starting training:

```bash
python scripts/baselines/run_sbind_visual_binary.py \
  --stage status \
  --device cuda \
  --workers 2
```

The command prints `completed=... pending=... total=...` and up to five pending task paths.

## Server commands

```bash
cd /data2/yuq1ngr/6.30project
git pull origin main
. /home/xlab/anaconda3/conda/etc/profile.d/conda.sh
conda activate /data2/yuq1ngr/conda_envs/fus
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python scripts/baselines/run_sbind_visual_binary.py --stage sanity --device cuda --workers 0
```

Start the formal resumable run inside GNU screen:

```bash
mkdir -p outputs/sbind_visual_binary_v1/logs
screen -S sbind_visual_v1
python scripts/baselines/run_sbind_visual_binary.py \
  --stage full \
  --device cuda \
  --workers 2 \
  2>&1 | tee -a outputs/sbind_visual_binary_v1/logs/server_full.log
```

Detach with `Ctrl-a d`. Reattach with:

```bash
screen -r sbind_visual_v1
```

Running the same full command again is the supported resume operation. Valid completed tasks are
printed as `SKIP`; incomplete or corrupt tasks are rerun. Do not use the local sanity outputs in
formal aggregation.
