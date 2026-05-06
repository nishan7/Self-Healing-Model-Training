# v5 (v3 + v4)

`v5` combines:
- `v3` reliability features: Slurm requeue flow, checkpoint/resume, DDP guardrails, structured logs.
- `v4` workload + telemetry: CNN synthetic DDP load and per-node GPU/IB telemetry CSV polling.

## Files
- `run.sh`: Slurm entrypoint (2 nodes, 1 GPU per node).
- `train.py`: DDP CNN training with checkpoint/resume.
- `telemetry_poller.py`: background per-node telemetry poller.

## Submit
```bash
cd /home/018280561/Self-Healing-Model-Training/v5
sbatch run.sh
```

## Useful overrides
```bash
MASTER_PORT=29501 TRAIN_EPOCHS=2000 TRAIN_BATCH_SIZE=16 CKPT_EVERY=5 sbatch run.sh
```

Telemetry knobs:
- `POLL_INTERVAL` (default `0.5`)
- `POLL_DURATION` (default `86400`)
- `IB_DEVICE` (default `mlx4_0`)
- `IB_PORT` (default `1`)

## Outputs
- Slurm logs: `v5/logs/ddp-lens-v5-<jobid>.out|err`
- Results: `v5/results/<jobid>/`
- Checkpoint: `v5/results/<jobid>/elastic_ckpt.pt`
- Telemetry CSVs: `v5/results/<jobid>/telemetry_<node>_<jobid>.csv`
