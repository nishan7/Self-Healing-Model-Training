#!/usr/bin/env python3
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from prometheus_client import Counter, Gauge, start_http_server


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def main() -> None:
    rank = get_int_env("SLURM_PROCID", 0)
    world_size = get_int_env("SLURM_NTASKS", 1)
    local_rank = get_int_env("SLURM_LOCALID", 0)
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = get_int_env("MASTER_PORT", 29500)
    backend = os.getenv("DIST_BACKEND", "gloo")
    metrics_port = get_int_env("METRICS_PORT", 9108)
    train_steps = get_int_env("TRAIN_STEPS", 200)
    scrape_hold_seconds = get_int_env("SCRAPE_HOLD_SECONDS", 120)
    sleep_per_step_ms = get_int_env("SLEEP_PER_STEP_MS", 0)

    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    device = torch.device("cpu")
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 16)).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model)
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if rank == 0:
        start_http_server(metrics_port)
        g_loss = Gauge("ddp_train_loss", "Average loss across all ranks")
        g_step_sec = Gauge("ddp_step_seconds", "Average step duration across all ranks")
        g_samples_per_sec = Gauge("ddp_samples_per_second", "Approximate global throughput")
        g_world = Gauge("ddp_world_size", "Number of distributed ranks")
        c_steps = Counter("ddp_steps_total", "Total completed train steps on rank 0")
        g_world.set(world_size)
        print(f"[rank0] Prometheus metrics: http://{master_addr}:{metrics_port}/metrics", flush=True)

    batch_size = 64
    for step in range(train_steps):
        step_t0 = time.time()

        x = torch.randn(batch_size, 128, device=device)
        y = torch.randn(batch_size, 16, device=device)
        pred = ddp_model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if sleep_per_step_ms > 0:
            time.sleep(sleep_per_step_ms / 1000.0)

        step_elapsed = time.time() - step_t0
        loss_tensor = torch.tensor([loss.item()], dtype=torch.float64, device=device)
        step_tensor = torch.tensor([step_elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            avg_loss = (loss_tensor.item() / world_size)
            avg_step = (step_tensor.item() / world_size)
            throughput = (batch_size * world_size / avg_step) if avg_step > 0 else 0.0
            g_loss.set(avg_loss)
            g_step_sec.set(avg_step)
            g_samples_per_sec.set(throughput)
            c_steps.inc()
            if step % 20 == 0 or step == train_steps - 1:
                print(
                    f"[rank0] step={step:04d} loss={avg_loss:.5f} "
                    f"step_sec={avg_step:.4f} samples_per_sec={throughput:.2f}",
                    flush=True,
                )

    dist.barrier()
    if rank == 0 and scrape_hold_seconds > 0:
        print(f"[rank0] Holding metrics endpoint for {scrape_hold_seconds}s", flush=True)
        time.sleep(scrape_hold_seconds)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
