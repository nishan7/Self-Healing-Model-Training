import os
import time
import socket
import signal
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

CKPT_PATH = Path(__file__).resolve().parent / "elastic_ckpt.pt"

TOTAL_STEPS = 100000
CKPT_EVERY = 5

terminate_requested = False


def log_event(log_type, subtype, message):
    print(f"[{log_type}][{subtype}] {message}", flush=True)


def handle_sigterm(signum, frame):
    global terminate_requested
    terminate_requested = True
    log_event("system", "signal", "SIGTERM received. Will checkpoint and exit.")


def setup():
    signal.signal(signal.SIGTERM, handle_sigterm)

    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    hostname = socket.gethostname()
    log_event(
        "system",
        "pre_init",
        f"node={hostname} pid={os.getpid()} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR', 'na')} "
        f"MASTER_PORT={os.environ.get('MASTER_PORT', 'na')} "
        f"RANK={os.environ.get('RANK', 'na')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK', 'na')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE', 'na')}",
    )

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=3))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA device is visible to this process.")
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {num_gpus} CUDA device(s) visible. "
            "Set torchrun --nproc-per-node <= GPUs per node, or request more GPUs in Slurm."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log_event(
        "system",
        "post_init",
        f"node={hostname} rank={rank} local_rank={local_rank} "
        f"world_size={world_size} cuda_devices={num_gpus}",
    )

    return local_rank, rank, world_size, device


def cleanup():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def save_ckpt(rank, model, optimizer, step):
    if rank == 0:
        tmp_path = CKPT_PATH.with_suffix(".tmp")
        payload = {
            "model": model.module.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, CKPT_PATH)
        log_event("training", "checkpoint", f"saved step={step} path={CKPT_PATH}")


def load_ckpt(model, optimizer):
    if CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location="cpu")
        model.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        start_step = int(state["step"]) + 1
        log_event("training", "checkpoint", f"loaded start_step={start_step} path={CKPT_PATH}")
        return start_step
    return 0


def main():
    local_rank, rank, world_size, device = setup()
    hostname = socket.gethostname()

    model = torch.nn.Linear(10, 10).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    try:
        dist.barrier()
        start_step = load_ckpt(model, optimizer)
        dist.barrier()

        log_event(
            "system",
            "startup",
            f"node={hostname} rank={rank} local_rank={local_rank} "
            f"world_size={world_size} start_step={start_step} "
            f"restart_count={os.environ.get('SLURM_RESTART_COUNT', '0')} "
            f"job_id={os.environ.get('SLURM_JOB_ID', 'na')}",
        )

        for step in range(start_step, TOTAL_STEPS):
            x = torch.randn(32, 10, device=device)
            y = torch.randn(32, 10, device=device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            log_event(
                "training",
                "training_step",
                f"node={hostname} rank={rank} step={step} loss={loss.item():.4f}",
            )

            if step % CKPT_EVERY == 0 or terminate_requested:
                dist.barrier()
                save_ckpt(rank, model, optimizer, step)
                dist.barrier()

            if terminate_requested:
                log_event("training", "exit", f"rank={rank} clean exit after checkpoint")
                break

            time.sleep(2)

    finally:
        cleanup()


if __name__ == "__main__":
    main()
