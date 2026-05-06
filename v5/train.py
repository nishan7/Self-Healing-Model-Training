import os
import signal
import socket
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ---- Prototype knobs (edit directly) ----
TOTAL_STEPS = 100000
CKPT_EVERY = 5
SLEEP_SECONDS = 2
BATCH_SIZE = 32
FEATURE_DIM = 10
LR = 1e-3
# -----------------------------------------

CKPT_PATH = Path(os.environ.get("CKPT_PATH", Path(__file__).resolve().parent / "elastic_ckpt.pt"))
terminate_requested = False


def log_event(log_type, subtype, message):
    print(f"[{log_type}][{subtype}] {message}", flush=True)


def handle_sigterm(signum, frame):
    global terminate_requested
    terminate_requested = True
    log_event("system", "signal", "SIGTERM received. Will checkpoint and exit.")


def setup():
    signal.signal(signal.SIGTERM, handle_sigterm)

    host = socket.gethostname()
    log_event(
        "system",
        "pre_init",
        f"node={host} pid={os.getpid()} master={os.environ.get('MASTER_ADDR', 'na')}:{os.environ.get('MASTER_PORT', 'na')} "
        f"rank={os.environ.get('RANK', 'na')} local_rank={os.environ.get('LOCAL_RANK', 'na')} world_size={os.environ.get('WORLD_SIZE', 'na')}",
    )

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=3))

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No CUDA device visible.")
    if local_rank >= gpu_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} but only {gpu_count} CUDA device(s) visible.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log_event(
        "system",
        "post_init",
        f"node={host} rank={rank} local_rank={local_rank} world_size={world_size} cuda_devices={gpu_count}",
    )
    return rank, device


def cleanup():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def save_ckpt(rank, model, optimizer, step):
    if rank != 0:
        return
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_PATH.with_suffix(".tmp")
    torch.save({"model": model.module.state_dict(), "optim": optimizer.state_dict(), "step": step}, tmp)
    os.replace(tmp, CKPT_PATH)
    log_event("training", "checkpoint", f"saved step={step} path={CKPT_PATH}")


def load_ckpt(model, optimizer):
    if not CKPT_PATH.exists():
        return 0
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.module.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optim"])
    start_step = int(state["step"]) + 1
    log_event("training", "checkpoint", f"loaded start_step={start_step} path={CKPT_PATH}")
    return start_step


def main():
    rank, device = setup()
    host = socket.gethostname()

    model = torch.nn.Linear(FEATURE_DIM, FEATURE_DIM).to(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    try:
        dist.barrier()
        start_step = load_ckpt(model, optimizer)
        dist.barrier()

        log_event(
            "system",
            "startup",
            f"node={host} rank={rank} start_step={start_step} restart_count={os.environ.get('SLURM_RESTART_COUNT', '0')} job_id={os.environ.get('SLURM_JOB_ID', 'na')}",
        )

        for step in range(start_step, TOTAL_STEPS):
            x = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
            y = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            log_event("training", "training_step", f"node={host} rank={rank} step={step} loss={loss.item():.4f}")

            if step % CKPT_EVERY == 0 or terminate_requested:
                dist.barrier()
                save_ckpt(rank, model, optimizer, step)
                dist.barrier()

            if terminate_requested:
                log_event("training", "exit", f"rank={rank} clean exit after checkpoint")
                break

            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)

    finally:
        cleanup()


if __name__ == "__main__":
    main()
