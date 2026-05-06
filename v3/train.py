import os
import time
import socket
import signal
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

CKPT_PATH = Path(__file__).resolve().parent / "elastic_ckpt.pt"

TOTAL_STEPS = 100000
CKPT_EVERY = 5

terminate_requested = False


def handle_sigterm(signum, frame):
    global terminate_requested
    terminate_requested = True
    print("[signal] SIGTERM received. Will checkpoint and exit.", flush=True)


def setup():
    signal.signal(signal.SIGTERM, handle_sigterm)

    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "^lo,docker0")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "^lo,docker0")

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

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
        print(f"[checkpoint] saved step={step} path={CKPT_PATH}", flush=True)


def load_ckpt(model, optimizer):
    if CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location="cpu")
        model.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        start_step = int(state["step"]) + 1
        print(f"[checkpoint] loaded start_step={start_step} path={CKPT_PATH}", flush=True)
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

        print(
            f"[startup] node={hostname} rank={rank} local_rank={local_rank} "
            f"world_size={world_size} start_step={start_step} "
            f"restart_count={os.environ.get('SLURM_RESTART_COUNT', '0')} "
            f"job_id={os.environ.get('SLURM_JOB_ID', 'na')}",
            flush=True,
        )

        for step in range(start_step, TOTAL_STEPS):
            x = torch.randn(32, 10, device=device)
            y = torch.randn(32, 10, device=device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            print(
                f"[train] node={hostname} rank={rank} step={step} loss={loss.item():.4f}",
                flush=True,
            )

            if step % CKPT_EVERY == 0 or terminate_requested:
                dist.barrier()
                save_ckpt(rank, model, optimizer, step)
                dist.barrier()

            if terminate_requested:
                print(f"[exit] rank={rank} clean exit after checkpoint", flush=True)
                break

            time.sleep(2)

    finally:
        cleanup()


if __name__ == "__main__":
    main()