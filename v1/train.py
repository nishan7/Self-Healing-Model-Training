import os
import time
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


CKPT_PATH = Path("./elastic_ckpt.pt")
TOTAL_STEPS = 100000


def setup():
    # Optional network workaround for clusters with flaky IB
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
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def save_ckpt(rank, model, optimizer, step):
    if rank == 0:
        tmp = {
            "model": model.module.state_dict(),
            "optim": optimizer.state_dict(),
            "step": step,
        }
        torch.save(tmp, CKPT_PATH)


def load_ckpt(device, model, optimizer):
    if CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location=device)
        model.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        return int(state["step"]) + 1
    return 0


def main():
    local_rank, rank, world_size, device = setup()
    hostname = socket.gethostname()

    model = torch.nn.Linear(10, 10).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    start_step = 0

    try:
        start_step = load_ckpt(device, model, optimizer)

        if rank == 0:
            print(f"[INFO] started world_size={world_size} start_step={start_step}")

        for step in range(start_step, TOTAL_STEPS):
            x = torch.randn(32, 10, device=device)
            y = torch.randn(32, 10, device=device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            print(
                f"node={hostname} rank={rank} local_rank={local_rank} "
                f"world_size={world_size} step={step} loss={loss.item():.4f}",
                flush=True,
            )

            # checkpoint every few steps
            if step % 5 == 0:
                dist.barrier()
                save_ckpt(rank, model, optimizer, step)
                dist.barrier()

            # slow down so you can kill a node and watch recovery
            time.sleep(2)

    except KeyboardInterrupt:
        print(f"node={hostname} rank={rank} interrupted", flush=True)
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    main()