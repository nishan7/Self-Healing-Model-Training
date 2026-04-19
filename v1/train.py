import os
import time
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 1. HPC Network Fixes (built directly into the script)
os.environ["NCCL_DISABLE_IPV6"] = "1"
os.environ["GLOO_DISABLE_IPV6"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"
os.environ["GLOO_SOCKET_IFNAME"] = "^lo,docker0"

# 2. Dataset that reports its index (DistributedSampler splits these across GPUs)
class FakeData(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return idx, torch.randn(10), torch.randn(10)

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)
    model = DDP(torch.nn.Linear(10, 10).cuda(local_rank), device_ids=[local_rank])
    
    dataset = FakeData()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    dist.barrier()
    if global_rank == 0:
        print(
            f"ready | host={hostname} | dataset={len(dataset)} | "
            f"ranks={world_size} | steps/rank/epoch={len(dataloader)} | "
            f"samples/rank/epoch={sampler.num_samples}"
        )

    for epoch in range(100):
        sampler.set_epoch(epoch)
        if global_rank == 0:
            print(f"epoch {epoch}")

        for step, (indices, data, target) in enumerate(dataloader):
            output = model(data.cuda(local_rank))
            loss = torch.nn.MSELoss()(output, target.cuda(local_rank))
            loss.backward()

            ids = indices.tolist()
            print(
                f"node={hostname} gpu={local_rank} rank={global_rank} "
                f"epoch={epoch} step={step} ids={ids}"
            )

            time.sleep(2)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()