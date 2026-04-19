import os
import time
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- HPC NETWORK FIXES ---
# Disable IPv6 and virtual networks to prevent timeout hangs
os.environ["NCCL_DISABLE_IPV6"] = "1"
os.environ["GLOO_DISABLE_IPV6"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"
os.environ["GLOO_SOCKET_IFNAME"] = "^lo,docker0"

# THE MAGIC BULLET: Disable InfiniBand to prevent hardware crashes on disconnect!
os.environ["NCCL_IB_DISABLE"] = "1"

# --- DUMMY DATASET ---
class FakeData(Dataset):
    def __len__(self): return 1000
    def __getitem__(self, idx): return idx, torch.randn(10), torch.randn(10)

def main():
    # 1. Initialize Elastic Process Group
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    hostname = socket.gethostname() 

    torch.cuda.set_device(local_rank)
    
    # 2. Setup Model & Data
    model = DDP(torch.nn.Linear(10, 10).cuda(local_rank), device_ids=[local_rank])
    dataset = FakeData()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    if global_rank == 0:
        print(f"\n[INFO] Cluster established! Total GPUs: {world_size}\n")

    # 3. Training Loop
    for epoch in range(100):
        sampler.set_epoch(epoch)
        
        for step, (indices, data, target) in enumerate(dataloader):
            output = model(data.cuda(local_rank))
            loss = torch.nn.MSELoss()(output, target.cuda(local_rank))
            loss.backward()
            
            # Print status to prove who is doing what
            print(f"[{hostname} | Rank {global_rank}] Epoch {epoch} Step {step} | Processing Data IDs: {indices.tolist()[0:3]}...")
            
            time.sleep(2) # Slow it down so we can easily interrupt it

    dist.destroy_process_group()

if __name__ == "__main__":
    main()