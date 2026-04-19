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

# 2. Dataset that reports its Index
class FakeData(Dataset):
    def __len__(self): 
        return 100  # Small dataset so we can see epochs finish quickly
        
    def __getitem__(self, idx): 
        # By returning 'idx', we can prove the nodes get different data!
        return idx, torch.randn(10), torch.randn(10)

def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Grab the HPC node name (e.g., cs002 or cs003)
    hostname = socket.gethostname() 

    torch.cuda.set_device(local_rank)
    model = DDP(torch.nn.Linear(10, 10).cuda(local_rank), device_ids=[local_rank])
    
    dataset = FakeData()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Make everyone wait here so the logs don't print out of order
    dist.barrier()
    if global_rank == 0:
        print(f"\n--- Cluster ready with {world_size} GPUs! ---")


    # 3. Verbose Training Loop
    for epoch in range(100):
        sampler.set_epoch(epoch)
        
        dist.barrier()
        if global_rank == 0:
            print(f"\n========== STARTING EPOCH {epoch} ==========")
        dist.barrier()
        
        for step, (indices, data, target) in enumerate(dataloader):
            # Run the step
            output = model(data.cuda(local_rank))
            loss = torch.nn.MSELoss()(output, target.cuda(local_rank))
            loss.backward()
            
            # THE REVEAL: Every GPU prints its location and its data chunk!
            print(f"[Node: {hostname} | GPU Rank: {global_rank}] Step {step} | Data IDs: {indices.tolist()}")
            
            time.sleep(2) # Pause for 1 second so you can watch it flow

    dist.destroy_process_group()

if __name__ == "__main__":
    main()