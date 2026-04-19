import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 1. Initialize the elastic process group
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    
    # 2. Setup a dummy model
    model = torch.nn.Linear(10, 10).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    print(f"\n[INFO] Node {global_rank} successfully joined! Current cluster size: {world_size} GPUs\n")

    # 3. Dummy Training Loop
    for step in range(1, 10000):
        # Generate dummy data
        data = torch.randn(32, 10).cuda(local_rank)
        target = torch.randn(32, 10).cuda(local_rank)
        
        # Fake forward/backward pass
        output = model(data)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()
        
        # Slow down the loop so you can read the output
        time.sleep(2) 
        
        if global_rank == 0:
            print(f"--> Training Step {step} | Active GPUs (WORLD_SIZE): {world_size}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()