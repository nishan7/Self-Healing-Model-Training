import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.cuda.empty_cache()

def setup():
    """Initializes the distributed backend."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

class DummyModel(nn.Module):
    """A trivial model just to occupy the GPU."""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.net(x)

def main():
    local_rank = setup()
    rank = int(os.environ["RANK"]) # Global rank across all nodes
    
    # Initialize model and move it to the correct GPU
    model = DummyModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    checkpoint_path = "dummy_checkpoint.pt"
    start_epoch = 0

    # Wait for all nodes to be ready before checking files
    dist.barrier()

    # --- CHECKPOINT LOADING LOGIC ---
    if os.path.exists(checkpoint_path):
        chkpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.module.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer'])
        # Resume from the epoch *after* the last saved one
        start_epoch = chkpt['epoch'] + 1 
        
        if rank == 0:
            print(f"\n[INFO] Checkpoint found! --- RESUMING FROM EPOCH {start_epoch} ---\n")
    else:
        if rank == 0:
            print("\n[INFO] No checkpoint found. Starting fresh.\n")

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, 100):
        # Generate fake data
        data = torch.randn(32, 10).cuda(local_rank)
        target = torch.randn(32, 10).cuda(local_rank)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # --- CHECKPOINT SAVING LOGIC ---
        if rank == 0:
            print(f"Completed Epoch {epoch}/100. Saving checkpoint...")
            
            # ATOMIC SAVE: Save to a temp file first, then rename.
            # This prevents a corrupted checkpoint if SLURM kills the job mid-save.
            temp_path = f"{checkpoint_path}.tmp"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, temp_path)
            os.replace(temp_path, checkpoint_path)
        
        # Ensure all nodes wait until Node 0 finishes saving
        dist.barrier()
        
        # ARTIFICIAL DELAY: Pauses for 5 seconds so you have time to interrupt it!
        time.sleep(5) 
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()