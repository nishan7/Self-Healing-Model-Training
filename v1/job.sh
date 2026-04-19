#!/bin/bash
#SBATCH --job-name=dummy_test
#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1    # Request 1 A100 GPU per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --output=train_log_%j.out

# --- Load your Python/PyTorch environment here ---
# e.g., module load miniconda3
# e.g., conda activate my_pytorch_env
module load python3
module load ml/torch/2.6

# Find the IP address of the master node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Master node is $head_node with IP $head_node_ip"

export CUDA_VISIBLE_DEVICES=0

# Launch the distributed job
# In your job.sh, update the srun command to match the GRES
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    --master_addr=$head_node_ip \
    --master_port=29500 \
    train.py