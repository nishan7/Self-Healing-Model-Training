#!/bin/bash
#SBATCH --job-name=dummy_multinode
#SBATCH --partition=gpuqs
#SBATCH --nodes=2                # CHANGED: Request 2 nodes
#SBATCH --gres=gpu:a100:1        # Request 1 GPU per node (2 GPUs total)
#SBATCH --ntasks-per-node=1      # Runs 1 torchrun instance per node
#SBATCH --cpus-per-task=8
#SBATCH --requeue                # Allows the job to be requeued (restarted) safely
#SBATCH --output=train_log_%j.out

module load python3
module load ml/torch/2.6

# Find the IP address of the master node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Testing connectivity to master node ($head_node_ip) from $(hostname)"
nc -zv $head_node_ip 29500
echo "Master node is $head_node with IP $head_node_ip"

export CUDA_VISIBLE_DEVICES=0

# Launch the distributed job across 2 nodes
# srun ensures this exact command runs on both Node 0 and Node 1
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    train.py