export NCCL_DEBUG=INFO
export GLOO_DISABLE_IPV6=1
export NCCL_DISABLE_IPV6=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export GLOO_SOCKET_IFNAME=^lo,docker0


module load python3
module load ml/torch/2.6

torchrun \
    --nnodes=1:2 \
    --nproc_per_node=1 \
    --max_restarts=10 \
    --rdzv_id=my_elastic_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=172.16.1.76:29500 \
    train.py