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
    --max_restarts=20 \
    --rdzv_id=my_elastic_job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=172.16.1.211:29500 \
    --rdzv-conf join_timeout=900,last_call_timeout=60,close_timeout=60,read_timeout=900 \
    train.py