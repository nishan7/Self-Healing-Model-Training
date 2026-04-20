module load python3
module load ml/torch/2.6

python -m pip install prometheus_client




rsync -avz ./v1 018280561@coe-hpc1.sjsu.edu:~/
<!-- rsync -avz scripts g17:~/scripts -->


salloc --nodes=3 --partition=gpuqs --cpus-per-task=1 --time=01:00:00 --exclude=cs001,cs003

hostname --ip-address


srun --jobid=32953 --nodelist=g11 --pty bash

pkill -9 -f "python -u train.py"; pkill -9 -f "torchrun.*train.py"