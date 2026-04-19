module load python3
module load ml/torch/2.6

python -m pip install prometheus_client




rsync -avz ./v1 018280561@coe-hpc1.sjsu.edu:~/
<!-- rsync -avz scripts g17:~/scripts -->


salloc --nodes=2 --partition=gpuqs --gres=gpu:a100:1 --cpus-per-task=2 --time=01:00:00

hostname --ip-address


srun --jobid=32693 --nodelist=cs004 --pty bash