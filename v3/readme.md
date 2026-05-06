


sacct -j 32958 --duplicates --format=JobID,State,ExitCode,Submit,Start,End,NodeList


To restart a job
scontrol requeue <Jobid>


todo:
    run a job in specific nodes
    run a job omititng specific nodes



kill a node running a job (sbatch with --reqeue)
srun --jobid=39394 -w g5 pkill -f "torchrun|train.py"


ssh into running node
srun --jobid=39395 -N1 -n1 -w g5 --overlap --pty /bin/bash -l


Requeue skipping a node 

scontrol requeuehold 39401
scontrol update JobId=39401 ExcNodeList=g7
scontrol release 39401
