#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --account=projectnucleus
#SBATCH --partition=booster
#SBATCH --time=24:00:00
#SBATCH --array=0-9%1
#SBATCH --job-name=job-name

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION"

ml purge
ml Stages/2024
ml StdEnv
ml Python/3.11.3
ml CUDA/12
ml cuDNN/8.9.5.29-CUDA-12
ml NCCL/default-CUDA-12
ml PyTorch/2.1.2
ml torchvision
ml tensorboard
source /p/project/projectnucleus/envs/dino/bin/activate

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Allow communication over InfiniBand cells.
# adding “i” to hostname is crucial, otherwise compute nodes will not be able to communicate
export MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_PORT=23456
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
srun --cpu_bind=none,v --accel-bind=gn --threads-per-core=1 python -u dino/run_pipeline.py

