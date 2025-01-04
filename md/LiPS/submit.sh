#!/bin/bash -l

#SBATCH --job-name=nose_hoover
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=wen
#SBATCH --time=1-00:00:00
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1


conda activate camp
python run_md.py 0.25 100

scontrol show job $SLURM_JOBID
    