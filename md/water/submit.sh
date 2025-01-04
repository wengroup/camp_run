#!/bin/bash -l

#SBATCH --job-name=water_md
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=wen
#SBATCH --time=5-00:00:00
#SBATCH --mem=40GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1


conda activate camp

python run_md.py
