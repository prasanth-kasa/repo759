#!/usr/bin/env bash
#SBATCH --job-name=hw04_task1
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out

# Load the required modules
module load nvidia/cuda/13.0.0
module load python

# Run the Python script which handles the execution and plotting
python generate_task1_plot.py