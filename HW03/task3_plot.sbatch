#!/bin/bash
#SBATCH --job-name=task3_plot
#SBATCH --output=task3_plot.out
#SBATCH --error=task3_plot.err
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:20:00

# Load modules
module load nvidia/cuda/13.0.0
# Only uncomment the next line if python is not available by default
# module load python/3.9

# Compile Task 3 (required for the python script to run the executable)
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run the python plotting script
python3 plot_task3.py