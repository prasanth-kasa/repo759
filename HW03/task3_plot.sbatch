#!/bin/bash
#SBATCH --job-name=task3_plot
#SBATCH --output=task3_plot.out
#SBATCH --error=task3_plot.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:20:00

# Load modules
module load nvidia/cuda/13.0.0
# Uncomment the following line if your cluster requires loading python separately:
# module load python/3.9  

# Compile Task 3 first (needed by the python script) [cite: 65]
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run the python plotting script
# This will execute ./task3 for n=2^10 to 2^29 and generate task3.pdf 
python3 plot_task3.py