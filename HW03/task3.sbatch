#!/bin/bash
#SBATCH --job-name=task3
#SBATCH --output=task3.out
#SBATCH --error=task3.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

# Load modules
module load nvidia/cuda/13.0.0

# Compile Task 3 [cite: 65]
# Includes both task3.cu and vscale.cu
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run Task 3 with n = 1024 [cite: 66]
./task3 1024