#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --output=task1.out
#SBATCH --error=task1.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

# Load modules
module load nvidia/cuda/13.0.0

# Compile Task 1 [cite: 23]
# Flags: -O3, -Wall, -Xptxas -O3, std=c++17
nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run Task 1 [cite: 24]
./task1