#!/bin/bash
#SBATCH --job-name=task3
#SBATCH --output=task3.out
#SBATCH --error=task3.err
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

# Load modules
module load nvidia/cuda/13.0.0

# Compile and Run Task 3 (Test run with n=1024)
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3
./task3 1024