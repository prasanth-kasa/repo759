#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --output=task2.out
#SBATCH --error=task2.err
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

# Load modules
module load nvidia/cuda/13.0.0

# Compile and Run Task 2
nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2
./task2