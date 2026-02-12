#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --output=task1.out
#SBATCH --error=task1.err
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

module load nvidia/cuda/13.0.0

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
./task1