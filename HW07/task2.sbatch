#!/bin/bash
#SBATCH --job-name=hw07_t2
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=task2-%j.out
#SBATCH --error=task2-%j.err

set -euo pipefail

module load nvidia/cuda/13.0

nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

N=${N:-1048576}
./task2 "${N}"
