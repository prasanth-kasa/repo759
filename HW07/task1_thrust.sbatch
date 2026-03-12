#!/bin/bash
#SBATCH --job-name=hw07_t1_thrust
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=task1_thrust-%j.out
#SBATCH --error=task1_thrust-%j.err

set -euo pipefail

module load nvidia/cuda/13.0

nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust

N=${N:-1048576}
./task1_thrust "${N}"
