#!/bin/bash
#SBATCH --job-name=hw07_t4
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --constraint="volta|turing|ampere"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=task4-%j.out
#SBATCH --error=task4-%j.err

set -euo pipefail

module load nvidia/cuda/13.0

nvcc task4.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 \
  -o task4 -lcublas \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_70,code=sm_70

N=${N:-1024}
THREADS=${THREADS:-512}
./task4 "${N}" "${THREADS}"
