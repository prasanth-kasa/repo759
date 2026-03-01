#!/usr/bin/env bash
#SBATCH --job-name=hw05_1c
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --output=task1c_results.txt

module load nvidia/cuda/13.0

echo "Testing block_dim = 8"
./task1 16384 8
echo "Testing block_dim = 16"
./task1 16384 16
echo "Testing block_dim = 32"
./task1 16384 32