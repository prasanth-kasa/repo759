#!/usr/bin/env bash
#SBATCH --job-name=hw04_task1
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=task1_results.txt

# Load the required module first
module load nvidia/cuda/13.0.0

# Run a single test
echo "Running single test:"
./task1 1024 32

# Automate the scaling loop for Part 1(c)
echo "-------------------"
echo "Scaling data (Threads = 1024):"
for i in {5..14}; do
    n=$((2**i))
    echo "Testing n = $n"
    ./task1 $n 1024
done