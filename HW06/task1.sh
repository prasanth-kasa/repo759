#!/usr/bin/env bash
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task1_slurm.out

# Load the required module
module load nvidia/cuda/13.0.0

nvcc task1.cu mmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -lcublas -std=c++17 -o task1

echo "n,time" > task1_data.csv

n_tests=10

# Loop through n = 2^5 to 2^11
for i in {5..11}; do
    n=$((2**i))
    time=$(./task1 $n $n_tests)
    echo "$n,$time" >> task1_data.csv
done

echo "Task 1 runs completed. Data saved to task1_data.csv."