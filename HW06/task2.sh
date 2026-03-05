#!/usr/bin/env bash
#SBATCH --job-name=task2
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task2_slurm.out

module load nvidia/cuda/13.0.0

# Compile the code
nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Task 2a: Run cuda-memcheck and save output for Canvas
echo "Running cuda-memcheck..."
cuda-memcheck ./task2 1024 1024 > memcheck_output.txt

# Create a CSV file for plotting
echo "n,time" > task2_data.csv
threads_per_block=1024

# Loop through n = 2^10 to 2^16
for i in {10..16}; do
    n=$((2**i))
    
    # Run the executable, capture both lines of output
    output=$(./task2 $n $threads_per_block)
    
    # Extract the second line (which is the time in ms)
    time=$(echo "$output" | sed -n '2p')
    
    echo "$n,$time" >> task2_data.csv
done

echo "Task 2 runs completed. Data saved to task2_data.csv."