#!/bin/bash
#SBATCH --job-name=task1_scaling
#SBATCH --output=task1_output.txt
#SBATCH --partition=instruction
#SBATCH --constraint=cpu    # Ensure we are on a CPU node
#SBATCH --time=00:10:00     # time limit (e.g., 10 mins)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

# 2. Run the loop for n = 2^10 to 2^30
for i in {10..30}
do
    # Calculate n = 2^i
    n=$((2**i))
    
    echo "Running for n = 2^$i ($n)..."
    ./task1 $n
done
