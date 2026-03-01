#!/usr/bin/env bash
#SBATCH --job-name=hw05_task1
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --output=task1_results.txt
#SBATCH --error=task1_error.txt

module load nvidia/cuda/13.0

# Define your block dimension here. You may need to change this and 
# re-run to answer question 1c regarding the best performing value.
BLOCK_DIM=16

echo "Starting Task 1 Sweep..."
echo "Block Dimension: $BLOCK_DIM"
echo "--------------------------------------------------"

# Loop from 5 to 14 to compute n = 2^i
for i in {5..14}; do
    n=$((2**i))
    echo "Running task1 with n=$n and block_dim=$BLOCK_DIM"
    
    ./task1 $n $BLOCK_DIM
    
    echo "--------------------------------------------------"
done

echo "Task 1 Sweep Complete."