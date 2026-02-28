#!/usr/bin/env bash
#SBATCH --job-name=hw05_task2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=task2_results.txt
#SBATCH --error=task2_error.txt

module load nvidia/cuda/13.0

THREADS_1=1024
THREADS_2=512

echo "Starting Task 2 Sweep..."
echo "--------------------------------------------------"

echo "--- RUNNING WITH THREADS_PER_BLOCK = $THREADS_1 ---"
for i in {10..30}; do
    N=$((2**i))
    echo "Running task2 with N=$N, threads=$THREADS_1"
    
    ./task2 $N $THREADS_1
done

echo "--------------------------------------------------"
echo "--- RUNNING WITH THREADS_PER_BLOCK = $THREADS_2 ---"
for i in {10..30}; do
    N=$((2**i))
    echo "Running task2 with N=$N, threads=$THREADS_2"
    
    ./task2 $N $THREADS_2
done

echo "Task 2 Sweep Complete."