#!/bin/bash
#SBATCH --job-name=hw07_t2_sweep
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=task2_sweep-%j.out
#SBATCH --error=task2_sweep-%j.err

set -euo pipefail

module load nvidia/cuda/13.0

nvcc task2.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

echo "n,power,time_ms,last_value,last_count" > task2_times.csv

for p in $(seq 5 20); do
    n=$((1 << p))
    out=$(./task2 "${n}")
    last_value=$(printf "%s\n" "${out}" | sed -n '1p')
    last_count=$(printf "%s\n" "${out}" | sed -n '2p')
    ms=$(printf "%s\n" "${out}" | sed -n '3p')
    echo "${n},${p},${ms},${last_value},${last_count}" >> task2_times.csv
    echo "done n=${n}"
done
