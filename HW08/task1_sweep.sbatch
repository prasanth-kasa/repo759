#!/bin/bash
#SBATCH --job-name=hw08_task1
#SBATCH --output=task1_sweep-%j.out
#SBATCH --error=task1_sweep-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction

set -euo pipefail

g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

echo "t,time_ms" > task1_times.csv
for t in $(seq 1 20); do
    run_out=$(./task1 1024 "${t}")
    time_ms=$(printf "%s\n" "${run_out}" | awk 'NR==3')
    echo "${t},${time_ms}" >> task1_times.csv
done

python3 plot_task1.py task1_times.csv hw8_task1.pdf
