#!/bin/bash
#SBATCH --job-name=hw08_task2
#SBATCH --output=task2_sweep-%j.out
#SBATCH --error=task2_sweep-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction

set -euo pipefail

g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

echo "t,time_ms" > task2_times.csv
for t in $(seq 1 20); do
    run_out=$(./task2 1024 "${t}")
    time_ms=$(printf "%s\n" "${run_out}" | awk 'NR==3')
    echo "${t},${time_ms}" >> task2_times.csv
done

python3 plot_task2.py task2_times.csv hw8_task2.pdf
