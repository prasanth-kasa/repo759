#!/bin/bash
#SBATCH --job-name=hw09_p3
#SBATCH --output=task3_sweep-%j.out
#SBATCH --error=task3_sweep-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=instruction

set -euo pipefail

module load mpi/mpich/4.0.2

mpicxx task3.cpp -Wall -O3 -o task3

avg_mpi_ms() {
    local n_val="$1"
    local sum=0.0
    local rep
    for rep in $(seq 1 10); do
        local ms
        ms=$(srun ./task3 "${n_val}" | awk 'NR==1')
        sum=$(awk -v s="${sum}" -v m="${ms}" 'BEGIN { printf "%.12f", s + m }')
    done
    awk -v s="${sum}" 'BEGIN { printf "%.12f", s / 10.0 }'
}

echo "n,time_ms" >task3_times.csv
for p in $(seq 1 25); do
    n=$((2 ** p))
    echo "n=${n}" >&2
    avg=$(avg_mpi_ms "${n}")
    echo "${n},${avg}" >>task3_times.csv
done

python3 plot_task3.py task3_times.csv task3.pdf
