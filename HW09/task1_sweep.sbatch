#!/bin/bash
#SBATCH --job-name=hw09_p1
#SBATCH --output=task1_sweep-%j.out
#SBATCH --error=task1_sweep-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=instruction

set -euo pipefail

export OMP_PROC_BIND=true
export OMP_PLACES=cores

g++ task1.cpp cluster.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# Mean of 10 runs
avg_time_ms() {
    local n_val="$1"
    local t_val="$2"
    local sum=0.0
    local rep
    for rep in $(seq 1 10); do
        local ms
        ms=$(./task1 "${n_val}" "${t_val}" | awk 'NR==3')
        sum=$(awk -v s="${sum}" -v m="${ms}" 'BEGIN { printf "%.12f", s + m }')
    done
    awk -v s="${sum}" 'BEGIN { printf "%.12f", s / 10.0 }'
}

echo "t,time_ms" >task1_times.csv
for t in $(seq 1 10); do
    echo "t=${t}" >&2
    avg=$(avg_time_ms 5040000 "${t}")
    echo "${t},${avg}" >>task1_times.csv
done

python3 plot_task1.py task1_times.csv task1.pdf
