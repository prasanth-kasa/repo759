#!/bin/bash
#SBATCH --job-name=hw09_p2
#SBATCH --output=task2_sweep-%j.out
#SBATCH --error=task2_sweep-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=instruction

set -euo pipefail

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2_nosimd -fopenmp \
    -fno-tree-vectorize -march=native -fopt-info-vec

g++ task2.cpp montecarlo.cpp -Wall -O3 -std=c++17 -o task2_simd -fopenmp \
    -fno-tree-vectorize -march=native -fopt-info-vec -DMONTECARLO_USE_SIMD

avg_line2() {
    local exe="$1"
    local n_val="$2"
    local t_val="$3"
    local sum=0.0
    local rep
    for rep in $(seq 1 10); do
        local ms
        ms=$("${exe}" "${n_val}" "${t_val}" | awk 'NR==2')
        sum=$(awk -v s="${sum}" -v m="${ms}" 'BEGIN { printf "%.12f", s + m }')
    done
    awk -v s="${sum}" 'BEGIN { printf "%.12f", s / 10.0 }'
}

echo "t,time_ms_nosimd,time_ms_simd" >task2_times.csv
for t in $(seq 1 10); do
    echo "t=${t}" >&2
    a=$(avg_line2 ./task2_nosimd 1000000 "${t}")
    b=$(avg_line2 ./task2_simd 1000000 "${t}")
    echo "${t},${a},${b}" >>task2_times.csv
done

python3 plot_task2.py task2_times.csv task2.pdf
