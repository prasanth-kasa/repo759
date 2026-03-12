#!/bin/bash
#SBATCH --job-name=hw07_t1_sweep
#SBATCH --partition=instruction
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=task1_sweep-%j.out
#SBATCH --error=task1_sweep-%j.err

set -euo pipefail

module load nvidia/cuda/13.0

nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust
nvcc task1_cub.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_cub

echo "n,power,time_ms,reduction" > task1_thrust_times.csv
echo "n,power,time_ms,reduction" > task1_cub_times.csv

for p in $(seq 10 20); do
    n=$((1 << p))

    out_t=$(./task1_thrust "${n}")
    red_t=$(printf "%s\n" "${out_t}" | sed -n '1p')
    ms_t=$(printf "%s\n" "${out_t}" | sed -n '2p')
    echo "${n},${p},${ms_t},${red_t}" >> task1_thrust_times.csv

    out_c=$(./task1_cub "${n}")
    red_c=$(printf "%s\n" "${out_c}" | sed -n '1p')
    ms_c=$(printf "%s\n" "${out_c}" | sed -n '2p')
    echo "${n},${p},${ms_c},${red_c}" >> task1_cub_times.csv

    echo "done n=${n}"
done
