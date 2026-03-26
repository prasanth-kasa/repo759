#!/bin/bash
#SBATCH --job-name=hw09_p4
#SBATCH --output=task4_sweep-%j.out
#SBATCH --error=task4_sweep-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=instruction

set -euo pipefail

module load cuda/13.0

clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
    -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_61 -march=native \
    -o task4 convolve.cpp task4.cpp

echo "n,time_ms" >task4_times.csv
for n in 512 1024 2048 4096; do
    ms=$(./task4 "${n}" | awk 'NR==1')
    echo "${n},${ms}" >>task4_times.csv
done

python3 plot_task4.py task4_times.csv task4.pdf
