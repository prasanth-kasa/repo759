#!/bin/bash
#SBATCH --job-name=hw07_t3
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=task3-%j.out
#SBATCH --error=task3-%j.err

set -euo pipefail

g++ task3.cpp -Wall -O3 -std=c++17 -fopenmp -o task3
./task3
