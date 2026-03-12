#!/bin/bash
#SBATCH --job-name=hw07_plot_t2
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=plot_task2-%j.out
#SBATCH --error=plot_task2-%j.err

set -euo pipefail

python3 plot_task2.py --out task2.pdf
