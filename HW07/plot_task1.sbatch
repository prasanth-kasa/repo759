#!/bin/bash
#SBATCH --job-name=hw07_plot_t1
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=plot_task1-%j.out
#SBATCH --error=plot_task1-%j.err

set -euo pipefail

python3 plot_task1.py --out task1.pdf
