#!/bin/bash
#SBATCH --job-name=hw08_task3
#SBATCH --output=task3_sweeps-%j.out
#SBATCH --error=task3_sweeps-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=instruction

set -euo pipefail

g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Sweep threshold ts = 2^1, 2^2, ..., 2^10 with n=10^6 and t=8
echo "ts,time_ms" > task3_ts_times.csv
for p in $(seq 1 10); do
    ts=$((2 ** p))
    run_out=$(./task3 1000000 8 "${ts}")
    time_ms=$(printf "%s\n" "${run_out}" | awk 'NR==3')
    echo "${ts},${time_ms}" >> task3_ts_times.csv
done

python3 plot_task3_ts.py task3_ts_times.csv hw8_task3_ts.pdf

# Pick best ts (minimum runtime) from the previous sweep.
best_ts=$(python3 - <<'PY'
import csv
with open("task3_ts_times.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
best = min(rows, key=lambda r: float(r["time_ms"]))
print(best["ts"])
PY
)
echo "Best threshold found: ${best_ts}"

# Sweep thread count t = 1..20 using best ts.
echo "t,time_ms" > task3_t_times.csv
for t in $(seq 1 20); do
    run_out=$(./task3 1000000 "${t}" "${best_ts}")
    time_ms=$(printf "%s\n" "${run_out}" | awk 'NR==3')
    echo "${t},${time_ms}" >> task3_t_times.csv
done

python3 plot_task3_t.py task3_t_times.csv hw8_task3_t.pdf
