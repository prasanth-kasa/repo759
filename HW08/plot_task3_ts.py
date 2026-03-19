#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 plot_task3_ts.py <input_csv> <output_pdf>", file=sys.stderr)
        return 1

    input_csv = sys.argv[1]
    output_pdf = sys.argv[2]

    thresholds = []
    times = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds.append(int(row["ts"]))
            times.append(float(row["time_ms"]))

    plt.figure(figsize=(7, 4.5))
    plt.semilogx(thresholds, times, marker="o", linewidth=1.5, base=2, color="tab:green")
    plt.title("HW08 Task 3: msort Runtime vs Threshold")
    plt.xlabel("Threshold (ts)")
    plt.ylabel("Time (ms)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
