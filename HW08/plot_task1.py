#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 plot_task1.py <input_csv> <output_pdf>", file=sys.stderr)
        return 1

    input_csv = sys.argv[1]
    output_pdf = sys.argv[2]

    threads = []
    times = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            threads.append(int(row["t"]))
            times.append(float(row["time_ms"]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(threads, times, marker="o", linewidth=1.5)
    plt.title("HW08 Task 1: mmul Runtime vs Threads")
    plt.xlabel("Threads (t)")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
