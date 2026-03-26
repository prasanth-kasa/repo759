#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 plot_task1.py <input_csv> <output_pdf>", file=sys.stderr)
        return 1

    threads = []
    times = []
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            threads.append(int(row["t"]))
            times.append(float(row["time_ms"]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(threads, times, marker="o", linewidth=1.5)
    plt.title("HW09 Problem 1: cluster() time vs threads")
    plt.xlabel("t")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
