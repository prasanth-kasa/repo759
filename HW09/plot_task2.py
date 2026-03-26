#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 plot_task2.py <input_csv> <output_pdf>", file=sys.stderr)
        return 1

    threads = []
    nosimd = []
    simd = []
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            threads.append(int(row["t"]))
            nosimd.append(float(row["time_ms_nosimd"]))
            simd.append(float(row["time_ms_simd"]))

    plt.figure(figsize=(7, 4.5))
    plt.plot(threads, nosimd, marker="o", linewidth=1.5, label="without simd")
    plt.plot(threads, simd, marker="s", linewidth=1.5, label="with simd")
    plt.title("HW09 Problem 2: montecarlo() time vs threads")
    plt.xlabel("t")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
