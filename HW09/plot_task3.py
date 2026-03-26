#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 plot_task3.py <input_csv> <output_pdf>", file=sys.stderr)
        return 1

    ns = []
    times = []
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ns.append(int(row["n"]))
            times.append(float(row["time_ms"]))

    plt.figure(figsize=(7, 4.5))
    plt.loglog(ns, times, marker="o", linewidth=1.5)
    plt.title("HW09 Problem 3: MPI Send/Recv time vs message length")
    plt.xlabel("n (floats)")
    plt.ylabel("Time (ms)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
