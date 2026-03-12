#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_xy(csv_path: Path):
    xs, ys = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["n"]))
            ys.append(float(row["time_ms"]))
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="Plot HW07 Task2 timings.")
    parser.add_argument("--infile", default="task2_times.csv", help="Task2 timing CSV")
    parser.add_argument("--out", default="task2.pdf", help="Output PDF filename")
    args = parser.parse_args()

    x_vals, y_vals = load_xy(Path(args.infile))

    plt.figure(figsize=(8, 5))
    plt.loglog(x_vals, y_vals, marker="o", label="count() runtime")
    plt.xlabel("n")
    plt.ylabel("time (ms)")
    plt.title("HW07 Task2 Scaling")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
