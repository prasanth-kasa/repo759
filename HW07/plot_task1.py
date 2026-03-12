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
    parser = argparse.ArgumentParser(description="Plot HW07 Task1 timings.")
    parser.add_argument("--thrust", default="task1_thrust_times.csv", help="Thrust timing CSV")
    parser.add_argument("--cub", default="task1_cub_times.csv", help="CUB timing CSV")
    parser.add_argument(
        "--hw05",
        default="hw05_task2_times.csv",
        help="Optional HW05 timing CSV with columns n,time_ms",
    )
    parser.add_argument("--out", default="task1.pdf", help="Output PDF filename")
    args = parser.parse_args()

    thrust_path = Path(args.thrust)
    cub_path = Path(args.cub)
    hw05_path = Path(args.hw05)

    x_t, y_t = load_xy(thrust_path)
    x_c, y_c = load_xy(cub_path)

    plt.figure(figsize=(8, 5))
    plt.loglog(x_t, y_t, marker="o", label="Thrust reduce")
    plt.loglog(x_c, y_c, marker="s", label="CUB DeviceReduce::Sum")

    if hw05_path.exists():
        x_h, y_h = load_xy(hw05_path)
        plt.loglog(x_h, y_h, marker="^", label="HW05 CUDA reduction")

    plt.xlabel("n")
    plt.ylabel("time (ms)")
    plt.title("HW07 Task1 Scaling")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
