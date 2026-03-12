#!/usr/bin/env python3
"""
Default output: task1.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_hw07_csv(csv_path: Path) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["n"]))
            ys.append(float(row["time_ms"]))
    return xs, ys


def read_hw05_csv(csv_path: Path) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = float(row["n"])
            t = float(row["time_ms"])
            if 2**10 <= n <= 2**20:
                xs.append(n)
                ys.append(t)
    return xs, ys


def parse_hw05_task2_results(results_path: Path) -> Tuple[List[float], List[float]]:
    """
    Parse HW05/task2_results.txt. For each block:
      Running task2 with N=<n>, threads=<t>
      <reduction_result>
      <time_ms>
    Keep only threads=1024 and n in [2^10, 2^20].
    """
    xs: List[float] = []
    ys: List[float] = []

    current_n = None
    current_threads = None
    numeric_lines: List[float] = []

    for raw in results_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("Running task2 with N=") and ", threads=" in line:
            # Flush previous block.
            if (
                current_n is not None
                and current_threads == 1024
                and len(numeric_lines) >= 2
                and 2**10 <= current_n <= 2**20
            ):
                xs.append(float(current_n))
                ys.append(float(numeric_lines[1]))

            numeric_lines = []
            # Expected format: Running task2 with N=1024, threads=1024
            left, right = line.split(", threads=", maxsplit=1)
            current_n = int(left.split("N=", maxsplit=1)[1])
            current_threads = int(right)
            continue

        try:
            value = float(line)
            if math.isfinite(value):
                numeric_lines.append(value)
        except ValueError:
            pass

    # Flush last block.
    if (
        current_n is not None
        and current_threads == 1024
        and len(numeric_lines) >= 2
        and 2**10 <= current_n <= 2**20
    ):
        xs.append(float(current_n))
        ys.append(float(numeric_lines[1]))

    return xs, ys


def must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create HW07 Task1 overlay plot (Thrust + CUB + HW05)."
    )
    parser.add_argument("--thrust", default="outputs/task1_thrust_times.csv")
    parser.add_argument("--cub", default="outputs/task1_cub_times.csv")
    parser.add_argument(
        "--hw05-csv",
        default="hw05_task2_times.csv",
        help="Optional prepared CSV with columns n,time_ms",
    )
    parser.add_argument(
        "--hw05-results",
        default="../HW05/task2_results.txt",
        help="Fallback raw HW05 results file to parse",
    )
    parser.add_argument("--out", default="outputs/task1.pdf")
    args = parser.parse_args()

    thrust_path = must_exist(Path(args.thrust), "HW07 Thrust CSV")
    cub_path = must_exist(Path(args.cub), "HW07 CUB CSV")
    hw05_csv_path = Path(args.hw05_csv)
    hw05_results_path = Path(args.hw05_results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_t, y_t = read_hw07_csv(thrust_path)
    x_c, y_c = read_hw07_csv(cub_path)

    if hw05_csv_path.exists():
        x_h, y_h = read_hw05_csv(hw05_csv_path)
        hw05_label = "HW05 CUDA reduction (CSV)"
    else:
        hw05_results_path = must_exist(hw05_results_path, "HW05 task2_results.txt")
        x_h, y_h = parse_hw05_task2_results(hw05_results_path)
        hw05_label = "HW05 CUDA reduction (parsed)"

    if not x_h:
        raise RuntimeError(
            "No valid HW05 points found for n=2^10..2^20 (threads=1024 expected)."
        )

    plt.figure(figsize=(8, 5))
    plt.loglog(x_t, y_t, marker="o", label="HW07 Thrust reduce")
    plt.loglog(x_c, y_c, marker="s", label="HW07 CUB reduce")
    plt.loglog(x_h, y_h, marker="^", label=hw05_label)
    plt.xlabel("n")
    plt.ylabel("time (ms)")
    plt.title("Task1: Reduction Scaling Comparison")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
