"""
Quick analysis utility for RCWA result CSVs produced by adapter_step1 or step3_batch.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return {key: np.asarray(values) for key, values in columns.items()}


def summarize(path: Path) -> None:
    data = load_csv(path)
    freq = data["freq_GHz"]
    R = data["R"]
    T = data["T"]
    A_raw = data.get("A_raw", 1 - R - T)
    A = data.get("A", np.clip(A_raw, 0.0, None))
    RL = data["RL_dB"]

    print(f"File: {path}")
    print(f"  freq range: {freq[0]:.3f} – {freq[-1]:.3f} GHz  ({len(freq)} points)")
    print(f"  R min/max:  {R.min():.4f} / {R.max():.4f}")
    print(f"  T min/max:  {T.min():.4e} / {T.max():.4e}")
    print(f"  A_raw min:  {A_raw.min():.4f}")
    print(f"  A clip min: {A.min():.4f}")
    print(f"  RL min/max: {RL.min():.2f} / {RL.max():.2f} dB")
    worst_idx = int(np.argmin(A_raw))
    print(f"  Worst energy point: {freq[worst_idx]:.3f} GHz -> R={R[worst_idx]:.4f}, T={T[worst_idx]:.4e}, A_raw={A_raw[worst_idx]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RCWA result CSVs.")
    parser.add_argument("csv", nargs="+", type=Path, help="Result CSV file(s).")
    args = parser.parse_args()
    for path in args.csv:
        summarize(path.resolve())


if __name__ == "__main__":
    main()
