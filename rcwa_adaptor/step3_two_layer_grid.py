"""
Grid search over two homogeneous layers (L1, L3) for the TMM-style absorber.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from adapter_step1 import run_step1_adapter


def _load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(data: Dict, destination: Path) -> None:
    destination.write_text(json.dumps(data, indent=4), encoding="utf-8")


def _read_results(csv_path: Path) -> Dict[str, np.ndarray]:
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return {key: np.asarray(values) for key, values in columns.items()}


def search(
    base_config: Dict,
    base_dir: Path,
    thickness_L1: np.ndarray,
    thickness_L3: np.ndarray,
    log_dir: Path | None = None,
) -> List[Tuple[float, float, float, float, float]]:
    results = []
    for t1 in thickness_L1:
        for t3 in thickness_L3:
            cfg = json.loads(json.dumps(base_config))
            layers = cfg.setdefault("layers", {})
            layers.setdefault("L1", {})["thickness_m"] = float(t1)
            layers.setdefault("L3", {})["thickness_m"] = float(t3)
            cfg["polarization"] = "TE"
            tol_cfg = cfg.setdefault("tolerances", {})
            tol_cfg.setdefault("strict", False)
            prefix = cfg.get("output_prefix", "grid")
            cfg["output_prefix"] = f"{prefix}_t1_{t1*1e3:.1f}mm_t3_{t3*1e3:.1f}mm"

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(base_dir)) as temp_cfg:
                temp_path = Path(temp_cfg.name)
            _write_config(cfg, temp_path)
            outputs = run_step1_adapter(temp_path, log_dir=log_dir)
            temp_path.unlink(missing_ok=True)

            data = _read_results(outputs["csv"])
            R = data["R"]
            T = data["T"]
            A_raw = data.get("A_raw", 1 - R - T)
            RL_dB = data["RL_dB"]
            avg_R = float(np.mean(R))
            avg_A = float(np.mean(A_raw))
            min_R = float(np.min(R))
            min_RL = float(np.min(RL_dB))
            print(
                f"L1={t1*1e3:.2f} mm, L3={t3*1e3:.2f} mm -> avg R={avg_R:.3f}, avg A_raw={avg_A:.3f}, "
                f"min R={min_R:.3f}, RL_min={min_RL:.2f} dB"
            )
            results.append((t1, t3, avg_R, avg_A, min_R))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for two-layer RCWA absorber.")
    parser.add_argument("config", type=Path, help="Base JSON configuration (two-layer).")
    parser.add_argument("--l1-mm", nargs=3, type=float, default=[2.0, 3.0, 4.0],
                        help="Start, stop, steps (mm) for L1 thickness sweep.")
    parser.add_argument("--l3-mm", nargs=3, type=float, default=[5.0, 10.0, 6],
                        help="Start, stop, steps (mm) for L3 thickness sweep.")
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    base_dir = args.config.parent.resolve()
    l1_vals = np.linspace(args.l1_mm[0], args.l1_mm[1], int(args.l1_mm[2])) * 1e-3
    l3_vals = np.linspace(args.l3_mm[0], args.l3_mm[1], int(args.l3_mm[2])) * 1e-3
    search(cfg, base_dir, l1_vals, l3_vals, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
