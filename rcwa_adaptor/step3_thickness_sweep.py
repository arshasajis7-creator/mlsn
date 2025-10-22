"""
Thickness sweep utility to experiment with top-layer thickness for RCWA setups.

The script clones a base configuration, replaces layers.L1.thickness_m with
values over a sweep, runs the TE-only adapter (for speed), and reports averaged
reflection/transmission/absorption metrics for each thickness.
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


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(data: Dict, destination: Path) -> None:
    destination.write_text(json.dumps(data, indent=4), encoding="utf-8")


def read_results(csv_path: Path) -> Dict[str, np.ndarray]:
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return {key: np.asarray(values) for key, values in columns.items()}


def thickness_sweep(
    base_config: Dict,
    base_dir: Path,
    thickness_values: np.ndarray,
    log_dir: Path | None = None,
) -> List[Tuple[float, float, float, float]]:
    results: List[Tuple[float, float, float, float]] = []
    for thickness in thickness_values:
        cfg_copy = json.loads(json.dumps(base_config))
        if "layers" not in cfg_copy or "L1" not in cfg_copy["layers"]:
            raise ValueError("Configuration must contain layers.L1.")
        cfg_copy["layers"]["L1"]["thickness_m"] = float(thickness)
        tol_cfg = cfg_copy.setdefault("tolerances", {})
        tol_cfg.setdefault("strict", False)
        cfg_copy["polarization"] = "TE"
        prefix = cfg_copy.get("output_prefix", "sweep")
        cfg_copy["output_prefix"] = f"{prefix}_t{thickness*1e3:.1f}mm"

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(base_dir)) as temp_cfg:
            temp_path = Path(temp_cfg.name)
        write_config(cfg_copy, temp_path)
        outputs = run_step1_adapter(temp_path, log_dir=log_dir)
        temp_path.unlink(missing_ok=True)

        data = read_results(outputs["csv"])
        R = data["R"]
        T = data["T"]
        A_raw = data.get("A_raw", 1 - R - T)
        RL = data["RL_dB"]
        avg_R = float(np.mean(R))
        avg_T = float(np.mean(T))
        avg_A = float(np.mean(A_raw))
        min_RL = float(np.min(RL))
        print(
            f"thickness={thickness*1e3:.2f} mm -> "
            f"avg R={avg_R:.2f}, avg T={avg_T:.4f}, avg A_raw={avg_A:.2f}, RL_min={min_RL:.2f} dB"
        )
        results.append((thickness, avg_R, avg_T, avg_A))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep layers.L1 thickness for RCWA configuration.")
    parser.add_argument("config", type=Path, help="Path to base JSON configuration.")
    parser.add_argument("--start", type=float, default=0.002, help="Start thickness (m).")
    parser.add_argument("--stop", type=float, default=0.012, help="Stop thickness (m).")
    parser.add_argument("--steps", type=int, default=6, help="Number of thickness points.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional log directory override.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = args.config.parent.resolve()
    thickness_values = np.linspace(args.start, args.stop, args.steps)
    thickness_sweep(cfg, base_dir, thickness_values, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
