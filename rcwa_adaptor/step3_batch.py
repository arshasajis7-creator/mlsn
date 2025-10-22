"""
Utility to run the RCWA adapter sequentially for multiple polarizations and
combine the results on resource-constrained systems.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from adapter_step1 import RESULT_FILENAME_SUFFIX, run_step1_adapter


@dataclass
class RunInfo:
    polarization: str
    csv_path: Path
    log_path: Path


def _load_config(config_path: Path) -> Dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


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


def _write_average_results(
    destination: Path,
    freq: np.ndarray,
    wavelengths: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    A_raw: np.ndarray,
    A: np.ndarray,
) -> None:
    RL = -10.0 * np.log10(np.maximum(R, 1e-12))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["freq_GHz", "wavelength_m", "R", "T", "A_raw", "A", "RL_dB"])
        for row in zip(freq, wavelengths, R, T, A_raw, A, RL):
            writer.writerow([f"{value:.10e}" for value in row])


def run_sequence(
    config_path: Path,
    polarizations: Sequence[str],
    output_suffix: str = "avg",
    log_dir: Path | None = None,
) -> Dict[str, Path]:
    base_config = _load_config(config_path)
    base_prefix = base_config.get("output_prefix", config_path.stem)
    base_dir = config_path.parent.resolve()

    run_infos: List[RunInfo] = []
    for pol in polarizations:
        config_copy = json.loads(json.dumps(base_config))
        config_copy["polarization"] = pol.upper()
        tol_cfg = config_copy.setdefault("tolerances", {})
        tol_cfg.setdefault("strict", False)
        config_copy["output_prefix"] = f"{base_prefix}_{pol.lower()}"

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", dir=str(base_dir)) as temp_cfg:
            temp_path = Path(temp_cfg.name)
        _write_config(config_copy, temp_path)
        outputs = run_step1_adapter(temp_path, log_dir=log_dir)
        run_infos.append(
            RunInfo(
                polarization=pol.upper(),
                csv_path=outputs["csv"],
                log_path=outputs["log"],
            )
        )
        temp_path.unlink(missing_ok=True)

    if len(run_infos) < 2:
        return {"csv": run_infos[0].csv_path, "log": run_infos[0].log_path}

    # Combine results by averaging power quantities
    freq = wavelengths = None
    R_stack: List[np.ndarray] = []
    T_stack: List[np.ndarray] = []
    A_raw_stack: List[np.ndarray] = []
    A_stack: List[np.ndarray] = []

    for info in run_infos:
        data = _read_results(info.csv_path)
        if freq is None:
            freq = data["freq_GHz"]
            wavelengths = data["wavelength_m"]
        R_stack.append(data["R"])
        T_stack.append(data["T"])
        A_raw_stack.append(data.get("A_raw", 1 - data["R"] - data["T"]))
        A_stack.append(data.get("A", np.clip(A_raw_stack[-1], 0.0, None)))

    R_avg = np.mean(np.vstack(R_stack), axis=0)
    T_avg = np.mean(np.vstack(T_stack), axis=0)
    A_raw_avg = np.mean(np.vstack(A_raw_stack), axis=0)
    A_avg = np.mean(np.vstack(A_stack), axis=0)

    avg_csv = base_dir / f"{base_prefix}_{output_suffix}{RESULT_FILENAME_SUFFIX}"
    _write_average_results(avg_csv, freq, wavelengths, R_avg, T_avg, A_raw_avg, A_avg)
    return {"csv": avg_csv, "log": run_infos[0].log_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RCWA adapter sequentially for multiple polarizations and combine results."
    )
    parser.add_argument("config", type=Path, help="Path to base JSON configuration.")
    parser.add_argument(
        "--polarizations",
        nargs="+",
        default=["TE", "TM"],
        help="Polarizations to run sequentially (default: TE TM).",
    )
    parser.add_argument(
        "--suffix",
        default="avg",
        help="Suffix for combined result files (default: avg).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional override directory for log output.",
    )
    args = parser.parse_args()
    outputs = run_sequence(args.config, args.polarizations, output_suffix=args.suffix, log_dir=args.log_dir)
    print(f"Combined results saved to {outputs['csv']}")


if __name__ == "__main__":
    main()
