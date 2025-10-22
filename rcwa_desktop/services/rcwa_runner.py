from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from models.configuration import Configuration, save_configuration


@dataclass
class SimulationResult:
    output_csv: Path
    stdout: str
    stderr: str
    freq_GHz: List[float]
    RL_dB: List[float]


def run_simulation(config: Configuration, repo_root: Path) -> SimulationResult:
    """
    Execute adapter_step1.py with the provided configuration.

    Parameters
    ----------
    config:
        The configuration to serialize and run.
    repo_root:
        Path to the repository root (folder containing rcwa_adaptor/).
    """

    adapter_path = repo_root / "rcwa_adaptor" / "adapter_step1.py"
    working_dir = repo_root / "rcwa_adaptor"

    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter_step1.py not found at {adapter_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "config.json"
        save_configuration(config, tmp_path)

        command = ["python", str(adapter_path), str(tmp_path)]
        completed = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=False,
        )

        stdout = completed.stdout
        stderr = completed.stderr

        if completed.returncode != 0:
            raise RuntimeError(
                f"Simulation failed with exit code {completed.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )

    # Expect results in rcwa_adaptor directory
    output_csv = working_dir / f"{config.output_prefix}_step1_results.csv"
    if not output_csv.exists():
        raise FileNotFoundError(f"Expected results CSV not found at {output_csv}")

    freq, rl = _load_rl_curve(output_csv)
    return SimulationResult(
        output_csv=output_csv,
        stdout=stdout,
        stderr=stderr,
        freq_GHz=freq,
        RL_dB=rl,
    )


def _load_rl_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    freq = []
    rl = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                freq.append(float(row["freq_GHz"]))
                rl.append(float(row["RL_dB"]))
            except (ValueError, KeyError):
                continue
    return freq, rl
