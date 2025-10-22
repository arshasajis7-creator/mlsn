from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Tuple

try:  # pragma: no cover - exercised in packaging environments
    from ..models.configuration import Configuration, MaskHole, save_configuration
except ImportError:  # pragma: no cover - fallback when run as script
    from models.configuration import Configuration, MaskHole, save_configuration


@dataclass
class SimulationResult:
    output_csv: Path
    stdout: str
    stderr: str
    freq_GHz: List[float]
    RL_dB: List[float]


def run_simulation(
    config: Configuration, repo_root: Path, *, log_dir: Path | None = None
) -> SimulationResult:
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

    adapter_config = _prepare_config_for_adapter(config, repo_root)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "config.json"
        save_configuration(adapter_config, tmp_path)

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

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "adapter_stdout.txt").write_text(stdout, encoding="utf-8")
        (log_dir / "adapter_stderr.txt").write_text(stderr, encoding="utf-8")
        save_configuration(adapter_config, log_dir / "adapter_config.json")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Simulation failed with exit code {completed.returncode}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "adapter_stdout.txt").write_text(stdout, encoding="utf-8")
        (log_dir / "adapter_stderr.txt").write_text(stderr, encoding="utf-8")
        save_configuration(config, log_dir / "config.json")

    # Expect results in rcwa_adaptor directory
    output_csv = working_dir / f"{config.output_prefix}_step1_results.csv"
    if not output_csv.exists():
        raise FileNotFoundError(f"Expected results CSV not found at {output_csv}")

    freq, rl = _load_rl_curve(output_csv)

    if log_dir is not None and output_csv.exists():
        destination = log_dir / output_csv.name
        if destination != output_csv:
            destination.write_text(output_csv.read_text(encoding="utf-8"), encoding="utf-8")

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


def _prepare_config_for_adapter(config: Configuration, repo_root: Path) -> Configuration:
    """Normalise material paths and hole definitions for the adapter."""

    def resolve_path(entry: str) -> str:
        if not entry:
            return entry
        candidate = Path(entry)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)

        search_roots = [repo_root, repo_root / "rcwa_adaptor"]
        for root in search_roots:
            candidate_path = (root / candidate).resolve()
            if candidate_path.exists():
                return str(candidate_path)
        return str(candidate.resolve())

    layer_top = replace(config.layer_top, material_csv=resolve_path(config.layer_top.material_csv))
    layer_bottom = replace(
        config.layer_bottom, material_csv=resolve_path(config.layer_bottom.material_csv)
    )

    mask_holes: list[MaskHole] = []
    hole_index_by_position: dict[tuple[float, float], int] = {}

    def quantise(value: float) -> float:
        """Round coordinates to avoid floating-point mismatch when deduplicating."""

        return round(value, 9)

    for hole in config.mask.holes:
        diameter = max(hole.adapter_diameter(), 0.0)
        if diameter <= 0.0:
            # Skip zero-area holes; the adapter treats them as invalid.
            continue

        key = (quantise(hole.x_m), quantise(hole.y_m))
        replacement = MaskHole(
            shape="circle",
            x_m=hole.x_m,
            y_m=hole.y_m,
            size1=diameter,
            size2=None,
        )

        existing_index = hole_index_by_position.get(key)
        if existing_index is None:
            hole_index_by_position[key] = len(mask_holes)
            mask_holes.append(replacement)
            continue

        # If multiple holes share a centre, keep the largest diameter to avoid
        # nested discs that destabilise the adapter's power calculations.
        if diameter > mask_holes[existing_index].size1:
            mask_holes[existing_index] = replacement

    mask = replace(
        config.mask,
        solid_csv=resolve_path(config.mask.solid_csv),
        hole_csv=resolve_path(config.mask.hole_csv),
        holes=mask_holes,
    )

    return replace(config, layer_top=layer_top, layer_bottom=layer_bottom, mask=mask)
