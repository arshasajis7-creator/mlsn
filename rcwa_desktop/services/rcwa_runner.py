from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
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
    warnings: List[str] = field(default_factory=list)
    loss_plot_image: Path | None = None
    mask_layout_image: Path | None = None


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

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if log_dir is not None:
        config_dir = log_dir
        config_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        config_dir = Path(temp_dir.name)

    config_path = config_dir / "config.json"
    save_configuration(adapter_config, config_path)

    command = [sys.executable, str(adapter_path), str(config_path)]
    completed = subprocess.run(
        command,
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = completed.stdout
    stderr = completed.stderr

    warnings = _extract_adapter_warnings(stderr)

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

    # Locate the results CSV emitted by the adapter
    filename = f"{config.output_prefix}_step1_results.csv"
    output_csv = config_dir / filename
    if not output_csv.exists():
        fallback = working_dir / filename
        if fallback.exists():
            output_csv = fallback
        else:
            raise FileNotFoundError(f"Expected results CSV not found at {output_csv}")

    final_csv = output_csv
    if log_dir is not None:
        destination = log_dir / filename
        if destination != output_csv:
            shutil.copy2(output_csv, destination)
        final_csv = destination
    elif output_csv.parent != working_dir:
        destination = working_dir / filename
        shutil.copy2(output_csv, destination)
        final_csv = destination

    freq, rl = _load_and_negate_rl(final_csv)

    loss_image: Path | None = None
    mask_image: Path | None = None
    if log_dir is not None:
        from .run_artifacts import write_mask_svg, write_reflection_loss_svg

        try:
            loss_path = log_dir / "reflection_loss.svg"
            write_reflection_loss_svg(freq, rl, loss_path)
            loss_image = loss_path
        except Exception as exc:  # pragma: no cover - best effort artefact
            warnings.append(f"Failed to generate reflection loss image: {exc}")

        try:
            mask_path = log_dir / "mask_layout.svg"
            write_mask_svg(config, mask_path)
            mask_image = mask_path
        except Exception as exc:  # pragma: no cover - best effort artefact
            warnings.append(f"Failed to generate mask layout image: {exc}")

    summary_lines: List[str] = [f"Results CSV: {final_csv}"]
    if getattr(config, "mask", None):
        summary_lines.append(
            f"Mask grid: {config.mask.grid_nx} x {config.mask.grid_ny} samples (rasterised for FFT)."
        )
        mask_holes = config.mask.holes
    else:
        mask_holes = []
    if freq and rl:
        samples = len(freq)
        min_rl = min(rl)
        max_rl = max(rl)
        min_idx = rl.index(min_rl)
        max_idx = rl.index(max_rl)
        summary_lines.append(f"Samples: {samples}")
        summary_lines.append(f"Minimum RL: {min_rl:.3f} dB @ {freq[min_idx]:.3f} GHz")
        summary_lines.append(f"Maximum RL: {max_rl:.3f} dB @ {freq[max_idx]:.3f} GHz")
    else:
        summary_lines.append("No reflection loss samples recorded.")

    for idx, hole in enumerate(mask_holes, start=1):
        parts = [
            f"Hole {idx}: shape={hole.shape}",
            f"x={hole.x_m * 1e3:.3f} mm",
            f"y={hole.y_m * 1e3:.3f} mm",
            f"size1={hole.size1 * 1e3:.3f} mm",
        ]
        if hole.size2 is not None or hole.shape in {"square", "rectangle", "ellipse"}:
            size2_display = hole.size2 if hole.size2 is not None else hole.size1
            parts.append(f"size2={size2_display * 1e3:.3f} mm")
        rotation = getattr(hole, "rotation_deg", 0.0) or 0.0
        if rotation:
            parts.append(f"rotation={rotation:.2f} deg")
        summary_lines.append(", ".join(parts))

    if loss_image:
        summary_lines.append(f"Reflection loss plot: {loss_image}")
    if mask_image:
        summary_lines.append(f"Mask layout image: {mask_image}")
    for warning in warnings:
        summary_lines.append(f"Warning: {warning}")

    if log_dir is not None:
        (log_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    if temp_dir is not None:
        temp_dir.cleanup()

    return SimulationResult(
        output_csv=final_csv,
        stdout=stdout,
        stderr=stderr,
        freq_GHz=freq,
        RL_dB=rl,
        warnings=warnings,
        loss_plot_image=loss_image,
        mask_layout_image=mask_image,
    )


def _load_and_negate_rl(csv_path: Path) -> Tuple[List[float], List[float]]:
    freq: List[float] = []
    rl: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = [dict(row) for row in reader]

    if not rows or fieldnames is None:
        return freq, rl

    for row in rows:
        try:
            freq_val = float(row.get("freq_GHz", "nan"))
            rl_val = float(row.get("RL_dB", "nan"))
        except (TypeError, ValueError):
            continue
        neg_rl = -abs(rl_val)
        row["RL_dB"] = f"{neg_rl:.10e}"
        freq.append(freq_val)
        rl.append(neg_rl)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

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
    hole_index_by_position: dict[tuple[float, float, str], int] = {}

    def quantise(value: float) -> float:
        """Round coordinates to avoid floating-point mismatch when deduplicating."""

        return round(value, 9)

    for hole in config.mask.holes:
        size1 = max(float(hole.size1), 0.0)
        size2 = hole.size2
        if size2 is not None:
            size2 = max(float(size2), 0.0)
        rotation = float(getattr(hole, "rotation_deg", 0.0) or 0.0)

        if size1 <= 0.0:
            # Skip zero-area entries
            continue

        key = (quantise(hole.x_m), quantise(hole.y_m), hole.shape)

        replacement = MaskHole(
            shape=hole.shape,
            x_m=hole.x_m,
            y_m=hole.y_m,
            size1=size1,
            size2=size2,
            rotation_deg=rotation,
        )

        existing_index = hole_index_by_position.get(key)
        if existing_index is None:
            hole_index_by_position[key] = len(mask_holes)
            mask_holes.append(replacement)
        else:
            # When shapes overlap at the same position, keep the larger primary dimension.
            if size1 > mask_holes[existing_index].size1:
                mask_holes[existing_index] = replacement

    mask = replace(
        config.mask,
        solid_csv=resolve_path(config.mask.solid_csv),
        hole_csv=resolve_path(config.mask.hole_csv),
        holes=mask_holes,
    )

    return replace(config, layer_top=layer_top, layer_bottom=layer_bottom, mask=mask)


def _extract_adapter_warnings(stderr: str) -> List[str]:
    """Return structured warnings emitted by the adapter."""

    warnings: List[str] = []

    negative_power_pattern = re.compile(
        r"Negative power component detected \(([-+0-9.eE]+)\).*Tolerance is ([0-9.eE+-]+)",
        re.IGNORECASE,
    )
    energy_violation_pattern = re.compile(
        r"Energy conservation violated by ([0-9.eE+-]+).*tolerance ([0-9.eE+-]+)",
        re.IGNORECASE,
    )

    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = negative_power_pattern.search(line)
        if match:
            value, tolerance = match.groups()
            warnings.append(
                "Negative power component detected ("
                f"{value}) which exceeds the adapter tolerance of {tolerance}."
            )
            continue

        match = energy_violation_pattern.search(line)
        if match:
            delta, tolerance = match.groups()
            warnings.append(
                "Energy conservation violated by "
                f"{delta} which is above the configured tolerance of {tolerance}."
            )

    return warnings
