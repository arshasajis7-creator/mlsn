"""
Step 1/2/3 RCWA adapter: homogeneous layers with optional periodic mask.

This module wraps the existing `rcwa` core without modifying it.  It parses a
JSON configuration, loads frequency-domain epsilon/mu tables, constructs the
layer stack (including an optional μ-aware periodic mask), runs the solver for a
requested sweep, and emits both CSV summaries and JSONL logs.  Guard rails catch
non-finite numbers and energy-budget violations so issues surface immediately.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Ensure the repository root (which contains the rcwa package) is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rcwa.crystal import Crystal
from rcwa.layer import Layer, LayerStack
from rcwa.material import Material
from rcwa.source import Source
from rcwa.solver import Solver

C0 = 299_792_458.0  # Speed of light in vacuum (m/s)
GHZ_TO_HZ = 1e9
RESULT_FILENAME_SUFFIX = "_step1_results.csv"
LOG_FILENAME_SUFFIX = "_step1_log.jsonl"

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
    module="rcwa.matrices",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
    module="rcwa.solver",
)


class ConfigError(ValueError):
    """Raised when the user configuration is invalid."""


@dataclass(frozen=True)
class FrequencySpec:
    start_GHz: float
    stop_GHz: float
    points: int

    def as_array(self) -> np.ndarray:
        if self.points < 2:
            raise ConfigError("freq_GHz.points must be at least 2 for a sweep.")
        if self.stop_GHz <= self.start_GHz:
            raise ConfigError("freq_GHz.stop must be greater than freq_GHz.start.")
        return np.linspace(self.start_GHz, self.stop_GHz, self.points, dtype=float)


@dataclass(frozen=True)
class LayerSpec:
    csv: Path
    thickness_m: float


@dataclass(frozen=True)
class MaskHole:
    shape: str
    x_m: float
    y_m: float
    size1: float
    size2: float | None
    rotation_deg: float
    adapter_diameter: float


@dataclass(frozen=True)
class MaskSpec:
    solid_csv: Path
    hole_csv: Path
    thickness_m: float
    grid_nx: int
    grid_ny: int
    holes: Tuple[MaskHole, ...]


@dataclass(frozen=True)
class AdapterConfig:
    base_dir: Path
    frequency: FrequencySpec
    layer1: LayerSpec
    layer3: Optional[LayerSpec]
    mask: Optional[MaskSpec]
    polarization: str
    n_harmonics: Union[int, Tuple[int, int]]
    backing_type: str
    backing_eps_imag: float
    output_prefix: str
    energy_tolerance: float
    negativity_tolerance: float
    strict_energy: bool
    theta_rad: float
    phi_rad: float
    solver_check_convergence: bool
    solver_atol: float
    solver_rtol: float
    solver_max_iters: int
    cell_Lx: float
    cell_Ly: float
    supercell: Tuple[int, int]
    log_dir: Path


@dataclass(frozen=True)
class AdapterTables:
    layer1: "DispersionTable"
    layer3: Optional["DispersionTable"]
    mask_solid: Optional["DispersionTable"]
    mask_hole: Optional["DispersionTable"]


def _resolve_path(config_dir: Path, entry: str, description: str) -> Path:
    if not entry:
        raise ConfigError(f"{description} must include 'csv'.")
    path = (config_dir / Path(entry)).resolve()
    if not path.is_file():
        raise ConfigError(f"Material table not found: {path}")
    return path


def _parse_layer_spec(config_dir: Path, node: Mapping, name: str) -> LayerSpec:
    if not isinstance(node, Mapping):
        raise ConfigError(f"{name} layer must be an object.")
    csv_path = _resolve_path(config_dir, str(node.get("csv", "")), f"{name} layer")
    thickness = float(node.get("thickness_m", 0.0))
    if thickness <= 0:
        raise ConfigError(f"{name}.thickness_m must be positive.")
    return LayerSpec(csv=csv_path, thickness_m=thickness)


def _parse_holes(node: Iterable[Mapping], cell_Lx: float, cell_Ly: float) -> Tuple[MaskHole, ...]:
    holes: List[MaskHole] = []
    half_Lx = cell_Lx / 2.0
    half_Ly = cell_Ly / 2.0
    valid_shapes = {"circle", "square", "rectangle", "ellipse"}

    for idx, entry in enumerate(node):
        if not isinstance(entry, Mapping):
            raise ConfigError(f"holes[{idx}] must be an object.")

        shape = str(entry.get("type", entry.get("shape", "circle"))).lower()
        if shape not in valid_shapes:
            raise ConfigError(f"holes[{idx}].type '{shape}' is not supported.")

        x_m = float(entry.get("x_m", entry.get("x", 0.0)))
        y_m = float(entry.get("y_m", entry.get("y", 0.0)))
        rotation_deg = float(entry.get("rotation_deg", entry.get("rotation", entry.get("theta_deg", 0.0))))

        def clamp_positive(value: float) -> float:
            val = float(value)
            if val <= 0.0:
                raise ConfigError(f"holes[{idx}] must have positive dimensions.")
            return val

        if shape == "circle":
            size1 = clamp_positive(entry.get("diameter_m", entry.get("size1", 0.0)))
            size2 = None
        elif shape in {"square", "rectangle"}:
            width = clamp_positive(entry.get("width_m", entry.get("size1", entry.get("diameter_m", 0.0))))
            height = clamp_positive(entry.get("height_m", entry.get("size2", width)))
            size1, size2 = width, height
        elif shape == "ellipse":
            axis_a = clamp_positive(entry.get("axis_a_m", entry.get("size1", entry.get("diameter_m", 0.0))))
            axis_b = clamp_positive(entry.get("axis_b_m", entry.get("size2", axis_a)))
            size1, size2 = axis_a, axis_b
        else:
            size1 = clamp_positive(entry.get("diameter_m", entry.get("size1", 0.0)))
            size2 = None

        if shape == "circle":
            effective_radius = size1 / 2.0
        elif shape in {"square", "rectangle"}:
            # Use the circumscribed circle radius for bounds checking
            effective_radius = 0.5 * math.hypot(size1, size2 if size2 is not None else size1)
        elif shape == "ellipse":
            effective_radius = max(size1, size2 if size2 is not None else size1) / 2.0
        else:
            effective_radius = size1 / 2.0

        if abs(x_m) + effective_radius > half_Lx + 1e-12:
            raise ConfigError(f"holes[{idx}] exceeds cell bounds in x-direction.")
        if abs(y_m) + effective_radius > half_Ly + 1e-12:
            raise ConfigError(f"holes[{idx}] exceeds cell bounds in y-direction.")

        if shape in {"square", "rectangle", "ellipse"}:
            other = size2 if size2 is not None else size1
            adapter_diameter = min(size1, other)
        else:
            adapter_diameter = size1

        holes.append(
            MaskHole(
                shape=shape,
                x_m=x_m,
                y_m=y_m,
                size1=size1,
                size2=size2,
                rotation_deg=rotation_deg,
                adapter_diameter=adapter_diameter,
            )
        )
    return tuple(holes)


def _parse_mask_spec(config_dir: Path, node: Mapping, cell_Lx: float, cell_Ly: float) -> MaskSpec:
    if not isinstance(node, Mapping):
        raise ConfigError("layers.L2_mask must be an object.")
    solid_csv = _resolve_path(config_dir, str(node.get("csv_solid", "")), "L2_mask")
    hole_csv = _resolve_path(config_dir, str(node.get("csv_hole", "")), "L2_mask")

    thickness = float(node.get("thickness_m", 0.0))
    if thickness <= 0:
        raise ConfigError("layers.L2_mask.thickness_m must be positive.")

    grid_section = node.get("grid")
    if not isinstance(grid_section, Mapping):
        raise ConfigError("layers.L2_mask.grid must be an object with Nx, Ny.")
    grid_nx = int(grid_section.get("Nx", 0))
    grid_ny = int(grid_section.get("Ny", 0))
    if grid_nx < 3 or grid_ny < 3:
        raise ConfigError("layers.L2_mask.grid.Nx/Ny must be ≥ 3.")

    holes_node = node.get("holes", [])
    if not isinstance(holes_node, Iterable):
        raise ConfigError("layers.L2_mask.holes must be a list when provided.")
    holes = _parse_holes(holes_node, cell_Lx, cell_Ly)

    return MaskSpec(
        solid_csv=solid_csv,
        hole_csv=hole_csv,
        thickness_m=thickness,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        holes=holes,
    )


def _parse_supercell(node: Mapping) -> Tuple[int, int]:
    if not isinstance(node, Mapping):
        raise ConfigError("supercell must be an object when provided.")
    nx = int(node.get("nx", 1))
    ny = int(node.get("ny", 1))
    if nx < 1 or ny < 1:
        raise ConfigError("supercell nx, ny must be ≥ 1.")
    return nx, ny


def load_config(path: Path, override_log_dir: Optional[Path] = None) -> AdapterConfig:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc

    freq_section = raw.get("freq_GHz")
    if not isinstance(freq_section, Mapping):
        raise ConfigError("freq_GHz section missing or not an object.")
    frequency = FrequencySpec(
        start_GHz=float(freq_section.get("start", 0.0)),
        stop_GHz=float(freq_section.get("stop", 0.0)),
        points=int(freq_section.get("points", 0)),
    )

    cell_section = raw.get("cell", {})
    if cell_section is None:
        cell_section = {}
    if not isinstance(cell_section, Mapping):
        raise ConfigError("cell must be an object when provided.")
    cell_Lx = float(cell_section.get("Lx_m", 1.0))
    cell_Ly = float(cell_section.get("Ly_m", 1.0))
    if cell_Lx <= 0 or cell_Ly <= 0:
        raise ConfigError("cell.Lx_m and cell.Ly_m must be positive.")

    config_dir = path.parent.resolve()
    layer1_spec: LayerSpec
    layer3_spec: Optional[LayerSpec] = None
    mask_spec: Optional[MaskSpec] = None

    layers_section = raw.get("layers")
    if layers_section is not None:
        if not isinstance(layers_section, Mapping):
            raise ConfigError("layers must be an object when provided.")
        if "L1" not in layers_section:
            raise ConfigError("layers.L1 must be provided.")
        layer1_spec = _parse_layer_spec(config_dir, layers_section["L1"], "layers.L1")
        if "L3" in layers_section and layers_section["L3"] is not None:
            layer3_spec = _parse_layer_spec(config_dir, layers_section["L3"], "layers.L3")
        if "L2_mask" in layers_section and layers_section["L2_mask"] is not None:
            mask_spec = _parse_mask_spec(config_dir, layers_section["L2_mask"], cell_Lx, cell_Ly)
    else:
        layer_section = raw.get("layer")
        if not isinstance(layer_section, Mapping):
            raise ConfigError("layer section missing or not an object.")
        layer1_spec = _parse_layer_spec(config_dir, layer_section, "layer")

    polarization = str(raw.get("polarization", "TE")).upper()
    if polarization not in {"TE", "TM", "AVG"}:
        raise ConfigError("polarization must be one of: TE, TM, AVG.")

    n_harmonics_value = raw.get("n_harmonics", 1)
    if isinstance(n_harmonics_value, (list, tuple)):
        if not (1 <= len(n_harmonics_value) <= 2):
            raise ConfigError("n_harmonics sequence must have one or two entries.")
        harmonics = tuple(int(h) for h in n_harmonics_value)
        for idx, h in enumerate(harmonics):
            if h < 1 or h % 2 == 0:
                raise ConfigError(f"n_harmonics[{idx}] must be a positive odd integer.")
        n_harmonics: Union[int, Tuple[int, int]]
        if len(harmonics) == 1:
            n_harmonics = harmonics[0]
        else:
            n_harmonics = harmonics  # type: ignore[assignment]
    elif isinstance(n_harmonics_value, (int, float)):
        n_harmonics = int(n_harmonics_value)
        if n_harmonics < 1 or n_harmonics % 2 == 0:
            raise ConfigError("n_harmonics must be a positive odd integer.")
    else:
        raise ConfigError("n_harmonics must be an int or a list/tuple of ints.")

    backing_section = raw.get("backing", {"type": "metal"})
    if not isinstance(backing_section, Mapping):
        raise ConfigError("backing section must be an object when provided.")
    backing_type = str(backing_section.get("type", "metal")).lower()
    if backing_type not in {"metal", "free"}:
        raise ConfigError("backing.type must be 'metal' or 'free'.")
    backing_eps_imag = float(backing_section.get("eps_imag_clamp", 1e8))
    if backing_type == "metal" and backing_eps_imag <= 0:
        raise ConfigError("backing.eps_imag_clamp must be positive for metal backing.")

    tolerances_section = raw.get("tolerances", {})
    if not isinstance(tolerances_section, Mapping):
        raise ConfigError("tolerances must be an object when provided.")
    energy_tol = float(tolerances_section.get("energy", 1e-3))
    if energy_tol <= 0:
        raise ConfigError("tolerances.energy must be positive.")
    negativity_tol = float(tolerances_section.get("nonnegativity", 1e-6))
    if negativity_tol <= 0:
        raise ConfigError("tolerances.nonnegativity must be positive.")
    strict_energy = bool(tolerances_section.get("strict", True))

    angles_section = raw.get("angles", {})
    if not isinstance(angles_section, Mapping):
        raise ConfigError("angles must be an object when provided.")
    theta_deg = float(angles_section.get("theta_deg", 0.0))
    phi_deg = float(angles_section.get("phi_deg", 0.0))
    theta_rad = math.radians(theta_deg)
    phi_rad = math.radians(phi_deg)
    if theta_rad < 0 or theta_rad > (math.pi / 2 + 1e-12):
        raise ConfigError("angles.theta_deg must be between 0 and 90 degrees.")

    solver_section = raw.get("solver", {})
    if not isinstance(solver_section, Mapping):
        raise ConfigError("solver must be an object when provided.")
    solver_check = bool(solver_section.get("check_convergence", False))
    solver_atol = float(solver_section.get("atol", 1e-3))
    solver_rtol = float(solver_section.get("rtol", 1e-2))
    solver_max_iters = int(solver_section.get("max_iters", 50))
    if solver_atol <= 0 or solver_rtol <= 0:
        raise ConfigError("solver tolerances must be positive.")
    if solver_max_iters <= 0:
        raise ConfigError("solver.max_iters must be positive.")

    supercell_section = raw.get("supercell", {"nx": 1, "ny": 1})
    supercell = _parse_supercell(supercell_section)

    output_prefix = str(raw.get("output_prefix", path.stem))
    base_dir = config_dir

    log_dir = override_log_dir or (base_dir / "step1_logs")
    log_dir = log_dir.resolve()

    return AdapterConfig(
        base_dir=base_dir,
        frequency=frequency,
        layer1=layer1_spec,
        layer3=layer3_spec,
        mask=mask_spec,
        polarization=polarization,
        n_harmonics=n_harmonics,
        backing_type=backing_type,
        backing_eps_imag=backing_eps_imag,
        output_prefix=output_prefix,
        energy_tolerance=energy_tol,
        negativity_tolerance=negativity_tol,
        strict_energy=strict_energy,
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        solver_check_convergence=solver_check,
        solver_atol=solver_atol,
        solver_rtol=solver_rtol,
        solver_max_iters=solver_max_iters,
        cell_Lx=cell_Lx,
        cell_Ly=cell_Ly,
        supercell=supercell,
        log_dir=log_dir,
    )


class DispersionTable:
    """Loads frequency-domain ε/μ data and interpolates across GHz."""

    REQUIRED_COLUMNS = {"freq_GHz", "eps_real", "eps_imag", "mu_real", "mu_imag"}

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._freq_GHz, self._eps_values, self._mu_values = self._load(csv_path)

    @staticmethod
    def _load(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        freqs: List[float] = []
        eps_vals: List[complex] = []
        mu_vals: List[complex] = []

        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ConfigError(f"CSV {csv_path} has no header.")
            missing_cols = DispersionTable.REQUIRED_COLUMNS.difference(reader.fieldnames)
            if missing_cols:
                raise ConfigError(f"CSV {csv_path} missing required columns: {sorted(missing_cols)}")

            for row in reader:
                try:
                    freq = float(row["freq_GHz"])
                    eps = complex(float(row["eps_real"]), float(row["eps_imag"]))
                    mu = complex(float(row["mu_real"]), float(row["mu_imag"]))
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"Failed parsing row {row} in {csv_path}") from exc
                freqs.append(freq)
                eps_vals.append(eps)
                mu_vals.append(mu)

        if len(freqs) < 2:
            raise ConfigError(f"CSV {csv_path} must contain at least two frequency samples.")

        order = np.argsort(freqs)
        freq_arr = np.asarray(freqs, dtype=float)[order]
        eps_arr = np.asarray(eps_vals, dtype=np.complex128)[order]
        mu_arr = np.asarray(mu_vals, dtype=np.complex128)[order]

        return freq_arr, eps_arr, mu_arr

    def permittivity(self, wavelength_m: np.ndarray) -> np.ndarray:
        freq = (C0 / np.asarray(wavelength_m, dtype=float)) / GHZ_TO_HZ
        return _interp_complex(freq, self._freq_GHz, self._eps_values)

    def permeability(self, wavelength_m: np.ndarray) -> np.ndarray:
        freq = (C0 / np.asarray(wavelength_m, dtype=float)) / GHZ_TO_HZ
        return _interp_complex(freq, self._freq_GHz, self._mu_values)

    def permittivity_scalar(self, wavelength_m: float) -> complex:
        return self.permittivity(np.asarray([wavelength_m], dtype=float))[0]

    def permeability_scalar(self, wavelength_m: float) -> complex:
        return self.permeability(np.asarray([wavelength_m], dtype=float))[0]

    def create_material(self) -> Material:
        def er_func(lam: float) -> np.ndarray:
            val = self.permittivity_scalar(lam)
            return np.array([[val]], dtype=np.complex128)

        def ur_func(lam: float) -> np.ndarray:
            val = self.permeability_scalar(lam)
            return np.array([[val]], dtype=np.complex128)

        return Material(er=er_func, ur=ur_func)


class MaskPattern:
    """Generates ε/μ cell data for a periodic mask at arbitrary wavelength."""

    def __init__(
        self,
        cfg: AdapterConfig,
        spec: MaskSpec,
        solid_dispersion: DispersionTable,
        hole_dispersion: DispersionTable,
        reference_wavelength: float,
    ):
        super_nx, super_ny = cfg.supercell
        self.Lx = cfg.cell_Lx * super_nx
        self.Ly = cfg.cell_Ly * super_ny
        self.grid_nx = spec.grid_nx
        self.grid_ny = spec.grid_ny
        self.holes = spec.holes
        self.solid_dispersion = solid_dispersion
        self.hole_dispersion = hole_dispersion
        self.reference_wavelength = reference_wavelength

        self._dx = self.Lx / self.grid_nx
        self._dy = self.Ly / self.grid_ny
        x_coords = (np.arange(self.grid_nx) + 0.5) * self._dx - self.Lx / 2.0
        y_coords = (np.arange(self.grid_ny) + 0.5) * self._dy - self.Ly / 2.0
        self._X, self._Y = np.meshgrid(x_coords, y_coords)

    def generate(self, wavelength: float) -> Tuple[np.ndarray, np.ndarray]:
        er_solid = self.solid_dispersion.permittivity_scalar(wavelength)
        ur_solid = self.solid_dispersion.permeability_scalar(wavelength)
        er_hole = self.hole_dispersion.permittivity_scalar(wavelength)
        ur_hole = self.hole_dispersion.permeability_scalar(wavelength)

        er_cell = np.full(
            (self.grid_ny, self.grid_nx, 1),
            er_solid,
            dtype=np.complex128,
        )
        ur_cell = np.full(
            (self.grid_ny, self.grid_nx, 1),
            ur_solid,
            dtype=np.complex128,
        )

        for hole in self.holes:
            x_rel = self._X - hole.x_m
            y_rel = self._Y - hole.y_m
            theta = math.radians(getattr(hole, "rotation_deg", 0.0) or 0.0)
            if theta:
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                x_local = x_rel * cos_t + y_rel * sin_t
                y_local = -x_rel * sin_t + y_rel * cos_t
            else:
                x_local = x_rel
                y_local = y_rel

            shape = hole.shape
            if shape == "circle":
                radius = hole.size1 / 2.0
                mask = (x_local ** 2 + y_local ** 2) <= (radius ** 2)
            elif shape in {"square", "rectangle"}:
                half_w = hole.size1 / 2.0
                half_h = (hole.size2 if hole.size2 is not None else hole.size1) / 2.0
                mask = (np.abs(x_local) <= half_w) & (np.abs(y_local) <= half_h)
            elif shape == "ellipse":
                axis_a = hole.size1 / 2.0
                axis_b = (hole.size2 if hole.size2 is not None else hole.size1) / 2.0
                mask = (x_local / axis_a) ** 2 + (y_local / axis_b) ** 2 <= 1.0
            else:
                # Fallback: treat unknown shapes as circle with adapter diameter
                radius = hole.adapter_diameter / 2.0
                mask = (x_local ** 2 + y_local ** 2) <= (radius ** 2)

            er_cell[mask, 0] = er_hole
            ur_cell[mask, 0] = ur_hole

        return er_cell, ur_cell


def _interp_complex(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    xr = np.asarray(x, dtype=float)
    real = _interp_with_linear_extrap(xr, xp, fp.real)
    imag = _interp_with_linear_extrap(xr, xp, fp.imag)
    return real + 1j * imag


def _interp_with_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if xp.size < 2:
        raise ConfigError("Interpolation requires at least two support points.")

    xp_sorted = np.asarray(xp, dtype=float)
    fp_sorted = np.asarray(fp, dtype=float)
    interp_vals = np.interp(x, xp_sorted, fp_sorted)

    left_mask = x < xp_sorted[0]
    if np.any(left_mask):
        slope = (fp_sorted[1] - fp_sorted[0]) / (xp_sorted[1] - xp_sorted[0])
        interp_vals[left_mask] = fp_sorted[0] + slope * (x[left_mask] - xp_sorted[0])

    right_mask = x > xp_sorted[-1]
    if np.any(right_mask):
        slope = (fp_sorted[-1] - fp_sorted[-2]) / (xp_sorted[-1] - xp_sorted[-2])
        interp_vals[right_mask] = fp_sorted[-1] + slope * (x[right_mask] - xp_sorted[-1])

    return interp_vals


def _build_backing_layer(cfg: AdapterConfig) -> Layer:
    if cfg.backing_type == "metal":
        eps = complex(1.0, cfg.backing_eps_imag)
        metal_material = Material(er=eps, ur=1.0)
        return Layer(material=metal_material)
    return Layer(er=1.0, ur=1.0)


def _prepare_tables(cfg: AdapterConfig) -> AdapterTables:
    layer1_table = DispersionTable(cfg.layer1.csv)

    layer3_table = DispersionTable(cfg.layer3.csv) if cfg.layer3 else None
    mask_solid_table = DispersionTable(cfg.mask.solid_csv) if cfg.mask else None
    mask_hole_table = DispersionTable(cfg.mask.hole_csv) if cfg.mask else None

    return AdapterTables(
        layer1=layer1_table,
        layer3=layer3_table,
        mask_solid=mask_solid_table,
        mask_hole=mask_hole_table,
    )


def _attach_mask_pattern(layer: Layer, pattern: MaskPattern) -> None:
    base_method = Layer.__dict__["set_convolution_matrices"]

    def set_convolution_matrices(self: Layer, n_harmonics):
        wavelength = self.source.wavelength if self.source is not None else pattern.reference_wavelength
        er_cell, ur_cell = pattern.generate(wavelength)
        self.crystal.permittivityCellData = er_cell
        self.crystal.permeabilityCellData = ur_cell
        return base_method(self, n_harmonics)

    layer.set_convolution_matrices = MethodType(set_convolution_matrices, layer)


def _mode_count(n_harmonics: Union[int, Tuple[int, int]]) -> int:
    arr = np.atleast_1d(np.array(n_harmonics, dtype=int))
    return int(np.prod(arr))


def _create_homogeneous_material(table: DispersionTable, n_harmonics: Union[int, Tuple[int, int]]) -> Material:
    modes = _mode_count(n_harmonics)

    def er_func(lam: float) -> np.ndarray:
        val = table.permittivity_scalar(lam)
        return val * np.identity(modes, dtype=np.complex128)

    def ur_func(lam: float) -> np.ndarray:
        val = table.permeability_scalar(lam)
        return val * np.identity(modes, dtype=np.complex128)

    return Material(er=er_func, ur=ur_func)


def _create_mask_layer(
    cfg: AdapterConfig,
    tables: AdapterTables,
    reference_wavelength: float,
) -> Layer:
    assert cfg.mask and tables.mask_solid and tables.mask_hole
    pattern = MaskPattern(cfg, cfg.mask, tables.mask_solid, tables.mask_hole, reference_wavelength)
    er_cell, ur_cell = pattern.generate(reference_wavelength)

    super_nx, super_ny = cfg.supercell
    crystal = Crystal(
        [cfg.cell_Lx * super_nx, 0.0, 0.0],
        [0.0, cfg.cell_Ly * super_ny, 0.0],
        er=er_cell,
        ur=ur_cell,
    )
    mask_layer = Layer(crystal=crystal, thickness=cfg.mask.thickness_m)
    _attach_mask_pattern(mask_layer, pattern)
    return mask_layer


def _build_layer_stack(
    cfg: AdapterConfig,
    tables: AdapterTables,
    reference_wavelength: float,
) -> LayerStack:
    layers: List[Layer] = []

    layers.append(
        Layer(
            material=_create_homogeneous_material(tables.layer1, cfg.n_harmonics),
            thickness=cfg.layer1.thickness_m,
        )
    )

    if cfg.mask:
        layers.append(_create_mask_layer(cfg, tables, reference_wavelength))

    if cfg.layer3 and tables.layer3:
        layers.append(
            Layer(
                material=_create_homogeneous_material(tables.layer3, cfg.n_harmonics),
                thickness=cfg.layer3.thickness_m,
            )
        )

    return LayerStack(
        *layers,
        incident_layer=Layer(er=1.0, ur=1.0),
        transmission_layer=_build_backing_layer(cfg),
    )


def _polarization_vector(name: str) -> np.ndarray:
    if name == "TE":
        return np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    if name == "TM":
        return np.array([0.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    raise ConfigError(f"Unsupported polarization: {name}")


def _real_array(values: Sequence[complex], label: str, tol: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{label} contains non-finite values.")
    max_imag = float(np.max(np.abs(arr.imag)))
    if max_imag > tol:
        raise RuntimeError(f"{label} has non-negligible imaginary parts (max {max_imag:.3e}).")
    return arr.real


def _check_energy(
    R: np.ndarray,
    T: np.ndarray,
    cfg: AdapterConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    A_raw = 1.0 - R - T
    power_sum = R + T + A_raw
    max_violation = float(np.max(np.abs(power_sum - 1.0)))
    min_power = float(np.min(A_raw))
    violation_msg = None
    if max_violation > cfg.energy_tolerance:
        violation_msg = (
            f"Energy conservation violated by {max_violation:.3e} "
            f"(tolerance {cfg.energy_tolerance})."
        )
    if min_power < -cfg.negativity_tolerance:
        extra = (
            f"Negative power component detected ({min_power:.3e}). "
            f"Tolerance is {cfg.negativity_tolerance}."
        )
        violation_msg = f"{violation_msg or ''} {extra}".strip()
    if violation_msg:
        if cfg.strict_energy:
            raise RuntimeError(violation_msg)
        warnings.warn(violation_msg, RuntimeWarning)

    if cfg.strict_energy:
        return A_raw, A_raw, max_violation, min_power
    return A_raw, np.clip(A_raw, 0.0, None), max_violation, min_power


def _run_solver(
    cfg: AdapterConfig,
    tables: AdapterTables,
    wavelengths_m: np.ndarray,
    reference_wavelength: float,
    polarization: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = _build_layer_stack(cfg, tables, reference_wavelength)

    src = Source(
        wavelength=float(wavelengths_m[0]),
        theta=cfg.theta_rad,
        phi=cfg.phi_rad,
        pTEM=_polarization_vector(polarization),
        layer=stack.incident_layer,
    )

    solver = Solver(stack, src, n_harmonics=cfg.n_harmonics)
    results = solver.solve(
        wavelength=wavelengths_m,
        check_convergence=cfg.solver_check_convergence,
        atol=cfg.solver_atol,
        rtol=cfg.solver_rtol,
        max_iters=cfg.solver_max_iters,
    )

    wavelengths = np.asarray(results["wavelength"], dtype=float)
    R = _real_array(results["RTot"], "RTot", tol=1e-8)
    T = _real_array(results["TTot"], "TTot", tol=1e-8)
    return wavelengths, R, T


def run_step1_adapter(config_path: Path, log_dir: Optional[Path] = None) -> Dict[str, Path]:
    cfg = load_config(config_path, override_log_dir=log_dir)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    tables = _prepare_tables(cfg)
    freq_GHz = cfg.frequency.as_array()
    wavelengths_m = C0 / (freq_GHz * GHZ_TO_HZ)
    reference_wavelength = float(np.median(wavelengths_m))

    if cfg.polarization == "AVG":
        wl_te, R_te, T_te = _run_solver(cfg, tables, wavelengths_m, reference_wavelength, "TE")
        _, R_tm, T_tm = _run_solver(cfg, tables, wavelengths_m, reference_wavelength, "TM")
        R = 0.5 * (R_te + R_tm)
        T = 0.5 * (T_te + T_tm)
        wavelengths = wl_te
    else:
        wavelengths, R, T = _run_solver(cfg, tables, wavelengths_m, reference_wavelength, cfg.polarization)

    A_raw, A, max_violation, min_power = _check_energy(R, T, cfg)
    RL = 10.0 * np.log10(np.maximum(R, 1e-16))

    csv_path = cfg.base_dir / f"{cfg.output_prefix}{RESULT_FILENAME_SUFFIX}"
    log_path = cfg.log_dir / f"{cfg.output_prefix}{LOG_FILENAME_SUFFIX}"

    _write_csv(csv_path, freq_GHz, wavelengths, R, T, A_raw, A, RL)
    _write_log(
        log_path=log_path,
        cfg=cfg,
        freq_GHz=freq_GHz,
        wavelengths=wavelengths,
        R=R,
        T=T,
        A_raw=A_raw,
        A=A,
        RL=RL,
        max_violation=max_violation,
        min_power=min_power,
    )

    return {"csv": csv_path, "log": log_path}


def _write_csv(
    destination: Path,
    freq_GHz: np.ndarray,
    wavelengths: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    A_raw: np.ndarray,
    A: np.ndarray,
    RL: np.ndarray,
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["freq_GHz", "wavelength_m", "R", "T", "A_raw", "A", "RL_dB"])
        for row in zip(freq_GHz, wavelengths, R, T, A_raw, A, RL):
            writer.writerow([f"{value:.10e}" for value in row])


def _write_log(
    log_path: Path,
    cfg: AdapterConfig,
    freq_GHz: np.ndarray,
    wavelengths: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    A_raw: np.ndarray,
    A: np.ndarray,
    RL: np.ndarray,
    max_violation: float,
    min_power: float,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    config_record = {
        "polarization": cfg.polarization,
        "n_harmonics": cfg.n_harmonics,
        "layer1_csv": str(cfg.layer1.csv),
        "layer1_thickness_m": cfg.layer1.thickness_m,
        "layer3_csv": str(cfg.layer3.csv) if cfg.layer3 else None,
        "layer3_thickness_m": cfg.layer3.thickness_m if cfg.layer3 else None,
        "mask": {
            "solid_csv": str(cfg.mask.solid_csv),
            "hole_csv": str(cfg.mask.hole_csv),
            "thickness_m": cfg.mask.thickness_m,
            "grid_nx": cfg.mask.grid_nx,
            "grid_ny": cfg.mask.grid_ny,
            "hole_count": len(cfg.mask.holes),
        } if cfg.mask else None,
        "backing_type": cfg.backing_type,
        "backing_eps_imag": cfg.backing_eps_imag,
        "cell_Lx_m": cfg.cell_Lx,
        "cell_Ly_m": cfg.cell_Ly,
        "supercell": {"nx": cfg.supercell[0], "ny": cfg.supercell[1]},
        "energy_tolerance": cfg.energy_tolerance,
        "negativity_tolerance": cfg.negativity_tolerance,
        "strict_energy": cfg.strict_energy,
        "theta_deg": math.degrees(cfg.theta_rad),
        "phi_deg": math.degrees(cfg.phi_rad),
        "solver": {
            "check_convergence": cfg.solver_check_convergence,
            "atol": cfg.solver_atol,
            "rtol": cfg.solver_rtol,
            "max_iters": cfg.solver_max_iters,
        },
    }

    start_record = {"event": "run_start", "timestamp": timestamp, "config": config_record}
    end_record = {
        "event": "run_complete",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "min_R": float(np.min(R)),
            "max_R": float(np.max(R)),
            "min_RL_dB": float(np.min(RL)),
            "max_RL_dB": float(np.max(RL)),
            "max_energy_violation": max_violation,
            "min_A_raw": min_power,
        },
    }

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(start_record) + "\n")
        for idx, freq in enumerate(freq_GHz):
            record = {
                "event": "sample",
                "index": int(idx),
                "freq_GHz": float(freq),
                "wavelength_m": float(wavelengths[idx]),
                "R": float(R[idx]),
                "T": float(T[idx]),
                "A_raw": float(A_raw[idx]),
                "A": float(A[idx]),
                "RL_dB": float(RL[idx]),
            }
            handle.write(json.dumps(record) + "\n")
        handle.write(json.dumps(end_record) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RCWA adapter (homogeneous layers with optional periodic mask)."
    )
    parser.add_argument("config", type=Path, help="Path to JSON configuration file.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional override directory for JSON log output.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        outputs = run_step1_adapter(args.config, log_dir=args.log_dir)
    except ConfigError as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - to ensure clean CLI exit
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved results to {outputs['csv']}")
    print(f"Wrote log to {outputs['log']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
