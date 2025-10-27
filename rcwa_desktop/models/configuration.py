from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Primitive dataclasses


@dataclass
class FrequencySpec:
    start: float = 1.0
    stop: float = 18.0
    points: int = 21

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "FrequencySpec":
        return cls(
            start=float(data.get("start", 1.0)),
            stop=float(data.get("stop", 18.0)),
            points=int(data.get("points", 21)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {"start": self.start, "stop": self.stop, "points": self.points}


@dataclass
class CellSpec:
    Lx_m: float = 0.03
    Ly_m: float = 0.03

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CellSpec":
        return cls(
            Lx_m=float(data.get("Lx_m", 0.03)),
            Ly_m=float(data.get("Ly_m", 0.03)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {"Lx_m": self.Lx_m, "Ly_m": self.Ly_m}


@dataclass
class LayerSpec:
    material_csv: str = ""
    thickness_m: float = 0.0

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LayerSpec":
        return cls(
            material_csv=str(data.get("csv", "")),
            thickness_m=float(data.get("thickness_m", 0.0)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {"csv": self.material_csv, "thickness_m": self.thickness_m}


@dataclass
class MaskHole:
    shape: str = "circle"  # circle, square, rectangle, ellipse
    x_m: float = 0.0
    y_m: float = 0.0
    size1: float = 0.0  # diameter, width or major axis
    size2: float | None = None  # height or minor axis
    rotation_deg: float = 0.0

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MaskHole":
        shape = str(data.get("type", data.get("shape", "circle"))).lower()
        valid_shapes = {"circle", "square", "rectangle", "ellipse"}
        if shape not in valid_shapes:
            shape = "circle"

        x = float(data.get("x_m", data.get("x", 0.0)))
        y = float(data.get("y_m", data.get("y", 0.0)))
        rotation = float(data.get("rotation_deg", data.get("rotation", data.get("theta_deg", 0.0))))

        def clamp_positive(value: float) -> float:
            return max(float(value), 0.0)

        diameter = float(data.get("diameter_m", data.get("size1", 0.0)))
        width = float(data.get("width_m", data.get("size1", diameter)))
        height = float(data.get("height_m", data.get("size2", width)))
        axis_a = float(data.get("axis_a_m", data.get("size1", diameter)))
        axis_b = float(data.get("axis_b_m", data.get("size2", axis_a)))

        if shape == "circle":
            size1 = clamp_positive(diameter)
            size2 = None
        elif shape in {"square", "rectangle"}:
            size1 = clamp_positive(width)
            size2 = clamp_positive(height)
        elif shape == "ellipse":
            size1 = clamp_positive(axis_a)
            size2 = clamp_positive(axis_b)
        else:  # fallback for unexpected shape values
            size1 = clamp_positive(diameter)
            size2 = None

        return cls(shape=shape, x_m=x, y_m=y, size1=size1, size2=size2, rotation_deg=rotation)

    def adapter_diameter(self) -> float:
        """Return a circle diameter compatible with the RCWA adapter."""

        if self.shape in {"square", "rectangle", "ellipse"}:
            other = self.size2 if self.size2 is not None else self.size1
            return max(min(self.size1, other), 0.0)
        return max(self.size1, 0.0)

    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": self.shape,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "rotation_deg": self.rotation_deg,
            "diameter_m": self.adapter_diameter(),
        }
        if self.shape == "circle":
            data["diameter_m"] = self.size1
        elif self.shape in {"square", "rectangle"}:
            height = self.size2 if self.size2 is not None else self.size1
            data.update({
                "width_m": self.size1,
                "height_m": height,
            })
        elif self.shape == "ellipse":
            axis_b = self.size2 if self.size2 is not None else self.size1
            data.update({
                "axis_a_m": self.size1,
                "axis_b_m": axis_b,
            })
        return data


@dataclass
class MaskSpec:
    solid_csv: str = ""
    hole_csv: str = ""
    thickness_m: float = 0.0015
    grid_nx: int = 48
    grid_ny: int = 48
    holes: List[MaskHole] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MaskSpec":
        holes = [MaskHole.from_json(item) for item in data.get("holes", [])]
        grid = data.get("grid", {})
        return cls(
            solid_csv=str(data.get("csv_solid", "")),
            hole_csv=str(data.get("csv_hole", "")),
            thickness_m=float(data.get("thickness_m", 0.0015)),
            grid_nx=int(grid.get("Nx", 48)),
            grid_ny=int(grid.get("Ny", 48)),
            holes=holes,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "csv_solid": self.solid_csv,
            "csv_hole": self.hole_csv,
            "thickness_m": self.thickness_m,
            "grid": {"Nx": self.grid_nx, "Ny": self.grid_ny},
            "holes": [hole.to_json() for hole in self.holes],
        }


@dataclass
class BackingSpec:
    type: str = "metal"
    eps_imag_clamp: float = 1e8

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "BackingSpec":
        return cls(
            type=str(data.get("type", "metal")),
            eps_imag_clamp=float(data.get("eps_imag_clamp", 1e8)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {"type": self.type, "eps_imag_clamp": self.eps_imag_clamp}


@dataclass
class SolverSpec:
    check_convergence: bool = False
    atol: float = 1e-3
    rtol: float = 1e-2
    max_iters: int = 50

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SolverSpec":
        return cls(
            check_convergence=bool(data.get("check_convergence", False)),
            atol=float(data.get("atol", 1e-3)),
            rtol=float(data.get("rtol", 1e-2)),
            max_iters=int(data.get("max_iters", 50)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "check_convergence": self.check_convergence,
            "atol": self.atol,
            "rtol": self.rtol,
            "max_iters": self.max_iters,
        }


@dataclass
class ToleranceSpec:
    energy: float = 0.5
    nonnegativity: float = 1e-3
    strict: bool = False

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ToleranceSpec":
        return cls(
            energy=float(data.get("energy", 0.5)),
            nonnegativity=float(data.get("nonnegativity", 1e-3)),
            strict=bool(data.get("strict", False)),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "energy": self.energy,
            "nonnegativity": self.nonnegativity,
            "strict": self.strict,
        }


# ---------------------------------------------------------------------------
# Complete configuration


@dataclass
class Configuration:
    freq: FrequencySpec = field(default_factory=FrequencySpec)
    cell: CellSpec = field(default_factory=CellSpec)
    layer_top: LayerSpec = field(
        default_factory=lambda: LayerSpec(material_csv="m1.csv", thickness_m=0.003)
    )
    mask: MaskSpec = field(
        default_factory=lambda: MaskSpec(
            solid_csv="m3.csv",
            hole_csv="mhole.csv",
            thickness_m=0.005,
            grid_nx=32,
            grid_ny=32,
            holes=[MaskHole(size1=0.01)],
        )
    )
    layer_bottom: LayerSpec = field(
        default_factory=lambda: LayerSpec(material_csv="m3.csv", thickness_m=0.004)
    )
    polarization: str = "TE"
    n_harmonics: List[int] = field(default_factory=lambda: [11, 11])
    theta_deg: float = 0.0
    phi_deg: float = 0.0
    output_prefix: str = "desktop_config"
    backing: BackingSpec = field(default_factory=BackingSpec)
    solver: SolverSpec = field(default_factory=SolverSpec)
    tolerances: ToleranceSpec = field(default_factory=ToleranceSpec)

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "Configuration":
        layers = payload.get("layers", {})
        return cls(
            freq=FrequencySpec.from_json(payload.get("freq_GHz", {})),
            cell=CellSpec.from_json(payload.get("cell", {})),
            layer_top=LayerSpec.from_json(layers.get("L1", {})),
            mask=MaskSpec.from_json(layers.get("L2_mask", {})),
            layer_bottom=LayerSpec.from_json(layers.get("L3", {})),
            polarization=str(payload.get("polarization", "TE")).upper(),
            n_harmonics=list(payload.get("n_harmonics", [11, 11])),
            theta_deg=float(payload.get("angles", {}).get("theta_deg", 0.0)),
            phi_deg=float(payload.get("angles", {}).get("phi_deg", 0.0)),
            output_prefix=str(payload.get("output_prefix", "desktop_config")),
            backing=BackingSpec.from_json(payload.get("backing", {})),
            solver=SolverSpec.from_json(payload.get("solver", {})),
            tolerances=ToleranceSpec.from_json(payload.get("tolerances", {})),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": "desktop-stage3",
            "freq_GHz": self.freq.to_json(),
            "cell": self.cell.to_json(),
            "layers": {
                "L1": self.layer_top.to_json(),
                "L2_mask": self.mask.to_json(),
                "L3": self.layer_bottom.to_json(),
            },
            "polarization": self.polarization,
            "n_harmonics": self.n_harmonics,
            "angles": {"theta_deg": self.theta_deg, "phi_deg": self.phi_deg},
            "output_prefix": self.output_prefix,
            "backing": self.backing.to_json(),
            "solver": self.solver.to_json(),
            "tolerances": self.tolerances.to_json(),
        }


# ---------------------------------------------------------------------------
# I/O utilities


def load_configuration(path: Path) -> Configuration:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Configuration.from_json(payload)


def save_configuration(config: Configuration, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_json(), handle, indent=2)
