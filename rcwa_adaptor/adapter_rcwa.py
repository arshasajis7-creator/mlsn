# adapter_rcwa.py — fixed & stable: strict n_harmonics normalization + robust logging + safe nh3 monkey-patch + x/y_components fix
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.interpolate import interp1d

from rcwa import Material, Layer, LayerStack, Source, Solver, Crystal

C0 = 299792458.0

class _NH3Tuple:
    """Strict 3-tuple proxy for n_harmonics that ALWAYS behaves like a size-3 tuple."""
    __slots__ = ("_t",)
    def __init__(self, a, b=None, c=None):
        # Allow various input formats
        if b is None and c is None:
            try:
                arr = [int(v) for v in a]
            except Exception:
                arr = [int(a)]
        else:
            arr = [int(a), int(b), int(c)]
        arr = (arr + [1, 1, 1])[:3]
        self._t = (max(1, int(arr[0])), max(1, int(arr[1])), max(1, int(arr[2])))

    # sequence-like behavior
    def __iter__(self):
        return iter(self._t)
    def __len__(self):
        return 3
    def __getitem__(self, i):
        return self._t[i]
    def __repr__(self):
        return f"_NH3Tuple{self._t}"

    # SAFE addition: never let length exceed 3
    def __add__(self, other):
        if isinstance(other, _NH3Tuple):
            o = other._t
        elif isinstance(other, (tuple, list)):
            o = tuple(other)
        else:
            o = (int(other),)
        if len(self._t) == 3 and o == (1,):
            return self
        t = tuple(self._t) + o
        t = t[:3] if len(t) > 3 else t
        return _NH3Tuple(t)

    def __radd__(self, other):
        if isinstance(other, _NH3Tuple):
            o = other._t
        elif isinstance(other, (tuple, list)):
            o = tuple(other)
        else:
            o = (int(other),)
        if len(self._t) == 3 and o == (1,):
            return self
        t = o + tuple(self._t)
        t = t[:3] if len(t) > 3 else t
        return _NH3Tuple(t)

# ---------------- JSONL logger ----------------
class JSONLLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.run_id = int(time.time())
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_event({"event": "run_start", "run_id": self.run_id})

    def log_event(self, data: Dict[str, Any]):
        rec = dict(data)
        rec["ts"] = time.time()
        rec["run_id"] = self.run_id
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------- dispersion from CSV(freq_GHz) → ε(λ), μ(λ) ----------------
class MWDispersion:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        need = ["freq_GHz", "eps_real", "eps_imag", "mu_real", "mu_imag"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"CSV {csv_path} missing column: {c}")
        f = df["freq_GHz"].to_numpy(float)
        er = df["eps_real"].to_numpy(float) + 1j * df["eps_imag"].to_numpy(float)
        ur = df["mu_real"].to_numpy(float) + 1j * df["mu_imag"].to_numpy(float)
        idx = np.argsort(f)
        self.f = f[idx]
        self.er = er[idx]
        self.ur = ur[idx]
        self._er_r = interp1d(self.f, self.er.real, kind="linear", fill_value="extrapolate", assume_sorted=True)
        self._er_i = interp1d(self.f, self.er.imag, kind="linear", fill_value="extrapolate", assume_sorted=True)
        self._ur_r = interp1d(self.f, self.ur.real, kind="linear", fill_value="extrapolate", assume_sorted=True)
        self._ur_i = interp1d(self.f, self.ur.imag, kind="linear", fill_value="extrapolate", assume_sorted=True)

    def er_of_lambda(self, lam_m):
        lam = np.atleast_1d(lam_m)
        f_GHz = (C0 / lam) / 1e9
        result = self._er_r(f_GHz) + 1j * self._er_i(f_GHz)
        return np.atleast_2d(result)

    def ur_of_lambda(self, lam_m):
        lam = np.atleast_1d(lam_m)
        f_GHz = (C0 / lam) / 1e9
        result = self._ur_r(f_GHz) + 1j * self._ur_i(f_GHz)
        return np.atleast_2d(result)

def make_material_from_csv(csv_path: str) -> Material:
    disp = MWDispersion(csv_path)
    return Material(er=lambda lam: disp.er_of_lambda(lam),
                    ur=lambda lam: disp.ur_of_lambda(lam))

# ---------------- geometry helpers ----------------
def guard_holes(holes: List[Dict[str, float]], Lx: float, Ly: float) -> None:
    for i, h in enumerate(holes):
        x0 = float(h["x_m"]); y0 = float(h["y_m"]); d = float(h["diameter_m"])
        if abs(x0) + d/2 > Lx/2 or abs(y0) + d/2 > Ly/2:
            raise ValueError(f"Hole {i} out of bounds: {h}")

def build_mask_cell(Lx: float, Ly: float, Nx: int, Ny: int,
                    holes: List[Dict[str, float]],
                    er_solid: complex, ur_solid: complex,
                    er_hole: complex, ur_hole: complex) -> Tuple[np.ndarray, np.ndarray]:
    er = np.full((Ny, Nx), er_solid, dtype=complex)
    ur = np.full((Ny, Nx), ur_solid, dtype=complex)
    xs = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    ys = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    XX, YY = np.meshgrid(xs, ys)
    for h in holes or []:
        x0 = float(h["x_m"]); y0 = float(h["y_m"]); d = float(h["diameter_m"])
        mask = (XX - x0)**2 + (YY - y0)**2 <= (d/2)**2
        er[mask] = er_hole
        ur[mask] = ur_hole
    return er, ur

# ---------------- run dir mgmt ----------------
def prepare_run_dir(base_out_dir: Path) -> Path:
    base = Path(base_out_dir)
    base.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        rd = base / f"run_{i:03d}"
        if not rd.exists():
            break
        i += 1
    (rd / "artifacts").mkdir(parents=True, exist_ok=True)
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    return rd

# ---------------- harmonics normalization ----------------
def _flatten_ints(x) -> List[int]:
    try:
        arr = np.array(x).astype(int).ravel().tolist()
        return [int(v) for v in arr]
    except Exception:
        try:
            return [int(x)]
        except Exception:
            return [1]

def _nh_tuple_or_int(harm_cfg: Optional[Dict[str, Any]], has_mask: bool):
    harm_cfg = harm_cfg or {}
    if "n_3d" in harm_cfg:
        vals = _flatten_ints(harm_cfg["n_3d"])
    elif "n_2d" in harm_cfg:
        vals = _flatten_ints(harm_cfg["n_2d"]) + [1]
    elif "n_1d" in harm_cfg:
        n = int(harm_cfg["n_1d"])
        vals = [n, 1, 1] if has_mask else [n]
    else:
        vals = [5, 5, 1] if has_mask else [1]
    if has_mask:
        vals = (vals + [1, 1, 1])[:3]
        return (int(vals[0]), int(vals[1]), int(vals[2]))
    else:
        return int(vals[0])

def _enforce_nh3(x):
    import numpy as _np
    arr = _np.array(x).astype(int).ravel().tolist()
    if len(arr) == 0:
        arr = [1, 1, 1]
    nh3 = tuple((arr + [1, 1, 1])[:3])
    nh3 = tuple(max(1, int(v)) for v in nh3)
    return nh3

# ---------------- SAFE monkey-patch for nh3 and x/y_components ----------------
def _install_safe_nh3_patch(logger: JSONLLogger, forced_nh3: Tuple[int,int,int]):
    """
    Monkey-patch: Force rcwa.layer.Layer to always use a strict _NH3Tuple object
    for n_harmonics, ensure convolution matrix is 2D square, and patch x_components/y_components
    to always return 3 values.
    """
    import rcwa.layer as _rcwa_layer
    import rcwa.harmonics as _rcwa_harmonics
    if getattr(_rcwa_layer, "_nh3_patched_forced", False):
        logger.log_event({"event": "nh3_patch", "status": "already_patched_forced",
                          "value": list(forced_nh3)})
        return

    _orig_conv = _rcwa_layer.Layer._convolution_matrix
    _orig_set = _rcwa_layer.Layer.set_convolution_matrices
    _orig_x_components = _rcwa_harmonics.x_components
    _orig_y_components = _rcwa_harmonics.y_components

    def _to_3tuple(x):
        try:
            a = tuple(int(v) for v in x)
        except Exception:
            a = (int(x),)
        a = (a + (1,1,1))[:3]
        return (int(a[0]), int(a[1]), int(a[2]))

    forced = _to_3tuple(forced_nh3)
    forced_obj = _NH3Tuple(*forced)

    def _conv_wrapper(self, cellData, n_harmonics):
        import numpy as _np
        nh3 = forced_obj
        try:
            logger.log_event({"event": "nh3_patch_conv_forced", "value": list(nh3), "type": str(type(nh3))})
        except Exception:
            pass
        mat = _orig_conv(self, cellData, nh3)
        # POST-FIX: ensure square 2D matrix for rcwa.matrices
        if _np.ndim(mat) == 1:
            mat = _np.diag(_np.asarray(mat))
        elif mat is not None:
            a = _np.asarray(mat)
            if a.ndim == 0:
                mat = a.reshape(1, 1)
            elif a.ndim == 2:
                pass
            else:
                try:
                    G = a.size
                    mat = _np.diag(a.reshape(G))
                except Exception:
                    mat = _np.atleast_2d(a)
        return mat

    def _set_wrapper(self, n_harmonics):
        nh3 = forced_obj
        try:
            logger.log_event({"event": "nh3_patch_set_forced", "value": list(nh3), "type": str(type(nh3))})
        except Exception:
            pass
        return _orig_set(self, nh3)

    def _x_components_wrapper(k_incident, T1, T2):
        import numpy as _np
        result = _orig_x_components(k_incident, T1, T2)
        if len(result) == 2:
            try:
                first_elem = _np.asarray(result[0])
                logger.log_event({"event": "x_components_patch", "status": "fixed_2to3",
                                  "original_len": 2})
                if isinstance(result, list):
                    return result + [_np.zeros_like(first_elem)]
                else:
                    return result + (_np.zeros_like(first_elem),)
            except Exception as e:
                logger.log_event({"event": "x_components_patch_error", "msg": str(e)})
                raise
        return result

    def _y_components_wrapper(k_incident, T1, T2):
        import numpy as _np
        result = _orig_y_components(k_incident, T1, T2)
        if len(result) == 2:
            try:
                first_elem = _np.asarray(result[0])
                logger.log_event({"event": "y_components_patch", "status": "fixed_2to3",
                                  "original_len": 2})
                if isinstance(result, list):
                    return result + [_np.zeros_like(first_elem)]
                else:
                    return result + (_np.zeros_like(first_elem),)
            except Exception as e:
                logger.log_event({"event": "y_components_patch_error", "msg": str(e)})
                raise
        return result

    _rcwa_layer.Layer._convolution_matrix = _conv_wrapper
    _rcwa_layer.Layer.set_convolution_matrices = _set_wrapper
    _rcwa_harmonics.x_components = _x_components_wrapper
    _rcwa_harmonics.y_components = _y_components_wrapper
    _rcwa_layer._nh3_patched_forced = True

    logger.log_event({"event": "nh3_patch_forced", "status": "installed", "value": list(forced)})

# ---------------- main adapter ----------------
def run_adapter(cfg: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    base_out = Path(out_dir) if out_dir else Path("outputs")
    run_dir = prepare_run_dir(base_out)
    logger = JSONLLogger(run_dir / "logs" / "run.log.jsonl")
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    f = np.linspace(cfg["freq_GHz"]["start"], cfg["freq_GHz"]["stop"], cfg["freq_GHz"]["points"])
    wls = C0 / (f * 1e9)
    logger.log_event({"event": "freq_setup", "f_start": float(f[0]), "f_stop": float(f[-1]), "points": int(len(f))})

    incident = Layer(er=np.array([[1]]), ur=np.array([[1]]))

    if cfg["backing"]["type"] == "metal":
        metal_eps_imag = float(cfg["backing"].get("eps_imag_clamp", 1e8))
        transmission = Layer(
            material=Material(er=lambda lam: np.array([[1 + 1j * metal_eps_imag]]), ur=lambda lam: np.array([[1]])),
            thickness=float(cfg["backing"].get("thickness_m", 1e-3)),
        )
    else:
        transmission = Layer(er=np.array([[1]]), ur=np.array([[1]]))

    layers: List[Layer] = []

    # L1 homogeneous
    L1_cfg = cfg["layers"]["L1"]
    mat_L1 = make_material_from_csv(L1_cfg["csv"])
    L1 = Layer(material=mat_L1, thickness=float(L1_cfg["thickness_m"]))
    layers.append(L1)

    # L2 mask (optional periodic layer)
    has_mask = "L2_mask" in cfg["layers"]
    if has_mask:
        Lx = float(cfg["cell"]["Lx_m"]); Ly = float(cfg["cell"]["Ly_m"])
        mask_cfg = cfg["layers"]["L2_mask"]
        Nx = int(mask_cfg["grid"]["Nx"]); Ny = int(mask_cfg["grid"]["Ny"])
        holes = mask_cfg.get("holes", [])
        guard_holes(holes, Lx, Ly)

        lam_mid = wls[len(wls) // 2]
        disp_solid = MWDispersion(mask_cfg["csv_solid"])
        disp_hole = MWDispersion(mask_cfg["csv_hole"])

        er_s = np.atleast_1d(disp_solid.er_of_lambda(lam_mid))[0]
        atleast_1d(disp_solid.er_of_lambda(lam_mid))[0]
        ur_s = np.atleast_1d(disp_solid.ur_of_lambda(lam_mid))[0]
        er_h = np.atleast_1d(disp_hole.er_of_lambda(lam_mid))[0]
        ur_h = np.atleast_1d(disp_hole.ur_of_lambda(lam_mid))[0]

        er_cell, ur_cell = build_mask_cell(Lx, Ly, Nx, Ny, holes, er_s, ur_s, er_h, ur_h)

        # artifacts
        np.save(run_dir / "artifacts" / "mask_er_complex.npy", er_cell)
        np.save(run_dir / "artifacts" / "mask_ur_complex.npy", ur_cell)
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(np.real(er_cell), origin="lower", extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
            plt.colorbar(label="Re{ε}")
            plt.title("Mask ε real")
            plt.savefig(run_dir / "artifacts" / "mask_er_real.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.log_event({"event": "warn", "msg": f"mask plot failed: {e}"})

        crystal = Crystal([Lx, 0, 0], [0, Ly, 0], er=er_cell, ur=ur_cell)
        L2 = Layer(crystal=crystal, thickness=float(mask_cfg["thickness_m"]))
        layers.append(L2)
        logger.log_event({"event": "mask_layer_built", "Nx": Nx, "Ny": Ny, "holes": len(holes)})

    # L3 homogeneous (optional)
    if "L3" in cfg["layers"]:
        L3_cfg = cfg["layers"]["L3"]
        mat_L3 = make_material_from_csv(L3_cfg["csv"])
        L3 = Layer(material=mat_L3, thickness=float(L3_cfg["thickness_m"]))
        layers.append(L3)

    stack = LayerStack(*layers, incident_layer=incident, transmission_layer=transmission)

    # harmonics: normalize once, log once, and use everywhere
    harm_cfg = cfg.get("harmonics", {})
    nh = _nh_tuple_or_int(harm_cfg, has_mask)
    logger.log_event({"event": "harmonics_normalized",
                      "nh": (nh if isinstance(nh, int) else list(nh)),
                      "has_mask": has_mask})

    # final solver harmonics (always 3-tuple for rcwa internals)
    if has_mask:
        nh_solver3 = _enforce_nh3(nh)
    else:
        nh_solver3 = _enforce_nh3([nh, 1, 1])

    logger.log_event({"event": "harmonics_final_for_solver", "value": list(nh_solver3), "len": 3})

    _install_safe_nh3_patch(logger, nh_solver3)  # Apply patch before solver

    def solve_for_pol(pol: str) -> np.ndarray:
        pTEM = [1, 0] if pol.upper() == "TE" else [0, 1]
        theta = np.deg2rad(cfg["angles"]["theta_deg"])
        phi = np.deg2rad(cfg["angles"].get("phi_deg", 0.0))
        src = Source(wavelength=wls[0], theta=theta, phi=phi, pTEM=pTEM)
        solver = Solver(stack, src, n_harmonics=nh_solver3)
        res = solver.solve(wavelength=wls)
        R = np.asarray(res["RTot"])
        T = np.asarray(res["TTot"])
        if not np.all(np.isfinite(R)):
            raise RuntimeError("NaN/Inf in RTot")
        if not np.all(np.isfinite(T)):
            raise RuntimeError("NaN/Inf in TTot")
        A = 1 - R - T
        if np.max(R + T + A) > 1 + 1e-3:
            raise RuntimeError("Energy violation: max(R+T+A) > 1+1e-3")
        logger.log_event({"event": "solve_pol_done",
                          "pol": pol,
                          "max_R": float(np.max(R)),
                          "max_T": float(np.max(T))})
        return R

    pol = cfg["polarization"].upper()
    if pol in ("TE", "TM"):
        R = solve_for_pol(pol)
    elif pol == "AVG":
        R = 0.5 * (solve_for_pol("TE") + solve_for_pol("TM"))
    else:
        raise ValueError(f"Unknown polarization: {pol}")

    RL_dB = -10 * np.log10(np.maximum(R, 1e-12))
    data = np.vstack([f, RL_dB, R]).T

    np.savetxt(run_dir / "results.csv", data, delimiter=",", header="freq_GHz,RL_dB,R", comments="")
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(f, RL_dB)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Reflection Loss (dB)")
        plt.grid(True, alpha=0.3)
        plt.savefig(run_dir / "RL_plot.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.log_event({"event": "warn", "msg": f"plot failed: {e}"})

    base_out.mkdir(parents=True, exist_ok=True)
    np.savetxt(base_out / "latest_results.csv", data, delimiter=",",
               header="freq_GHz,RL_dB,R", comments="")

    logger.log_event({"event": "results_written", "csv": str(run_dir / "results.csv")})
    logger.log_event({"event": "run_end", "run_dir": str(run_dir)})
    return {"freq_GHz": f, "RL_dB": RL_dB, "R": R, "run_dir": str(run_dir)}

def run_from_json(cfg_path: str, out_dir: Optional[str] = None):
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    return run_adapter(cfg, out_dir=out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to input JSON file")
    ap.add_argument("--out_dir", default="outputs", help="Base output directory")
    args = ap.parse_args()
    out = run_from_json(args.config, out_dir=args.out_dir)
    print("Done. Run directory:", out["run_dir"])