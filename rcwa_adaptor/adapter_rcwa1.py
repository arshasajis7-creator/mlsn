# adapter_rcwa.py — fixed & stable: strict n_harmonics normalization + robust logging
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.interpolate import interp1d

from rcwa import Material, Layer, LayerStack, Source, Solver, Crystal

C0 = 299792458.0

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
        return self._er_r(f_GHz) + 1j * self._er_i(f_GHz)

    def ur_of_lambda(self, lam_m):
        lam = np.atleast_1d(lam_m)
        f_GHz = (C0 / lam) / 1e9
        return self._ur_r(f_GHz) + 1j * self._ur_i(f_GHz)


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
    """Convert x into a flat list of ints (robust to list/tuple/numpy scalar)."""
    try:
        arr = np.array(x).astype(int).ravel().tolist()
        return [int(v) for v in arr]
    except Exception:
        try:
            return [int(x)]
        except Exception:
            return [1]


def _nh_tuple_or_int(harm_cfg: Optional[Dict[str, Any]], has_mask: bool):
    """Return a 3-tuple (P,Q,R) if periodic layer exists; else a single int.
    Pads with 1s or truncates to length 3 deterministically.
    """
    harm_cfg = harm_cfg or {}
    if "n_3d" in harm_cfg:
        vals = _flatten_ints(harm_cfg["n_3d"])
    elif "n_2d" in harm_cfg:
        vals = _flatten_ints(harm_cfg["n_2d"]) + [1]  # -> [nx, ny, 1]
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
    """Force any input to a 3-int tuple (P,Q,R): flatten, cast, pad, truncate."""
    import numpy as _np
    arr = _np.array(x).astype(int).ravel().tolist()
    # اگر خالی شد، پیش‌فرض
    if len(arr) == 0:
        arr = [1, 1, 1]
    # فلت + پد + تریم به طول 3
    nh3 = tuple((arr + [1, 1, 1])[:3])
    return (int(nh3[0]), int(nh3[1]), int(nh3[2]))

# ---------------- main adapter ----------------
def run_adapter(cfg: Dict[str, Any], out_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    base_out = Path(out_dir) if out_dir else Path("outputs")
    run_dir = prepare_run_dir(base_out)
    logger = JSONLLogger(run_dir / "logs" / "run.log.jsonl")
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    f = np.linspace(cfg["freq_GHz"]["start"], cfg["freq_GHz"]["stop"], cfg["freq_GHz"]["points"])
    wls = C0 / (f * 1e9)
    logger.log_event({"event": "freq_setup", "f_start": float(f[0]), "f_stop": float(f[-1]), "points": int(len(f))})

    incident = Layer(er=1, ur=1)

    if cfg["backing"]["type"] == "metal":
        metal_eps_imag = float(cfg["backing"].get("eps_imag_clamp", 1e8))
        transmission = Layer(
            material=Material(er=lambda lam: 1 + 1j * metal_eps_imag, ur=1),
            thickness=float(cfg["backing"].get("thickness_m", 1e-3)),
        )
    else:
        transmission = Layer(er=1, ur=1)

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
                     # --- FORCE final harmonics exactly 3-tuple for RCWA periodic layers ---
if has_mask:
    nh_solver3 = _enforce_nh3(nh)           # هرچی هست → (P,Q,R)
else:
    nh_solver3 = _enforce_nh3([nh, 1, 1])   # ساختار همگن → (n,1,1) برای سازگاری
logger.log_event({
    "event": "harmonics_final_for_solver",
    "value": list(nh_solver3),
    "len": 3
})
# (اختیاری ولی مفید) اطمینان سفت‌وسخت:
assert isinstance(nh_solver3, tuple) and len(nh_solver3) == 3, "nh must be 3-tuple"
 
    # --- FORCE final harmonics exactly as RCWA expects ---
    if has_mask:
        # سه‌تایی دقیق (P,Q,R) بساز؛ اگر اشتباهی اسکالر بود با  (n,1,1) پر کن
        nh_solver = (int(nh[0]), int(nh[1]), int(nh[2])) if not isinstance(nh, int) else (int(nh), 1, 1)
        logger.log_event({"event": "harmonics_final_for_solver", "type": "tuple3", "value": list(nh_solver)})
    else:
        # بدون لایهٔ دوره‌ای، یک اسکالر کافی است
        nh_solver = int(nh) if isinstance(nh, int) else int((nh[0] if len(nh) > 0 else 1))
        logger.log_event({"event": "harmonics_final_for_solver", "type": "int", "value": nh_solver})

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

    # convenience copy
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
