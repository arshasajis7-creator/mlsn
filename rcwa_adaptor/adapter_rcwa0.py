
# adapter_rcwa.py  — final, stable adapter using installed rcwa core (v2 with nh normalization)
import json, time, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.interpolate import interp1d

# Import directly from the rcwa package (your __init__.py is fixed)
from rcwa import Material, Layer, LayerStack, Source, Solver, Crystal

def _flatten_ints(x):
    \"\"\"Convert x into a flat list of ints (robust to list/tuple/numpy scalar).\"\"\"
    try:
        import numpy as _np
        arr = _np.array(x).astype(int).ravel().tolist()
        return [int(v) for v in arr]
    except Exception:
        try:
            return [int(x)]
        except Exception:
            return [1]

def _nh_tuple_or_int(harm_cfg, has_mask: bool):
    \"\"\"Return a 3-tuple (P,Q,R) if periodic layer exists; else an int.
    Pads with 1s or truncates to length 3 deterministically.\"\"\"
    harm_cfg = harm_cfg or {}
    if 'n_3d' in harm_cfg:
        vals = _flatten_ints(harm_cfg['n_3d'])
    elif 'n_2d' in harm_cfg:
        vals = _flatten_ints(harm_cfg['n_2d']) + [1]      # -> [nx, ny, 1]
    elif 'n_1d' in harm_cfg:
        n = int(harm_cfg['n_1d'])
        vals = [n, 1, 1] if has_mask else [n]
    else:
        vals = [5, 5, 1] if has_mask else [1]
    if has_mask:
        vals = (vals + [1, 1, 1])[:3]
        return (int(vals[0]), int(vals[1]), int(vals[2]))
    else:
        return int(vals[0])


C0 = 299792458.0

# ---------- Simple structured logger (JSONL)
class JSONLLogger:
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.run_id = int(time.time())
        self._ensure_parent()
        self.log_event({"event": "run_start", "run_id": self.run_id})

    def _ensure_parent(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, data: Dict[str, Any]):
        data = dict(data)
        data["ts"] = time.time()
        data["run_id"] = self.run_id
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

# ---------- Dispersion loader: CSV(freq_GHz, eps_real, eps_imag, mu_real, mu_imag) → ε(λ), μ(λ)
class MWDispersion:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        need = ['freq_GHz', 'eps_real', 'eps_imag', 'mu_real', 'mu_imag']
        for col in need:
            if col not in df.columns:
                raise ValueError(f"CSV {csv_path} missing column: {col}")
        f = df['freq_GHz'].to_numpy(float)
        er = df['eps_real'].to_numpy(float) + 1j * df['eps_imag'].to_numpy(float)
        ur = df['mu_real' ].to_numpy(float) + 1j * df['mu_imag' ].to_numpy(float)
        idx = np.argsort(f)
        self.f = f[idx]; self.er = er[idx]; self.ur = ur[idx]
        self._er_r = interp1d(self.f, self.er.real, kind='linear', fill_value='extrapolate', assume_sorted=True)
        self._er_i = interp1d(self.f, self.er.imag, kind='linear', fill_value='extrapolate', assume_sorted=True)
        self._ur_r = interp1d(self.f, self.ur.real, kind='linear', fill_value='extrapolate', assume_sorted=True)
        self._ur_i = interp1d(self.f, self.ur.imag, kind='linear', fill_value='extrapolate', assume_sorted=True)

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

# ---------- Geometry helpers
def guard_holes(holes: List[Dict[str, float]], Lx: float, Ly: float) -> None:
    for i, h in enumerate(holes):
        x0 = float(h['x_m']); y0 = float(h['y_m']); d = float(h['diameter_m'])
        if abs(x0) + d/2 > Lx/2 or abs(y0) + d/2 > Ly/2:
            raise ValueError(f"Hole {i} out of bounds: {h}")

def build_mask_cell(Lx: float, Ly: float, Nx: int, Ny: int,
                    holes: List[Dict[str, float]],
                    er_solid: complex, ur_solid: complex,
                    er_hole: complex,  ur_hole: complex) -> Tuple[np.ndarray, np.ndarray]:
    er = np.full((Ny, Nx), er_solid, dtype=complex)
    ur = np.full((Ny, Nx), ur_solid, dtype=complex)
    xs = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    ys = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    XX, YY = np.meshgrid(xs, ys)
    for h in holes or []:
        x0 = float(h['x_m']); y0 = float(h['y_m']); d = float(h['diameter_m'])
        mask = (XX - x0)**2 + (YY - y0)**2 <= (d/2)**2
        er[mask] = er_hole
        ur[mask] = ur_hole
    return er, ur

# ---------- Run dir management
def prepare_run_dir(base_out_dir: Path) -> Path:
    base_out_dir = Path(base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in base_out_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    next_idx = 1
    if existing:
        def idx(p):
            s = p.name.split("_")[-1]
            try:
                return int(s)
            except:
                return -1
        last = max(existing, key=idx)
        last_i = idx(last)
        if last_i >= 1:
            next_idx = last_i + 1
    run_dir = base_out_dir / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir

def _normalize_nh(nh_in, has_mask: bool):
    """Ensure n_harmonics matches what the RCWA layer expects:
       - periodic layer present (has_mask=True): 3-tuple (P,Q,R)
       - no periodic layer: single int
    """
    # numpy arrays / lists → python list
    if isinstance(nh_in, (tuple, list, np.ndarray)):
        vals = [int(v) for v in nh_in]
    else:
        vals = [int(nh_in)]
    if has_mask:
        # expand/truncate to exactly 3
        if len(vals) < 3:
            vals = vals + [1] * (3 - len(vals))
        elif len(vals) > 3:
            vals = vals[:3]
        return tuple(vals)
    else:
        # no periodicity → int
        return int(vals[0]) if vals else 1

# ---------- Adapter main
def run_adapter(cfg: Dict[str, Any], out_dir: Optional[str]=None) -> Dict[str, np.ndarray]:
    base_out = Path(out_dir) if out_dir else Path("outputs")
    run_dir = prepare_run_dir(base_out)
    logger = JSONLLogger(run_dir / "logs" / "run.log.jsonl")
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    f = np.linspace(cfg['freq_GHz']['start'], cfg['freq_GHz']['stop'], cfg['freq_GHz']['points'])
    wls = C0 / (f * 1e9)
    logger.log_event({"event":"freq_setup", "f_start": float(f[0]), "f_stop": float(f[-1]), "points": int(len(f))})

    incident = Layer(er=1, ur=1)

    if cfg['backing']['type'] == 'metal':
        metal_eps_imag = float(cfg['backing'].get('eps_imag_clamp', 1e8))
        transmission = Layer(material=Material(er=lambda lam: 1+1j*metal_eps_imag, ur=1),
                             thickness=float(cfg['backing'].get('thickness_m', 1e-3)))
    else:
        transmission = Layer(er=1, ur=1)

    layers: List[Layer] = []

    # L1 homogeneous
    L1_cfg = cfg['layers']['L1']
    mat_L1 = make_material_from_csv(L1_cfg['csv'])
    L1 = Layer(material=mat_L1, thickness=float(L1_cfg['thickness_m']))
    layers.append(L1)

    # L2 mask (optional)
    if 'L2_mask' in cfg['layers']:
        Lx = float(cfg['cell']['Lx_m']); Ly = float(cfg['cell']['Ly_m'])
        mask_cfg = cfg['layers']['L2_mask']
        Nx = int(mask_cfg['grid']['Nx']); Ny = int(mask_cfg['grid']['Ny'])
        holes = mask_cfg.get('holes', [])
        guard_holes(holes, Lx, Ly)

        lam_mid = wls[len(wls)//2]

        # read dispersion directly (no Material.er dependency on source)
        disp_solid = MWDispersion(mask_cfg['csv_solid'])
        disp_hole  = MWDispersion(mask_cfg['csv_hole'])
        er_s = np.atleast_1d(disp_solid.er_of_lambda(lam_mid))[0]
        ur_s = np.atleast_1d(disp_solid.ur_of_lambda(lam_mid))[0]
        er_h = np.atleast_1d(disp_hole.er_of_lambda(lam_mid))[0]
        ur_h = np.atleast_1d(disp_hole.ur_of_lambda(lam_mid))[0]

        er_cell, ur_cell = build_mask_cell(Lx, Ly, Nx, Ny, holes, er_s, ur_s, er_h, ur_h)
        # Save mask artifacts
        np.save(run_dir / "artifacts" / "mask_er_complex.npy", er_cell)
        np.save(run_dir / "artifacts" / "mask_ur_complex.npy", ur_cell)
        try:
            import matplotlib.pyplot as plt
            fig1 = plt.figure()
            plt.imshow(np.real(er_cell), origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
            plt.colorbar(label='Re{ε}')
            plt.title('Mask ε real')
            fig1.savefig(run_dir / "artifacts" / "mask_er_real.png", dpi=150, bbox_inches="tight")
            plt.close(fig1)

            fig2 = plt.figure()
            plt.imshow(np.real(ur_cell), origin='lower', extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
            plt.colorbar(label='Re{μ}')
            plt.title('Mask μ real')
            fig2.savefig(run_dir / "artifacts" / "mask_ur_real.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)
        except Exception as e:
            logger.log_event({"event":"warn", "msg": f"mask plot failed: {e}"})

        crystal = Crystal([Lx,0,0],[0,Ly,0], er=er_cell, ur=ur_cell)
        L2 = Layer(crystal=crystal, thickness=float(mask_cfg['thickness_m']))
        layers.append(L2)
        logger.log_event({"event":"mask_layer_built", "Nx":Nx, "Ny":Ny, "holes":len(holes)})

    # L3 homogeneous (optional)
    if 'L3' in cfg['layers']:
        L3_cfg = cfg['layers']['L3']
        mat_L3 = make_material_from_csv(L3_cfg['csv'])
        L3 = Layer(material=mat_L3, thickness=float(L3_cfg['thickness_m']))
        layers.append(L3)

    stack = LayerStack(*layers, incident_layer=incident, transmission_layer=transmission)

    
    # --- harmonics handling (normalized) ---
    has_mask = ('L2_mask' in cfg.get('layers', {}))
    harm_cfg = cfg.get('harmonics', {})
    nh = _nh_tuple_or_int(harm_cfg, has_mask)
    try:
        logger.log_event({"event":"harmonics_normalized","nh": (nh if isinstance(nh,int) else list(nh)), "has_mask": has_mask})
    except Exception:
        pass
# --- harmonics handling: always pass a 3-tuple when a periodic layer exists
    has_mask = ('L2_mask' in cfg['layers'])
    harm = cfg.get('harmonics', {})
    if 'n_3d' in harm:
        nh_raw = harm['n_3d']
    elif 'n_2d' in harm:
        nh_raw = list(harm['n_2d']) + [1]  # [nx, ny] → [nx, ny, 1]
    elif 'n_1d' in harm:
        n = int(harm['n_1d'])
        nh_raw = (n, 1, 1) if has_mask else n
    else:
        nh_raw = (5, 5, 1) if has_mask else 1  # safe default
    nh = _normalize_nh(nh_raw, has_mask)

    def solve_for_pol(pol: str) -> np.ndarray:
        pTEM = [1,0] if pol.upper()=="TE" else [0,1]
        theta = np.deg2rad(cfg['angles']['theta_deg'])
        phi   = np.deg2rad(cfg['angles'].get('phi_deg', 0.0))
        src = Source(wavelength=wls[0], theta=theta, phi=phi, pTEM=pTEM)
        solver = Solver(stack, src, n_harmonics=nh)
        res = solver.solve(wavelength=wls)
        R = np.asarray(res['RTot'])
        T = np.asarray(res['TTot'])
        if not np.all(np.isfinite(R)): raise RuntimeError("NaN/Inf in RTot")
        if not np.all(np.isfinite(T)): raise RuntimeError("NaN/Inf in TTot")
        A = 1 - R - T
        drift = float(np.max(R+T+A) - 1.0)
        if np.max(R+T+A) > 1 + 1e-3:
            raise RuntimeError("Energy violation: max(R+T+A) > 1+1e-3")
        logger.log_event({"event":"solve_pol_done", "pol":pol, "max_energy_excess": drift})
        return R

    pol = cfg['polarization'].upper()
    if pol in ("TE","TM"):
        R = solve_for_pol(pol)
    elif pol == "AVG":
        R = 0.5*(solve_for_pol("TE") + solve_for_pol("TM"))
    else:
        raise ValueError(f"Unknown polarization: {pol}")

    RL_dB = -10*np.log10(np.maximum(R, 1e-12))

    data = np.vstack([f, RL_dB, R]).T
    # convenience copy in base_out
    (base_out := Path(out_dir) if out_dir else Path("outputs")).mkdir(parents=True, exist_ok=True)
    np.savetxt(base_out / "latest_results.csv", data, delimiter=",", header="freq_GHz,RL_dB,R", comments="")

    run_csv = Path(logger.log_path).parent.parent / "results.csv"  # logs/.. = run_dir
    np.savetxt(run_csv, data, delimiter=",", header="freq_GHz,RL_dB,R", comments="")
    logger.log_event({"event":"results_written", "csv": str(run_csv)})

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(f, RL_dB)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Reflection Loss (dB)")
        plt.grid(True, alpha=0.3)
        out_png = run_csv.parent / "RL_plot.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        logger.log_event({"event":"plot_written", "png": str(out_png)})
    except Exception as e:
        logger.log_event({"event":"warn", "msg": f"plot failed: {e}"})

    logger.log_event({"event":"run_end", "run_dir": str(run_csv.parent)})
    return {"freq_GHz": f, "RL_dB": RL_dB, "R": R, "run_dir": str(run_csv.parent)}

def run_from_json(cfg_path: str, out_dir: Optional[str] = None):
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    return run_adapter(cfg, out_dir=out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to input JSON file")
    ap.add_argument("--out_dir", default="outputs", help="Base output directory")
    args = ap.parse_args()
    out = run_from_json(args.config, out_dir=args.out_dir)
    print("Done. Run directory:", out["run_dir"])
