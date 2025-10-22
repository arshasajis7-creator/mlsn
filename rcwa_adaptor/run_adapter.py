
# run_adapter.py
import sys
from adapter_rcwa import run_from_json

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "example_config.json"
    out = run_from_json(cfg, out_dir="outputs")
    print("Done. Run directory:", out["run_dir"])
