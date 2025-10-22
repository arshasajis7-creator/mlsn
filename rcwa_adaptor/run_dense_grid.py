import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, r"c:\mlsn")
from step3_two_layer_grid import search, _load_config
cfg = _load_config(Path('step3_example_layered.json'))
base_dir = Path('.').resolve()
thickness_L1 = np.linspace(0.002, 0.006, 9)
thickness_L3 = np.linspace(0.007, 0.012, 11)
results = search(cfg, base_dir, thickness_L1, thickness_L3)
pd.DataFrame(results, columns=['t1','t3','avg_R','avg_A','min_R']).to_csv('two_layer_refined_grid.csv', index=False)
print('saved', len(results), 'rows')
