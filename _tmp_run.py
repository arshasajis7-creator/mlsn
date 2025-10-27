from pathlib import Path
from rcwa_desktop.models.configuration import Configuration, MaskHole
from rcwa_desktop.services.rcwa_runner import run_simulation
from rcwa_desktop.services.run_logger import RunLogger

cfg = Configuration()
cfg.mask.holes = [cfg.mask.holes[0], MaskHole(shape='square', x_m=0.002, y_m=0.0, size1=0.003, size2=0.004)]
repo = Path('.').resolve()
logger = RunLogger(repo)
ctx = logger.start_run(cfg.output_prefix + '_cli_square')
ctx.record_configuration(cfg)
res = run_simulation(cfg, repo, log_dir=ctx.directory)
print('run_dir=' + str(ctx.directory))
print('square warning?', any('Square mask holes' in w for w in res.warnings))
