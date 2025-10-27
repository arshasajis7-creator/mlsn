import os

file_location = os.path.dirname(__file__)
nk_location = os.path.join(file_location, os.pardir, 'nkData/')

from rcwa.utils.nk_loaders import *
from rcwa.utils.fresnel import *

# Plotting helpers depend on matplotlib, which is optional for core simulations.
# Import lazily to avoid hard dependency when only numeric solvers are used.
try:  # pragma: no cover - optional dependency
    from rcwa.utils.plotter import *  # type: ignore[F401,F403]
except Exception:  # pragma: no cover - plotting is optional
    pass
