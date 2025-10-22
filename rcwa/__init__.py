# rcwa/__init__.py  — minimal, safe, no absolute imports
# Export only high-level symbols; avoid circular imports.

from .material import Material
from .crystal import Crystal

# matrices module might have different names; try candidates
MatrixCalculator = None
try:
    from .matrices import MatrixCalculator  # common name
except Exception:
    try:
        from .matrix import MatrixCalculator
    except Exception:
        pass  # leave as None if not present

from .layer import Layer, LayerStack
from .source import Source
from .solver import Solver

# Results is optional in some repos
try:
    from .results import Results
except Exception:
    Results = None

__all__ = [
    "Material",
    "Crystal",
    "MatrixCalculator",
    "Layer", "LayerStack",
    "Source",
    "Solver",
    "Results",
]
