"""Single import seam for the legacy SNIFS runtime characterization tests.

The minimization work should only need to update this module when the retained
implementation moves below ``scene_model._compat``.
"""

import sys
from pathlib import Path

# The compatibility packages are deliberately top-level in the baseline tree
# but are not installed in the active development environment used by all
# worktrees. Keep path setup confined to this import seam.
sys.path.insert(0, str(Path(__file__).parents[1]))

from scene_model._compat.arrays import metaslice
from scene_model._compat.coords import altaz2hadec, hadec2zdpar
from scene_model._compat.atmosphere import ADR
from scene_model._compat import plotting as MPL
from scene_model._compat.warnings import warning2stdout
from scene_model._compat.snifs_io import SNIFS_cube, spectrum

from scene_model.snifs import evaluate_power_law, fit_power_law

__all__ = [
    "ADR",
    "MPL",
    "SNIFS_cube",
    "altaz2hadec",
    "evaluate_power_law",
    "fit_power_law",
    "hadec2zdpar",
    "metaslice",
    "spectrum",
    "warning2stdout",
]
