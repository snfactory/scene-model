# Vendored extract-star runtime subset

Scene-model historically imports the top-level `ToolBox` and `pySNIFS`
namespaces.  They were not independently installable dependencies.  This
repository therefore carries only the modules needed by scene-model, sourced
from:

- repository: `https://github.com/snfactory/extract-star`
- immutable revision: `8152df0579c0f660e6534177c0b2fd8a39df5a95`
- upstream license: MIT (preserved in
  `THIRD_PARTY_LICENSES/extract-star-MIT.txt`)

Source mapping:

- `extract_star/pySNIFS.py` -> `pySNIFS/__init__.py`
- `extract_star/extern/{Arrays,Atmosphere,Coords,IO,Misc,Optimizer}.py` ->
  `ToolBox/`
- `ToolBox/Astro/Coords.py` is a compatibility re-export.
- `ToolBox/MPL.py` is the minimal plotting adapter previously used by the
  local Python 3 scene-model port.

The source modules contain only the audited Python 3 compatibility changes
needed by the scene-model runtime: integer slice division, `range`, `str`, and
`dict.items`; current Astropy FITS constants and overwrite keywords; and removal
of a dead reference to the Python 2 `file` builtin.  No scene-model fitting or
covariance mathematics is implemented in this compatibility layer.
