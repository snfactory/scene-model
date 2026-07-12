# Private extract-star compatibility subset

Scene-model historically imported the top-level `ToolBox` and `pySNIFS`
namespaces from the surrounding SNfactory runtime. They were not independently
installable dependencies. This repository now carries only the behavior used
by scene-model under the private `scene_model._compat` namespace, sourced from:

- repository: `https://github.com/snfactory/extract-star`
- immutable revision: `8152df0579c0f660e6534177c0b2fd8a39df5a95`
- upstream license: MIT (preserved in
  `THIRD_PARTY_LICENSES/extract-star-MIT.txt`)

Current source mapping:

- required `extract_star/pySNIFS.py` behavior ->
  `scene_model/_compat/snifs_io.py`;
- required `extract_star/extern/{Arrays,Atmosphere,Coords,Misc}.py` behavior ->
  the corresponding private compatibility modules;
- the general `Optimizer` and `IO` modules are no longer shipped because the
  one live power-law fit now uses its equivalent direct SciPy residual;
- the accepted local plotting adapter -> `scene_model/_compat/plotting.py`.

The retained code includes only the audited Python 3 compatibility changes
needed by the scene-model runtime: integer slice division, `range`, current
Astropy FITS construction and overwrite APIs, and removal of a dead reference
to the Python 2 `file` builtin. No scene-model PSF, JAX, or covariance
mathematics is implemented in this compatibility layer.

Artifact-visible provenance is also installed as
`scene_model/_compat/NOTICE.md`.
