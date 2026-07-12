# Private SNIFS compatibility runtime

This private package contains the minimal historical SNIFS support required by
scene-model. Portions are derived from the Nearby Supernova Factory's public
`extract-star` repository at immutable revision
`8152df0579c0f660e6534177c0b2fd8a39df5a95`, distributed under the MIT license
preserved as `THIRD_PARTY_LICENSES/extract-star-MIT.txt`.

Source lineage:

- `arrays.py`: required `metaslice` behavior from `extract_star/extern/Arrays.py`;
- `coords.py`: required coordinate transforms from `extract_star/extern/Coords.py`;
- `atmosphere.py`: required modified-Edlen/ADR behavior from
  `extract_star/extern/Atmosphere.py`;
- `snifs_io.py`: required cube and spectrum I/O from `extract_star/pySNIFS.py`;
- `warnings.py`: required warning formatting from `extract_star/extern/Misc.py`;
- `plotting.py`: the accepted local plotting adapter for colors and error bands.

The retained third-party code has been reduced to the behavior used by
scene-model and its three scripts. It includes Python 3 and current-Astropy
compatibility updates. The private namespace is an implementation detail and
does not preserve the former top-level `ToolBox` or `pySNIFS` APIs.
