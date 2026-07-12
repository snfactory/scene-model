# Why the scene-model fork contains `ToolBox` and `pySNIFS`

## Plain-language answer

The fork did **not** acquire these dependencies because of the covariance or
JAX work.  Boone's original public `scene_model` already imported `ToolBox`
and `pySNIFS` in the SNIFS-specific extraction code and scripts.  However, the
original repository did not contain those modules, declare them as installable
dependencies, or explain how to obtain them.  Its packaging installed only
`scene_model` and three scripts.  In practice, SNfactory users had to run it in
an existing SNIFS software environment where top-level imports such as
`import pySNIFS` and `from ToolBox ...` were already available on Python's
module search path (normally through the surrounding installation or
`PYTHONPATH`).

The fork first made the project a conventional, standalone Python package and
then copied the legacy runtime modules into the repository so a fresh install
could actually run the existing SNIFS code.  That solved a real installation
problem, but it also turned an implicit dependency into 2,730 lines of
top-level, publicly packaged compatibility code.  The intended cleanup is to
retain only the behavior scene-model actually uses and place it under the
private namespace `scene_model._compat`.

This is therefore a packaging/self-containment issue, not a new scientific
dependency introduced by JAX and not evidence that the original extractor did
the same work without these routines.

## Commit and path evidence

The comparison points used for this audit are:

| State | Immutable commit | Relevant fact |
|---|---|---|
| Boone upstream `origin/master` | `7e1a03125f55009b882843d85bc0ecf177210a25` | Imports the legacy modules but does not ship them. |
| Fork immediately before vendoring | `29a6278aaaca6135e6cf42a7f8ac2f91d298cad1` | Modern metadata exists, but packages only `scene_model` and `scripts`. |
| Vendoring commit | `7083de0a3d3aab4b3113432c580e7032e6f6ff26` | Adds `ToolBox`, `pySNIFS`, their license, and package entries. |
| Audited fork baseline | `88f6c31f9c929f2b8c96c4d2c3adda34b44a4d54` | Current merged covariance/JAX branch before minimization. |

At `origin/master`, `scene_model/snifs.py` lines 30--34 import:

```python
from ToolBox.Arrays import metaslice
from ToolBox.Astro import Coords
from ToolBox.Atmosphere import ADR
from ToolBox import MPL
import pySNIFS
```

The same upstream file imports `ToolBox.Optimizer` inside `fit_power_law` at
line 272.  The upstream scripts also contain direct imports:

- `scripts/extract_star2.py:14` imports `ToolBox.Misc.warning2stdout`;
- `scripts/extract_fixed_star2.py:9` imports `pySNIFS`;
- `scripts/subtract_psf2.py:9` imports `pySNIFS`.

None of `ToolBox` or `pySNIFS` exists in the 16 tracked files at that upstream
commit.  Its `setup.py` names only `packages=['scene_model']`, lists the three
scripts, and supplies no `install_requires`.  Its short README acknowledges
that scene-model was heavily inspired by and borrowed code from
`extract_star.py`, but provides no installation instructions for these imports.
The `try/except ImportError` around the imports in `scene_model/snifs.py` merely
prints that some functionality is disabled; it does not supply a substitute.
The actual SNIFS cube and extraction paths later dereference these names.

That combination is the concrete basis for the "ambient SNfactory runtime"
conclusion: it is an inference from unresolved top-level imports plus packaging
that neither includes nor declares their providers.  It does not rely on a
claim that Boone documented a particular deployment mechanism.

Commit `bf1d5ca8f0e39f285f378f5c03ec47bd1391e6dd` added modern Python project
metadata and a console entry point.  Its `pyproject.toml` declared Astropy,
Matplotlib, NumPy, and SciPy and packaged `scene_model` plus `scripts`, but the
legacy imports still had no installable provider.  Its successor `7083de0`
closed that gap by adding 2,809 lines in 15 files, including all 2,730 lines of
the compatibility packages, and by adding these package names to
`pyproject.toml`:

```toml
"ToolBox",
"ToolBox.Astro",
"pySNIFS",
```

The ordering is unambiguous: the vendoring commit precedes the first covariance
commit, `cf96cd3` (`feat: propagate scene-model flux covariance`), and precedes
the JAX prototype, `b65472b` (`Prototype JAX variable-projection Jacobian`).
Neither covariance nor JAX caused the original `ToolBox`/`pySNIFS` imports.

## Source and provenance mapping

The recorded public source is the Nearby Supernova Factory
`extract-star` repository at
`8152df0579c0f660e6534177c0b2fd8a39df5a95`.  The source snapshot used for
this audit carries the MIT license, copyright 2016 Nearby Supernova Factory.
The fork preserves that text in
`THIRD_PARTY_LICENSES/extract-star-MIT.txt`.

| Fork path | Public source at `extract-star@8152df0` | Audit result |
|---|---|---|
| `pySNIFS/__init__.py` | `extract_star/pySNIFS.py` | Same implementation with narrow Python 3/current-Astropy edits: integer slicing, `range`, FITS overwrite/boolean constants, and removal of a dead Python 2 `file` reference. |
| `ToolBox/Arrays.py` | `extract_star/extern/Arrays.py` | One Python 3 integer-division correction. |
| `ToolBox/Atmosphere.py` | `extract_star/extern/Atmosphere.py` | Exact copy in this comparison. |
| `ToolBox/Coords.py` | `extract_star/extern/Coords.py` | Python 2 `basestring` changed to `str`. |
| `ToolBox/IO.py` | `extract_star/extern/IO.py` | Exact copy in this comparison. |
| `ToolBox/Misc.py` | `extract_star/extern/Misc.py` | Python 2 `iteritems` changed to `items` plus whitespace cleanup. |
| `ToolBox/Optimizer.py` | `extract_star/extern/Optimizer.py` | Exact copy in this comparison. |
| `ToolBox/Astro/Coords.py` and `ToolBox/Astro/__init__.py` | No separate source file | Small compatibility re-exports preserving Boone's existing `ToolBox.Astro` import shape. |
| `ToolBox/MPL.py` | No file at the cited public revision | A 46-line plotting adapter used for colors and `Axes.errorband`; it is not fitting, extraction, covariance, or JAX mathematics.  The project owner has accepted retaining this plotting implementation. |

The compatibility layer contains operational support code used by the
pre-existing SNIFS interface: cube/spectrum I/O, wavelength metaslicing,
coordinate and atmospheric-refraction calculations, a power-law fitting
wrapper, warning formatting, and plotting helpers.  It does not contain the
new JAX differentiation or covariance propagation implementations.

## What the current distributions expose

A clean offline build at `88f6c31` produced a 123 KiB wheel and a 125 KiB
source archive.  This audit inspected their file lists, not just the source
configuration.

The wheel installs all of the following as importable **top-level packages**:

- `scene_model`;
- `scripts`;
- `ToolBox` (eight Python files plus the two-file `ToolBox.Astro` package);
- `pySNIFS` (the complete 1,372-line module).

The wheel includes the project license and the extract-star MIT license under
distribution metadata.  It does **not** include `VENDORED_EXTRACT_STAR.md`, so
the detailed source revision and modification mapping are not visible in an
installed wheel.

The source archive includes the same complete `ToolBox` and `pySNIFS` trees,
the three script modules, tests, and both license files.  It also omits
`VENDORED_EXTRACT_STAR.md`.  Thus both artifacts distribute considerably more
legacy public API than scene-model intentionally promises or needs, while the
best existing provenance explanation is absent from both artifacts.

## Current and target dependency shapes

Current upstream behavior was structurally this:

```text
Boone scene_model repository/install
  scene_model.snifs + three scripts
       |
       +-- import ToolBox.*  ----\
       +-- import pySNIFS    -----+--> supplied by the surrounding SNIFS
                                      Python environment, not by the repo
```

The current fork made those imports self-contained, but public:

```text
scene-model wheel
  scene_model + scripts
       |
       +-- top-level ToolBox package (1,358 lines)
       +-- top-level pySNIFS package (1,372 lines)
```

The cleanup target keeps standalone installation while narrowing ownership and
surface area:

```text
scene-model wheel
  scene_model
    _compat/                 # private, minimal retained behavior
      snifs_io               # required cube/spectrum reads and writes
      atmosphere/coords      # required ADR and coordinate operations
      arrays                 # metaslice only
      plotting               # accepted colors/errorband behavior
      warnings               # CLI warning formatting
  scripts                    # import only scene_model-private interfaces

  no top-level ToolBox package
  no top-level pySNIFS package
```

The target must preserve the existing extraction and file-format behavior.  It
changes where the supporting code lives and how much is shipped; it does not
replace the analytic Gaussian/Moffat or Fourier-domain PSF algorithms, alter
the covariance calculation, or change the accepted NumPy flux result.

## Audit conclusion

The apparent paradox has a simple resolution: the original repository needed
these modules too, but did not make a clean installation reproducible.  The
fork exposed that hidden dependency by vendoring a public legacy implementation
before covariance/JAX development began.  The fork therefore should not remove
all of the behavior, but it also need not continue distributing the complete
legacy namespaces as accidental public APIs.  Minimizing them into
`scene_model._compat`, preserving the third-party license and artifact-visible
provenance, is the appropriate forward-only correction.

## Implemented cleanup result

The `minimize-vendored-runtime` branch implements that correction without
changing the PSF, covariance, or JAX algorithms:

- the 2,730-line top-level compatibility packages are removed;
- 836 lines of reachable support code remain under the private
  `scene_model._compat` namespace;
- all three installed scripts import that private namespace, while the
  general `ToolBox.Optimizer` wrapper is replaced at its sole call site by a
  direct SciPy residual function;
- the wheel and source archive contain neither top-level `ToolBox` nor
  top-level `pySNIFS`, and both contain `_compat/NOTICE.md` plus the public
  extract-star MIT license;
- clean artifacts measured 107,248 bytes for the wheel and 114,674 bytes for
  the source archive in the validation build;
- the full repository suite passes, including successful private-runtime
  workflows for `extract_star2`, `extract_fixed_star2`, and `subtract_psf2`,
  plus both dynamic cube writers;
- an external real-data comparison on the locked B and R cubes passed at
  exact tolerance for classic and Fourier PSFs: fitted parameters, extracted
  flux, variance, covariance, and selected headers all had zero difference.

The reproducible gates are
`validation/inventory_runtime_artifact.py --mode minimized` and
`validation/compare_runtime_baseline.py`. The real observation cubes remain
external; the latter report records only their SHA-256 identities.
