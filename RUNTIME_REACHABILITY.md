# ToolBox and pySNIFS runtime reachability

## Purpose and method

This ledger freezes the compatibility surface that must be preserved before
the top-level `ToolBox` and `pySNIFS` packages are minimized.  It is based on
static references from `scene_model/`, `scripts/`, and `tests/`, followed
through calls inside the vendored modules.  Static analysis cannot prove that
an external consumer does not import these accidentally public packages; this
ledger deliberately covers the scene-model distribution itself.

The classifications are:

- **fitting-critical**: needed to load, fit, extract, or write the normal
  scene-model result;
- **plotting-only**: reached only by diagnostic plotting;
- **auxiliary-script-only**: reached only by one of the three installed
  scripts or its optional outputs;
- **transitive**: not imported by scene-model directly, but called by a
  retained implementation;
- **test-only**: preserved solely by a packaging test, with no production
  caller;
- **unused**: no caller in this distribution.

## ToolBox ledger

| Module / symbol | Classification | Retention decision and reason |
|---|---|---|
| `Arrays.metaslice` | fitting-critical | Retain privately; constructs wavelength meta-slices in `SnifsCubeFitter`. |
| `Arrays.rebin` | test-only | Remove; `test_packaging` is its only consumer. |
| `Arrays.unsqueeze`, `count`, `isTriangular`, `isDiagonal` | unused | Remove. `isDiagonal -> isTriangular` is reachable only within this unused group. |
| `Coords.altaz2hadec`, `hadec2zdpar` | fitting-critical | Retain privately; derive hour angle, zenith distance, and parallactic angle from headers. |
| `Coords.rec2pol` | transitive | Retain privately because both active coordinate transforms call it. |
| `Coords.ten` | test-only for the active runtime | Remove from the minimized runtime. Its other internal callers are the unused `ADR.blurring`/`plot` methods. |
| `Astro.Coords` wildcard re-export and `Astro.__init__` | transitive packaging shim | Remove the re-export hierarchy after consumers import the private coordinate module directly. |
| `Atmosphere.ADR.__init__`, `set_ref`, `set_param`, `get_scale`, `refract`, `get_airmass`, `get_parangle` | fitting-critical | Retain privately; these implement the ADR used by analytic Gaussian/Moffat and Fourier-domain fitting. |
| `Atmosphere.refractiveIndexMEdlen` (and alias `refractiveIndex`) | transitive | Retain privately; used by `ADR.set_ref` and `ADR.get_scale`. |
| `Atmosphere.saturationVaporPressure`, `_saturationVaporPressureOverWater`, `_saturationVaporPressureOverIce` | transitive | Retain privately; humidity correction in `refractiveIndexMEdlen` reaches them even though normal headers commonly use zero humidity. |
| `ADR.__str__`, `get_zd`, `blurring`, `plot` | unused | Remove. They do not participate in scene-model diagnostics; the ADR diagnostics in `scene_model.snifs` call `refract` directly. |
| `MPL` color constants and `MPL.errorband` | plotting-only | Retain privately with accepted behavior. Import currently patches `matplotlib.axes.Axes`. |
| `Misc.make_method` | transitive plotting | Retain only if the `Axes.errorband` monkey patch remains; otherwise replace with explicit local plotting calls. |
| `Misc.warning2stdout` | auxiliary-script-only | Retain privately for `extract_star2` warning routing. |
| `Misc.add_attrs` | test-only | Remove; the packaging test is its only consumer. |
| `Misc.deprecated`, `cached_property` | transitive to `Optimizer` | Remove when `Optimizer` is replaced. They have no independent production consumer. |
| `Misc.catch` | unused | Remove. |
| `Optimizer.Model`, `DataSet`, `Fitter` | fitting-critical wrapper | Do not vendor these general classes. Preserve only `fit_power_law` behavior using its existing direct SciPy residual and Jacobian. The live surface is construction plus `Fitter.residuals`; the remainder is unused. |
| `Optimizer.approx_deriv`, `vec2corr`, `cov2corr`, `corr2cov` | unused/transitive to unused optimizer APIs | Remove with the general optimizer. None is reached by the current power-law path. |
| `IO.str_magn` | transitive to unused optimizer presentation | Remove with optimizer reporting; scene-model never calls the reporting APIs that use it. |

### Locked ToolBox retention set

Retain private equivalents of `metaslice`, the three active coordinate
functions, the ADR core and refractive-index helpers, plotting colors and
error bands, and `warning2stdout`.  Replace `Optimizer` at its single call
site.  Everything else is removable after parity tests pass.

## pySNIFS ledger

| Module / symbol | Classification | Retention decision and reason |
|---|---|---|
| `spectrum.__init__` and its data attributes | fitting-critical and auxiliary-script-only | Retain privately. Extraction constructs output spectra; `extract_fixed_star2` and `subtract_psf2` also read spectra. Preserve `data`, `var`, `has_var`, `x`, `start`, `step`, `len`, and arbitrary `cov` attachment. |
| `spectrum.WR_fits_file` | auxiliary-script-only | Retain for `extract_fixed_star2`; normal extraction uses `write_pysnifs_spectrum`. |
| `spectrum.subset`, `reset_interval`, `index` | unused | Remove. |
| `SNIFS_cube.__init__` and its established attributes | fitting-critical | Retain private E3D, FITS3D, meta-slice, and empty-model construction. Preserve wavelength, lens geometry, data/variance, header, and format-origin attributes consumed by fitting and scripts. |
| `SNIFS_cube.slice2d` | plotting-only plus writer-transitive | Retain; diagnostics and the FITS3D writer use it. The weighted-spectrum branch has no current caller and may be removed if characterization confirms this. |
| `SNIFS_cube.WR_3d_fits` | auxiliary-script-only | Retain for `--keepmodel` and `subtract_psf2`. |
| `SNIFS_cube.WR_e3d_file` | auxiliary-script-only | Retain for `subtract_psf2`; it transitively reaches module-level `WR_e3d_file`. |
| module-level `WR_e3d_file` | transitive | Retain only as a private helper for the cube writer. |
| `SNIFS_cube.spec`, `get_spec`, `get_no`, `get_ij`, `get_lindex` | unused | Remove. (`spec -> get_lindex` and `get_spec -> spec` form an otherwise unreachable group.) |
| `spec_list`, `image_array` and their writers | unused | Remove. |
| `SNIFS_mask` and all methods | unused | Remove; it also invokes obsolete external SNfactory commands and temporary files. |
| `convert_tab`, `histogram`, `common_bounds_cube`, `common_bounds_spec`, `common_lens`, `fit_poly`, `gaus_array`, `comp_cdg`, `zerolike` | unused | Remove. |

### Locked pySNIFS retention set

Retain private, reduced spectrum and cube implementations, the cube slice
operation, the spectrum/FITS3D/Euro3D writers needed by installed scripts,
and only their internal helpers.  No other public pySNIFS container or
utility is retained.

## Dynamic and behavioral hazards

These paths need black-box coverage before removal because a text search is
not sufficient:

1. Importing `ToolBox.MPL` mutates the global Matplotlib `Axes` class by
   installing `errorband`; plotting calls use `ax.errorband` without an
   explicit helper import.
2. `ToolBox.Astro.Coords` is a wildcard re-export.  Consumer imports must be
   made explicit before that package hierarchy disappears.
3. `subtract_psf2` dynamically assigns `cube.writeto` to either
   `WR_e3d_file` or `WR_3d_fits`; both formats must remain functional.
4. E3D and FITS3D constructors establish different headers and provenance
   attributes. Meta-slicing takes schema-dependent branches and uses SciPy
   filtering for data and variance.
5. FITS3D loading drops all-NaN lenslets and rewrites geometry arrays in
   lockstep. Sparse-lens and missing-variance fixtures are required.
6. `spectrum` is used as a mutable record: covariance is attached after
   construction and downstream writing checks attributes rather than a
   declared schema.
7. Cube model output is constructed empty and later populated. A minimized
   constructor must not assume it always read a file.
8. The current broad import guard in `scene_model.snifs` can turn any
   compatibility import failure into a delayed `ImportError`; fresh-wheel
   script tests are needed to detect packaging omissions.
9. `pyproject.toml` explicitly exposes `ToolBox`, `ToolBox.Astro`, and
   `pySNIFS` as top-level packages. Source cleanup alone will not remove them
   from built artifacts.

## Removal gate

The minimized branch may remove the top-level packages only after tests prove
parity for B/R E3D and FITS3D loading, meta-slicing, ADR and power-law fitting,
normal and covariance extraction, diagnostic plotting, `--keepmodel`, all
three scripts, and spectrum/cube writes. Built wheel and sdist inspection must
then pass `validation/inventory_runtime_artifact.py --mode minimized`.

### Gate result on `minimize-vendored-runtime`

The gate is satisfied. The full suite covers the retained numerical and I/O
contracts, successful mocked orchestration for all three scripts, both E3D and
FITS3D dynamic writers, sparse FITS3D lenslets, missing variance, plotting,
and package imports. Built wheel and source archives pass minimized inventory
with the private notice and extract-star license present. Finally, locked real
B/R cubes were compared against `robust-flux-cov@88f6c31` for classic and
Fourier PSFs at `rtol=0`, `atol=0`; fitted parameters, flux, variance,
covariance, and selected headers were exactly equal in all four cases.
