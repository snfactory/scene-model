"""Scene-model covariance hand-off sidecar (``<spectrum>.scenecov.npz``).

The spectrum FITS carries the assembled flux covariance (COVAR) only.  The
sidecar keeps every part of it separately addressable so that a downstream
pipeline can rebuild the scene-stage covariance with a non-diagonal cube
covariance:

* ``H`` -- the fixed native extraction map (CSR, ``(nwave, nwave*nx*ny)``,
  columns wavelength-major / spaxel-minor over the ``cube_to_arrays`` grid,
  masked spaxels as zero columns), with ``flux == H @ nan_to_num(fixed_data)``;
* ``conditional_variance`` ``D_cond`` -- native per-wavelength amplitude
  variance (``diag(H diag(fixed_variance) H.T)`` to roundoff);
* ``propagated_factor`` ``U = J L`` with ``C_p = L L.T``, so that
  ``COVAR == diag(D_cond) + U U.T``.

The sidecar records the SHA-256 of the spectrum file it belongs to; the
spectrum header records the sidecar name under ``COVSIDE``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile

import numpy as np
from scipy import sparse

from .utils import SceneModelException

SCENE_SIDECAR_SCHEMA = 'SNIFS-SCENE-COV-NPZ-1.2'
SCENE_SIDECAR_SUFFIX = '.scenecov.npz'
COVARIANCE_VERSION = 'SNIFS-COV-1.0'
# Flux-map parity contract, shared verbatim with
# snifs_pipe.scene_covariance.fixed_scene_covariance (reference_flux):
#   FLUX-MAP-PARITY-1: pass iff for every wavelength i
#     |(H d)_i - f_i| <= max(1024 * eps * (|H| |d|)_i, 1e-6 * sqrt(D_cond_i))
#   eps = np.finfo(float).eps, |H| and |d| elementwise absolute values,
#   d = nan_to_num(fixed_data), D_cond the conditional variance.
FLUX_MAP_PARITY = 'FLUX-MAP-PARITY-1'
FLUX_MAP_ROUNDOFF_FACTOR = 1024.0
FLUX_MAP_SIGMA_FRACTION = 1e-6
# conditional_variance_check above this is reported as not at roundoff.
CONDITIONAL_VARIANCE_ROUNDOFF = 1e-8

_STRING_KEYS = (
    'schema', 'spectrum_name', 'spectrum_sha256', 'jacobian_model',
    'jacobian_backend', 'covariance_version', 'scene_model_revision',
)


def sidecar_path(spectrum_path):
    """Return the sidecar path that belongs to ``spectrum_path``."""
    return os.fspath(spectrum_path) + SCENE_SIDECAR_SUFFIX


def file_sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scene_model_revision():
    """Git HEAD of the scene-model checkout (read-only), or 'unknown'.

    A '+dirty' suffix marks a checkout with uncommitted tracked changes,
    so the revision never claims code that did not produce the product.
    """
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        revision = subprocess.run(
            ['git', '-C', directory, 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not revision:
            return 'unknown'
        status = subprocess.run(
            ['git', '-C', directory, 'status', '--porcelain',
             '--untracked-files=no'],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return revision + ('+dirty' if status else '')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def _float_or_nan(value):
    return float('nan') if value is None else float(value)


def _spaxel_map(cube, nx, ny):
    """Map Euro3D rows onto the flattened ``i*ny + j`` grid index."""
    spaxel_i = np.asarray(cube.i, dtype=np.int64).ravel()
    spaxel_j = np.asarray(cube.j, dtype=np.int64).ravel()
    try:
        row_spec_id = np.asarray(cube.no, dtype=np.int64).ravel()
    except AttributeError:
        raise SceneModelException(
            "Cube has no Euro3D SPEC_ID ('no') per spaxel row"
        )
    if not (len(spaxel_i) == len(spaxel_j) == len(row_spec_id)):
        raise SceneModelException("Cube spaxel i/j/SPEC_ID lengths differ")
    if (np.any(spaxel_i < 0) or np.any(spaxel_i >= nx) or
            np.any(spaxel_j < 0) or np.any(spaxel_j >= ny)):
        raise SceneModelException("Cube spaxel i/j fall outside the grid")
    flat = spaxel_i * ny + spaxel_j
    if len(np.unique(flat)) != len(flat):
        raise SceneModelException("Two Euro3D rows share one grid spaxel")
    if np.any(row_spec_id < 0):
        raise SceneModelException("Negative Euro3D SPEC_ID")
    spaxel_spec_id = np.full(nx * ny, -1, dtype=np.int64)
    spaxel_spec_id[flat] = row_spec_id
    return spaxel_spec_id, spaxel_i, spaxel_j, row_spec_id


def extraction_map(payload):
    """Rebuild the CSR fixed extraction map ``H`` from a payload."""
    shape = tuple(int(value) for value in payload['extraction_map_shape'])
    return sparse.csr_matrix(
        (payload['extraction_map_data'], payload['extraction_map_indices'],
         payload['extraction_map_indptr']), shape=shape,
    )


def _scaled_max(difference, scale):
    """max_i difference_i / scale_i with 0/0 -> 0 and x/0 -> inf."""
    if difference.size == 0:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(difference == 0, 0.0, difference / scale)
    return float(np.max(ratio))


def flux_map_check(matrix, fixed_data, flux, conditional_variance):
    """Gate ``H @ nan_to_num(d)`` against the accepted flux ``f``.

    Contract FLUX-MAP-PARITY-1 (identical in snifs_pipe's
    fixed_scene_covariance): pass iff for every wavelength i

        |(H d)_i - f_i| <= max(1024 * eps * (|H| |d|)_i,
                               1e-6 * sqrt(D_cond_i))

    with eps = np.finfo(float).eps, |H| and |d| elementwise absolute values
    (d = nan_to_num(fixed_data)) and D_cond the conditional variance.  The
    first term is the summation-roundoff scale, which stays meaningful when
    the mixed-sign weights cancel; the second is a negligible fraction of
    the statistical error.

    Returns (passed, details) with details keys ``max_relative``
    (max |Hd - f| / |f|), ``max_roundoff_units`` (max |Hd - f| /
    (eps |H||d|)), ``max_sigma`` (max |Hd - f| / sqrt(D_cond)),
    ``worst_index`` (largest difference / allowance) and ``worst_flux``.
    """
    data = np.nan_to_num(np.ravel(fixed_data), nan=0.0)
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.sqrt(np.asarray(conditional_variance, dtype=np.float64))
    mapped = matrix.dot(data)
    difference = np.abs(mapped - flux)
    eps = np.finfo(float).eps
    roundoff_scale = eps * abs(matrix).dot(np.abs(data))
    allowed = np.maximum(FLUX_MAP_ROUNDOFF_FACTOR * roundoff_scale,
                         FLUX_MAP_SIGMA_FRACTION * sigma)
    passed = bool(np.all(difference <= allowed))
    if difference.size:
        with np.errstate(divide='ignore', invalid='ignore'):
            excess = np.where(difference == 0, 0.0, difference / allowed)
        worst = int(np.argmax(excess))
    else:
        worst = -1
    details = {
        'max_relative': _scaled_max(difference, np.abs(flux)),
        'max_roundoff_units': _scaled_max(difference, roundoff_scale),
        'max_sigma': _scaled_max(difference, sigma),
        'worst_index': worst,
        'worst_flux': float(flux[worst]) if worst >= 0 else float('nan'),
    }
    return passed, details


def conditional_variance_check(matrix, fixed_variance, conditional_variance):
    """max_w |D_cond - diag(H diag(v) H.T)| / D_cond (v NaN -> 0)."""
    variance = np.nan_to_num(np.ravel(fixed_variance), nan=0.0)
    mapped = np.asarray(matrix.multiply(matrix).dot(variance)).ravel()
    difference = np.abs(conditional_variance - mapped)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative = np.where(difference == 0, 0.0,
                            difference / np.abs(conditional_variance))
    return float(np.max(relative)) if relative.size else 0.0


def validate_scene_sidecar(payload):
    """Raise SceneModelException unless the payload is self-consistent."""
    if str(payload.get('schema')) != SCENE_SIDECAR_SCHEMA:
        raise SceneModelException(
            "Scene sidecar schema %r is not %r" %
            (str(payload.get('schema')), SCENE_SIDECAR_SCHEMA)
        )
    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.dtype == object:
            raise SceneModelException(
                "Scene sidecar key %s has object dtype" % key
            )

    def shape(key):
        return tuple(np.shape(payload[key]))

    def expect(key, expected):
        if shape(key) != tuple(expected):
            raise SceneModelException(
                "Scene sidecar %s has shape %s, expected %s" %
                (key, shape(key), tuple(expected))
            )

    nwave = len(payload['wavelength'])
    npar = len(payload['parameter_names'])
    rank = shape('propagated_factor')[1] if \
        len(shape('propagated_factor')) == 2 else -1
    fixed_input_shape = tuple(int(value)
                              for value in payload['fixed_input_shape'])
    if len(fixed_input_shape) != 3 or fixed_input_shape[0] != nwave:
        raise SceneModelException(
            "Scene sidecar fixed_input_shape %s does not match %d wavelengths"
            % (fixed_input_shape, nwave)
        )
    nspaxel = fixed_input_shape[1] * fixed_input_shape[2]
    for key in ('wavelength', 'flux', 'conditional_variance'):
        expect(key, (nwave,))
    expect('propagated_factor', (nwave, rank))
    expect('jacobian', (nwave, npar))
    expect('covariance_factor', (npar, rank))
    expect('parameter_values', (npar,))
    expect('global_covariance', (npar, npar))
    expect('extraction_map_shape', (2,))
    if tuple(int(v) for v in payload['extraction_map_shape']) != \
            (nwave, nwave * nspaxel):
        raise SceneModelException(
            "Scene sidecar extraction_map_shape %s is not (%d, %d)" %
            (tuple(payload['extraction_map_shape']), nwave, nwave * nspaxel)
        )
    expect('extraction_map_indptr', (nwave + 1,))
    expect('extraction_map_indices', shape('extraction_map_data'))
    expect('fixed_variance', (nwave * nspaxel,))
    expect('fixed_data', (nwave * nspaxel,))
    expect('spaxel_spec_id', (nspaxel,))
    nrow = len(payload['row_spec_id'])
    for key in ('spaxel_i', 'spaxel_j', 'row_spec_id'):
        expect(key, (nrow,))
    if int(np.sum(np.asarray(payload['spaxel_spec_id']) >= 0)) != nrow:
        raise SceneModelException(
            "Scene sidecar spaxel_spec_id does not cover the Euro3D rows"
        )
    product = np.dot(payload['jacobian'], payload['covariance_factor'])
    scale = float(np.max(np.abs(product))) if product.size else 0.0
    if not np.allclose(product, payload['propagated_factor'], rtol=1e-12,
                       atol=1e-12 * scale):
        raise SceneModelException(
            "Scene sidecar propagated_factor != jacobian @ covariance_factor"
        )
    return payload


def build_scene_sidecar(fitter):
    """Collect the scene covariance hand-off of a covariance extraction.

    Returns a dict of plain numpy arrays and strings.  ``spectrum_name`` and
    ``spectrum_sha256`` are filled by write_scene_sidecar once the spectrum
    file exists.  Raises SceneModelException if ``H @ fixed_data`` does not
    reproduce the accepted flux.
    """
    diagnostics = getattr(fitter, 'covariance_diagnostics', None)
    if diagnostics is None:
        raise SceneModelException(
            "Scene sidecar requires a covariance extraction"
        )
    for key in ('fixed_extraction_map', 'propagated_factor', 'jacobian',
                'conditional_variance', 'covariance_factor', 'fixed_data',
                'fixed_variance', 'fixed_input_shape', 'derivatives'):
        if key not in diagnostics:
            raise SceneModelException(
                "Covariance diagnostics lack %s for the scene sidecar" % key
            )

    fixed_input_shape = tuple(int(v) for v in diagnostics['fixed_input_shape'])
    nwave, nx, ny = fixed_input_shape
    fixed_data = np.asarray(diagnostics['fixed_data'], dtype=np.float64)
    fixed_variance = np.asarray(diagnostics['fixed_variance'],
                                dtype=np.float64)
    if fixed_data.shape != fixed_input_shape or \
            fixed_variance.shape != fixed_input_shape:
        raise SceneModelException(
            "Fixed data/variance do not have the fixed input shape %s" %
            (fixed_input_shape,)
        )
    matrix = sparse.csr_matrix(diagnostics['fixed_extraction_map'],
                               dtype=np.float64)
    matrix.sort_indices()
    flux = np.array(fitter.point_source_spectrum.data, dtype=np.float64)
    conditional_variance = np.array(diagnostics['conditional_variance'],
                                    dtype=np.float64)
    parameter_info = tuple(fitter.fit_global_parameter_info)
    spaxel_spec_id, spaxel_i, spaxel_j, row_spec_id = _spaxel_map(
        fitter.cube, nx, ny
    )
    baseline_parity = diagnostics.get('baseline_parity') or {}
    jacobian_model = diagnostics.get('jacobian_model')

    # Both checks are computed and printed before either can raise, so a
    # failed run still reports the conditional-variance number.
    flux_passed, flux_details = flux_map_check(
        matrix, fixed_data, flux, conditional_variance
    )
    flux_summary = (
        "max |H d - f|/|f| = %.3g; max |H d - f|/(eps |H||d|) = %.3g; "
        "max |H d - f|/sqrt(D_cond) = %.3g; worst wavelength index %d "
        "(flux %.6g)" %
        (flux_details['max_relative'], flux_details['max_roundoff_units'],
         flux_details['max_sigma'], flux_details['worst_index'],
         flux_details['worst_flux'])
    )
    print("  Scene sidecar check (%s): %s: %s" %
          (FLUX_MAP_PARITY, flux_summary, 'pass' if flux_passed else 'FAIL'))
    variance_relative = conditional_variance_check(
        matrix, fixed_variance, conditional_variance
    )
    print("  Scene sidecar check: max |D_cond - diag(H V H^T)| / D_cond = "
          "%.3g%s" % (variance_relative,
                      '' if variance_relative <= CONDITIONAL_VARIANCE_ROUNDOFF
                      else ' (NOT at roundoff)'))
    if not flux_passed:
        raise SceneModelException(
            "Fixed extraction map does not reproduce the accepted flux "
            "under %s (%s)" % (FLUX_MAP_PARITY, flux_summary)
        )

    payload = {
        'schema': SCENE_SIDECAR_SCHEMA,
        'spectrum_name': '',
        'spectrum_sha256': '',
        'wavelength': np.array(fitter.cube.lbda, dtype=np.float64),
        'flux': flux,
        'conditional_variance': conditional_variance,
        'propagated_factor': np.array(diagnostics['propagated_factor'],
                                      dtype=np.float64),
        'jacobian': np.array(diagnostics['jacobian'], dtype=np.float64),
        'covariance_factor': np.array(diagnostics['covariance_factor'],
                                      dtype=np.float64),
        'parameter_names': np.array([str(p.name) for p in parameter_info],
                                    dtype=str),
        'parameter_values': np.array([p.value for p in parameter_info],
                                     dtype=np.float64),
        'global_covariance': np.array(fitter.fit_global_covariance,
                                      dtype=np.float64),
        'coefficient_names': np.array(
            [str(name) for name in diagnostics.get('coefficient_names', ())],
            dtype=str),
        'extraction_map_data': np.array(matrix.data, dtype=np.float64),
        'extraction_map_indices': np.array(matrix.indices, dtype=np.int64),
        'extraction_map_indptr': np.array(matrix.indptr, dtype=np.int64),
        'extraction_map_shape': np.array(matrix.shape, dtype=np.int64),
        'fixed_input_shape': np.array(fixed_input_shape, dtype=np.int64),
        'fixed_variance': fixed_variance.ravel().copy(),
        'fixed_data': fixed_data.ravel().copy(),
        'spaxel_spec_id': spaxel_spec_id,
        'spaxel_i': spaxel_i,
        'spaxel_j': spaxel_j,
        'row_spec_id': row_spec_id,
        'jacobian_model': 'none' if jacobian_model is None
        else str(jacobian_model),
        'jacobian_backend': str(diagnostics['derivatives'].backend),
        'covariance_version': COVARIANCE_VERSION,
        'scene_model_revision': scene_model_revision(),
        'baseline_parity_max_relative': _float_or_nan(
            baseline_parity.get('max_relative')),
        'baseline_parity_max_sigma': _float_or_nan(
            baseline_parity.get('max_sigma_scaled')),
        'flux_map_max_relative': flux_details['max_relative'],
        'flux_map_max_roundoff_units': flux_details['max_roundoff_units'],
        'flux_map_max_sigma': flux_details['max_sigma'],
        'conditional_variance_check': variance_relative,
    }
    return validate_scene_sidecar(payload)


def write_scene_sidecar(spectrum_path, payload):
    """Write ``payload`` next to an existing spectrum file; return its path.

    Fills ``spectrum_name`` and ``spectrum_sha256`` from the spectrum file,
    writes to a temporary file in the same directory and atomically
    replaces the destination.
    """
    spectrum_path = os.fspath(spectrum_path)
    if not os.path.isfile(spectrum_path):
        raise SceneModelException(
            "Spectrum %s must exist before its scene sidecar" % spectrum_path
        )
    payload = dict(payload)
    payload['spectrum_name'] = os.path.basename(spectrum_path)
    payload['spectrum_sha256'] = file_sha256(spectrum_path)
    arrays = {key: np.asarray(value) for key, value in payload.items()}
    for key, value in arrays.items():
        if value.dtype == object:
            raise SceneModelException(
                "Scene sidecar key %s has object dtype" % key
            )
    validate_scene_sidecar(arrays)

    path = sidecar_path(spectrum_path)
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix='.%s.' % os.path.basename(path), suffix='.tmp', dir=directory,
    )
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return path


def read_scene_sidecar(spectrum_path, *, verify_hash=True):
    """Read and validate the sidecar of ``spectrum_path``.

    Returns a dict with every stored key (string keys as ``str``) plus
    ``extraction_map``, the CSR matrix ``H``.
    """
    spectrum_path = os.fspath(spectrum_path)
    path = sidecar_path(spectrum_path)
    if not os.path.isfile(path):
        raise SceneModelException("Scene sidecar %s does not exist" % path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    for key in _STRING_KEYS:
        if key in payload:
            payload[key] = str(payload[key])
    validate_scene_sidecar(payload)
    if verify_hash:
        if not os.path.isfile(spectrum_path):
            raise SceneModelException(
                "Spectrum %s of scene sidecar does not exist" % spectrum_path
            )
        actual = file_sha256(spectrum_path)
        if actual != payload['spectrum_sha256']:
            raise SceneModelException(
                "Scene sidecar %s belongs to a different spectrum "
                "(SHA-256 %s, file has %s)" %
                (path, payload['spectrum_sha256'], actual)
            )
    payload['extraction_map'] = extraction_map(payload)
    return payload
