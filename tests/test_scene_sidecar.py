from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from scene_model import snifs
from scene_model.covariance import (GlobalParameter, JacobianDiagnostics,
                                    fixed_extraction_map)
from scene_model.scene_sidecar import (SCENE_SIDECAR_SCHEMA,
                                       build_scene_sidecar,
                                       read_scene_sidecar, sidecar_path,
                                       write_scene_sidecar)
from scene_model.utils import SceneModelException


def _fake_fitter(seed=3, nwave=4, nx=3, ny=3, missing=(1, 2)):
    rng = np.random.default_rng(seed)
    nspaxel = nx * ny
    grid_i, grid_j = np.indices((nx, ny))
    present = np.ones((nx, ny), dtype=bool)
    present[missing] = False
    spaxel_i = grid_i[present]
    spaxel_j = grid_j[present]
    spec_id = 100 + np.arange(len(spaxel_i))[::-1]

    data = rng.normal(5.0, 1.0, (nwave, nx, ny))
    variance = rng.uniform(0.5, 2.0, (nwave, nx, ny))
    data[:, ~present] = np.nan
    variance[:, ~present] = np.nan
    estimator = rng.normal(0.0, 0.1, (nwave, 2, nspaxel))
    estimator[:, :, ~present.ravel()] = 0.0

    matrix = fixed_extraction_map(estimator, 0)
    flux = matrix.dot(np.nan_to_num(data.ravel()))
    conditional_variance = np.asarray(
        matrix.multiply(matrix).dot(np.nan_to_num(variance.ravel()))
    ).ravel()
    npar = 3
    names = ("alpha", "beta", "gamma")
    values = rng.normal(size=npar)
    root = rng.normal(size=(npar, npar))
    global_covariance = root @ root.T + np.eye(npar)
    factor = np.linalg.cholesky(global_covariance)
    jacobian = rng.normal(size=(nwave, npar))
    diagnostics = {
        "fixed_extraction_map": matrix,
        "propagated_factor": jacobian @ factor,
        "jacobian": jacobian,
        "conditional_variance": conditional_variance,
        "covariance_factor": factor,
        "fixed_data": data,
        "fixed_variance": variance,
        "fixed_input_shape": (nwave, nx, ny),
        "coefficient_names": ("amplitude", "background"),
        "derivatives": JacobianDiagnostics((), (), (), backend="jax"),
        "jacobian_model": "folded",
        "baseline_parity": {"max_relative": 2e-15, "max_sigma_scaled": 1e-12},
    }
    return SimpleNamespace(
        covariance_diagnostics=diagnostics,
        point_source_spectrum=SimpleNamespace(data=flux.copy()),
        cube=SimpleNamespace(lbda=3300.0 + 10.0 * np.arange(nwave),
                             i=spaxel_i, j=spaxel_j, no=spec_id),
        fit_global_parameter_info=tuple(
            GlobalParameter(name, value, (None, None))
            for name, value in zip(names, values)
        ),
        fit_global_covariance=global_covariance,
    )


def _spectrum_file(tmp_path, content=b"SIMPLE spectrum bytes"):
    path = tmp_path / "spec_B.fits"
    path.write_bytes(content)
    return path


def test_round_trip_preserves_every_array(tmp_path):
    fitter = _fake_fitter()
    payload = build_scene_sidecar(fitter)
    spectrum = _spectrum_file(tmp_path)

    written = write_scene_sidecar(spectrum, payload)
    assert written == sidecar_path(spectrum) == str(spectrum) + ".scenecov.npz"
    result = read_scene_sidecar(spectrum)

    assert result["schema"] == SCENE_SIDECAR_SCHEMA
    assert result["spectrum_name"] == "spec_B.fits"
    assert len(result["spectrum_sha256"]) == 64
    assert result["jacobian_backend"] == "jax"
    assert result["jacobian_model"] == "folded"
    for key, value in payload.items():
        if key in ("spectrum_name", "spectrum_sha256"):
            continue
        stored = result[key]
        if isinstance(value, str):
            assert stored == value
        else:
            np.testing.assert_array_equal(stored, np.asarray(value))
            assert stored.dtype != object
    matrix = fitter.covariance_diagnostics["fixed_extraction_map"]
    assert (result["extraction_map"] != matrix).nnz == 0
    np.testing.assert_array_equal(result["flux"],
                                  fitter.point_source_spectrum.data)
    # Missing spaxel (1, 2) of the 3x3 grid has no Euro3D row.
    assert result["spaxel_spec_id"][1 * 3 + 2] == -1
    assert np.sum(result["spaxel_spec_id"] == -1) == 1
    flat = result["spaxel_i"] * 3 + result["spaxel_j"]
    np.testing.assert_array_equal(result["spaxel_spec_id"][flat],
                                  result["row_spec_id"])
    assert result["conditional_variance_check"] < 1e-12
    assert result["flux_map_max_relative"] < 1e-13


def test_hash_mismatch_is_detected(tmp_path):
    spectrum = _spectrum_file(tmp_path)
    write_scene_sidecar(spectrum, build_scene_sidecar(_fake_fitter()))
    spectrum.write_bytes(b"a different spectrum")

    with pytest.raises(SceneModelException, match="different spectrum"):
        read_scene_sidecar(spectrum)
    read_scene_sidecar(spectrum, verify_hash=False)


def test_shape_mismatch_is_detected(tmp_path):
    spectrum = _spectrum_file(tmp_path)
    payload = build_scene_sidecar(_fake_fitter())
    write_scene_sidecar(spectrum, payload)

    with np.load(sidecar_path(spectrum), allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    stored["spaxel_spec_id"] = stored["spaxel_spec_id"][:-1]
    with open(sidecar_path(spectrum), "wb") as handle:
        np.savez_compressed(handle, **stored)
    with pytest.raises(SceneModelException, match="spaxel_spec_id"):
        read_scene_sidecar(spectrum)

    broken = dict(payload)
    broken["propagated_factor"] = payload["propagated_factor"][:-1]
    with pytest.raises(SceneModelException, match="propagated_factor"):
        write_scene_sidecar(spectrum, broken)


def test_map_that_does_not_reproduce_flux_raises():
    fitter = _fake_fitter()
    fitter.point_source_spectrum.data[2] *= 1 + 1e-3

    with pytest.raises(SceneModelException, match="FLUX-MAP-PARITY-1"):
        build_scene_sidecar(fitter)


def _cancelling_fitter():
    """Fake fitter whose wavelength-0 flux cancels to ~1e-8 of |H||d|."""
    fitter = _fake_fitter()
    diagnostics = fitter.covariance_diagnostics
    matrix = diagnostics["fixed_extraction_map"]
    data = np.array(diagnostics["fixed_data"], copy=True)
    flat = data.reshape(-1)
    row = matrix.getrow(0)
    column = int(row.indices[np.argmax(np.abs(row.data))])
    weight = float(matrix[0, column])
    flat[column] -= (matrix.dot(np.nan_to_num(flat))[0] - 1e-7) / weight
    diagnostics["fixed_data"] = data
    flux = matrix.dot(np.nan_to_num(flat))
    fitter.point_source_spectrum.data = flux.copy()
    scale = abs(matrix).dot(np.abs(np.nan_to_num(flat)))
    return fitter, flux, scale


def test_ulp_scale_difference_under_cancellation_passes():
    fitter, flux, scale = _cancelling_fitter()
    eps = np.finfo(float).eps
    assert abs(flux[0]) < 1e-6 * scale[0]
    # A few ulps of the summation scale: far beyond rtol 1e-13 of the
    # cancelled flux, well inside 1024 eps |H||d|.
    fitter.point_source_spectrum.data[0] += 8 * eps * scale[0]
    assert abs(8 * eps * scale[0] / flux[0]) > 1e-13

    payload = build_scene_sidecar(fitter)

    assert payload["flux_map_max_roundoff_units"] == pytest.approx(8, rel=0.2)
    assert payload["flux_map_max_relative"] > 1e-13
    assert payload["flux_map_max_sigma"] < 1e-6


def test_swapped_spaxel_columns_fail():
    fitter = _fake_fitter()
    diagnostics = fitter.covariance_diagnostics
    matrix = diagnostics["fixed_extraction_map"].tolil()
    nspaxel = 9
    # Swap two present spaxels' columns at every wavelength.
    for wave in range(matrix.shape[0]):
        left, right = wave * nspaxel + 0, wave * nspaxel + 4
        column_left = matrix[:, left].toarray()
        matrix[:, left] = matrix[:, right].toarray()
        matrix[:, right] = column_left
    diagnostics["fixed_extraction_map"] = matrix.tocsr()

    with pytest.raises(SceneModelException, match="FLUX-MAP-PARITY-1"):
        build_scene_sidecar(fitter)


def test_conditional_variance_mismatch_is_stored_not_raised():
    fitter = _fake_fitter()
    fitter.covariance_diagnostics["conditional_variance"][1] *= 1 + 1e-3

    payload = build_scene_sidecar(fitter)

    assert payload["conditional_variance_check"] == pytest.approx(
        1e-3 / (1 + 1e-3), rel=1e-6)


def test_object_dtype_is_refused(tmp_path):
    spectrum = _spectrum_file(tmp_path)
    payload = build_scene_sidecar(_fake_fitter())
    payload["parameter_names"] = np.array(["alpha", "beta", "gamma"],
                                          dtype=object)

    with pytest.raises(SceneModelException, match="object dtype"):
        write_scene_sidecar(spectrum, payload)
    assert not os.path.exists(sidecar_path(spectrum))


# --- the real SnifsCubeFitter.write_spectrum path --------------------------

def _writable_fitter():
    pytest.importorskip("jax")
    from test_jax_production_integration import _production_fitter

    fitter = _production_fitter("fourier")
    cube = fitter.cube
    # Drop grid spaxel (0, 0) so one grid position has no Euro3D row.
    keep = np.arange(1, cube.data.shape[1])
    cube.data = cube.data[:, keep]
    cube.var = cube.var[:, keep]
    cube.i = cube.i[keep]
    cube.j = cube.j[keep]
    cube.no = 1000 + keep
    fitter.header = fits.Header({"EFFTIME": 20.0})
    fitter.fitter_3d = SimpleNamespace(fit_chi_square=1.0)
    fitter.adr_model = SimpleNamespace(get_airmass=lambda: 1.2,
                                       get_parangle=lambda: 30.0)
    fitter.meta_cube = SimpleNamespace(lstart=3650.0, lend=7050.0)
    fitter.prior_scale = 0.0
    fitter.seeing_prior = None
    fitter.subsampling = 2
    fitter.border = 2
    fitter.path = "synthetic_cube.fits"
    return fitter


def test_write_spectrum_writes_matching_sidecar(tmp_path):
    fitter = _writable_fitter()
    fitter.extract(method="psf", covariance=True, jacobian_backend="jax",
                   jax_wavelength_batch=None)
    output = tmp_path / "spec_B.fits"

    fitter.write_spectrum(str(output), str(tmp_path / "sky_B.fits"))

    header = fits.getheader(output)
    assert header["COVSIDE"] == "spec_B.fits.scenecov.npz"
    result = read_scene_sidecar(output)
    with fits.open(output) as hdus:
        flux = hdus[0].data.astype(float)
        covar = np.tril(hdus["COVAR"].data.astype(float))
    covar = covar + np.tril(covar, -1).T
    np.testing.assert_array_equal(result["flux"], flux)
    expected = (np.diag(result["conditional_variance"]) +
                result["propagated_factor"] @ result["propagated_factor"].T)
    np.testing.assert_allclose(covar, expected, rtol=1e-12, atol=0.0)
    assert result["spaxel_spec_id"][0] == -1
    assert np.sum(result["spaxel_spec_id"] == -1) == 1
    assert np.isnan(result["fixed_data"][0])
    assert result["conditional_variance_check"] < 1e-8


def test_failed_sidecar_removes_spectrum(tmp_path, monkeypatch):
    fitter = _writable_fitter()
    fitter.extract(method="psf", covariance=True, jacobian_backend="jax",
                   jax_wavelength_batch=None)
    output = tmp_path / "spec_B.fits"

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(snifs, "write_scene_sidecar", fail)
    with pytest.raises(OSError, match="disk full"):
        fitter.write_spectrum(str(output), str(tmp_path / "sky_B.fits"))
    assert not output.exists()
    assert not os.path.exists(sidecar_path(output))


def test_covariance_off_writes_no_sidecar_or_keyword(tmp_path):
    fitter = _writable_fitter()
    fitter.extract(method="psf", covariance=False)
    output = tmp_path / "spec_B.fits"

    fitter.write_spectrum(str(output), str(tmp_path / "sky_B.fits"))

    assert "COVSIDE" not in fits.getheader(output)
    assert not os.path.exists(sidecar_path(output))
    assert os.listdir(tmp_path) == ["spec_B.fits"]
