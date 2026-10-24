"""SCALA pupil-mask instrumental core (scene_model.scala_mask).

Oracle: Daniel Kuesters' ``otf_analytic`` / ``MaskPsfElement`` as sent by
e-mail (2026-10), reproduced verbatim in ``_daniel_otf_analytic`` below.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from scene_model import scala_mask
from scene_model.jax_scene import (
    FOURIER_PARAMETER_NAMES,
    build_fourier_fixed_arrays,
    build_polynomial_background_bases,
    fourier_flux,
    fourier_mask_arrays,
    fourier_profile_constants,
    fourier_source_pixels,
)
from scene_model.scala_mask import (
    APERTURE_RADIUS,
    ScalaMaskPsfElement,
    circular_aperture_otf,
    load_base_vectors,
    scala_mask_otf,
    select_base_vectors,
)
from scene_model.snifs import SnifsFourierSceneModel
from scene_model.utils import SceneModelException

ORIGIN = np.array([[0.0, 0.0]])


def _daniel_otf_analytic(xx, yy, x0, y0, R, lbda):
    """Verbatim copy of Daniel's function (NaN warnings silenced)."""
    scale = 45 * 1.2 * 0.645 * 0.9 / (lbda / 10000)
    rho_p = np.sqrt((xx / scale - x0) ** 2 + (yy / scale - y0) ** 2) / R
    with np.errstate(invalid="ignore"):
        ot = 2 / np.pi * (np.arccos(rho_p) - rho_p * np.sqrt(1 - rho_p ** 2))
    ot[rho_p > 1] = 0.0
    return ot


class _FixedAdr:
    def set_param(self, **kwargs):
        pass

    def get_scale(self, wavelength):
        wavelength = np.asarray(wavelength)
        return (wavelength - 5000.0) / 1800.0


def _model(inst_core, base_vectors=ORIGIN, subsampling=2, border=2,
           background_degree=0, **kwargs):
    return SnifsFourierSceneModel(
        use_empirical_parameters=True, wavelength_dependence=True,
        background_degree=background_degree, adr_model=_FixedAdr(),
        spaxel_size=0.43, grid_size=(15, 15), subsampling=subsampling,
        border=border, inst_core=inst_core, scala_base_vectors=base_vectors,
        **kwargs)


# ----------------------------------------------------------------- functions

@pytest.mark.parametrize("wavelength", [3300.0, 5000.0, 9700.0])
@pytest.mark.parametrize("x0,y0", [(0.0, 0.0), (0.7, -0.4)])
def test_circular_aperture_otf_matches_daniel(wavelength, x0, y0):
    k = 2 * np.pi * np.fft.fftfreq(60, 1.0 / 2)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    expected = _daniel_otf_analytic(kx, ky, x0, y0, 0.16, wavelength)
    actual = circular_aperture_otf(kx, ky, x0, y0, APERTURE_RADIUS,
                                   wavelength)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    # The OTF vanishes beyond k = scale * R (plus the vector offset); the
    # support shrinks like 1/wavelength, so with this grid it is cut only in
    # the red.
    cutoff = scala_mask.pupil_scale(wavelength) * APERTURE_RADIUS
    assert (np.count_nonzero(actual) < actual.size) == (
        np.abs(k).max() > cutoff - np.hypot(x0, y0) * cutoff / APERTURE_RADIUS)


def test_otf_is_one_at_zero_frequency_and_bounded():
    k = np.linspace(-40, 40, 201)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    otf = scala_mask_otf(kx, ky, ORIGIN, 4000.0)
    assert otf[100, 100] == 1.0
    assert otf.min() >= 0.0 and otf.max() <= 1.0
    # radially symmetric for the origin vector
    np.testing.assert_allclose(otf, otf.T, rtol=0, atol=1e-15)


def test_otf_with_jax_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    k = 2 * np.pi * np.fft.fftfreq(40, 0.5)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    expected = circular_aperture_otf(kx, ky, 0.3, 0.1, APERTURE_RADIUS, 4500.0)
    actual = np.asarray(circular_aperture_otf(
        jnp.asarray(kx), jnp.asarray(ky), 0.3, 0.1, APERTURE_RADIUS, 4500.0,
        xp=jnp))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_select_base_vectors_reproduces_daniels_loop():
    vectors = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 0.0], [0.5, 0.5]])
    # production: first vector only
    np.testing.assert_array_equal(select_base_vectors(vectors), ORIGIN)
    # all vectors, radius cut 2.5 + 0.372 + 0.02 drops (3, 0)
    selected = select_base_vectors(vectors, max_vectors=None)
    np.testing.assert_array_equal(selected, vectors[[0, 1, 3]])
    with pytest.raises(SceneModelException):
        select_base_vectors(np.array([[5.0, 5.0]]))
    with pytest.raises(SceneModelException):
        select_base_vectors(vectors, max_vectors=0)


def test_load_base_vectors_reads_npy_and_python2_style_pickle(tmp_path,
                                                             monkeypatch):
    vectors = np.array([[0.0, 0.0], [0.9, -0.3]])
    npy = tmp_path / "v.npy"
    np.save(npy, vectors)
    np.testing.assert_array_equal(load_base_vectors(str(npy)), vectors)
    pkl = tmp_path / "v.pkl"
    with open(pkl, "wb") as handle:
        pickle.dump([tuple(v) for v in vectors], handle, protocol=2)
    np.testing.assert_array_equal(load_base_vectors(str(pkl)), vectors)
    monkeypatch.setenv(scala_mask.BASE_VECTORS_ENV, str(npy))
    np.testing.assert_array_equal(load_base_vectors(), vectors)
    monkeypatch.setenv(scala_mask.BASE_VECTORS_ENV, str(tmp_path / "no"))
    with pytest.raises(SceneModelException, match="not found"):
        load_base_vectors()


def test_installed_base_vectors_are_daniels_file(monkeypatch):
    """scene_model/data/scala_base_vectors.pkl is Daniel's
    base_vectors_python2.pkl: 18 x 18 pairwise difference vectors of an
    18-aperture mask, the origin first, so the production selection (first
    vector only) is normalized."""
    monkeypatch.delenv(scala_mask.BASE_VECTORS_ENV, raising=False)
    if not any(os.path.exists(scala_mask.DEFAULT_BASE_VECTORS_PATH + ext)
               for ext in (".npy", ".pkl")):
        pytest.skip("Daniel's base-vector file is not installed (kept out "
                    "of git; see scene_model/data/README.md)")
    vectors = load_base_vectors()
    assert vectors.shape == (324, 2)
    np.testing.assert_array_equal(vectors[0], [0.0, 0.0])
    assert np.all(np.hypot(vectors[:, 0], vectors[:, 1]) < 1.81)
    assert (np.all(vectors == 0, axis=1)).sum() == 18
    np.testing.assert_array_equal(select_base_vectors(vectors), ORIGIN)
    assert float(scala_mask_otf(np.zeros(1), np.zeros(1), vectors,
                                10000.0)[0]) == pytest.approx(18.0)


# ------------------------------------------------------------------- element

def test_element_evaluates_daniels_otf_at_the_shared_wavelength():
    element = ScalaMaskPsfElement(ORIGIN, prefix="inst_core",
                                  fits_keyword_prefix="C")
    assert list(element._parameter_info) == ["wavelength"]
    k = 2 * np.pi * np.fft.fftfreq(30, 0.5)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    actual = element._evaluate_fourier(kx, ky, 2, {}, wavelength=6000.0)
    np.testing.assert_array_equal(
        actual, _daniel_otf_analytic(kx, ky, 0.0, 0.0, 0.16, 6000.0))
    with pytest.raises(SceneModelException, match="wavelength"):
        element._evaluate_fourier(kx, ky, 2, {}, wavelength=None)


def test_element_rejects_unnormalized_selections():
    with pytest.raises(SceneModelException, match="not 1"):
        ScalaMaskPsfElement(np.array([[0.0, 0.0], [0.05, 0.0]]))
    with pytest.raises(SceneModelException, match="not 1"):
        ScalaMaskPsfElement(np.array([[1.0, 0.0]]))
    with pytest.raises(SceneModelException):
        ScalaMaskPsfElement(np.zeros((0, 2)))


# --------------------------------------------------------------- scene model

def test_mask_model_replaces_the_gaussian_core_only():
    gaussian = _model("gaussian")
    mask = _model("scala_mask")
    assert mask.inst_core == "scala_mask"
    assert isinstance(mask.inst_core_element, ScalaMaskPsfElement)
    assert set(gaussian.parameters) - set(mask.parameters) == {
        "inst_core_sigma_x", "inst_core_sigma_y", "inst_core_rho"}
    assert set(mask.parameters) <= set(gaussian.parameters)
    # the other empirical constants are still fixed
    for name in ("inst_wings_power", "inst_wings_width", "seeing_power"):
        assert mask._parameter_info[name]["fixed"]
        assert mask[name] == gaussian[name]
    with pytest.raises(SceneModelException, match="wavelength_dependence"):
        SnifsFourierSceneModel(wavelength_dependence=False,
                               inst_core="scala_mask",
                               scala_base_vectors=ORIGIN)
    with pytest.raises(SceneModelException, match="Unknown instrumental"):
        SnifsFourierSceneModel(inst_core="airy")


def test_mask_model_images_are_finite_normalized_and_chromatic():
    parameters = dict(zip(FOURIER_PARAMETER_NAMES,
                          [0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09]))
    gaussian = _model("gaussian", background_degree=-1, border=15)
    mask = _model("scala_mask", background_degree=-1, border=15)
    previous = None
    for wavelength in (3650.0, 5000.0, 8800.0):
        image_g, image_m = [m.evaluate(wavelength=wavelength, amplitude=1.0,
                                       **parameters)
                            for m in (gaussian, mask)]
        assert np.isfinite(image_m).all()
        # unit amplitude: the PSF integrates to 1 on the padded grid, and
        # nearly all of it falls inside the 15x15 field for these widths
        assert 0.8 < image_m.sum() <= 1.0 + 1e-12
        # the mask core is wider than the Gaussian core: the peak is lower
        assert image_m.max() < image_g.max()
        # and chromatic in a way the fixed Gaussian core is not
        if previous is not None:
            assert not np.allclose(image_m / image_m.max(),
                                   previous / previous.max(), atol=1e-4)
        previous = image_m


@pytest.mark.parametrize("subsampling,border", [(2, 2), (4, 3)])
def test_jax_source_pixels_match_numpy_mask_model(subsampling, border):
    pytest.importorskip("jax")
    model = _model("scala_mask", subsampling=subsampling, border=border,
                   background_degree=-1)
    values = np.array([0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09])
    parameters = dict(zip(FOURIER_PARAMETER_NAMES, values))
    wavelengths = np.array([3650.0, 5000.0, 7050.0, 9500.0])
    expected = np.asarray([
        model.evaluate(wavelength=w, amplitude=1.0, **parameters)
        for w in wavelengths])
    grid = model.grid_info
    adr_scale = np.ravel(_FixedAdr().get_scale(wavelengths)) / 0.43
    mask_vectors, mask_radius = fourier_mask_arrays(model)
    np.testing.assert_array_equal(mask_vectors, ORIGIN)
    assert mask_radius == APERTURE_RADIUS
    constants = fourier_profile_constants(model.parameters, "scala_mask")
    assert constants[:3].tolist() == [0.0, 0.0, 0.0]
    for fold in (False, True):
        actual = np.asarray(fourier_source_pixels(
            values, wavelengths, adr_scale, grid["pad_grid_kx"],
            grid["pad_grid_ky"], grid["pad_ifft_shift"], constants,
            native_shape=(15, 15), subsampling=subsampling, border=border,
            fold=fold, mask_vectors=mask_vectors, mask_radius=mask_radius))
        np.testing.assert_allclose(actual, expected, rtol=0,
                                   atol=1e-12 * np.max(np.abs(expected)))
    # and the Gaussian-core evaluator differs: the mask is really in use
    other = np.asarray(fourier_source_pixels(
        values, wavelengths, adr_scale, grid["pad_grid_kx"],
        grid["pad_grid_ky"], grid["pad_ifft_shift"],
        fourier_profile_constants(_model("gaussian").parameters),
        native_shape=(15, 15), subsampling=subsampling, border=border))
    assert np.max(np.abs(other - expected)) > 1e-3 * np.max(np.abs(expected))


def test_jax_fixed_arrays_carry_the_mask_and_flux_matches_numpy():
    pytest.importorskip("jax")
    model = _model("scala_mask", background_degree=0)
    values = np.array([0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09])
    parameters = dict(zip(FOURIER_PARAMETER_NAMES, values))
    wavelengths = np.array([3650.0, 5000.0, 7050.0])
    amplitudes = np.array([4.4, 7.3, 5.2])
    images = np.asarray([
        model.evaluate(wavelength=w, amplitude=a, background=0.3,
                       **parameters)
        for w, a in zip(wavelengths, amplitudes)])
    variance = 0.35 + np.arange(images.size).reshape(images.shape) / images.size
    grid = model.grid_info
    bases = build_polynomial_background_bases(grid["grid_x"], grid["grid_y"],
                                              0)
    adr_scale = np.ravel(_FixedAdr().get_scale(wavelengths)) / 0.43
    mask_vectors, mask_radius = fourier_mask_arrays(model)
    fixed = build_fourier_fixed_arrays(
        images, variance, wavelengths, adr_scale, grid, bases,
        fourier_profile_constants(model.parameters, "scala_mask"),
        mask_vectors=mask_vectors, mask_radius=mask_radius)
    np.testing.assert_array_equal(fixed.mask_vectors, ORIGIN)
    assert fixed.mask_vectors.flags.writeable is False
    flux = fourier_flux(values, FOURIER_PARAMETER_NAMES, fixed)
    np.testing.assert_allclose(flux, amplitudes, rtol=1e-9)
    # default (no mask) arrays still build and are the Gaussian core
    plain = build_fourier_fixed_arrays(
        images, variance, wavelengths, adr_scale, grid, bases,
        fourier_profile_constants(_model("gaussian").parameters))
    assert plain.mask_vectors.shape == (0, 2)
    with pytest.raises(SceneModelException, match="shape"):
        build_fourier_fixed_arrays(
            images, variance, wavelengths, adr_scale, grid, bases,
            fixed.profile_constants, mask_vectors=np.zeros((2, 3)))


def test_mask_model_header_round_trip(monkeypatch, tmp_path):
    from astropy.io import fits
    model = _model("scala_mask")
    header = fits.Header()
    for key, value, comment in model.get_fits_header_items():
        header[key] = (value, comment)
    assert header["ES_CORE"] == "scala_mask"
    assert header["ES_MSKN"] == 1
    assert header["ES_MSKR"] == APERTURE_RADIUS
    assert "ES_CSIGX" not in header
    header["ES_SDEG"] = 0
    header["PRESSURE"] = 616.0
    header["TEMP"] = 2.0
    # from_fits_header loads the installed base vectors
    npy = tmp_path / "v.npy"
    np.save(npy, np.array([[0.0, 0.0], [1.0, 1.0]]))
    monkeypatch.setenv(scala_mask.BASE_VECTORS_ENV, str(npy))
    loaded = SnifsFourierSceneModel.from_fits_header(header)
    assert loaded.inst_core == "scala_mask"
    np.testing.assert_array_equal(loaded.scala_base_vectors, ORIGIN)
    # Gaussian headers (no ES_CORE, as written before this option) still load
    gaussian = _model("gaussian")
    header = fits.Header()
    for key, value, comment in gaussian.get_fits_header_items():
        if key != "ES_CORE":
            header[key] = (value, comment)
    header["ES_SDEG"] = 0
    header["PRESSURE"] = 616.0
    header["TEMP"] = 2.0
    assert SnifsFourierSceneModel.from_fits_header(header).inst_core == \
        "gaussian"


def test_jax_fit_objective_accepts_the_mask_model():
    pytest.importorskip("jax")
    from scene_model.fit import MultipleImageFitter
    from scene_model.jax_fit import SceneObjective
    model = _model("scala_mask", background_degree=0)
    values = np.array([0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09])
    parameters = dict(zip(FOURIER_PARAMETER_NAMES, values))
    wavelengths = np.array([3650.0, 5000.0, 7050.0])
    images = np.asarray([
        model.evaluate(wavelength=w, amplitude=a, background=0.3,
                       **parameters)
        for w, a in zip(wavelengths, (4.4, 7.3, 5.2))])
    variance = 0.35 + np.arange(images.size).reshape(images.shape) / images.size
    fitter = MultipleImageFitter(model, images, variance)
    fitter.fix(wavelength=wavelengths)
    for name in FOURIER_PARAMETER_NAMES:
        fitter.add_global_fit_parameter(name, parameters[name])
    names = list(FOURIER_PARAMETER_NAMES)
    objective = SceneObjective(
        fitter.scene_models, names, analytic_coefficients=True,
        apply_fit_scale=False, priors_by_scale=(),
        fixed_values={})
    np.testing.assert_array_equal(objective.mask_vectors, ORIGIN)
    expected = fitter.chi_square(dict(zip(names, values)),
                                 do_analytic_coefficients=True,
                                 apply_fit_scale=False)
    actual = objective.value(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-6)
    _, gradient = objective.value_and_grad(values)
    assert np.all(np.isfinite(gradient))
