"""Oracles for the JAX fit backend (scene_model.jax_fit).

The NumPy element graph is the reference: the JAX objective must reproduce
SceneModel.chi_square / MultipleImageFitter.chi_square, its gradient must
match centred finite differences of the NumPy chi-square, its Hessian-based
covariance must match the NumPy finite-difference covariance, and a fit with
fit_backend='jax' must reach the NumPy fit's minimum.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scene_model.fit import MultipleImageFitter, check_fit_backend
from scene_model.jax_fit import (
    multiple_image_objective,
    scene_model_objective,
)
from scene_model.snifs import (
    SnifsClassicSceneModel,
    SnifsClassicSeeingPrior,
    SnifsFourierSceneModel,
    SnifsFourierSeeingPrior,
)
from scene_model.utils import SceneModelException

pytest.importorskip("jax")

CHI2_RTOL = 1e-10
GRADIENT_RTOL = 1e-5
SIGMA_RTOL = 1e-4

TRUTH_2D = {
    "fourier": dict(center_x=0.31, center_y=-0.42, seeing_width=0.9,
                    ellipticity_x=0.2, ellipticity_y=0.45),
    "classic": dict(center_x=0.31, center_y=-0.42, alpha=2.3, ell=1.1,
                    xy=0.03),
}
TRUTH_3D = {
    "fourier": dict(ref_center_x=0.17, ref_center_y=-0.23, adr_delta=1.1,
                    adr_theta=0.31, seeing_ref_power=-0.3,
                    seeing_ref_width=0.95, ellipticity_x=0.2,
                    ellipticity_y=0.4),
    "classic": dict(ref_center_x=0.17, ref_center_y=-0.23, adr_delta=1.1,
                    adr_theta=0.31, A0=0.08, A1=-0.19, A2=2.35, ell=1.08,
                    xy=0.035),
}


class _LinearAdr:
    def get_scale(self, wavelength):
        return (np.asarray(wavelength, dtype=float) - 5000.0) / 10000.0

    def set_param(self, **kwargs):
        pass


def _model(psf, **kwargs):
    common = dict(grid_size=(15, 15), subsampling=3, border=5, **kwargs)
    if psf == "fourier":
        return SnifsFourierSceneModel(**common)
    return SnifsClassicSceneModel(exposure_time=20.0, **common)


def _noisy(model_image, seed):
    rng = np.random.default_rng(seed)
    variance = 0.5 + 0.01 * np.abs(model_image)
    return model_image + rng.standard_normal(model_image.shape) * \
        np.sqrt(variance), variance


def _two_d(psf, background_degree=0):
    truth = _model(psf, background_degree=background_degree)
    parameters = dict(TRUTH_2D[psf], amplitude=150.0)
    if background_degree == 0:
        parameters["background"] = 2.0
    image, variance = _noisy(truth.evaluate(**parameters), 1)
    model = _model(psf, background_degree=background_degree)
    model.load_image(image, variance)
    return model


def _three_d(psf, priors=False):
    wavelengths = np.linspace(3600.0, 5100.0, 5)
    adr = _LinearAdr()
    base = _model(psf, wavelength_dependence=True, background_degree=0,
                  adr_model=adr, spaxel_size=0.43)
    parameters = dict(TRUTH_3D[psf], wavelength=wavelengths,
                      amplitude=np.linspace(120.0, 160.0, 5), background=1.5)
    images, variances = _noisy(base.evaluate_multi(**parameters), 2)
    prior_list = []
    if priors:
        prior_list = [SnifsFourierSeeingPrior("B", 300.0, 1.4)
                      if psf == "fourier" else
                      SnifsClassicSeeingPrior("B", 1.4)]
    fitter = MultipleImageFitter(base, images, variances,
                                 priors=prior_list, prior_scale=1.0)
    fitter.fix(wavelength=wavelengths)
    for name, value in TRUTH_3D[psf].items():
        fitter.add_global_fit_parameter(name, value * 1.03 + 0.01)
    return fitter


def _points(x0, bounds, n=4, seed=3):
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    points = [x0]
    for _ in range(n - 1):
        x = x0 + 0.03 * np.maximum(np.abs(x0), 0.05) * \
            rng.standard_normal(len(x0))
        for i, (low, high) in enumerate(bounds):
            if low is not None:
                x[i] = max(x[i], low)
            if high is not None:
                x[i] = min(x[i], high)
        points.append(x)
    return points


def _centred_gradient(function, x):
    gradient = np.empty(len(x))
    for i in range(len(x)):
        step = 1e-6 * max(abs(x[i]), 1e-2)
        plus, minus = x.copy(), x.copy()
        plus[i] += step
        minus[i] -= step
        gradient[i] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def _single_setup(model, analytic):
    info = model._get_fit_parameters(do_analytic_coefficients=analytic)
    names = list(info)
    x0 = np.array([v["value"] for v in info.values()], float)
    bounds = [v["bounds"] for v in info.values()]

    def numpy_chi2(x):
        return model.chi_square(dict(zip(names, x)),
                                do_analytic_coefficients=analytic,
                                apply_fit_scale=True, apply_priors=True)

    objective = scene_model_objective(model, names,
                                      analytic_coefficients=analytic)
    return names, x0, bounds, numpy_chi2, objective


def _multi_setup(fitter, analytic):
    global_info, individual = fitter._get_fit_parameters(analytic)
    names, x0, bounds, scales = fitter._parse_start_fit_parameters(
        global_info, individual)

    def numpy_chi2(x):
        g, i = fitter._map_fit_parameters(x, global_info, individual)
        return fitter.chi_square(g, i, do_analytic_coefficients=analytic,
                                 apply_fit_scale=True, apply_priors=True)

    objective = multiple_image_objective(fitter, names,
                                         analytic_coefficients=analytic)
    return names, np.asarray(x0, float), bounds, numpy_chi2, objective


@pytest.mark.parametrize("psf", ["fourier", "classic"])
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("background_degree", [-1, 0])
def test_two_d_chi2_and_gradient_match_numpy(psf, analytic,
                                             background_degree):
    model = _two_d(psf, background_degree)
    names, x0, bounds, numpy_chi2, objective = _single_setup(model, analytic)
    for x in _points(x0, bounds):
        np.testing.assert_allclose(objective.value(x), numpy_chi2(x),
                                   rtol=CHI2_RTOL)
    value, gradient = objective.value_and_grad(x0)
    expected = _centred_gradient(numpy_chi2, x0)
    assert np.linalg.norm(gradient - expected) <= \
        GRADIENT_RTOL * np.linalg.norm(expected)


@pytest.mark.parametrize("psf", ["fourier", "classic"])
@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("priors", [False, True])
def test_three_d_chi2_and_gradient_match_numpy(psf, analytic, priors):
    fitter = _three_d(psf, priors=priors)
    names, x0, bounds, numpy_chi2, objective = _multi_setup(fitter, analytic)
    for x in _points(x0, bounds):
        np.testing.assert_allclose(objective.value(x), numpy_chi2(x),
                                   rtol=CHI2_RTOL)
    value, gradient = objective.value_and_grad(x0)
    expected = _centred_gradient(numpy_chi2, x0)
    assert np.linalg.norm(gradient - expected) <= \
        GRADIENT_RTOL * np.linalg.norm(expected)


@pytest.mark.parametrize("psf", ["fourier", "classic"])
def test_two_d_fit_and_covariance_match_numpy(psf):
    reference = _two_d(psf)
    jax_model = deepcopy(reference)
    reference_parameters, _ = reference.fit()
    jax_parameters, _ = jax_model.fit(fit_backend="jax")
    np.testing.assert_allclose(jax_model.fit_chi_square,
                               reference.fit_chi_square, rtol=1e-6)
    for name in TRUTH_2D[psf]:
        assert abs(jax_parameters[name] - reference_parameters[name]) <= \
            1e-3 * max(abs(reference_parameters[name]), 1.0)

    names, covariance = reference.calculate_covariance()
    jax_names, jax_covariance = reference.calculate_covariance(method="jax")
    assert jax_names == names
    np.testing.assert_allclose(np.sqrt(np.diag(jax_covariance)),
                               np.sqrt(np.diag(covariance)), rtol=SIGMA_RTOL)


@pytest.mark.parametrize("psf", ["fourier", "classic"])
def test_three_d_fit_and_covariance_match_numpy(psf):
    reference = _three_d(psf)
    jax_fitter = deepcopy(reference)
    reference.fit()
    jax_fitter.fit(fit_backend="jax")
    np.testing.assert_allclose(jax_fitter.fit_chi_square,
                               reference.fit_chi_square, rtol=1e-6)

    names, covariance = reference.calculate_covariance()
    jax_names, jax_covariance = reference.calculate_covariance(method="jax")
    assert list(jax_names) == list(names)
    np.testing.assert_allclose(np.sqrt(np.diag(jax_covariance)),
                               np.sqrt(np.diag(covariance)), rtol=SIGMA_RTOL)


def test_unsupported_configurations_raise():
    with pytest.raises(SceneModelException):
        check_fit_backend("torch")
    model = _two_d("fourier", background_degree=0)
    with pytest.raises(SceneModelException):
        scene_model_objective(model, ["not_a_parameter"],
                              analytic_coefficients=True)
    with pytest.raises(SceneModelException):
        scene_model_objective(model, ["amplitude"],
                              analytic_coefficients=True)


@pytest.mark.parametrize("psf", ["fourier", "classic"])
@pytest.mark.parametrize("analytic", [True, False])
def test_structured_hessian_matches_full_jax_hessian(psf, analytic):
    import jax
    from scene_model.jax_fit import _chi2

    fitter = _three_d(psf, priors=True)
    names, x0, bounds, numpy_chi2, objective = _multi_setup(fitter, analytic)
    arguments, static = objective._arguments(), objective._static()
    full = np.asarray(jax.hessian(
        lambda y: _chi2(y, arguments, **static))(x0))
    np.testing.assert_allclose(objective.hessian(x0), full, rtol=1e-9,
                               atol=1e-9 * np.max(np.abs(full)))


@pytest.mark.parametrize("psf", ["fourier", "classic"])
def test_jax_psf_extraction_matches_numpy(psf):
    from scene_model.jax_fit import extract_psf

    wavelengths = np.linspace(3500.0, 5100.0, 7)
    model = _model(psf, wavelength_dependence=True, background_degree=0,
                   adr_model=_LinearAdr(), spaxel_size=0.43)
    model.set_parameters(update_derived=False, **TRUTH_3D[psf])
    parameters = dict(TRUTH_3D[psf], wavelength=wavelengths,
                      amplitude=np.linspace(80.0, 140.0, 7), background=1.2)
    images, variances = _noisy(model.evaluate_multi(**parameters), 4)
    images[2, 3, 4] = np.nan          # masked spaxel
    variances[5, 7, 7] = 0.0          # masked spaxel
    expected = model.extract(images, variances, method="psf",
                             wavelength=wavelengths)
    actual = extract_psf(model, images, variances, wavelengths, batch=4)
    assert actual.colnames == expected.colnames
    for name in expected.colnames:
        np.testing.assert_allclose(np.asarray(actual[name]),
                                   np.asarray(expected[name]), rtol=1e-10)
