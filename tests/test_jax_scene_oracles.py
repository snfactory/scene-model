"""Independent production-equation oracles for the fixed JAX scene map.

These tests intentionally use the mutable NumPy element graph and a plain
centered finite difference as the oracle.  They do not share PSF equations or
normal-equation implementation with :mod:`scene_model.jax_scene`.
"""

from __future__ import annotations

import numpy as np
import pytest

from scene_model.jax_scene import (
    CLASSIC_PARAMETER_NAMES,
    ClassicFixedArrays,
    build_classic_fixed_arrays,
    build_polynomial_background_bases,
    classic_flux,
    classic_flux_jacobian,
)
from scene_model.snifs import SnifsClassicSceneModel
from scene_model.utils import SceneModelException


BASELINE_RTOL = 1e-12
BASELINE_ATOL = 0.0
JACOBIAN_RELATIVE_FROBENIUS_LIMIT = 1e-6


class _LinearAdr:
    def get_scale(self, wavelength):
        return (np.asarray(wavelength, dtype=float) - 5000.0) / 10000.0

    def set_param(self, **kwargs):
        pass


def _classic_fixture(*, channel="B", exposure_time=20.0,
                     background_degree=0):
    wavelengths = (
        np.linspace(3500.0, 5100.0, 5) if channel == "B"
        else np.linspace(5400.0, 8500.0, 5)
    )
    adr = _LinearAdr()
    spaxel_size = 0.43
    model = SnifsClassicSceneModel(
        exposure_time=exposure_time, wavelength_dependence=True,
        background_degree=background_degree, adr_model=adr,
        spaxel_size=spaxel_size, grid_size=(15, 15), subsampling=3, border=5,
    )
    values_by_name = {
        "ref_center_x": 0.17,
        "ref_center_y": -0.23,
        "adr_delta": 1.12,
        "adr_theta": 0.31,
        "A0": 0.08,
        "A1": -0.19,
        "A2": 2.35,
        "ell": 1.08,
        "xy": 0.035,
    }
    values = np.array([values_by_name[name]
                       for name in CLASSIC_PARAMETER_NAMES])
    parameters = dict(values_by_name, wavelength=wavelengths,
                      amplitude=np.linspace(60.0, 95.0, len(wavelengths)))
    if background_degree == 0:
        parameters["background"] = 2.25
    elif background_degree > 0:
        parameters["background"] = 0.1
        for x_degree in range(background_degree + 1):
            for y_degree in range(background_degree + 1 - x_degree):
                if x_degree or y_degree:
                    parameters["background_%d_%d" %
                               (x_degree, y_degree)] = (
                        0.03 * x_degree - 0.02 * y_degree
                    )
    data = model.evaluate_multi(**parameters)
    # Heteroscedastic weights exercise the complete variable-projection solve.
    pixel = np.arange(225, dtype=float).reshape(15, 15)
    variance = 0.7 + 0.002 * pixel[None, :, :] \
        + 0.04 * np.arange(len(wavelengths))[:, None, None]
    adr_scale = adr.get_scale(wavelengths) / spaxel_size
    backgrounds = build_polynomial_background_bases(
        model.grid_info["grid_x"], model.grid_info["grid_y"],
        background_degree,
    )
    fixed = build_classic_fixed_arrays(
        data, variance, wavelengths, adr_scale, model.grid_info, backgrounds,
        exposure_time,
    )
    assert isinstance(fixed, ClassicFixedArrays)
    return model, data, variance, wavelengths, values, fixed


def _numpy_classic_flux(model, data, variance, wavelengths, values, names):
    parameters = dict(zip(names, np.asarray(values, dtype=float)))
    result = model.extract(
        data, variance, method="psf", wavelength=wavelengths,
        return_covariance=True, **parameters
    )
    return np.asarray(result.table["amplitude"], dtype=float)


def _centered_jacobian(function, values):
    values = np.asarray(values, dtype=float)
    baseline = np.asarray(function(values), dtype=float)
    jacobian = np.empty((len(baseline), len(values)))
    for index, value in enumerate(values):
        step = 1e-5 * max(abs(value), 1.0)
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (function(plus) - function(minus)) / (2 * step)
    return jacobian


@pytest.mark.parametrize("channel", ["B", "R"])
@pytest.mark.parametrize("exposure_time", [8.0, 20.0])
@pytest.mark.parametrize("background_degree", [-1, 0, 2])
def test_classic_baseline_matches_independent_numpy_extraction(
        channel, exposure_time, background_degree):
    model, data, variance, wavelengths, values, fixed = _classic_fixture(
        channel=channel, exposure_time=exposure_time,
        background_degree=background_degree,
    )
    expected = _numpy_classic_flux(
        model, data, variance, wavelengths, values, CLASSIC_PARAMETER_NAMES
    )
    actual = np.asarray(classic_flux(
        values, CLASSIC_PARAMETER_NAMES, fixed
    ))
    np.testing.assert_allclose(
        actual, expected, rtol=BASELINE_RTOL, atol=BASELINE_ATOL
    )


@pytest.mark.parametrize(
    "channel,exposure_time,background_degree",
    [("B", 20.0, 0), ("R", 8.0, 2)],
)
def test_classic_jacobian_matches_independent_centered_difference(
        channel, exposure_time, background_degree):
    model, data, variance, wavelengths, values, fixed = _classic_fixture(
        channel=channel, exposure_time=exposure_time,
        background_degree=background_degree,
    )
    oracle = _centered_jacobian(
        lambda candidate: _numpy_classic_flux(
            model, data, variance, wavelengths, candidate,
            CLASSIC_PARAMETER_NAMES,
        ),
        values,
    )
    actual = np.asarray(classic_flux_jacobian(
        values, CLASSIC_PARAMETER_NAMES, fixed
    ))
    relative_error = np.linalg.norm(actual - oracle) / np.linalg.norm(oracle)
    assert actual.shape == oracle.shape
    assert np.all(np.isfinite(actual))
    assert relative_error <= JACOBIAN_RELATIVE_FROBENIUS_LIMIT


def test_classic_parameter_mapping_is_name_based_and_strict():
    _, _, _, _, values, fixed = _classic_fixture()
    permutation = np.array([8, 2, 5, 0, 7, 3, 6, 1, 4])
    names = tuple(CLASSIC_PARAMETER_NAMES[index] for index in permutation)
    expected = np.asarray(classic_flux(
        values, CLASSIC_PARAMETER_NAMES, fixed
    ))
    reordered = np.asarray(classic_flux(values[permutation], names, fixed))
    np.testing.assert_allclose(
        reordered, expected, rtol=BASELINE_RTOL, atol=BASELINE_ATOL
    )

    with pytest.raises(SceneModelException):
        classic_flux(values[:-1], CLASSIC_PARAMETER_NAMES[:-1], fixed)
    duplicate = tuple(CLASSIC_PARAMETER_NAMES[:-1]) + \
        (CLASSIC_PARAMETER_NAMES[-2],)
    with pytest.raises(SceneModelException):
        classic_flux(values, duplicate, fixed)


def test_classic_masked_pixel_value_cannot_affect_flux_or_jacobian():
    model, data, variance, wavelengths, values, _ = _classic_fixture()
    masked_variance = variance.copy()
    masked_variance[:, 3, 11] = np.nan
    first_data = data.copy()
    second_data = data.copy()
    first_data[:, 3, 11] = np.nan
    second_data[:, 3, 11] = 1e200
    backgrounds = build_polynomial_background_bases(
        model.grid_info["grid_x"], model.grid_info["grid_y"], 0
    )
    adr_scale = _LinearAdr().get_scale(wavelengths) / 0.43
    first = build_classic_fixed_arrays(
        first_data, masked_variance, wavelengths, adr_scale, model.grid_info,
        backgrounds, 20.0,
    )
    second = build_classic_fixed_arrays(
        second_data, masked_variance, wavelengths, adr_scale, model.grid_info,
        backgrounds, 20.0,
    )
    np.testing.assert_array_equal(
        np.asarray(classic_flux(values, CLASSIC_PARAMETER_NAMES, first)),
        np.asarray(classic_flux(values, CLASSIC_PARAMETER_NAMES, second)),
    )
    first_jacobian = np.asarray(classic_flux_jacobian(
        values, CLASSIC_PARAMETER_NAMES, first
    ))
    second_jacobian = np.asarray(classic_flux_jacobian(
        values, CLASSIC_PARAMETER_NAMES, second
    ))
    relative_error = (
        np.linalg.norm(first_jacobian - second_jacobian)
        / max(np.linalg.norm(first_jacobian), np.finfo(float).tiny)
    )
    assert relative_error <= JACOBIAN_RELATIVE_FROBENIUS_LIMIT
