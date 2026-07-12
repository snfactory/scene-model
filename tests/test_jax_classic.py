from __future__ import annotations

import numpy as np
import pytest

from scene_model.jax_scene import (
    CLASSIC_PARAMETER_NAMES,
    build_classic_fixed_arrays,
    build_polynomial_background_bases,
    classic_flux,
    classic_flux_jacobian,
)
from scene_model.snifs import SnifsClassicSceneModel
from scene_model.utils import SceneModelException


class _FixedAdr:
    def get_scale(self, wavelength):
        wavelength = np.asarray(wavelength)
        return (wavelength - 5000.0) / 1800.0


def _classic_fixture(exposure_time=20.0, background_degree=1,
                     subsampling=2, border=1):
    adr = _FixedAdr()
    spaxel_size = 0.43
    model = SnifsClassicSceneModel(
        exposure_time=exposure_time,
        wavelength_dependence=True,
        background_degree=background_degree,
        adr_model=adr,
        spaxel_size=spaxel_size,
        grid_size=(15, 15),
        subsampling=subsampling,
        border=border,
    )
    wavelengths = np.array([3750.0, 5000.0, 6750.0])
    parameter_values = np.array([
        0.17, -0.23, 1.08, 0.31, -0.045, -0.19, 2.35, 1.14, 0.08
    ])
    parameters = dict(zip(CLASSIC_PARAMETER_NAMES, parameter_values))
    amplitudes = np.array([4.2, 7.1, 5.6])
    backgrounds = np.array([0.3, -0.2, 0.5])
    slopes_x = np.array([0.15, -0.08, 0.04])
    slopes_y = np.array([-0.03, 0.06, 0.11])
    images = []
    for wavelength, amplitude, background, slope_x, slope_y in zip(
            wavelengths, amplitudes, backgrounds, slopes_x, slopes_y):
        coefficient_parameters = {"amplitude": amplitude}
        if background_degree >= 0:
            coefficient_parameters["background"] = background
        if background_degree >= 1:
            coefficient_parameters.update(
                background_1_0=slope_x, background_0_1=slope_y
            )
        images.append(model.evaluate(
            wavelength=wavelength,
            **coefficient_parameters,
            **parameters,
        ))
    images = np.asarray(images)
    # Non-uniform weights and different masks exercise fixed-axis handling.
    variance = 0.4 + np.arange(images.size).reshape(images.shape) / images.size
    images[0, 1, 9] = np.nan
    variance[1, 11, 2] = np.nan
    variance[2, 4, 12] = -1.0
    bases = build_polynomial_background_bases(
        model.grid_info["grid_x"], model.grid_info["grid_y"],
        background_degree,
    )
    fixed = build_classic_fixed_arrays(
        images, variance, wavelengths,
        adr.get_scale(wavelengths) / spaxel_size,
        model.grid_info, bases, exposure_time,
    )
    return model, images, variance, wavelengths, parameter_values, fixed


def _numpy_flux(model, images, variance, wavelengths, values, names):
    parameters = dict(zip(names, values))
    return np.asarray(model.extract(
        images, variance, wavelength=wavelengths, **parameters
    )["amplitude"])


def test_background_bases_match_production_component_order():
    model = SnifsClassicSceneModel(
        exposure_time=20.0, wavelength_dependence=False,
        background_degree=2, grid_size=(15, 15), subsampling=1, border=0,
    )
    components = model.evaluate(
        separate_components=True, apply_coefficients=False,
        center_x=0.0, center_y=0.0, alpha=2.0, ell=1.0, xy=0.0,
    )
    expected = np.asarray(components[1:])
    actual = build_polynomial_background_bases(
        model.grid_info["grid_x"], model.grid_info["grid_y"], 2
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.writeable is False


@pytest.mark.parametrize("exposure_time", [8.0, 20.0])
def test_classic_fixed_jax_flux_matches_production_numpy(exposure_time):
    pytest.importorskip("jax")
    model, images, variance, wavelengths, values, fixed = _classic_fixture(
        exposure_time=exposure_time
    )
    expected = _numpy_flux(
        model, images, variance, wavelengths, values, CLASSIC_PARAMETER_NAMES
    )
    actual = classic_flux(values, CLASSIC_PARAMETER_NAMES, fixed)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    # Invalid samples remain on the fixed 225-spaxel axis as exact zeros.
    assert fixed.data.shape[1:] == (15, 15)
    assert fixed.data[0, 1, 9] == 0.0
    assert fixed.inverse_variance[0, 1, 9] == 0.0
    assert fixed.inverse_variance[1, 11, 2] == 0.0
    assert fixed.inverse_variance[2, 4, 12] == 0.0


def test_classic_fixed_jax_flux_supports_no_background():
    pytest.importorskip("jax")
    model, images, variance, wavelengths, values, fixed = _classic_fixture(
        background_degree=-1
    )
    expected = _numpy_flux(
        model, images, variance, wavelengths, values, CLASSIC_PARAMETER_NAMES
    )
    actual = classic_flux(values, CLASSIC_PARAMETER_NAMES, fixed)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    assert fixed.background_bases.shape == (0, 15, 15)


def test_classic_named_parameter_reordering_preserves_flux_and_jacobian():
    pytest.importorskip("jax")
    _, _, _, _, values, fixed = _classic_fixture()
    permutation = np.array([8, 3, 5, 0, 7, 2, 6, 1, 4])
    permuted_names = tuple(CLASSIC_PARAMETER_NAMES[index]
                           for index in permutation)
    permuted_values = values[permutation]

    canonical_flux = classic_flux(values, CLASSIC_PARAMETER_NAMES, fixed)
    canonical_jacobian = classic_flux_jacobian(
        values, CLASSIC_PARAMETER_NAMES, fixed
    )
    permuted_flux = classic_flux(permuted_values, permuted_names, fixed)
    permuted_jacobian = classic_flux_jacobian(
        permuted_values, permuted_names, fixed
    )
    np.testing.assert_array_equal(permuted_flux, canonical_flux)
    np.testing.assert_array_equal(
        permuted_jacobian, canonical_jacobian[:, permutation]
    )


def test_classic_jax_jacobian_matches_centered_production_oracle():
    pytest.importorskip("jax")
    model, images, variance, wavelengths, values, fixed = _classic_fixture()
    numerical = np.empty((len(wavelengths), len(values)))
    for index, value in enumerate(values):
        step = 2e-5 * max(abs(value), 1.0)
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        numerical[:, index] = (
            _numpy_flux(model, images, variance, wavelengths, plus,
                        CLASSIC_PARAMETER_NAMES)
            - _numpy_flux(model, images, variance, wavelengths, minus,
                          CLASSIC_PARAMETER_NAMES)
        ) / (2.0 * step)

    automatic = classic_flux_jacobian(
        values, CLASSIC_PARAMETER_NAMES, fixed
    )
    relative_error = (
        np.linalg.norm(automatic - numerical) / np.linalg.norm(numerical)
    )
    assert relative_error <= 1e-6
    np.testing.assert_allclose(automatic, numerical, rtol=2e-6, atol=2e-8)


def test_classic_rejects_missing_names_and_non_225_axis():
    _, _, _, _, values, fixed = _classic_fixture()
    with pytest.raises(SceneModelException, match="parameter names differ"):
        classic_flux(values[:-1], CLASSIC_PARAMETER_NAMES[:-1], fixed)
    with pytest.raises(SceneModelException, match="225-spaxel"):
        build_classic_fixed_arrays(
            np.ones((2, 5, 5)), np.ones((2, 5, 5)), np.ones(2), np.ones(2),
            {
                "subsampling": 1,
                "border": 0,
                "pad_grid_x": np.ones((5, 5)),
                "pad_grid_y": np.ones((5, 5)),
                "pad_grid_kx": np.ones((5, 5)),
                "pad_grid_ky": np.ones((5, 5)),
                "pad_fft_shift": np.ones((5, 5), dtype=complex),
                "pad_ifft_shift": np.ones((5, 5), dtype=complex),
            },
            np.ones((1, 5, 5)), 20.0,
        )
