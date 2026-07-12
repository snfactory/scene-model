from __future__ import annotations

import numpy as np
import pytest

from scene_model.jax_scene import (
    FOURIER_PARAMETER_NAMES,
    FOURIER_PROFILE_NAMES,
    batched_flux_jacobian,
    build_fourier_fixed_arrays,
    build_polynomial_background_bases,
    fourier_flux,
    fourier_flux_jacobian,
)
from scene_model.snifs import SnifsFourierSceneModel
from scene_model.utils import SceneModelException


class _FixedAdr:
    def get_scale(self, wavelength):
        wavelength = np.asarray(wavelength)
        return (wavelength - 5000.0) / 1800.0


def _fourier_fixture(background_degree=1, subsampling=2, border=2):
    adr = _FixedAdr()
    spaxel_size = 0.43
    model = SnifsFourierSceneModel(
        use_empirical_parameters=True,
        wavelength_dependence=True,
        background_degree=background_degree,
        adr_model=adr,
        spaxel_size=spaxel_size,
        grid_size=(15, 15),
        subsampling=subsampling,
        border=border,
    )
    wavelengths = np.array([3650.0, 5000.0, 7050.0])
    parameter_values = np.array([
        0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09
    ])
    parameters = dict(zip(FOURIER_PARAMETER_NAMES, parameter_values))
    amplitudes = np.array([4.4, 7.3, 5.2])
    backgrounds = np.array([0.25, -0.15, 0.45])
    slopes_x = np.array([0.12, -0.06, 0.03])
    slopes_y = np.array([-0.02, 0.05, 0.09])
    images = []
    for wavelength, amplitude, background, slope_x, slope_y in zip(
            wavelengths, amplitudes, backgrounds, slopes_x, slopes_y):
        images.append(model.evaluate(
            wavelength=wavelength,
            amplitude=amplitude,
            background=background,
            background_1_0=slope_x,
            background_0_1=slope_y,
            **parameters,
        ))
    images = np.asarray(images)
    variance = 0.35 + np.arange(images.size).reshape(images.shape) / images.size
    images[0, 2, 10] = np.nan
    variance[1, 12, 3] = np.nan
    variance[2, 5, 11] = 0.0
    bases = build_polynomial_background_bases(
        model.grid_info["grid_x"], model.grid_info["grid_y"],
        background_degree,
    )
    profile_constants = {
        name: model.parameters[name] for name in FOURIER_PROFILE_NAMES
    }
    fixed = build_fourier_fixed_arrays(
        images, variance, wavelengths,
        adr.get_scale(wavelengths) / spaxel_size,
        model.grid_info, bases, profile_constants,
    )
    return model, images, variance, wavelengths, parameter_values, fixed


def _numpy_flux(model, images, variance, wavelengths, values, names):
    parameters = dict(zip(names, values))
    return np.asarray(model.extract(
        images, variance, wavelength=wavelengths, **parameters
    )["amplitude"])


@pytest.mark.parametrize("background_degree", [-1, 0, 1])
def test_fourier_fixed_jax_flux_matches_production_numpy(background_degree):
    pytest.importorskip("jax")
    model, images, variance, wavelengths, values, fixed = _fourier_fixture(
        background_degree=background_degree
    )
    expected = _numpy_flux(
        model, images, variance, wavelengths, values, FOURIER_PARAMETER_NAMES
    )
    actual = fourier_flux(values, FOURIER_PARAMETER_NAMES, fixed)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    assert fixed.data.shape[1:] == (15, 15)
    assert fixed.data[0, 2, 10] == 0.0
    assert fixed.inverse_variance[0, 2, 10] == 0.0
    assert fixed.inverse_variance[1, 12, 3] == 0.0
    assert fixed.inverse_variance[2, 5, 11] == 0.0
    assert fixed.profile_constants.flags.writeable is False


def test_fourier_named_parameter_reordering_preserves_flux_and_jacobian():
    pytest.importorskip("jax")
    _, _, _, _, values, fixed = _fourier_fixture()
    permutation = np.array([7, 3, 5, 0, 2, 6, 1, 4])
    permuted_names = tuple(FOURIER_PARAMETER_NAMES[index]
                           for index in permutation)
    permuted_values = values[permutation]

    canonical_flux = fourier_flux(values, FOURIER_PARAMETER_NAMES, fixed)
    canonical_jacobian = fourier_flux_jacobian(
        values, FOURIER_PARAMETER_NAMES, fixed
    )
    permuted_flux = fourier_flux(permuted_values, permuted_names, fixed)
    permuted_jacobian = fourier_flux_jacobian(
        permuted_values, permuted_names, fixed
    )
    np.testing.assert_array_equal(permuted_flux, canonical_flux)
    np.testing.assert_array_equal(
        permuted_jacobian, canonical_jacobian[:, permutation]
    )


def test_fourier_batched_jacobian_matches_unbatched_map():
    pytest.importorskip("jax")
    _, _, _, _, values, fixed = _fourier_fixture()

    flux, jacobian = batched_flux_jacobian(
        values, FOURIER_PARAMETER_NAMES, fixed,
        profile="fourier", wavelength_batch=2,
    )

    np.testing.assert_allclose(
        flux, fourier_flux(values, FOURIER_PARAMETER_NAMES, fixed),
        rtol=1e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        jacobian,
        fourier_flux_jacobian(values, FOURIER_PARAMETER_NAMES, fixed),
        rtol=1e-13, atol=1e-13,
    )


def test_fourier_jax_jacobian_matches_centered_production_oracle():
    pytest.importorskip("jax")
    model, images, variance, wavelengths, values, fixed = _fourier_fixture()
    numerical = np.empty((len(wavelengths), len(values)))
    for index, value in enumerate(values):
        step = 2e-5 * max(abs(value), 1.0)
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        numerical[:, index] = (
            _numpy_flux(model, images, variance, wavelengths, plus,
                        FOURIER_PARAMETER_NAMES)
            - _numpy_flux(model, images, variance, wavelengths, minus,
                          FOURIER_PARAMETER_NAMES)
        ) / (2.0 * step)

    automatic = fourier_flux_jacobian(
        values, FOURIER_PARAMETER_NAMES, fixed
    )
    relative_error = (
        np.linalg.norm(automatic - numerical) / np.linalg.norm(numerical)
    )
    assert relative_error <= 1e-6
    np.testing.assert_allclose(automatic, numerical, rtol=2e-6, atol=2e-8)


def test_fourier_profile_constants_are_exported_in_exact_named_order():
    _, _, _, _, _, fixed = _fourier_fixture()
    expected = np.array([
        0.2376935 * np.sqrt(2),
        0.154173 * np.sqrt(2),
        0.0,
        1.07,
        0.209581,
        1.52,
    ])
    np.testing.assert_array_equal(fixed.profile_constants, expected)


def test_fourier_rejects_missing_parameters_and_profile_constants():
    _, images, variance, wavelengths, values, fixed = _fourier_fixture()
    with pytest.raises(SceneModelException, match="parameter names differ"):
        fourier_flux(values[:-1], FOURIER_PARAMETER_NAMES[:-1], fixed)
    with pytest.raises(SceneModelException, match="profile is missing"):
        build_fourier_fixed_arrays(
            images, variance, wavelengths, np.ones(len(wavelengths)),
            {
                "subsampling": fixed.subsampling,
                "border": fixed.border,
                "pad_grid_kx": fixed.pad_grid_kx,
                "pad_grid_ky": fixed.pad_grid_ky,
                "pad_ifft_shift": fixed.pad_ifft_shift,
            },
            fixed.background_bases,
            {name: 1.0 for name in FOURIER_PROFILE_NAMES[:-1]},
        )
