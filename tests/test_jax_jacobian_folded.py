"""Folded, one-direction-at-a-time JAX flux Jacobian vs the unfolded oracle.

The unfolded ``jacfwd`` path (``classic_flux_jacobian`` /
``fourier_flux_jacobian``) is the validated oracle.  The folded path
(``flux_and_jacobian_folded``; Fourier-space subpixel fold, forward-mode JVP
one parameter direction at a time) must evaluate the same map to roundoff,
on the grid configurations of the October fold test plus subsampling 3 at
the production border 15.
"""

from __future__ import annotations

import numpy as np
import pytest

from scene_model.covariance import check_baseline_parity
from scene_model.jax_scene import (
    CLASSIC_PARAMETER_NAMES,
    FOURIER_PARAMETER_NAMES,
    FOURIER_PROFILE_NAMES,
    batched_flux_jacobian,
    build_classic_fixed_arrays,
    build_fourier_fixed_arrays,
    build_polynomial_background_bases,
    classic_flux,
    classic_flux_jacobian,
    flux_and_jacobian_folded,
    fourier_flux,
    fourier_flux_jacobian,
)
from scene_model.snifs import SnifsClassicSceneModel, SnifsFourierSceneModel
from scene_model.utils import SceneModelException

pytest.importorskip("jax")

TOLERANCE = 1e-10
NWAVE = 5
PARTIAL_BATCH = 2          # 5 wavelengths -> batches of 2, 2 and 1

VALUES = {
    "fourier": np.array([0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09]),
    "classic": np.array([0.17, -0.23, 1.08, 0.31, -0.045, -0.19, 2.35, 1.14,
                         0.08]),
}
NAMES = {"fourier": FOURIER_PARAMETER_NAMES, "classic": CLASSIC_PARAMETER_NAMES}
UNFOLDED = {
    "fourier": (fourier_flux, fourier_flux_jacobian),
    "classic": (classic_flux, classic_flux_jacobian),
}
# The October fold test's three configurations, plus subsampling 3 at the
# production border 15.  (16, 15) is the production grid.
GRIDS = [(3, 5), (4, 0), (16, 15), (3, 15)]


class _FixedAdr:
    def get_scale(self, wavelength):
        return (np.asarray(wavelength) - 5000.0) / 1800.0


def _fixture(psf, subsampling, border, background_degree=1):
    adr = _FixedAdr()
    spaxel_size = 0.43
    common = dict(wavelength_dependence=True,
                  background_degree=background_degree, adr_model=adr,
                  spaxel_size=spaxel_size, grid_size=(15, 15),
                  subsampling=subsampling, border=border)
    if psf == "fourier":
        model = SnifsFourierSceneModel(use_empirical_parameters=True,
                                       **common)
    else:
        model = SnifsClassicSceneModel(exposure_time=20.0, **common)
    values = VALUES[psf]
    parameters = dict(zip(NAMES[psf], values))
    wavelengths = np.linspace(3500.0, 9000.0, NWAVE)
    amplitudes = np.linspace(4.0, 7.0, NWAVE)
    images = np.asarray([
        model.evaluate(wavelength=wavelength, amplitude=amplitude,
                       background=0.3, background_1_0=0.1,
                       background_0_1=-0.05, **parameters)
        for wavelength, amplitude in zip(wavelengths, amplitudes)
    ])
    rng = np.random.default_rng(7)
    variance = 0.35 + rng.uniform(0.0, 0.3, images.shape)
    images = images + rng.standard_normal(images.shape) * np.sqrt(variance)
    images[0, 2, 10] = np.nan          # masked spaxels
    variance[3, 12, 3] = 0.0
    bases = build_polynomial_background_bases(
        model.grid_info["grid_x"], model.grid_info["grid_y"],
        background_degree)
    adr_scale = adr.get_scale(wavelengths) / spaxel_size
    if psf == "fourier":
        constants = {name: model.parameters[name]
                     for name in FOURIER_PROFILE_NAMES}
        fixed = build_fourier_fixed_arrays(images, variance, wavelengths,
                                           adr_scale, model.grid_info,
                                           bases, constants)
    else:
        fixed = build_classic_fixed_arrays(images, variance, wavelengths,
                                           adr_scale, model.grid_info,
                                           bases, 20.0)
    return values, fixed


def _flux_relative(actual, expected):
    return np.max(np.abs(actual - expected) / np.abs(expected))


def _frobenius_relative(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("psf", ["fourier", "classic"])
@pytest.mark.parametrize("subsampling,border", GRIDS)
def test_folded_flux_and_jacobian_match_unfolded_oracle(psf, subsampling,
                                                        border):
    values, fixed = _fixture(psf, subsampling, border)
    names = NAMES[psf]
    flux_function, jacobian_function = UNFOLDED[psf]

    oracle_flux = flux_function(values, names, fixed)
    oracle_jacobian = jacobian_function(values, names, fixed)
    flux, jacobian = flux_and_jacobian_folded(values, names, fixed,
                                              profile=psf)
    assert flux.shape == (NWAVE,)
    assert jacobian.shape == (NWAVE, len(names))
    assert _flux_relative(flux, oracle_flux) <= TOLERANCE
    assert _frobenius_relative(jacobian, oracle_jacobian) <= TOLERANCE

    # Production batching with a partial last batch (2, 2, 1).
    batched_unfolded = batched_flux_jacobian(
        values, names, fixed, profile=psf, wavelength_batch=PARTIAL_BATCH)
    batched_folded = batched_flux_jacobian(
        values, names, fixed, profile=psf, wavelength_batch=PARTIAL_BATCH,
        folded=True)
    assert _flux_relative(batched_folded[0], batched_unfolded[0]) <= TOLERANCE
    assert _frobenius_relative(batched_folded[1], batched_unfolded[1]) \
        <= TOLERANCE
    assert _flux_relative(batched_folded[0], flux) <= TOLERANCE
    assert _frobenius_relative(batched_folded[1], jacobian) <= TOLERANCE


@pytest.mark.parametrize("psf", ["fourier", "classic"])
def test_folded_jacobian_respects_caller_parameter_order(psf):
    values, fixed = _fixture(psf, 3, 5)
    names = NAMES[psf]
    permutation = np.roll(np.arange(len(names)), 3)[::-1]
    permuted_names = tuple(names[index] for index in permutation)
    canonical = flux_and_jacobian_folded(values, names, fixed, profile=psf)
    permuted = flux_and_jacobian_folded(values[permutation], permuted_names,
                                        fixed, profile=psf)
    np.testing.assert_array_equal(permuted[0], canonical[0])
    np.testing.assert_array_equal(permuted[1], canonical[1][:, permutation])


def test_folded_rejects_unknown_profile():
    values, fixed = _fixture("fourier", 3, 5)
    with pytest.raises(SceneModelException, match="Unknown JAX scene profile"):
        flux_and_jacobian_folded(values, FOURIER_PARAMETER_NAMES, fixed,
                                 profile="moffat")
    with pytest.raises(SceneModelException, match="Unknown JAX scene profile"):
        batched_flux_jacobian(values, FOURIER_PARAMETER_NAMES, fixed,
                              profile="moffat", folded=True)


def test_baseline_parity_gate_tolerates_roundoff_and_rejects_real_errors():
    accepted = np.array([100.0, 250.0, 40.0])
    variance = np.array([1.0, 4.0, 0.25])        # S/N 100-125

    roundoff = accepted * (1.0 + 1e-11)
    diagnostics = check_baseline_parity(roundoff, accepted, variance)
    assert diagnostics["max_relative"] == pytest.approx(1e-11, rel=1e-3)
    assert diagnostics["max_sigma_scaled"] == pytest.approx(
        np.max(1e-11 * accepted / np.sqrt(variance)), rel=1e-3)
    assert diagnostics["rtol"] == 1e-9
    assert diagnostics["sigma_fraction"] == 1e-6

    with pytest.raises(SceneModelException,
                       match="baseline failed accepted NumPy parity"):
        check_baseline_parity(accepted * (1.0 + 1e-6), accepted, variance)

    # The looser allowance wins: at S/N 0.1 a 1e-6 relative difference is
    # 1e-7 sigma, inside the 1e-6 sigma allowance.
    noisy_variance = (10.0 * accepted) ** 2
    check_baseline_parity(accepted * (1.0 + 1e-6), accepted, noisy_variance)

    with pytest.raises(SceneModelException, match="flux axes differ"):
        check_baseline_parity(accepted[:2], accepted, variance)


@pytest.mark.parametrize("psf", ["classic", "fourier"])
@pytest.mark.parametrize("wavelength_batch", [None, 128])
def test_jax_fit_backend_selects_folded_jacobian(psf, wavelength_batch):
    from test_jax_production_integration import _production_fitter

    results = {}
    for fit_backend in ("numpy", "jax"):
        fitter = _production_fitter(psf)
        fitter.fit_backend = fit_backend
        fitter.extract(method="psf", covariance=True, jacobian_backend="jax",
                       jax_wavelength_batch=wavelength_batch)
        results[fit_backend] = fitter

    for fit_backend, model in (("numpy", "unfolded"), ("jax", "folded")):
        diagnostics = results[fit_backend].covariance_diagnostics
        assert diagnostics["jacobian_model"] == model
        assert dict(diagnostics["derivatives"].provenance)[
            "jacobian_model"] == model
        parity = diagnostics["baseline_parity"]
        assert parity["max_relative"] <= parity["rtol"]
    assert _frobenius_relative(
        results["jax"].covariance_diagnostics["jacobian"],
        results["numpy"].covariance_diagnostics["jacobian"],
    ) <= TOLERANCE
    np.testing.assert_array_equal(results["jax"].point_source_spectrum.data,
                                  results["numpy"].point_source_spectrum.data)


def test_finite_difference_path_records_no_jax_model():
    from test_jax_production_integration import _production_fitter

    fitter = _production_fitter("fourier")
    fitter.fit_backend = "jax"
    fitter.extract(method="psf", covariance=True,
                   jacobian_backend="finite-difference")
    diagnostics = fitter.covariance_diagnostics
    assert diagnostics["jacobian_model"] is None
    assert diagnostics["baseline_parity"] is None
