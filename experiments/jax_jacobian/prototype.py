"""Pure-array prototype of the native variable-projection extraction seam.

The profiles are deliberately small representatives of the two production
paths: a real-space Gaussian/Moffat mixture (classic) and an FFT-generated
Gaussian/exponential-power mixture (Fourier).  The important production
operation is preserved exactly: every wavelength constructs a source column
and analytically re-solves all source/background linear coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised on installations sans JAX
    jax = None
    jnp = None


@dataclass(frozen=True)
class Fixture:
    data: np.ndarray
    inverse_variance: np.ndarray
    wavelength: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    background: np.ndarray
    parameters: np.ndarray
    profile: str


def require_jax():
    if jax is None:
        raise RuntimeError("The JAX experiment requires jax and jaxlib")


def _classic_source(parameters, wavelength, grid_x, grid_y):
    center_x, center_y, log_width, chromatic_index, ellipticity = parameters
    ratio = wavelength[:, None, None] / 5000.0
    width = jnp.exp(log_width) * ratio ** chromatic_index
    dx = grid_x[None, :, :] - center_x
    dy = grid_y[None, :, :] - center_y
    axis_x = width * jnp.exp(ellipticity)
    axis_y = width * jnp.exp(-ellipticity)
    radius2 = (dx / axis_x) ** 2 + (dy / axis_y) ** 2
    gaussian = jnp.exp(-0.5 * radius2)
    moffat = (1.0 + radius2 / 3.2) ** -2.4
    profile = 0.72 * gaussian + 0.28 * moffat
    return profile / jnp.sum(profile, axis=(1, 2), keepdims=True)


def _fourier_source(parameters, wavelength, grid_x, grid_y):
    center_x, center_y, log_width, chromatic_index, wing_fraction = parameters
    nx, ny = grid_x.shape
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny)
    kx, ky = jnp.meshgrid(kx, ky, indexing="ij")
    ratio = wavelength[:, None, None] / 5000.0
    width = jnp.exp(log_width) * ratio ** chromatic_index
    k2 = kx[None, :, :] ** 2 + ky[None, :, :] ** 2
    core = jnp.exp(-0.5 * width ** 2 * k2)
    wings = jnp.exp(-(1.8 * width * jnp.sqrt(k2 + 1e-30)) ** 1.52)
    shift = jnp.exp(-1j * (center_x * kx + center_y * ky))[None, :, :]
    transform = ((1.0 - wing_fraction) * core + wing_fraction * wings) * shift
    profile = jnp.real(jnp.fft.ifft2(transform, axes=(-2, -1)))
    profile = jnp.fft.fftshift(profile, axes=(-2, -1))
    # FFT ringing in the representative wing model can be weakly negative.
    # A smooth positive floor keeps the weighted solve well conditioned.
    profile = jax.nn.softplus(40.0 * profile) / 40.0
    return profile / jnp.sum(profile, axis=(1, 2), keepdims=True)


def source_profiles(parameters, wavelength, grid_x, grid_y, profile):
    if profile == "classic":
        return _classic_source(parameters, wavelength, grid_x, grid_y)
    if profile == "fourier":
        return _fourier_source(parameters, wavelength, grid_x, grid_y)
    raise ValueError("Unknown profile %r" % profile)


def variable_projection_flux(parameters, data, inverse_variance, wavelength,
                             grid_x, grid_y, background, profile):
    """Return native amplitudes after a weighted solve at every wavelength."""
    source = source_profiles(parameters, wavelength, grid_x, grid_y, profile)
    source = source.reshape((source.shape[0], -1))
    repeated_background = jnp.broadcast_to(
        background[None, :, :],
        (source.shape[0], background.shape[0], background.shape[1]),
    )
    design = jnp.concatenate((source[:, :, None], repeated_background), axis=2)
    normal = jnp.einsum("wpi,wp,wpj->wij", design, inverse_variance, design)
    rhs = jnp.einsum("wpi,wp,wp->wi", design, inverse_variance, data)
    coefficients = jnp.linalg.solve(normal, rhs[..., None]).squeeze(-1)
    return coefficients[:, 0]


@partial(jax.jit, static_argnames=("profile",)) if jax else (lambda f: f)
def jitted_flux(parameters, data, inverse_variance, wavelength, grid_x, grid_y,
                background, profile):
    return variable_projection_flux(
        parameters, data, inverse_variance, wavelength, grid_x, grid_y,
        background, profile,
    )


if jax:
    jitted_jacobian = jax.jit(
        # There are only 5--9 global inputs but hundreds of wavelength
        # outputs, so forward mode avoids one reverse sweep per output.
        jax.jacfwd(variable_projection_flux, argnums=0),
        static_argnames=("profile",),
    )
else:  # pragma: no cover
    def jitted_jacobian(*args, **kwargs):
        require_jax()


def make_fixture(profile="classic", nwave=96, grid_size=15, seed=20260712):
    """Create a deterministic SNIFS-sized-slice representative fixture."""
    if profile not in ("classic", "fourier"):
        raise ValueError("Unknown profile %r" % profile)
    rng = np.random.default_rng(seed + (profile == "fourier"))
    axis = np.arange(grid_size, dtype=float) - (grid_size - 1.0) / 2.0
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    wavelength = np.linspace(3300.0, 5200.0, nwave)
    background = np.column_stack((
        np.ones(grid_size * grid_size),
        grid_x.ravel() / grid_size,
        grid_y.ravel() / grid_size,
    ))
    if profile == "classic":
        parameters = np.array([0.17, -0.23, np.log(1.35), -0.20, 0.06])
    else:
        parameters = np.array([0.17, -0.23, np.log(1.45), -0.20, 0.23])

    require_jax()
    source = np.asarray(source_profiles(
        jnp.asarray(parameters), jnp.asarray(wavelength), jnp.asarray(grid_x),
        jnp.asarray(grid_y), profile,
    )).reshape(nwave, -1)
    amplitudes = 800.0 + 120.0 * np.sin(np.linspace(0, 2 * np.pi, nwave))
    background_coefficients = np.column_stack((
        np.full(nwave, 12.0), np.full(nwave, 1.2), np.full(nwave, -0.8)
    ))
    noiseless = amplitudes[:, None] * source + background_coefficients @ background.T
    variance = 9.0 + 0.015 * np.maximum(noiseless, 0.0)
    data = noiseless + rng.normal(scale=np.sqrt(variance))
    return Fixture(
        data=data, inverse_variance=1.0 / variance, wavelength=wavelength,
        grid_x=grid_x, grid_y=grid_y, background=background,
        parameters=parameters, profile=profile,
    )


def fixture_arguments(fixture):
    require_jax()
    return (
        jnp.asarray(fixture.parameters),
        jnp.asarray(fixture.data),
        jnp.asarray(fixture.inverse_variance),
        jnp.asarray(fixture.wavelength),
        jnp.asarray(fixture.grid_x),
        jnp.asarray(fixture.grid_y),
        jnp.asarray(fixture.background),
    )


def centered_difference_jacobian(fixture, relative_step=2e-5):
    """Independent finite-difference oracle using the same pure-array flux."""
    args = list(fixture_arguments(fixture))
    parameters = fixture.parameters
    result = np.empty((fixture.data.shape[0], len(parameters)))
    for index, value in enumerate(parameters):
        step = relative_step * max(1.0, abs(value))
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        args[0] = jnp.asarray(plus)
        high = np.asarray(jitted_flux(*args, profile=fixture.profile))
        args[0] = jnp.asarray(minus)
        low = np.asarray(jitted_flux(*args, profile=fixture.profile))
        result[:, index] = (high - low) / (2.0 * step)
    return result
