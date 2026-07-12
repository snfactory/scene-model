"""Pure-array JAX evaluators for fixed SNIFS scene extraction.

NumPy remains authoritative for fitting and point estimates.  This module
reproduces the accepted, fixed classic scene extraction so JAX can provide its
global-parameter Jacobian without tracing the mutable model element graph.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache

import numpy as np

from .utils import SceneModelException


CLASSIC_PARAMETER_NAMES = (
    "ref_center_x",
    "ref_center_y",
    "adr_delta",
    "adr_theta",
    "A0",
    "A1",
    "A2",
    "ell",
    "xy",
)

FOURIER_PARAMETER_NAMES = (
    "ref_center_x",
    "ref_center_y",
    "adr_delta",
    "adr_theta",
    "seeing_ref_power",
    "seeing_ref_width",
    "ellipticity_x",
    "ellipticity_y",
)

FOURIER_PROFILE_NAMES = (
    "inst_core_sigma_x",
    "inst_core_sigma_y",
    "inst_core_rho",
    "inst_wings_power",
    "inst_wings_width",
    "seeing_power",
)

REFERENCE_WAVELENGTH = 5000.0
NATIVE_SPAXELS = 225
PRODUCTION_WAVELENGTH_BATCH = 128


def _readonly_array(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ClassicFixedArrays:
    """Immutable arrays and constants for one fixed classic extraction."""

    data: np.ndarray
    inverse_variance: np.ndarray
    wavelengths: np.ndarray
    adr_scale: np.ndarray
    pad_grid_x: np.ndarray
    pad_grid_y: np.ndarray
    pad_grid_kx: np.ndarray
    pad_grid_ky: np.ndarray
    pad_fft_shift: np.ndarray
    pad_ifft_shift: np.ndarray
    background_bases: np.ndarray
    profile_constants: np.ndarray
    subsampling: int
    border: int

    def __post_init__(self):
        real_fields = (
            "data", "inverse_variance", "wavelengths", "adr_scale",
            "pad_grid_x", "pad_grid_y", "pad_grid_kx", "pad_grid_ky",
            "background_bases", "profile_constants",
        )
        complex_fields = ("pad_fft_shift", "pad_ifft_shift")
        for name in real_fields:
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        for name in complex_fields:
            object.__setattr__(
                self, name, _readonly_array(getattr(self, name), np.complex128)
            )
        object.__setattr__(self, "subsampling", int(self.subsampling))
        object.__setattr__(self, "border", int(self.border))

        if self.data.ndim != 3 or self.data.shape != self.inverse_variance.shape:
            raise SceneModelException("Classic fixed data axes are inconsistent")
        nwave, native_x, native_y = self.data.shape
        if native_x * native_y != NATIVE_SPAXELS:
            raise SceneModelException(
                "Classic extraction requires the fixed 225-spaxel axis"
            )
        if self.wavelengths.shape != (nwave,) or self.adr_scale.shape != (nwave,):
            raise SceneModelException("Classic wavelength axes are inconsistent")
        if self.background_bases.ndim != 3 or \
                self.background_bases.shape[1:] != (native_x, native_y):
            raise SceneModelException("Classic background axes are inconsistent")
        padded_shape = self.pad_grid_x.shape
        padded_names = (
            "pad_grid_y", "pad_grid_kx", "pad_grid_ky", "pad_fft_shift",
            "pad_ifft_shift",
        )
        if len(padded_shape) != 2 or any(
                getattr(self, name).shape != padded_shape
                for name in padded_names):
            raise SceneModelException("Classic padded grid axes are inconsistent")
        if self.subsampling < 1 or self.border < 0:
            raise SceneModelException("Invalid classic grid configuration")
        expected_padded = (
            (native_x + 2 * self.border) * self.subsampling,
            (native_y + 2 * self.border) * self.subsampling,
        )
        if padded_shape != expected_padded:
            raise SceneModelException(
                "Classic padded grid does not match subsampling and border"
            )
        if self.profile_constants.shape != (6,):
            raise SceneModelException("Classic profile constants are invalid")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
                raise SceneModelException(
                    "Classic fixed field %s is non-finite" % field.name
                )
        if np.any(self.inverse_variance < 0):
            raise SceneModelException("Classic inverse variance is negative")


@dataclass(frozen=True)
class FourierFixedArrays:
    """Immutable arrays and fixed empirical terms for Fourier extraction."""

    data: np.ndarray
    inverse_variance: np.ndarray
    wavelengths: np.ndarray
    adr_scale: np.ndarray
    pad_grid_kx: np.ndarray
    pad_grid_ky: np.ndarray
    pad_ifft_shift: np.ndarray
    background_bases: np.ndarray
    profile_constants: np.ndarray
    subsampling: int
    border: int

    def __post_init__(self):
        real_fields = (
            "data", "inverse_variance", "wavelengths", "adr_scale",
            "pad_grid_kx", "pad_grid_ky", "background_bases",
            "profile_constants",
        )
        for name in real_fields:
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))
        object.__setattr__(
            self, "pad_ifft_shift",
            _readonly_array(self.pad_ifft_shift, np.complex128),
        )
        object.__setattr__(self, "subsampling", int(self.subsampling))
        object.__setattr__(self, "border", int(self.border))

        if self.data.ndim != 3 or self.data.shape != self.inverse_variance.shape:
            raise SceneModelException("Fourier fixed data axes are inconsistent")
        nwave, native_x, native_y = self.data.shape
        if native_x * native_y != NATIVE_SPAXELS:
            raise SceneModelException(
                "Fourier extraction requires the fixed 225-spaxel axis"
            )
        if self.wavelengths.shape != (nwave,) or self.adr_scale.shape != (nwave,):
            raise SceneModelException("Fourier wavelength axes are inconsistent")
        if self.background_bases.ndim != 3 or \
                self.background_bases.shape[1:] != (native_x, native_y):
            raise SceneModelException("Fourier background axes are inconsistent")
        padded_shape = self.pad_grid_kx.shape
        if len(padded_shape) != 2 or self.pad_grid_ky.shape != padded_shape or \
                self.pad_ifft_shift.shape != padded_shape:
            raise SceneModelException("Fourier padded grid axes are inconsistent")
        if self.subsampling < 1 or self.border < 0:
            raise SceneModelException("Invalid Fourier grid configuration")
        expected_padded = (
            (native_x + 2 * self.border) * self.subsampling,
            (native_y + 2 * self.border) * self.subsampling,
        )
        if padded_shape != expected_padded:
            raise SceneModelException(
                "Fourier padded grid does not match subsampling and border"
            )
        if self.profile_constants.shape != (len(FOURIER_PROFILE_NAMES),):
            raise SceneModelException("Fourier profile constants are invalid")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
                raise SceneModelException(
                    "Fourier fixed field %s is non-finite" % field.name
                )
        if np.any(self.inverse_variance < 0):
            raise SceneModelException("Fourier inverse variance is negative")


def build_polynomial_background_bases(grid_x, grid_y, degree,
                                      normalization_scale=10.0):
    """Build production-ordered, unit-coefficient background components."""
    grid_x = np.asarray(grid_x, dtype=np.float64)
    grid_y = np.asarray(grid_y, dtype=np.float64)
    degree = int(degree)
    if grid_x.shape != grid_y.shape or grid_x.ndim != 2:
        raise SceneModelException("Background grids are inconsistent")
    if degree < -1:
        raise SceneModelException("Invalid background degree")
    if not np.isfinite(normalization_scale) or normalization_scale <= 0:
        raise SceneModelException("Invalid background normalization")
    if degree == -1:
        return _readonly_array(np.empty((0,) + grid_x.shape))

    bases = [np.ones(grid_x.shape, dtype=np.float64)]
    norm_x = grid_x / normalization_scale
    norm_y = grid_y / normalization_scale
    for x_degree in range(degree + 1):
        for y_degree in range(degree + 1 - x_degree):
            if x_degree == 0 and y_degree == 0:
                continue
            bases.append(norm_x**x_degree * norm_y**y_degree)
    return _readonly_array(np.asarray(bases))


def _classic_profile_constants(exposure_time):
    if not np.isfinite(exposure_time):
        raise SceneModelException("Classic exposure time is non-finite")
    if exposure_time < 12:
        values = (1.395, 0.415, 0.56, 0.2, 0.6, 0.16)
    else:
        values = (1.685, 0.345, 0.545, 0.215, 1.04, 0.0)
    return np.asarray(values, dtype=np.float64)


def build_classic_fixed_arrays(data, variance, wavelengths, adr_scale,
                               grid_info, background_bases, exposure_time):
    """Export and sanitize the fixed inputs used by classic extraction."""
    data = np.asarray(data, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if data.shape != variance.shape or data.ndim != 3:
        raise SceneModelException("Classic data and variance axes differ")
    accepted = np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    fixed_data = np.where(accepted, data, 0.0)
    inverse_variance = np.zeros(variance.shape, dtype=np.float64)
    np.divide(1.0, variance, out=inverse_variance, where=accepted)

    required_grid_fields = (
        "subsampling", "border", "pad_grid_x", "pad_grid_y",
        "pad_grid_kx", "pad_grid_ky", "pad_fft_shift", "pad_ifft_shift",
    )
    missing = [name for name in required_grid_fields if name not in grid_info]
    if missing:
        raise SceneModelException(
            "Classic grid is missing %s" % ", ".join(missing)
        )
    return ClassicFixedArrays(
        data=fixed_data,
        inverse_variance=inverse_variance,
        wavelengths=wavelengths,
        adr_scale=adr_scale,
        pad_grid_x=grid_info["pad_grid_x"],
        pad_grid_y=grid_info["pad_grid_y"],
        pad_grid_kx=grid_info["pad_grid_kx"],
        pad_grid_ky=grid_info["pad_grid_ky"],
        pad_fft_shift=grid_info["pad_fft_shift"],
        pad_ifft_shift=grid_info["pad_ifft_shift"],
        background_bases=background_bases,
        profile_constants=_classic_profile_constants(exposure_time),
        subsampling=grid_info["subsampling"],
        border=grid_info["border"],
    )


def build_fourier_fixed_arrays(data, variance, wavelengths, adr_scale,
                               grid_info, background_bases,
                               profile_constants):
    """Export fixed Fourier inputs, including accepted empirical constants."""
    data = np.asarray(data, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if data.shape != variance.shape or data.ndim != 3:
        raise SceneModelException("Fourier data and variance axes differ")
    accepted = np.isfinite(data) & np.isfinite(variance) & (variance > 0)
    fixed_data = np.where(accepted, data, 0.0)
    inverse_variance = np.zeros(variance.shape, dtype=np.float64)
    np.divide(1.0, variance, out=inverse_variance, where=accepted)

    required_grid_fields = (
        "subsampling", "border", "pad_grid_kx", "pad_grid_ky",
        "pad_ifft_shift",
    )
    missing = [name for name in required_grid_fields if name not in grid_info]
    if missing:
        raise SceneModelException(
            "Fourier grid is missing %s" % ", ".join(missing)
        )
    if hasattr(profile_constants, "keys"):
        missing_constants = [
            name for name in FOURIER_PROFILE_NAMES
            if name not in profile_constants
        ]
        if missing_constants:
            raise SceneModelException(
                "Fourier profile is missing %s" %
                ", ".join(missing_constants)
            )
        profile_constants = [
            profile_constants[name] for name in FOURIER_PROFILE_NAMES
        ]
    return FourierFixedArrays(
        data=fixed_data,
        inverse_variance=inverse_variance,
        wavelengths=wavelengths,
        adr_scale=adr_scale,
        pad_grid_kx=grid_info["pad_grid_kx"],
        pad_grid_ky=grid_info["pad_grid_ky"],
        pad_ifft_shift=grid_info["pad_ifft_shift"],
        background_bases=background_bases,
        profile_constants=profile_constants,
        subsampling=grid_info["subsampling"],
        border=grid_info["border"],
    )


def _ensure_jax():
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except ImportError as error:
        raise SceneModelException(
            "Classic JAX extraction requires the JAX covariance dependency"
        ) from error
    return jax, jnp


def _classic_flux_impl(canonical_parameters, data, inverse_variance,
                       wavelengths, adr_scale, pad_grid_x, pad_grid_y,
                       pad_grid_kx, pad_grid_ky, pad_fft_shift,
                       pad_ifft_shift, background_bases, profile_constants,
                       *, subsampling, border):
    """Evaluate the exact fixed classic variable-projection map in JAX."""
    _, jnp = _ensure_jax()
    (ref_center_x, ref_center_y, adr_delta, adr_theta,
     A0, A1, A2, ell, xy) = canonical_parameters
    beta0, beta1, sigma0, sigma1, eta0, eta1 = profile_constants

    wavelength_ratio = wavelengths / REFERENCE_WAVELENGTH
    alpha = A2 * wavelength_ratio ** (A0 * (wavelength_ratio - 1.0) + A1)
    beta = beta0 + beta1 * alpha
    sigma = sigma0 + sigma1 * alpha
    eta = eta0 + eta1 * alpha

    max_xy = 0.99999 * jnp.sqrt(ell)
    bounded_xy = jnp.clip(xy, -max_xy, max_xy)
    radius_squared = (
        pad_grid_x**2 + ell * pad_grid_y**2
        + 2.0 * bounded_xy * pad_grid_x * pad_grid_y
    )
    radius_squared = radius_squared[jnp.newaxis, :, :]
    alpha_grid = alpha[:, jnp.newaxis, jnp.newaxis]
    beta_grid = beta[:, jnp.newaxis, jnp.newaxis]
    sigma_grid = sigma[:, jnp.newaxis, jnp.newaxis]
    eta_grid = eta[:, jnp.newaxis, jnp.newaxis]
    profile = (
        (1.0 + radius_squared / alpha_grid**2) ** (-beta_grid)
        + eta_grid * jnp.exp(-0.5 * radius_squared / sigma_grid**2)
    )
    normalization = (
        jnp.pi / jnp.sqrt(ell - bounded_xy**2)
        * (2.0 * eta * sigma**2 + alpha**2 / (beta - 1.0))
    )
    profile = profile / subsampling**2
    profile = profile / normalization[:, jnp.newaxis, jnp.newaxis]

    profile_fourier = (
        jnp.fft.fft2(profile, axes=(-2, -1))
        * pad_fft_shift[jnp.newaxis, :, :]
    )
    point_source = jnp.exp(-1j * (
        ref_center_x * pad_grid_kx + ref_center_y * pad_grid_ky
    ))
    shift_x = adr_delta * jnp.sin(adr_theta) * adr_scale
    shift_y = -adr_delta * jnp.cos(adr_theta) * adr_scale
    adr = jnp.exp(-1j * (
        shift_x[:, jnp.newaxis, jnp.newaxis] * pad_grid_kx
        + shift_y[:, jnp.newaxis, jnp.newaxis] * pad_grid_ky
    ))
    source_fourier = profile_fourier * point_source * adr
    source_subpixels = jnp.real(jnp.fft.ifft2(
        source_fourier * pad_ifft_shift[jnp.newaxis, :, :], axes=(-2, -1)
    ))

    nwave = data.shape[0]
    native_x = data.shape[1]
    native_y = data.shape[2]
    expanded_x = native_x + 2 * border
    expanded_y = native_y + 2 * border
    source_pixels = source_subpixels.reshape(
        nwave, expanded_x, subsampling, expanded_y, subsampling
    ).sum(axis=(2, 4))
    if border:
        source_pixels = source_pixels[:, border:-border, border:-border]

    source_basis = source_pixels.reshape(nwave, -1)
    fixed_background = jnp.broadcast_to(
        background_bases[jnp.newaxis, :, :, :],
        (nwave,) + background_bases.shape,
    ).reshape(nwave, background_bases.shape[0], native_x * native_y)
    design = jnp.concatenate(
        (source_basis[:, jnp.newaxis, :], fixed_background), axis=1
    ).transpose(0, 2, 1)
    weight = inverse_variance.reshape(nwave, -1)
    fixed_data = data.reshape(nwave, -1)
    normal = jnp.einsum("npc,np,npd->ncd", design, weight, design)
    right_hand_side = jnp.einsum(
        "npc,np,np->nc", design, weight, fixed_data
    )
    coefficients = jnp.einsum(
        "ncd,nd->nc", jnp.linalg.inv(normal), right_hand_side
    )
    return coefficients[:, 0]


@lru_cache(maxsize=16)
def _compiled_classic_flux(subsampling, border):
    jax, _ = _ensure_jax()

    def fixed_grid_flux(parameters, *arrays):
        return _classic_flux_impl(
            parameters, *arrays, subsampling=subsampling, border=border
        )

    return jax.jit(fixed_grid_flux)


@lru_cache(maxsize=16)
def _compiled_classic_jacobian(subsampling, border):
    jax, _ = _ensure_jax()

    def fixed_grid_flux(parameters, *arrays):
        return _classic_flux_impl(
            parameters, *arrays, subsampling=subsampling, border=border
        )

    return jax.jit(jax.jacfwd(fixed_grid_flux, argnums=0))


def _canonical_parameters(parameter_values, parameter_names):
    names = tuple(parameter_names)
    values = np.asarray(parameter_values, dtype=np.float64)
    if values.shape != (len(names),) or len(set(names)) != len(names):
        raise SceneModelException("Classic parameter axes are inconsistent")
    if set(names) != set(CLASSIC_PARAMETER_NAMES):
        missing = sorted(set(CLASSIC_PARAMETER_NAMES) - set(names))
        extra = sorted(set(names) - set(CLASSIC_PARAMETER_NAMES))
        raise SceneModelException(
            "Classic parameter names differ (missing=%s, extra=%s)" %
            (missing, extra)
        )
    if not np.all(np.isfinite(values)):
        raise SceneModelException("Classic parameters are non-finite")
    indices = np.asarray([names.index(name) for name in CLASSIC_PARAMETER_NAMES])
    return values[indices], indices


def _fixed_arguments(fixed):
    if not isinstance(fixed, ClassicFixedArrays):
        raise SceneModelException("Expected immutable classic fixed arrays")
    return (
        fixed.data,
        fixed.inverse_variance,
        fixed.wavelengths,
        fixed.adr_scale,
        fixed.pad_grid_x,
        fixed.pad_grid_y,
        fixed.pad_grid_kx,
        fixed.pad_grid_ky,
        fixed.pad_fft_shift,
        fixed.pad_ifft_shift,
        fixed.background_bases,
        fixed.profile_constants,
    )


def classic_flux(parameter_values, parameter_names, fixed):
    """Evaluate fixed classic amplitudes in the caller's named coordinates."""
    canonical, _ = _canonical_parameters(parameter_values, parameter_names)
    result = _compiled_classic_flux(fixed.subsampling, fixed.border)(
        canonical, *_fixed_arguments(fixed)
    )
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(fixed.wavelengths),) or not np.all(np.isfinite(result)):
        raise SceneModelException("Classic JAX baseline flux is invalid")
    return result


def classic_flux_jacobian(parameter_values, parameter_names, fixed):
    """Return the classic fixed-map Jacobian in caller-declared name order."""
    canonical, canonical_indices = _canonical_parameters(
        parameter_values, parameter_names
    )
    canonical_jacobian = _compiled_classic_jacobian(
        fixed.subsampling, fixed.border
    )(canonical, *_fixed_arguments(fixed))
    canonical_jacobian = np.asarray(canonical_jacobian, dtype=np.float64)
    expected_shape = (len(fixed.wavelengths), len(CLASSIC_PARAMETER_NAMES))
    if canonical_jacobian.shape != expected_shape or \
            not np.all(np.isfinite(canonical_jacobian)):
        raise SceneModelException("Classic JAX flux Jacobian is invalid")
    jacobian = np.empty_like(canonical_jacobian)
    jacobian[:, canonical_indices] = canonical_jacobian
    return jacobian


def _fourier_flux_impl(canonical_parameters, data, inverse_variance,
                       wavelengths, adr_scale, pad_grid_kx, pad_grid_ky,
                       pad_ifft_shift, background_bases, profile_constants,
                       *, subsampling, border):
    """Evaluate the exact fixed Fourier variable-projection map in JAX."""
    _, jnp = _ensure_jax()
    (ref_center_x, ref_center_y, adr_delta, adr_theta,
     seeing_ref_power, seeing_ref_width,
     ellipticity_x, ellipticity_y) = canonical_parameters
    (inst_core_sigma_x, inst_core_sigma_y, inst_core_rho,
     inst_wings_power, inst_wings_width, seeing_power) = profile_constants

    point_source = jnp.exp(-1j * (
        ref_center_x * pad_grid_kx + ref_center_y * pad_grid_ky
    ))
    instrumental_core = jnp.exp(-0.5 * (
        pad_grid_kx**2 * inst_core_sigma_x**2
        + pad_grid_ky**2 * inst_core_sigma_y**2
        + 2.0 * pad_grid_kx * pad_grid_ky * inst_core_rho
        * inst_core_sigma_x * inst_core_sigma_y
    ))
    radial_frequency = jnp.sqrt(pad_grid_kx**2 + pad_grid_ky**2)
    instrumental_wings = jnp.exp(
        -inst_wings_width**inst_wings_power
        * radial_frequency**inst_wings_power
    )

    shift_x = adr_delta * jnp.sin(adr_theta) * adr_scale
    shift_y = -adr_delta * jnp.cos(adr_theta) * adr_scale
    adr = jnp.exp(-1j * (
        shift_x[:, jnp.newaxis, jnp.newaxis] * pad_grid_kx
        + shift_y[:, jnp.newaxis, jnp.newaxis] * pad_grid_ky
    ))
    seeing_width = seeing_ref_width * (
        wavelengths / REFERENCE_WAVELENGTH
    )**seeing_ref_power
    seeing = jnp.exp(
        -seeing_width[:, jnp.newaxis, jnp.newaxis]**seeing_power
        * radial_frequency[jnp.newaxis, :, :]**seeing_power
    )
    tracking = jnp.exp(-0.5 * (
        pad_grid_kx**2 * ellipticity_x**2
        + pad_grid_ky**2 * ellipticity_y**2
        + 2.0 * pad_grid_kx * pad_grid_ky * 0.99
        * ellipticity_x * ellipticity_y
    ))
    fourier_pixel = (
        jnp.sinc(pad_grid_kx / jnp.pi / 2.0 / subsampling)
        * jnp.sinc(pad_grid_ky / jnp.pi / 2.0 / subsampling)
    )
    source_fourier = (
        point_source[jnp.newaxis, :, :]
        * instrumental_core[jnp.newaxis, :, :]
        * instrumental_wings[jnp.newaxis, :, :]
        * adr
        * seeing
        * tracking[jnp.newaxis, :, :]
        * fourier_pixel[jnp.newaxis, :, :]
    )
    source_subpixels = jnp.real(jnp.fft.ifft2(
        source_fourier * pad_ifft_shift[jnp.newaxis, :, :], axes=(-2, -1)
    ))

    nwave = data.shape[0]
    native_x = data.shape[1]
    native_y = data.shape[2]
    expanded_x = native_x + 2 * border
    expanded_y = native_y + 2 * border
    source_pixels = source_subpixels.reshape(
        nwave, expanded_x, subsampling, expanded_y, subsampling
    ).sum(axis=(2, 4))
    if border:
        source_pixels = source_pixels[:, border:-border, border:-border]

    source_basis = source_pixels.reshape(nwave, -1)
    fixed_background = jnp.broadcast_to(
        background_bases[jnp.newaxis, :, :, :],
        (nwave,) + background_bases.shape,
    ).reshape(nwave, background_bases.shape[0], native_x * native_y)
    design = jnp.concatenate(
        (source_basis[:, jnp.newaxis, :], fixed_background), axis=1
    ).transpose(0, 2, 1)
    weight = inverse_variance.reshape(nwave, -1)
    fixed_data = data.reshape(nwave, -1)
    normal = jnp.einsum("npc,np,npd->ncd", design, weight, design)
    right_hand_side = jnp.einsum(
        "npc,np,np->nc", design, weight, fixed_data
    )
    coefficients = jnp.einsum(
        "ncd,nd->nc", jnp.linalg.inv(normal), right_hand_side
    )
    return coefficients[:, 0]


@lru_cache(maxsize=16)
def _compiled_fourier_flux(subsampling, border):
    jax, _ = _ensure_jax()

    def fixed_grid_flux(parameters, *arrays):
        return _fourier_flux_impl(
            parameters, *arrays, subsampling=subsampling, border=border
        )

    return jax.jit(fixed_grid_flux)


@lru_cache(maxsize=16)
def _compiled_fourier_jacobian(subsampling, border):
    jax, _ = _ensure_jax()

    def fixed_grid_flux(parameters, *arrays):
        return _fourier_flux_impl(
            parameters, *arrays, subsampling=subsampling, border=border
        )

    return jax.jit(jax.jacfwd(fixed_grid_flux, argnums=0))


def _canonical_fourier_parameters(parameter_values, parameter_names):
    names = tuple(parameter_names)
    values = np.asarray(parameter_values, dtype=np.float64)
    if values.shape != (len(names),) or len(set(names)) != len(names):
        raise SceneModelException("Fourier parameter axes are inconsistent")
    if set(names) != set(FOURIER_PARAMETER_NAMES):
        missing = sorted(set(FOURIER_PARAMETER_NAMES) - set(names))
        extra = sorted(set(names) - set(FOURIER_PARAMETER_NAMES))
        raise SceneModelException(
            "Fourier parameter names differ (missing=%s, extra=%s)" %
            (missing, extra)
        )
    if not np.all(np.isfinite(values)):
        raise SceneModelException("Fourier parameters are non-finite")
    indices = np.asarray([names.index(name) for name in FOURIER_PARAMETER_NAMES])
    return values[indices], indices


def _fourier_fixed_arguments(fixed):
    if not isinstance(fixed, FourierFixedArrays):
        raise SceneModelException("Expected immutable Fourier fixed arrays")
    return (
        fixed.data,
        fixed.inverse_variance,
        fixed.wavelengths,
        fixed.adr_scale,
        fixed.pad_grid_kx,
        fixed.pad_grid_ky,
        fixed.pad_ifft_shift,
        fixed.background_bases,
        fixed.profile_constants,
    )


def fourier_flux(parameter_values, parameter_names, fixed):
    """Evaluate fixed Fourier amplitudes in caller-declared coordinates."""
    canonical, _ = _canonical_fourier_parameters(
        parameter_values, parameter_names
    )
    result = _compiled_fourier_flux(fixed.subsampling, fixed.border)(
        canonical, *_fourier_fixed_arguments(fixed)
    )
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(fixed.wavelengths),) or not np.all(np.isfinite(result)):
        raise SceneModelException("Fourier JAX baseline flux is invalid")
    return result


def fourier_flux_jacobian(parameter_values, parameter_names, fixed):
    """Return the Fourier fixed-map Jacobian in caller-declared name order."""
    canonical, canonical_indices = _canonical_fourier_parameters(
        parameter_values, parameter_names
    )
    canonical_jacobian = _compiled_fourier_jacobian(
        fixed.subsampling, fixed.border
    )(canonical, *_fourier_fixed_arguments(fixed))
    canonical_jacobian = np.asarray(canonical_jacobian, dtype=np.float64)
    expected_shape = (len(fixed.wavelengths), len(FOURIER_PARAMETER_NAMES))
    if canonical_jacobian.shape != expected_shape or \
            not np.all(np.isfinite(canonical_jacobian)):
        raise SceneModelException("Fourier JAX flux Jacobian is invalid")
    jacobian = np.empty_like(canonical_jacobian)
    jacobian[:, canonical_indices] = canonical_jacobian
    return jacobian


def _fixed_wavelength_slice(fixed, start, stop):
    """Return one immutable wavelength slice while sharing fixed grids."""
    return replace(
        fixed,
        data=fixed.data[start:stop],
        inverse_variance=fixed.inverse_variance[start:stop],
        wavelengths=fixed.wavelengths[start:stop],
        adr_scale=fixed.adr_scale[start:stop],
    )


def batched_flux_jacobian(parameter_values, parameter_names, fixed,
                          *, profile,
                          wavelength_batch=PRODUCTION_WAVELENGTH_BATCH):
    """Evaluate production flux and its Jacobian in bounded wavelength batches.

    Native coefficient solves are independent between wavelengths.  Batching
    therefore changes only peak JAX tangent storage, not the extraction map.
    """
    wavelength_batch = int(wavelength_batch)
    if wavelength_batch < 1:
        raise SceneModelException("JAX wavelength batch must be positive")
    if profile == "classic":
        flux_function = classic_flux
        jacobian_function = classic_flux_jacobian
    elif profile == "fourier":
        flux_function = fourier_flux
        jacobian_function = fourier_flux_jacobian
    else:
        raise SceneModelException("Unknown JAX scene profile %s" % profile)

    flux_parts = []
    jacobian_parts = []
    for start in range(0, len(fixed.wavelengths), wavelength_batch):
        stop = min(start + wavelength_batch, len(fixed.wavelengths))
        batch = _fixed_wavelength_slice(fixed, start, stop)
        flux_parts.append(flux_function(
            parameter_values, parameter_names, batch
        ))
        jacobian_parts.append(jacobian_function(
            parameter_values, parameter_names, batch
        ))
    if not flux_parts:
        raise SceneModelException("JAX scene extraction has no wavelengths")
    return np.concatenate(flux_parts), np.concatenate(jacobian_parts, axis=0)
