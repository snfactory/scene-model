"""Guarded JAX differentiation for fixed pure-array extraction maps."""

from __future__ import annotations

from functools import lru_cache
import platform

import numpy as np

from .covariance import JacobianDiagnostics
from .utils import SceneModelException


JAX_VERSION = "0.10.2"


def jax_runtime_provenance():
    """Return the reproducibility fields required for a JAX covariance run."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jaxlib
    except ImportError as error:
        raise SceneModelException(
            "JAX covariance backend is unavailable; install jax==0.10.2 or "
            "select jacobian_backend='finite-difference'"
        ) from error
    if str(jax.__version__) != JAX_VERSION or str(jaxlib.__version__) != JAX_VERSION:
        raise SceneModelException(
            "JAX covariance backend requires jax==jaxlib==%s; found %s/%s" %
            (JAX_VERSION, jax.__version__, jaxlib.__version__)
        )
    devices = jax.devices()
    device = devices[0] if devices else None
    return (
        ("jax", str(jax.__version__)),
        ("jaxlib", str(jaxlib.__version__)),
        ("backend", str(jax.default_backend())),
        ("device", str(device) if device is not None else "none"),
        ("architecture", platform.machine()),
        ("x64", str(bool(jax.config.read("jax_enable_x64"))).lower()),
    )


@lru_cache(maxsize=32)
def _compile_jax_jacobian(cache_key, function):
    import jax
    return jax.jit(jax.jacfwd(function, argnums=0))


def jax_forward_jacobian(evaluate_numpy, evaluate_jax, parameter_values,
                         *, cache_key, expected_flux=None,
                         parity_rtol=1e-12, parity_atol=0.0):
    """Differentiate an exported pure-JAX fixed extraction map."""
    provenance = jax_runtime_provenance()
    import jax.numpy as jnp

    parameter_values = np.asarray(parameter_values, dtype=np.float64)
    if expected_flux is None:
        expected_flux = np.asarray(evaluate_numpy(parameter_values),
                                   dtype=np.float64)
    else:
        expected_flux = np.asarray(expected_flux, dtype=np.float64)
    jax_values = jnp.asarray(parameter_values, dtype=jnp.float64)
    baseline = np.asarray(evaluate_jax(jax_values), dtype=np.float64)
    if baseline.shape != expected_flux.shape or not np.allclose(
            baseline, expected_flux, rtol=parity_rtol, atol=parity_atol):
        difference = (np.max(np.abs(baseline - expected_flux))
                      if baseline.shape == expected_flux.shape else np.inf)
        raise SceneModelException(
            "JAX fixed-map baseline failed accepted NumPy parity "
            "(maximum absolute difference=%g)" % difference
        )
    compiled = _compile_jax_jacobian(tuple(cache_key), evaluate_jax)
    jacobian = np.asarray(compiled(jax_values), dtype=np.float64)
    expected_shape = (len(expected_flux), len(parameter_values))
    if jacobian.shape != expected_shape or not np.all(np.isfinite(jacobian)):
        raise SceneModelException("JAX flux Jacobian is invalid")
    diagnostics = JacobianDiagnostics(
        steps=tuple(), stability=tuple(), stencils=tuple(), backend="jax",
        provenance=provenance,
    )
    return jacobian, diagnostics
