from __future__ import annotations

import numpy as np
import pytest

from scene_model.jax_backend import jax_forward_jacobian, jax_runtime_provenance
from scene_model.utils import SceneModelException


def test_jax_provenance_is_strict_or_reports_install_remedy():
    try:
        provenance = dict(jax_runtime_provenance())
    except SceneModelException as error:
        assert "0.10.2" in str(error)
    else:
        assert {"jax", "jaxlib", "backend", "device", "architecture", "x64"} \
            <= provenance.keys()
        assert provenance["x64"] == "true"


def test_jax_fixed_map_has_strict_parity_and_exact_forward_jacobian():
    pytest.importorskip("jax")
    try:
        provenance = dict(jax_runtime_provenance())
    except SceneModelException as error:
        pytest.skip(str(error))
    import jax.numpy as jnp

    matrix = np.array([[1.0, 2.0], [-0.5, 3.0], [2.5, -1.0]])

    def numpy_flux(parameters):
        return matrix @ np.asarray(parameters) ** 2

    jax_matrix = jnp.asarray(matrix)

    def jax_flux(parameters):
        return jax_matrix @ parameters ** 2

    values = np.array([0.7, -1.2])
    jacobian, diagnostics = jax_forward_jacobian(
        numpy_flux, jax_flux, values,
        cache_key=("classic", "B", 3, (3, 2), 2, 3),
        expected_flux=numpy_flux(values),
    )
    np.testing.assert_allclose(jacobian, matrix * (2.0 * values)[None, :],
                               rtol=1e-14, atol=1e-14)
    assert diagnostics.backend == "jax"
    assert dict(diagnostics.provenance) == provenance

    with pytest.raises(SceneModelException, match="baseline failed"):
        jax_forward_jacobian(
            numpy_flux, jax_flux, values,
            cache_key=("classic", "B", 3, (3, 2), 2, 3),
            expected_flux=numpy_flux(values) + 1e-6,
        )
