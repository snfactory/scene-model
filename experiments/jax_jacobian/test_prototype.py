import numpy as np
import pytest

from .prototype import (
    centered_difference_jacobian,
    fixture_arguments,
    jax,
    jitted_flux,
    jitted_jacobian,
    make_fixture,
)


pytestmark = pytest.mark.skipif(jax is None, reason="JAX is not installed")


@pytest.mark.parametrize("profile", ["classic", "fourier"])
def test_jax_jacobian_matches_centered_difference(profile):
    fixture = make_fixture(profile=profile, nwave=12, grid_size=9)
    arguments = fixture_arguments(fixture)
    jacobian = np.asarray(jitted_jacobian(*arguments, profile=profile))
    oracle = centered_difference_jacobian(fixture)

    assert jacobian.shape == (12, len(fixture.parameters))
    np.testing.assert_allclose(jacobian, oracle, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("profile", ["classic", "fourier"])
def test_jax_evaluation_is_repeatable_and_does_not_mutate_inputs(profile):
    fixture = make_fixture(profile=profile, nwave=8, grid_size=9)
    data = fixture.data.copy()
    parameters = fixture.parameters.copy()
    arguments = fixture_arguments(fixture)

    first = np.asarray(jitted_flux(*arguments, profile=profile))
    second = np.asarray(jitted_flux(*arguments, profile=profile))

    assert np.array_equal(first, second)
    assert np.array_equal(fixture.data, data)
    assert np.array_equal(fixture.parameters, parameters)
