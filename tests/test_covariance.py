from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from types import SimpleNamespace

from scene_model.covariance import (
    GlobalParameter,
    assemble_flux_covariance,
    factor_covariance,
    finite_difference_jacobian,
    select_marginal_covariance,
)
from scene_model.models import GaussianSceneModel
from scene_model.scene import SceneModel
from scene_model.models import Background, PointSource
from scene_model.utils import SceneModelException
from scene_model.snifs import write_pysnifs_spectrum


def test_marginal_covariance_selection_uses_exact_reordered_names():
    names = ("amplitude[0]", "shape", "background[0]", "position")
    covariance = np.arange(16.0).reshape(4, 4)
    covariance = covariance + covariance.T

    selected = select_marginal_covariance(
        names, covariance, ("position", "shape")
    )

    np.testing.assert_array_equal(selected, covariance[np.ix_([3, 1], [3, 1])])


@pytest.mark.parametrize("names", [
    ("shape", "shape"),
    ("other", "parameter"),
])
def test_marginal_covariance_selection_rejects_duplicate_or_missing(names):
    with pytest.raises(SceneModelException):
        select_marginal_covariance(names, np.eye(2), ("shape",))


def test_complete_native_coefficient_covariance_matches_dense_inverse():
    model = SceneModel([PointSource, Background], grid_size=(2, 2))
    source = np.array([[1.0, 0.5], [0.25, 0.1]])
    background = np.ones((2, 2))
    variance = np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)
    data = 3.0 * source + 2.0 * background
    mask = np.ones((2, 2), dtype=bool)

    names, values, variances, covariance = model._calculate_coefficients(
        [source, background], data[mask], variance[mask], mask,
        return_covariance=True,
    )
    design = np.vstack([source[mask], background[mask]]).T
    expected = np.linalg.inv(design.T @ ((1.0 / variance[mask])[:, None] * design))

    assert names == ["amplitude", "background"]
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_allclose(variances, np.diag(expected))
    assert covariance[0, 1] != 0
    np.testing.assert_allclose(values, [3.0, 2.0])


def test_structured_extraction_has_exact_legacy_table_parity():
    model = GaussianSceneModel(grid_size=(5, 5), subsampling=1, border=0)
    parameters = dict(center_x=0.1, center_y=-0.2, sigma_x=0.9,
                      sigma_y=1.1, rho=0.05)
    image = model.evaluate(amplitude=7.0, background=1.5, **parameters)
    variance = np.ones_like(image) * 0.25

    legacy = model.extract(image, variance, **parameters)
    structured = model.extract(
        image, variance, return_covariance=True, **parameters
    )

    assert structured.coefficient_names == ("amplitude", "background")
    assert structured.coefficient_covariance.shape == (1, 2, 2)
    assert structured.coefficient_covariance.flags.writeable is False
    assert legacy.colnames == structured.table.colnames
    for name in legacy.colnames:
        assert np.array_equal(legacy[name], structured.table[name])


def test_flux_jacobian_matches_analytic_derivative_and_bound_stencil():
    parameters = (
        GlobalParameter("x", 2.0, (None, None), 1.0),
        GlobalParameter("y", 0.0, (0.0, 10.0), 1.0),
    )

    def evaluate(values):
        x, y = values
        return np.array([x * x + 3.0 * y, 2.0 * x - y])

    jacobian, diagnostics = finite_difference_jacobian(
        evaluate, parameters, np.diag([0.2, 0.3])
    )

    np.testing.assert_allclose(jacobian, [[4.0, 3.0], [2.0, -1.0]],
                               rtol=2e-6, atol=2e-6)
    assert diagnostics.stencils == ("centered", "forward")
    assert max(diagnostics.stability) <= 5e-3


def test_psd_factorization_and_assembly_match_dense_product():
    scene_covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    jacobian = np.array([[1.0, 2.0], [-0.5, 0.25], [3.0, -1.0]])
    conditional = np.array([0.2, 0.3, 0.4])

    factor, diagnostics = factor_covariance(scene_covariance)
    covariance, propagated = assemble_flux_covariance(
        conditional, jacobian, factor
    )

    expected = np.diag(conditional) + jacobian @ scene_covariance @ jacobian.T
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_allclose(propagated @ propagated.T,
                               jacobian @ scene_covariance @ jacobian.T)
    assert diagnostics.rank == 2


def test_factorization_clips_only_tiny_negative_eigenvalues():
    factor, diagnostics = factor_covariance(np.diag([1.0, -1e-12]))
    assert diagnostics.clipped
    assert diagnostics.rank == 1
    np.testing.assert_allclose(factor @ factor.T, np.diag([1.0, 0.0]))

    with pytest.raises(SceneModelException):
        factor_covariance(np.diag([1.0, -1e-5]))


def test_structured_covariance_rejects_aperture_estimator():
    model = GaussianSceneModel(grid_size=(5, 5), subsampling=1, border=0)
    with pytest.raises(SceneModelException):
        model.extract(np.ones((5, 5)), np.ones((5, 5)), method="aperture",
                      return_covariance=True)


def test_fits_writer_round_trips_total_variance_and_lower_triangle(tmp_path):
    covariance = np.array([[2.0, 0.25], [0.25, 3.0]])
    spectrum = SimpleNamespace(
        start=4000.0,
        step=2.0,
        data=np.array([10.0, 20.0]),
        var=np.diag(covariance),
        cov=covariance,
        has_var=True,
    )
    output = tmp_path / "spectrum.fits"

    write_pysnifs_spectrum(spectrum, output, fits.Header(),
                           transactional=True)

    with fits.open(output) as hdus:
        np.testing.assert_array_equal(hdus["VARIANCE"].data,
                                      np.diag(covariance))
        np.testing.assert_array_equal(hdus["COVAR"].data,
                                      np.tril(covariance))
    assert not list(tmp_path.glob(".*.tmp"))
