"""Independent dense and analytic oracles for the Cov-M1 algebra."""

from __future__ import annotations

import numpy as np
import pytest

from scene_model.covariance import (
    GlobalParameter,
    assemble_flux_covariance,
    factor_covariance,
    finite_difference_jacobian,
    select_marginal_covariance,
)
from scene_model.utils import SceneModelException


def _variable_projection_solution(parameter, source_0, source_1,
                                  background, data, weight):
    design = np.column_stack((source_0 + parameter * source_1, background))
    normal = design.T @ weight @ design
    return np.linalg.solve(normal, design.T @ weight @ data)


def _variable_projection_derivative(parameter, source_0, source_1,
                                    background, data, weight):
    design = np.column_stack((source_0 + parameter * source_1, background))
    design_prime = np.column_stack((source_1, np.zeros_like(background)))
    normal = design.T @ weight @ design
    coefficients = np.linalg.solve(normal, design.T @ weight @ data)
    residual = data - design @ coefficients
    right_hand_side = (
        design_prime.T @ weight @ residual
        - design.T @ weight @ design_prime @ coefficients
    )
    return np.linalg.solve(normal, right_hand_side)


def test_full_coefficient_covariance_matches_dense_normal_inverse():
    design = np.array([
        [1.0, 1.0],
        [0.3, 1.0],
        [0.05, 1.0],
        [0.0, 1.0],
    ])
    variance = np.array([1.2, 0.8, 2.0, 1.5])
    covariance = np.linalg.inv(design.T @ np.diag(1.0 / variance) @ design)

    assert covariance.shape == (2, 2)
    assert covariance[0, 1] != 0.0  # source/background covariance is retained
    np.testing.assert_allclose(
        covariance,
        np.linalg.solve(design.T @ np.diag(1.0 / variance) @ design,
                        np.eye(2)),
        rtol=2e-15,
        atol=2e-15,
    )


def test_global_selection_is_named_marginal_block_not_conditional_inverse():
    # Precision has global/local coupling, so these two constructions differ.
    precision = np.array([
        [5.0, 0.8, -0.4],
        [0.8, 3.0, 0.6],
        [-0.4, 0.6, 2.5],
    ])
    complete_covariance = np.linalg.inv(precision)
    names = ("local[0]", "shape", "position")
    permutation = np.array([2, 0, 1])
    shuffled_names = tuple(names[index] for index in permutation)
    shuffled_covariance = complete_covariance[np.ix_(permutation, permutation)]

    selected = select_marginal_covariance(
        shuffled_names, shuffled_covariance, ("position", "shape")
    )
    expected = complete_covariance[np.ix_([2, 1], [2, 1])]
    conditional_wrong_answer = np.linalg.inv(precision[np.ix_([2, 1], [2, 1])])

    np.testing.assert_allclose(selected, expected, rtol=0.0, atol=0.0)
    assert not np.allclose(selected, conditional_wrong_answer)


@pytest.mark.parametrize(
    "names,global_names",
    [
        (("position", "position", "shape"), ("position",)),
        (("position", "local[0]"), ("shape",)),
    ],
)
def test_global_selection_rejects_duplicate_or_missing_names(names,
                                                              global_names):
    with pytest.raises(SceneModelException):
        select_marginal_covariance(names, np.eye(len(names)), global_names)


@pytest.mark.parametrize("at_bound", [False, True])
def test_flux_jacobian_matches_analytic_variable_projection(at_bound):
    source_0 = np.array([0.2, 1.0, 0.6, 0.1, -0.1])
    source_1 = np.array([0.3, -0.2, 0.15, 0.05, 0.1])
    background = np.ones(5)
    data = np.array([1.7, 4.2, 3.1, 1.2, 0.4])
    weight = np.diag([1.0, 0.7, 1.3, 0.5, 2.0])
    value = 0.0 if at_bound else 0.35
    bounds = (0.0, 2.0) if at_bound else (-2.0, 2.0)
    parameter = GlobalParameter("shape", value, bounds, scale=1.0)

    def evaluate(values):
        return _variable_projection_solution(
            values[0], source_0, source_1, background, data, weight
        )

    numerical, diagnostics = finite_difference_jacobian(
        evaluate, (parameter,), np.array([[0.04]])
    )
    analytic = _variable_projection_derivative(
        value, source_0, source_1, background, data, weight
    )

    np.testing.assert_allclose(numerical[:, 0], analytic, rtol=2e-6, atol=2e-8)
    assert diagnostics.stability[0] <= 5e-3
    assert diagnostics.stencils[0] == ("forward" if at_bound else "centered")


def test_low_rank_assembly_matches_direct_dense_product_and_rank():
    jacobian = np.array([
        [1.0, -0.3],
        [0.2, 0.8],
        [-0.1, 0.4],
        [0.7, 0.0],
    ])
    scene_covariance = np.array([[0.7, 0.2], [0.2, 0.4]])
    conditional = np.array([0.2, 0.3, 0.1, 0.5])
    factor, diagnostics = factor_covariance(scene_covariance)

    total, propagated = assemble_flux_covariance(
        conditional, jacobian, factor
    )
    expected_scene = jacobian @ scene_covariance @ jacobian.T

    np.testing.assert_allclose(propagated @ propagated.T, expected_scene,
                               rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(total, np.diag(conditional) + expected_scene,
                               rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(np.diag(total),
                               conditional + np.diag(expected_scene))
    assert np.linalg.matrix_rank(propagated @ propagated.T) <= diagnostics.rank
    assert np.linalg.eigvalsh(total).min() >= -1e-13


def test_factorization_clips_only_roundoff_scale_negative_modes():
    covariance = np.diag([2.0, 0.5, -1e-11])
    factor, diagnostics = factor_covariance(covariance)

    assert diagnostics.clipped
    assert diagnostics.rank == 2
    np.testing.assert_allclose(factor @ factor.T, np.diag([2.0, 0.5, 0.0]))

    with pytest.raises(SceneModelException):
        factor_covariance(np.diag([2.0, 0.5, -3e-10]))
