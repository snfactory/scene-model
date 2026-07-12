"""Two-stage Laplace covariance propagation for scene-model extraction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .utils import SceneModelException


@dataclass(frozen=True)
class ExtractionResult:
    """Native extraction table and its complete linear-coefficient covariance."""

    table: object
    coefficient_names: tuple[str, ...]
    coefficient_covariance: np.ndarray
    coefficient_estimator: np.ndarray


@dataclass(frozen=True)
class GlobalParameter:
    name: str
    value: float
    bounds: tuple[float | None, float | None]
    scale: float = 1.0


@dataclass(frozen=True)
class FactorizationDiagnostics:
    rank: int
    condition: float
    clipped: bool
    minimum_eigenvalue: float
    maximum_eigenvalue: float


@dataclass(frozen=True)
class JacobianDiagnostics:
    steps: tuple[float, ...]
    stability: tuple[float, ...]
    stencils: tuple[str, ...]
    backend: str = "finite-difference"
    provenance: tuple[tuple[str, str], ...] = ()


def select_marginal_covariance(names, covariance, global_names):
    """Select an ordering-safe marginal principal covariance block."""
    names = tuple(names)
    global_names = tuple(global_names)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(names), len(names)):
        raise SceneModelException("Named covariance has inconsistent shape")
    indices = []
    for name in global_names:
        matches = [idx for idx, candidate in enumerate(names)
                   if candidate == name]
        if len(matches) != 1:
            raise SceneModelException(
                "Global covariance parameter %s occurs %d times" %
                (name, len(matches))
            )
        indices.append(matches[0])
    result = covariance[np.ix_(indices, indices)].copy()
    if not np.all(np.isfinite(result)):
        raise SceneModelException("Global covariance contains non-finite values")
    return result


def factor_covariance(covariance, relative_negative_tolerance=1e-10):
    """Return L with covariance=L L.T, permitting only roundoff negativity."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise SceneModelException("Covariance must be a square matrix")
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise SceneModelException("Covariance contains non-finite values")

    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    maximum = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    if maximum < 0 or minimum < -relative_negative_tolerance * maximum:
        raise SceneModelException(
            "Covariance is not positive semidefinite (min=%g, max=%g)" %
            (minimum, maximum)
        )

    try:
        factor = np.linalg.cholesky(covariance)
        positive = eigenvalues[eigenvalues > 0]
        condition = (maximum / positive.min()) if len(positive) else np.inf
        diagnostics = FactorizationDiagnostics(
            rank=len(eigenvalues), condition=float(condition), clipped=False,
            minimum_eigenvalue=minimum, maximum_eigenvalue=maximum,
        )
        return factor, diagnostics
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        clipped = bool(np.any(eigenvalues < 0))
        eigenvalues = np.maximum(eigenvalues, 0.0)
        threshold = maximum * np.finfo(float).eps * max(1, len(eigenvalues))
        retained = eigenvalues > threshold
        factor = eigenvectors[:, retained] * np.sqrt(eigenvalues[retained])
        if np.any(retained):
            condition = maximum / eigenvalues[retained].min()
        else:
            condition = np.inf
        diagnostics = FactorizationDiagnostics(
            rank=int(np.sum(retained)), condition=float(condition),
            clipped=clipped, minimum_eigenvalue=minimum,
            maximum_eigenvalue=maximum,
        )
        return factor, diagnostics


def _relative_difference(left, right):
    denominator = max(np.linalg.norm(left), np.linalg.norm(right),
                      np.finfo(float).tiny)
    return float(np.linalg.norm(left - right) / denominator)


def finite_difference_jacobian(evaluate_flux, parameters, covariance,
                               relative_tolerance=5e-3, max_retries=5):
    """Differentiate native flux with stable centered/bounded stencils."""
    parameters = tuple(parameters)
    covariance = np.asarray(covariance, dtype=float)
    base_values = np.array([parameter.value for parameter in parameters],
                           dtype=float)
    base_flux = np.asarray(evaluate_flux(base_values), dtype=float)
    if base_flux.ndim != 1 or not np.all(np.isfinite(base_flux)):
        raise SceneModelException("Baseline extraction flux is invalid")

    jacobian = np.empty((len(base_flux), len(parameters)), dtype=float)
    used_steps, stabilities, stencils = [], [], []

    for index, parameter in enumerate(parameters):
        sigma = np.sqrt(max(0.0, covariance[index, index]))
        characteristic = max(abs(parameter.value), abs(parameter.scale),
                             sigma, 1.0)
        step = max(1e-3 * characteristic,
                   np.sqrt(np.finfo(float).eps) * characteristic)
        lower, upper = parameter.bounds

        def permitted(multiplier, direction):
            trial = parameter.value + direction * multiplier * step
            return ((lower is None or trial >= lower) and
                    (upper is None or trial <= upper))

        success = False
        last_stability = np.inf
        for _ in range(max_retries + 1):
            centered = permitted(2.0, -1) and permitted(2.0, +1)
            if centered:
                def derivative(h):
                    plus = base_values.copy()
                    minus = base_values.copy()
                    plus[index] += h
                    minus[index] -= h
                    return (np.asarray(evaluate_flux(plus)) -
                            np.asarray(evaluate_flux(minus))) / (2.0 * h)
                stencil = "centered"
            else:
                if permitted(4.0, +1):
                    direction = 1.0
                    stencil = "forward"
                elif permitted(4.0, -1):
                    direction = -1.0
                    stencil = "backward"
                else:
                    step *= 0.5
                    continue

                def derivative(h):
                    one = base_values.copy()
                    two = base_values.copy()
                    one[index] += direction * h
                    two[index] += direction * 2.0 * h
                    return direction * (-3.0 * base_flux +
                                        4.0 * np.asarray(evaluate_flux(one)) -
                                        np.asarray(evaluate_flux(two))) / (2.0*h)

            columns = [np.asarray(derivative(scale * step), dtype=float)
                       for scale in (0.5, 1.0, 2.0)]
            if any(column.shape != base_flux.shape or
                   not np.all(np.isfinite(column)) for column in columns):
                raise SceneModelException(
                    "Invalid derivative for parameter %s" % parameter.name
                )
            last_stability = max(_relative_difference(columns[0], columns[1]),
                                 _relative_difference(columns[1], columns[2]))
            if last_stability <= relative_tolerance:
                jacobian[:, index] = columns[0]
                success = True
                break
            step *= 0.5

        if not success:
            raise SceneModelException(
                "Unstable derivative for %s (relative change=%g)" %
                (parameter.name, last_stability)
            )
        used_steps.append(0.5 * step)
        stabilities.append(last_stability)
        stencils.append(stencil)

    return jacobian, JacobianDiagnostics(
        steps=tuple(used_steps), stability=tuple(stabilities),
        stencils=tuple(stencils), backend="finite-difference",
    )


def assemble_flux_covariance(conditional_variance, jacobian, factor):
    conditional_variance = np.asarray(conditional_variance, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    factor = np.asarray(factor, dtype=float)
    if conditional_variance.ndim != 1 or jacobian.ndim != 2 or factor.ndim != 2:
        raise SceneModelException("Covariance assembly inputs have invalid rank")
    if jacobian.shape[0] != len(conditional_variance):
        raise SceneModelException("Jacobian and native variance axes differ")
    if jacobian.shape[1] != factor.shape[0]:
        raise SceneModelException("Jacobian and scene covariance axes differ")
    if np.any(~np.isfinite(conditional_variance)) or np.any(conditional_variance < 0):
        raise SceneModelException("Conditional variances are invalid")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(factor)):
        raise SceneModelException("Covariance propagation inputs are non-finite")
    propagated_factor = np.dot(jacobian, factor)
    covariance = np.diag(conditional_variance) + \
        np.dot(propagated_factor, propagated_factor.T)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise SceneModelException("Assembled flux covariance is non-finite")
    if not np.array_equal(covariance, covariance.T):
        raise SceneModelException("Assembled flux covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(eigenvalues[-1]), 1.0) if len(eigenvalues) else 1.0
    if len(eigenvalues) and eigenvalues[0] < -1e-10 * scale:
        raise SceneModelException("Assembled flux covariance is not PSD")
    if not np.allclose(np.diag(covariance),
                       conditional_variance +
                       np.sum(propagated_factor**2, axis=1),
                       rtol=1e-13, atol=0.0):
        raise SceneModelException("Assembled covariance diagonal is inconsistent")
    return covariance, propagated_factor


def fixed_extraction_map(coefficient_estimator, coefficient: int):
    """Return the native block-diagonal fixed extraction map ``H``.

    Input columns use wavelength-major, fixed-spaxel-minor order.  Masked
    spaxels remain present as zero columns.  This map is the derivative of the
    accepted coefficient solve with scene parameters and decisions frozen.
    """
    from scipy import sparse

    estimator = np.asarray(coefficient_estimator, dtype=np.float64)
    if estimator.ndim != 3 or not 0 <= coefficient < estimator.shape[1]:
        raise SceneModelException("Fixed coefficient estimator has invalid axes")
    if not np.all(np.isfinite(estimator)):
        raise SceneModelException("Fixed coefficient estimator is non-finite")
    nwave, _, nspaxel = estimator.shape
    values = estimator[:, coefficient, :]
    rows = np.repeat(np.arange(nwave, dtype=np.int64), nspaxel)
    columns = np.arange(nwave * nspaxel, dtype=np.int64)
    matrix = sparse.csr_matrix(
        (values.ravel(), (rows, columns)), shape=(nwave, nwave * nspaxel)
    )
    matrix.eliminate_zeros()
    return matrix
