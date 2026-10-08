"""Exact JAX objectives for the SNIFS scene-model fits (opt-in fit backend).

The NumPy element graph (scene.py / fit.py) remains the reference
implementation.  This module re-expresses the chi-square that
``SceneModel.chi_square`` and ``MultipleImageFitter.chi_square`` compute for
the two SNIFS scene models, so that the L-BFGS-B fits get exact gradients and
the covariances exact Hessians instead of finite differences:

* ``SnifsFourierSceneModel`` and ``SnifsClassicSceneModel``;
* single-image fits without wavelength dependence (the 2D metaslice fits) and
  multiple-image fits with ADR and chromatic seeing (the 3D metaslice fit);
* coefficients (point-source amplitude, background polynomial) either solved
  analytically per image (fits) or explicit fit parameters (covariances);
* Gaussian / multivariate-Gaussian priors scaled by ``prior_scale``.

The forward models are the validated ones in ``jax_scene``
(``fourier_source_pixels`` / ``classic_source_pixels``).  A 2D model is the 3D
model at the reference wavelength with no ADR and no chromatic terms.  Any
configuration outside this set raises ``SceneModelException``; there is no
silent fallback to NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .jax_scene import (
    CLASSIC_PARAMETER_NAMES,
    FOURIER_PARAMETER_NAMES,
    FOURIER_PROFILE_NAMES,
    REFERENCE_WAVELENGTH,
    _ensure_jax,
    build_polynomial_background_bases,
    check_separable_grid,
    classic_source_pixels,
    fourier_source_pixels,
)
from .utils import SceneModelException

FIT_BACKENDS = ("numpy", "jax")

# 2D (wavelength-independent) parameter names -> canonical 3D names.
_TWO_D_ALIASES = {
    "fourier": {"center_x": "ref_center_x", "center_y": "ref_center_y",
                "seeing_width": "seeing_ref_width"},
    "classic": {"center_x": "ref_center_x", "center_y": "ref_center_y",
                "alpha": "A2"},
}
# Canonical values that make the 3D model reduce to the 2D one.
_TWO_D_NEUTRAL = {
    "fourier": {"adr_delta": 0.0, "adr_theta": 0.0, "seeing_ref_power": 0.0},
    "classic": {"adr_delta": 0.0, "adr_theta": 0.0, "A0": 0.0, "A1": 0.0},
}
_CANONICAL = {"fourier": FOURIER_PARAMETER_NAMES,
              "classic": CLASSIC_PARAMETER_NAMES}
# Background coefficient names in build_polynomial_background_bases order.
_BACKGROUND_NAMES = {-1: (), 0: ("background",)}


def check_backend(fit_backend):
    if fit_backend not in FIT_BACKENDS:
        raise SceneModelException("Unknown fit backend %r (choose from %s)"
                                  % (fit_backend, ", ".join(FIT_BACKENDS)))
    return fit_backend


def _psf_kind(scene_model):
    # Imported lazily: snifs.py imports this module's users.
    from .snifs import SnifsClassicSceneModel, SnifsFourierSceneModel
    if isinstance(scene_model, SnifsFourierSceneModel):
        return "fourier"
    if isinstance(scene_model, SnifsClassicSceneModel):
        return "classic"
    raise SceneModelException(
        "JAX fit backend supports only the SNIFS Fourier/classic scene "
        "models, not %s" % type(scene_model).__name__)


def _background_degree(scene_model):
    names = tuple(scene_model._coefficient_info)
    coefficient_names = names[1:]
    if names[:1] != ("amplitude",):
        raise SceneModelException(
            "JAX fit backend expects the point-source amplitude as the first "
            "coefficient, got %s" % (names,))
    for degree, expected in _BACKGROUND_NAMES.items():
        if coefficient_names == expected:
            return degree, names
    raise SceneModelException(
        "JAX fit backend supports background degree -1 or 0 only, got "
        "coefficients %s" % (names,))


def _model_constants(scene_model, psf):
    """(profile constants, padded grids) of a SNIFS scene model."""
    grid = scene_model.grid_info
    check_separable_grid(grid["pad_grid_kx"], grid["pad_grid_ky"])
    if psf == "fourier":
        parameters = scene_model.parameters
        constants = np.asarray(
            [float(parameters[name]) for name in FOURIER_PROFILE_NAMES])
        grids = (np.asarray(grid["pad_grid_kx"], float),
                 np.asarray(grid["pad_grid_ky"], float),
                 np.asarray(grid["pad_ifft_shift"], complex))
    else:
        # beta/sigma/eta = c0 + c1 * alpha, set from the exposure time
        element = next(e for e in scene_model.elements if hasattr(e, "beta0"))
        constants = np.asarray([element.beta0, element.beta1, element.sigma0,
                                element.sigma1, element.eta0, element.eta1])
        grids = tuple(np.asarray(grid[k]) for k in (
            "pad_grid_x", "pad_grid_y", "pad_grid_kx", "pad_grid_ky",
            "pad_fft_shift", "pad_ifft_shift"))
    return constants, grids


@dataclass
class _Image:
    data: np.ndarray            # (nx, ny), fit scale applied, 0 outside mask
    inverse_variance: np.ndarray
    wavelength: float
    adr_scale: float
    coefficient_values: np.ndarray   # current values, fit scale applied


def _image_arrays(scene_model, apply_fit_scale, psf, wavelength_dependence):
    if scene_model.fit_variance is None or scene_model.fit_mask is None:
        raise SceneModelException(
            "JAX fit backend requires statistical variances (chi2 fits)")
    mask = np.asarray(scene_model.fit_mask, dtype=bool)
    scale = float(scene_model.fit_scale) if apply_fit_scale else 1.0
    data = np.zeros(mask.shape)
    inverse_variance = np.zeros(mask.shape)
    data[mask] = np.asarray(scene_model.fit_data, float) / scale
    inverse_variance[mask] = scale**2 / np.asarray(scene_model.fit_variance,
                                                   float)
    if wavelength_dependence:
        wavelength = float(scene_model["wavelength"])
        adr = scene_model.adr_element
        adr_scale = float(np.ravel(adr.adr_model.get_scale(wavelength))[0]
                          / adr.spaxel_size)
    else:
        wavelength, adr_scale = REFERENCE_WAVELENGTH, 0.0
    coefficients = np.array(
        [scene_model._coefficient_info[name]["value"] or 0.0
         for name in scene_model._coefficient_info], float) / scale
    return _Image(data, inverse_variance, wavelength, adr_scale, coefficients)


class SceneObjective:
    """chi2(x) for a flat parameter vector, with exact gradient/Hessian.

    ``names`` is the flat vector order used by the NumPy fit/covariance code;
    individual parameters of image i are named ``name[i]`` when there are
    several images (MultipleImageFitter convention) and ``name`` otherwise.
    """

    def __init__(self, scene_models, names, *, analytic_coefficients,
                 apply_fit_scale, priors_by_scale, fixed_values):
        self.names = tuple(names)
        model0 = scene_models[0]
        self.psf = _psf_kind(model0)
        self.wavelength_dependence = bool(model0.wavelength_dependence)
        degree, coefficient_names = _background_degree(model0)
        for model in scene_models[1:]:
            if _psf_kind(model) != self.psf or \
                    bool(model.wavelength_dependence) != \
                    self.wavelength_dependence or \
                    _background_degree(model)[0] != degree:
                raise SceneModelException("Inconsistent scene models")
        self.analytic = bool(analytic_coefficients)
        nimage = len(scene_models)
        multi = nimage > 1 or any("[" in n for n in self.names)

        images = [_image_arrays(m, apply_fit_scale, self.psf,
                                self.wavelength_dependence)
                  for m in scene_models]
        grid = model0.grid_info
        self.subsampling = int(grid["subsampling"])
        self.border = int(grid["border"])
        self.native_shape = images[0].data.shape

        canonical = _CANONICAL[self.psf]
        aliases = {} if self.wavelength_dependence else _TWO_D_ALIASES[self.psf]
        neutral = {} if self.wavelength_dependence else _TWO_D_NEUTRAL[self.psf]

        # Base canonical vector from the current (fixed or starting) values.
        base = []
        for name in canonical:
            if name in neutral:
                base.append(neutral[name])
                continue
            source = next((k for k, v in aliases.items() if v == name), name)
            value = fixed_values.get(source)
            if value is None:
                value = model0[source]
            if value is None:
                raise SceneModelException("No value for %s" % source)
            base.append(float(value))
        self.base_theta = np.asarray(base, float)

        # Flat-vector layout.
        theta_slots, theta_index = [], []
        coefficient_slots, coefficient_index = [], []
        for position, name in enumerate(self.names):
            image = None
            if "[" in name:
                stem, rest = name.split("[", 1)
                image = int(rest.rstrip("]"))
            else:
                stem = name
            stem = aliases.get(stem, stem)
            if stem in canonical:
                if image is not None:
                    raise SceneModelException(
                        "Per-image PSF parameter %s is not supported" % name)
                theta_slots.append(position)
                theta_index.append(canonical.index(stem))
            elif stem in coefficient_names:
                if self.analytic:
                    raise SceneModelException(
                        "Coefficient %s given with analytic coefficients"
                        % name)
                if image is None:
                    if multi:
                        raise SceneModelException("Ambiguous %s" % name)
                    image = 0
                coefficient_slots.append(position)
                coefficient_index.append(
                    (image, coefficient_names.index(stem)))
            else:
                raise SceneModelException(
                    "JAX fit backend cannot map parameter %s for the %s%s "
                    "model" % (name, self.psf,
                               "" if self.wavelength_dependence else " 2D"))
        self.theta_slots = np.asarray(theta_slots, int)
        self.theta_index = np.asarray(theta_index, int)
        self.coefficient_slots = np.asarray(coefficient_slots, int)
        self.coefficient_index = np.asarray(coefficient_index, int).reshape(
            -1, 2)

        if not self.analytic:
            # Every coefficient must be either free or fixed at its value.
            for model in scene_models:
                for name, info in model._coefficient_info.items():
                    if info["fixed"]:
                        raise SceneModelException(
                            "Fixed coefficient %s is not supported" % name)

        self.data = np.stack([i.data for i in images])
        self.inverse_variance = np.stack([i.inverse_variance for i in images])
        self.wavelengths = np.asarray([i.wavelength for i in images])
        self.adr_scale = np.asarray([i.adr_scale for i in images])
        self.base_coefficients = np.stack([i.coefficient_values
                                           for i in images])
        self.background_bases = np.asarray(build_polynomial_background_bases(
            grid["grid_x"], grid["grid_y"], degree))

        self.profile_constants, self.grids = _model_constants(model0,
                                                              self.psf)

        # Priors: (scale, canonical indices or flat slots, centre, inverse cov)
        self.priors = []
        for scale, priors in priors_by_scale:
            if not scale:
                continue
            for prior in priors:
                names_ = getattr(prior, "parameter_names", None)
                if names_ is None and hasattr(prior, "parameter_name"):
                    names_ = [prior.parameter_name]
                    centre = np.atleast_1d(prior.central_value)
                    inv = np.atleast_2d(1.0 / prior.sigma**2)
                elif names_ is not None:
                    centre = np.asarray(prior.central_values, float)
                    inv = np.asarray(prior.inv_covariance, float)
                else:
                    raise SceneModelException(
                        "Unsupported prior %s" % type(prior).__name__)
                index = []
                for n in names_:
                    n = aliases.get(n, n)
                    if n not in canonical or n in neutral:
                        raise SceneModelException(
                            "Prior on %s is not supported by the JAX fit "
                            "backend" % n)
                    index.append(canonical.index(n))
                self.priors.append((float(scale), np.asarray(index, int),
                                    centre, inv))

    # --------------------------------------------------------- NumPy-facing
    def _arguments(self):
        return dict(
            base_theta=self.base_theta, theta_index=self.theta_index,
            theta_slots=self.theta_slots,
            base_coefficients=self.base_coefficients,
            coefficient_index=self.coefficient_index,
            coefficient_slots=self.coefficient_slots,
            data=self.data, inverse_variance=self.inverse_variance,
            wavelengths=self.wavelengths, adr_scale=self.adr_scale,
            background_bases=self.background_bases,
            profile_constants=self.profile_constants, grids=self.grids,
            priors=tuple(self.priors),
        )

    def _static(self):
        return dict(psf=self.psf, analytic=self.analytic,
                    native_shape=tuple(self.native_shape),
                    subsampling=self.subsampling, border=self.border)

    def value(self, x):
        return float(_compiled("value")(np.asarray(x, float),
                                        self._arguments(), **self._static()))

    def value_and_grad(self, x):
        value, gradient = _compiled("value_and_grad")(
            np.asarray(x, float), self._arguments(), **self._static())
        return float(value), np.asarray(gradient, dtype=np.float64)

    def hessian(self, x):
        return np.asarray(_compiled("hessian")(
            np.asarray(x, float), self._arguments(), **self._static()),
            dtype=np.float64)


_STATIC = ("psf", "analytic", "native_shape", "subsampling", "border")


def _sources(theta, a, *, psf, native_shape, subsampling, border,
             fold=True):
    """Unit-amplitude point-source images (nimage, nx, ny) for ``theta``.

    fold=True (Fourier-space subpixel summation) is used for the
    differentiated objectives, where it is faster; extraction is forward
    only and uses the inverse FFT on the subsampled grid."""
    if psf == "fourier":
        return fourier_source_pixels(
            theta, a["wavelengths"], a["adr_scale"], *a["grids"],
            a["profile_constants"], native_shape=native_shape,
            subsampling=subsampling, border=border, fold=fold)
    return classic_source_pixels(
        theta, a["wavelengths"], a["adr_scale"], *a["grids"],
        a["profile_constants"], native_shape=native_shape,
        subsampling=subsampling, border=border, fold=fold)


def _design(theta, a, *, psf, native_shape, subsampling, border,
            fold=True):
    """(nimage, ncoefficient, npixel) design: source then background bases."""
    _, jnp = _ensure_jax()
    sources = _sources(theta, a, psf=psf, native_shape=native_shape,
                       subsampling=subsampling, border=border, fold=fold)
    nimage = sources.shape[0]
    npix = native_shape[0] * native_shape[1]
    bases = jnp.asarray(a["background_bases"])
    background = jnp.broadcast_to(
        bases[jnp.newaxis], (nimage,) + bases.shape,
    ).reshape(nimage, bases.shape[0], npix)
    return jnp.concatenate((sources.reshape(nimage, 1, npix), background),
                           axis=1)


def _theta(x, a):
    _, jnp = _ensure_jax()
    return jnp.asarray(a["base_theta"]).at[a["theta_index"]].set(
        x[a["theta_slots"]])


def _chi2(x, a, *, psf, analytic, native_shape, subsampling, border):
    """chi2 of one or several images; ``a`` holds the arrays of a
    SceneObjective (see SceneObjective._arguments)."""
    _, jnp = _ensure_jax()
    theta = _theta(x, a)
    design = _design(theta, a, psf=psf, native_shape=native_shape,
                     subsampling=subsampling, border=border)
    nimage = design.shape[0]
    npix = native_shape[0] * native_shape[1]
    weight = a["inverse_variance"].reshape(nimage, npix)
    data = a["data"].reshape(nimage, npix)
    if analytic:
        normal = jnp.einsum("ncp,np,ndp->ncd", design, weight, design)
        rhs = jnp.einsum("ncp,np,np->nc", design, weight, data)
        coefficients = jnp.einsum("ncd,nd->nc", jnp.linalg.inv(normal), rhs)
    else:
        index = a["coefficient_index"]
        coefficients = jnp.asarray(a["base_coefficients"]).at[
            index[:, 0], index[:, 1]].set(x[a["coefficient_slots"]])
    model = jnp.einsum("nc,ncp->np", coefficients, design)
    chi2 = jnp.sum(weight * (data - model) ** 2)
    for scale, prior_index, centre, inverse in a["priors"]:
        diff = theta[prior_index] - centre
        chi2 = chi2 + scale * diff @ inverse @ diff
    return chi2


def _hessian(x, a, *, psf, analytic, native_shape, subsampling, border):
    """Exact Hessian of _chi2.

    Hessian-vector products (one forward-over-reverse pass at a time, which
    bounds peak memory) are taken only along the PSF-parameter directions;
    they give those rows (and, by symmetry, columns) in full.  Coefficients
    enter the model linearly, so the coefficient-coefficient block is exactly
    2 D^T W D per image (zero between images).
    """
    jax, jnp = _ensure_jax()
    static = dict(psf=psf, analytic=analytic, native_shape=native_shape,
                  subsampling=subsampling, border=border)
    gradient = jax.grad(lambda y: _chi2(y, a, **static))
    n = x.shape[0]
    theta_slots = a["theta_slots"]
    k = theta_slots.shape[0]
    directions = jnp.zeros((k, n), dtype=x.dtype).at[
        jnp.arange(k), theta_slots].set(1.0)
    rows = jax.lax.map(lambda v: jax.jvp(gradient, (x,), (v,))[1],
                       directions)
    hessian = jnp.zeros((n, n), dtype=x.dtype)
    hessian = hessian.at[theta_slots, :].set(rows)
    hessian = hessian.at[:, theta_slots].set(rows.T)
    coefficient_slots = a["coefficient_slots"]
    if not analytic and coefficient_slots.shape[0]:
        design = _design(_theta(x, a), a, psf=psf, native_shape=native_shape,
                         subsampling=subsampling, border=border)
        nimage = design.shape[0]
        weight = a["inverse_variance"].reshape(nimage, -1)
        gram = 2.0 * jnp.einsum("ncp,np,ndp->ncd", design, weight, design)
        index = a["coefficient_index"]
        image, column = index[:, 0], index[:, 1]
        block = gram[image[:, None], column[:, None], column[None, :]] * (
            image[:, None] == image[None, :])
        hessian = hessian.at[coefficient_slots[:, None],
                             coefficient_slots[None, :]].set(block)
    return hessian


_COMPILED = {}


def _compiled(kind):
    """Process-wide jitted functions; JAX caches one compilation per array
    shape/static configuration, so e.g. all 2D metaslice fits of a cube share
    one compilation."""
    if kind not in _COMPILED:
        jax, jnp = _ensure_jax()
        if kind == "value":
            function = _chi2
        elif kind == "value_and_grad":
            function = jax.value_and_grad(_chi2)
        elif kind == "hessian":
            function = _hessian
        else:
            raise SceneModelException("Unknown compiled kind %s" % kind)
        _COMPILED[kind] = jax.jit(function, static_argnames=_STATIC)
    return _COMPILED[kind]


def scene_model_objective(scene_model, names, *, analytic_coefficients,
                          apply_fit_scale=True, apply_priors=True):
    """Objective matching SceneModel.chi_square for a single model."""
    fixed = {k: v["value"] for k, v in scene_model._parameter_info.items()}
    priors = [(scene_model.prior_scale, scene_model.priors)] \
        if apply_priors else []
    return SceneObjective(
        [scene_model], names, analytic_coefficients=analytic_coefficients,
        apply_fit_scale=apply_fit_scale, priors_by_scale=priors,
        fixed_values=fixed)


def multiple_image_objective(fitter, names, *, analytic_coefficients,
                             apply_fit_scale=True, apply_priors=True):
    """Objective matching MultipleImageFitter.chi_square."""
    fixed = dict(fitter.scene_models[0].parameters)
    fixed.update({k: v["value"]
                  for k, v in fitter._global_fit_parameters.items()})
    priors = []
    if apply_priors:
        model_priors = {(id(m.priors), m.prior_scale): m
                        for m in fitter.scene_models if m.priors}
        if model_priors:
            raise SceneModelException(
                "Per-image priors in a multiple-image fit are not supported "
                "by the JAX fit backend")
        priors.append((fitter.prior_scale, fitter.priors))
    return SceneObjective(
        fitter.scene_models, names,
        analytic_coefficients=analytic_coefficients,
        apply_fit_scale=apply_fit_scale, priors_by_scale=priors,
        fixed_values=fixed)


EXTRACTION_BATCH = 128


def _extract_batch(theta, a, data, inverse_variance, *, psf, native_shape,
                   subsampling, border):
    """Per-wavelength weighted least squares for a fixed PSF: coefficient
    values and their variances (diagonal of the inverse normal matrix)."""
    _, jnp = _ensure_jax()
    design = _design(theta, a, psf=psf, native_shape=native_shape,
                     subsampling=subsampling, border=border, fold=False)
    nimage = design.shape[0]
    weight = inverse_variance.reshape(nimage, -1)
    data = data.reshape(nimage, -1)
    normal = jnp.einsum("ncp,np,ndp->ncd", design, weight, design)
    rhs = jnp.einsum("ncp,np,np->nc", design, weight, data)
    covariance = jnp.linalg.inv(normal)
    values = jnp.einsum("ncd,nd->nc", covariance, rhs)
    return values, jnp.diagonal(covariance, axis1=1, axis2=2)


def extract_psf(scene_model, images, variances, wavelengths, *,
                batch=EXTRACTION_BATCH):
    """JAX version of ``SceneModel.extract(images, variances, method='psf',
    wavelength=wavelengths)`` for a fitted, wavelength-dependent SNIFS
    scene model (chi2 extraction, no covariance).  Same pixel mask
    (non-NaN image and positive variance) and same output columns."""
    from astropy.table import Table
    jax, jnp = _ensure_jax()
    psf = _psf_kind(scene_model)
    if not scene_model.wavelength_dependence:
        raise SceneModelException("JAX extraction needs the 3D scene model")
    if variances is None:
        raise SceneModelException("JAX extraction needs statistical variances")
    degree, coefficient_names = _background_degree(scene_model)
    for name, info in scene_model._coefficient_info.items():
        if info["fixed"]:
            raise SceneModelException(
                "Fixed coefficient %s is not supported" % name)
    images = np.asarray(images, float)
    variances = np.asarray(variances, float)
    wavelengths = np.asarray(wavelengths, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mask = ~np.isnan(images) & (variances > 0)
        data = np.where(mask, images, 0.0)
        inverse_variance = np.where(mask, 1.0 / variances, 0.0)
    if not mask.reshape(len(images), -1).any(axis=1).all():
        raise SceneModelException("A wavelength slice has no valid spaxel")

    canonical = _CANONICAL[psf]
    theta = np.asarray([float(scene_model[name]) for name in canonical])
    adr = scene_model.adr_element
    adr_scale = np.ravel(adr.adr_model.get_scale(wavelengths)) / \
        adr.spaxel_size
    constants, grids = _model_constants(scene_model, psf)
    grid = scene_model.grid_info
    bases = np.asarray(build_polynomial_background_bases(
        grid["grid_x"], grid["grid_y"], degree))
    static = dict(psf=psf, native_shape=tuple(images.shape[1:]),
                  subsampling=int(grid["subsampling"]),
                  border=int(grid["border"]))
    if "extract" not in _COMPILED:
        _COMPILED["extract"] = jax.jit(
            _extract_batch, static_argnames=("psf", "native_shape",
                                             "subsampling", "border"))
    function = _COMPILED["extract"]

    nwave = len(wavelengths)
    values, variances_out = [], []
    for start in range(0, nwave, batch):
        stop = min(start + batch, nwave)
        pad = batch - (stop - start)   # fixed batch shape: one compilation

        def padded(array, fill):
            if not pad:
                return array[start:stop]
            filler = np.full((pad,) + array.shape[1:], fill)
            return np.concatenate([array[start:stop], filler])
        a = dict(wavelengths=padded(wavelengths, REFERENCE_WAVELENGTH),
                 adr_scale=padded(adr_scale, 0.0), grids=grids,
                 profile_constants=constants, background_bases=bases)
        v, var = function(theta, a, padded(data, 0.0),
                          padded(inverse_variance, 1.0), **static)
        values.append(np.asarray(v)[:stop - start])
        variances_out.append(np.asarray(var)[:stop - start])
    values = np.concatenate(values)
    variances_out = np.concatenate(variances_out)
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(variances_out))):
        raise SceneModelException("JAX extraction produced non-finite values")
    table = Table()
    for column, name in enumerate(coefficient_names):
        table[name] = values[:, column]
        table["%s_variance" % name] = variances_out[:, column]
    return table
