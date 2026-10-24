# -*- coding: utf-8 -*-
"""SCALA pupil-mask instrumental core for the SNIFS Fourier scene model.

Standard stars observed for the SCALA calibration programme see the telescope
pupil through the SCALA aperture mask, so their instrumental core is the
diffraction pattern of that mask rather than the empirical Gaussian core of
``SnifsFourierSceneModel``.  Daniel Kuesters' production reduction replaced the
Gaussian core by the analytic optical transfer function (OTF) of the mask in
the wavelength-dependent (3D) fit.  This module is that code, lifted out of
his private copies of ``models.py`` / ``snifs.py`` and made a configurable
element:

* ``circular_aperture_otf`` is his ``otf_analytic`` unchanged: the OTF of a
  circular aperture of radius ``radius`` (the autocorrelation of a disc),
  centred on one mask "base vector" and evaluated at the pupil coordinate
  ``k / scale(wavelength)``.
* ``ScalaMaskPsfElement`` sums the OTF over the base vectors that pass the
  radius cut.  Production used the first base vector only (``max_vectors=1``,
  his ``for i in range(0, 1)``); the cut radius and the comment that the full
  sum needs subsampling 32 are his.

The numerical constants are copied verbatim from his code.  Their provenance
(pupil geometry of SCALA / the UH 2.2 m) is not documented here; see
``SCALE_COEFFICIENT`` below.  The base vectors come from his
``base_vectors_python2.pkl`` and are loaded by ``load_base_vectors``.
"""
from __future__ import print_function

import os
import pickle

import numpy as np

from .element import PsfElement
from .utils import SceneModelException

__all__ = [
    "APERTURE_RADIUS", "SCALE_COEFFICIENT", "SCALE_REFERENCE_WAVELENGTH",
    "PRODUCTION_MAX_VECTORS", "PRODUCTION_MAX_RADIUS",
    "BASE_VECTORS_ENV", "DEFAULT_BASE_VECTORS_PATH",
    "pupil_scale", "circular_aperture_otf", "scala_mask_otf",
    "load_base_vectors", "select_base_vectors", "ScalaMaskPsfElement",
]

# --- constants copied from Daniel Kuesters' otf_analytic / MaskPsfElement ---

#: Aperture radius ``R`` in the base-vector units (his ``0.16``).
APERTURE_RADIUS = 0.16

#: ``scale = SCALE_COEFFICIENT / (wavelength / SCALE_REFERENCE_WAVELENGTH)``
#: converts the model's angular frequencies (radians per spaxel) to the pupil
#: coordinate of the base vectors.  His expression is ``45*1.2*0.645*0.9``,
#: with the comment "reference wavelength for scale is 1 mum"; the factors
#: are kept separate here so they can be identified later.  [provenance
#: unverified]
SCALE_COEFFICIENT = 45.0 * 1.2 * 0.645 * 0.9
SCALE_REFERENCE_WAVELENGTH = 10000.0   # Angstrom

#: Production used only the first base vector (``for i in range(0, 1)``).
PRODUCTION_MAX_VECTORS = 1

#: His radius cut on the base vectors, ``rad < (2.5 + 0.372 + 0.02)``, with
#: the comment: "cut the mask base vectors to save time, if you want all the
#: max radius is smaller 2.4 but you also need a subsampling of 32, this will
#: take forever."
PRODUCTION_MAX_RADIUS = 2.5 + 0.372 + 0.02

#: Environment variable naming the base-vector file; otherwise
#: ``DEFAULT_BASE_VECTORS_PATH`` (``scene_model/data/scala_base_vectors.*``).
BASE_VECTORS_ENV = "SCENE_MODEL_SCALA_BASE_VECTORS"
DEFAULT_BASE_VECTORS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "scala_base_vectors"
)


def pupil_scale(wavelength):
    """Daniel's ``scale``: radians per spaxel per base-vector unit."""
    return SCALE_COEFFICIENT / (wavelength / SCALE_REFERENCE_WAVELENGTH)


def circular_aperture_otf(kx, ky, x0, y0, radius, wavelength, xp=np):
    """OTF of a circular aperture of ``radius`` centred on ``(x0, y0)``.

    This is Daniel's ``otf_analytic``: ``2/pi (arccos(rho) - rho sqrt(1 -
    rho^2))`` for ``rho = |k / scale - (x0, y0)| / radius < 1`` and 0 outside.
    It is 1 at ``k / scale == (x0, y0)``.

    ``xp`` is ``numpy`` or ``jax.numpy``; the out-of-support region is handled
    with ``where`` on a clipped argument so neither backend produces NaNs.
    """
    scale = pupil_scale(wavelength)
    rho = xp.sqrt((kx / scale - x0) ** 2 + (ky / scale - y0) ** 2) / radius
    inside = rho < 1.0
    clipped = xp.where(inside, rho, 0.0)
    otf = 2.0 / np.pi * (xp.arccos(clipped)
                         - clipped * xp.sqrt(1.0 - clipped ** 2))
    return xp.where(inside, otf, 0.0)


def scala_mask_otf(kx, ky, base_vectors, wavelength, radius=APERTURE_RADIUS,
                   xp=np):
    """Sum of ``circular_aperture_otf`` over ``base_vectors`` (n, 2)."""
    base_vectors = np.asarray(base_vectors, dtype=np.float64)
    if base_vectors.ndim != 2 or base_vectors.shape[1] != 2 or \
            len(base_vectors) == 0:
        raise SceneModelException(
            "SCALA mask base vectors must have shape (n >= 1, 2), got %s"
            % (base_vectors.shape,))
    otf = None
    for x0, y0 in base_vectors:
        term = circular_aperture_otf(kx, ky, float(x0), float(y0), radius,
                                     wavelength, xp=xp)
        otf = term if otf is None else otf + term
    return otf


def load_base_vectors(path=None):
    """Load the SCALA mask base vectors as an (n, 2) float array.

    ``path`` defaults to ``$SCENE_MODEL_SCALA_BASE_VECTORS`` or, failing that,
    ``scene_model/data/scala_base_vectors.{npy,pkl}``.  ``.npy`` holds the
    array directly; ``.pkl`` is Daniel's Python-2 pickle
    (``base_vectors_python2.pkl``: a sequence of (x, y) pairs) and is read
    with ``encoding='latin1'``.
    """
    if path is None:
        path = os.environ.get(BASE_VECTORS_ENV)
    if path is None:
        for extension in (".npy", ".pkl"):
            candidate = DEFAULT_BASE_VECTORS_PATH + extension
            if os.path.exists(candidate):
                path = candidate
                break
    if path is None or not os.path.exists(path):
        raise SceneModelException(
            "SCALA mask base vectors not found. Set %s to Daniel Kuesters' "
            "base_vectors_python2.pkl (or an .npy copy) or install it as %s"
            ".{npy,pkl}" % (BASE_VECTORS_ENV, DEFAULT_BASE_VECTORS_PATH))
    if path.endswith(".npy"):
        vectors = np.load(path)
    else:
        with open(path, "rb") as handle:
            vectors = pickle.load(handle, encoding="latin1")
    vectors = np.asarray(
        [(float(v[0]), float(v[1])) for v in vectors], dtype=np.float64
    ).reshape(-1, 2)
    if len(vectors) == 0 or not np.all(np.isfinite(vectors)):
        raise SceneModelException("SCALA mask base vectors in %s are empty "
                                  "or non-finite" % path)
    return vectors


def select_base_vectors(base_vectors, max_vectors=PRODUCTION_MAX_VECTORS,
                        max_radius=PRODUCTION_MAX_RADIUS):
    """Daniel's selection loop: the first ``max_vectors`` entries, keeping
    those with ``|b| < max_radius``.  ``max_vectors=None`` takes all."""
    base_vectors = np.asarray(base_vectors, dtype=np.float64).reshape(-1, 2)
    if max_vectors is not None:
        if max_vectors < 1:
            raise SceneModelException("max_vectors must be >= 1")
        base_vectors = base_vectors[:max_vectors]
    radii = np.hypot(base_vectors[:, 0], base_vectors[:, 1])
    selected = base_vectors[radii < max_radius]
    if len(selected) == 0:
        raise SceneModelException(
            "No SCALA mask base vector passes the radius cut %g" % max_radius)
    return selected


class ScalaMaskPsfElement(PsfElement):
    """Instrumental core of a point source seen through the SCALA pupil mask.

    Replaces ``GaussianPsfElement`` as the ``inst_core`` element of
    ``SnifsFourierSceneModel(inst_core='scala_mask')``.  The element has no
    free parameters: its Fourier profile is ``scala_mask_otf`` at the shared,
    fixed ``wavelength`` parameter (so it needs a wavelength-dependent
    model).  ``base_vectors`` are the selected mask base vectors (see
    ``select_base_vectors``); ``radius`` is the aperture radius.

    The OTF must be 1 at zero frequency so the PSF is normalized like every
    other element.  That holds for the production selection (the first base
    vector, at the origin); a selection that breaks it raises.
    """
    def __init__(self, base_vectors, radius=APERTURE_RADIUS, *args, **kwargs):
        base_vectors = np.asarray(base_vectors, dtype=np.float64)
        if base_vectors.ndim != 2 or base_vectors.shape[1] != 2 or \
                len(base_vectors) == 0:
            raise SceneModelException(
                "ScalaMaskPsfElement needs base vectors of shape (n >= 1, 2)")
        if not np.all(np.isfinite(base_vectors)) or not radius > 0:
            raise SceneModelException("Invalid SCALA mask configuration")
        self.base_vectors = base_vectors
        self.radius = float(radius)
        # OTF at k = 0 does not depend on wavelength (k / scale = 0).
        zero = float(scala_mask_otf(np.zeros(1), np.zeros(1), base_vectors,
                                    SCALE_REFERENCE_WAVELENGTH, radius)[0])
        if abs(zero - 1.0) > 1e-12:
            raise SceneModelException(
                "SCALA mask OTF is %.6g at zero frequency, not 1: the PSF "
                "would not be normalized. Production uses the first base "
                "vector only; a wider selection needs a normalization that "
                "this element does not define." % zero)
        super(ScalaMaskPsfElement, self).__init__(*args, **kwargs)

    def _setup_parameters(self):
        self._add_parameter('wavelength', None, (None, None), 'WAVE',
                            'wavelength [A]', fixed=True, apply_prefix=False)

    def _evaluate_fourier(self, kx, ky, subsampling, grid_info, wavelength,
                          **kwargs):
        if wavelength is None:
            raise SceneModelException(
                "Must set wavelength for the SCALA mask instrumental core!")
        return scala_mask_otf(kx, ky, self.base_vectors, wavelength,
                              self.radius)
