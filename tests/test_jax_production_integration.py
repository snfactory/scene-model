from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scene_model.covariance import GlobalParameter
from scene_model.jax_scene import CLASSIC_PARAMETER_NAMES, FOURIER_PARAMETER_NAMES
from scene_model.snifs import (SnifsClassicSceneModel, SnifsCubeFitter,
                               SnifsFourierSceneModel)
from scene_model.utils import SceneModelException


class _FixedAdr:
    def set_param(self, **kwargs):
        pass

    def get_scale(self, wavelength):
        wavelength = np.asarray(wavelength)
        return (wavelength - 5000.0) / 1800.0


def _production_fitter(psf):
    wavelengths = np.array([3650.0, 5000.0, 7050.0])
    spaxel_size = 0.43
    adr = _FixedAdr()
    if psf == "classic":
        names = CLASSIC_PARAMETER_NAMES
        values = np.array([
            0.17, -0.23, 1.08, 0.31, -0.045, -0.19, 2.35, 1.14, 0.08
        ])
        model = SnifsClassicSceneModel(
            exposure_time=20.0, wavelength_dependence=True,
            background_degree=-1, adr_model=adr, spaxel_size=spaxel_size,
            grid_size=(15, 15), subsampling=2, border=2,
        )
    else:
        names = FOURIER_PARAMETER_NAMES
        values = np.array([
            0.14, -0.21, 1.11, 0.27, -0.24, 0.48, 0.13, -0.09
        ])
        model = SnifsFourierSceneModel(
            use_empirical_parameters=True, wavelength_dependence=True,
            background_degree=-1, adr_model=adr, spaxel_size=spaxel_size,
            grid_size=(15, 15), subsampling=2, border=2,
        )
    parameters = dict(zip(names, values))
    model.set_parameters(update_derived=False, **parameters)
    images = np.asarray([
        model.evaluate(wavelength=wavelength, amplitude=amplitude)
        for wavelength, amplitude in zip(wavelengths, (4.4, 7.3, 5.2))
    ])
    variance = 0.4 + np.arange(images.size).reshape(images.shape) / images.size

    grid_i, grid_j = np.indices((15, 15))
    cube = SimpleNamespace(
        data=images.reshape(len(wavelengths), -1),
        var=variance.reshape(len(wavelengths), -1),
        i=grid_i.ravel(),
        j=grid_j.ravel(),
        lbda=wavelengths,
        lstep=float(wavelengths[1] - wavelengths[0]),
        spxSize=spaxel_size,
    )
    parameter_info = tuple(
        GlobalParameter(name, value, tuple(model._parameter_info[name]["bounds"]))
        for name, value in zip(names, values)
    )

    fitter = SnifsCubeFitter.__new__(SnifsCubeFitter)
    fitter.fit_scene_model = model
    fitter.fit_global_covariance = np.eye(len(names)) * 1e-4
    fitter.fit_global_parameter_info = parameter_info
    fitter.background_degree = -1
    fitter.least_squares = False
    fitter.filter_variance = False
    fitter.reference_seeing = 1.0
    fitter.cube = cube
    fitter.psf = psf
    fitter.header = {"EFFTIME": 20.0}
    fitter.covariance_diagnostics = None
    fitter.extraction = None
    fitter.point_source_spectrum = None
    fitter.background_spectrum = None
    fitter.sky_spectrum = None
    fitter.extraction_method = None
    fitter.extraction_radius = None
    return fitter


@pytest.mark.parametrize("psf", ["classic", "fourier"])
@pytest.mark.parametrize("wavelength_batch", ["default", None, 128])
def test_production_extraction_uses_requested_jax_backend(
        psf, wavelength_batch, capsys):
    pytest.importorskip("jax")
    fitter = _production_fitter(psf)

    kwargs = {} if wavelength_batch == "default" else {
        "jax_wavelength_batch": wavelength_batch
    }
    fitter.extract(method="psf", covariance=True, jacobian_backend="jax",
                   **kwargs)

    diagnostics = fitter.covariance_diagnostics
    expected_batch = 128 if wavelength_batch == "default" else wavelength_batch
    assert diagnostics["derivatives"].backend == "jax"
    assert dict(diagnostics["derivatives"].provenance)["x64"] == "true"
    assert diagnostics["jax_wavelength_batch"] == expected_batch
    warning = "may change the JAX surrogate and covariance"
    assert (warning in capsys.readouterr().out) == (expected_batch is not None)
    assert diagnostics["jacobian"].shape == (
        len(fitter.cube.lbda), len(fitter.fit_global_parameter_info)
    )
    np.testing.assert_array_equal(
        fitter.point_source_spectrum.var,
        np.diag(fitter.point_source_spectrum.cov),
    )
    np.testing.assert_allclose(
        diagnostics["fixed_extraction_map"].dot(fitter.cube.data.ravel()),
        fitter.point_source_spectrum.data,
        rtol=1e-13,
        atol=0.0,
    )


def test_production_extraction_rejects_unknown_jacobian_backend():
    fitter = _production_fitter("fourier")

    with pytest.raises(SceneModelException,
                       match="Unknown Jacobian backend typo"):
        fitter.extract(method="psf", covariance=True, jacobian_backend="typo")
