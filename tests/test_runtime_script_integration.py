"""Integration contracts for the three installed SNIFS command scripts."""

from __future__ import annotations

import runpy
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from astropy.io import fits


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _checkout_environment():
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), existing) if value
    )
    return environment


@pytest.mark.parametrize(
    "script, expected",
    [
        ("extract_star2.py", ("--psf", "--keepmodel", "--covariance")),
        ("extract_fixed_star2.py", ("--cube", "--spec", "--method")),
        ("subtract_psf2.py", ("--ref", "--psfname", "--nosubtract")),
    ],
)
def test_script_help_contracts(script, expected):
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        cwd=ROOT,
        env=_checkout_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    for option in expected:
        assert option in completed.stdout


@pytest.mark.parametrize(
    "script, arguments, message",
    [
        ("extract_star2.py", [], "No input datacube specified"),
        ("subtract_psf2.py", ["synthetic-spectrum.fits"],
         "Reference cube not specified"),
        ("subtract_psf2.py", ["--ref", "cube.fits", "spectrum.fits"],
         "Name for output point-source subtracted cube not specified"),
    ],
)
def test_script_required_argument_contracts(script, arguments, message):
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *arguments],
        cwd=ROOT,
        env=_checkout_environment(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 2
    assert message in completed.stderr


class _RecordingCube:
    def __init__(self, calls, label):
        self.calls = calls
        self.label = label

    def WR_3d_fits(self, path, **kwargs):
        self.calls.append((self.label, Path(path).name, kwargs))


class _RecordingFitter:
    instances = []

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.channel = "B"
        self.calls = []
        self.meta_cube = _RecordingCube(self.calls, "meta")
        self.meta_cube_model = _RecordingCube(self.calls, "model")
        self.__class__.instances.append(self)

    def fit_metaslices_2d(self, **kwargs):
        self.calls.append(("fit2d", kwargs))

    def fit_metaslices_3d(self):
        self.calls.append(("fit3d",))

    def check_validity(self):
        self.calls.append(("valid",))

    def extract(self, **kwargs):
        self.calls.append(("extract", kwargs))

    def write_spectrum(self, output, sky):
        self.calls.append(("spectra", Path(output).name, Path(sky).name))

    @staticmethod
    def _plot(path):
        # Exercise the plotting-only MPL compatibility behavior through the
        # same methods called by the production script.
        from matplotlib import pyplot as plt
        from scene_model._compat import plotting as MPL

        fig, ax = plt.subplots()
        polygon = ax.errorband(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]),
            np.array([0.1, 0.2]), color=MPL.blue,
        )
        assert polygon.axes is ax
        if path is not None:
            fig.savefig(path)
        plt.close(fig)

    def plot_spectrum(self, path):
        self.calls.append(("plot_spectrum", Path(path).name))
        self._plot(path)

    def plot_slice_fit(self, path):
        self.calls.append(("plot_slice_fit", Path(path).name))
        self._plot(path)

    def plot_row_column_sums(self, path):
        self.calls.append(("plot_row_column_sums", Path(path).name))
        self._plot(path)

    def plot_adr(self, path):
        self.calls.append(("plot_adr", Path(path).name))
        self._plot(path)

    def plot_residuals(self, path):
        self.calls.append(("plot_residuals", Path(path).name))
        self._plot(path)

    def plot_seeing(self, path):
        self.calls.append(("plot_seeing", Path(path).name))
        self._plot(path)

    def plot_radial_profile(self, path):
        self.calls.append(("plot_radial_profile", Path(path).name))
        self._plot(path)

    def plot_contours(self, path):
        self.calls.append(("plot_contours", Path(path).name))
        self._plot(path)


def test_extract_star_plot_and_keepmodel_orchestration(tmp_path, monkeypatch):
    """Lock orchestration without requiring a committed observation cube."""
    from scene_model import snifs

    _RecordingFitter.instances.clear()
    monkeypatch.setattr(snifs, "SnifsCubeFitter", _RecordingFitter)
    output = tmp_path / "spectrum.fits"
    monkeypatch.setattr(
        sys, "argv",
        [
            "extract_star2", "--in", "synthetic-cube.fits",
            "--out", str(output), "--sky", str(tmp_path / "sky.fits"),
            "--keepmodel", "--graph", "png",
        ],
    )

    runpy.run_path(str(SCRIPTS / "extract_star2.py"), run_name="__main__")

    fitter = _RecordingFitter.instances[-1]
    assert ("meta", "meta_spectrum.fits", {}) in fitter.calls
    assert ("model", "psf_spectrum.fits", {"header": []}) in fitter.calls
    assert ("extract", {
        "method": "psf", "radius": -5.0, "covariance": False,
        "jacobian_backend": "finite-difference", "jax_wavelength_batch": None,
    }) in fitter.calls
    plot_calls = [call for call in fitter.calls if call[0].startswith("plot_")]
    assert len(plot_calls) == 8
    assert all((tmp_path / call[1]).is_file() for call in plot_calls)


class _FixedModel:
    @classmethod
    def from_fits_header(cls, header):
        return cls()

    def setup_grid(self, **kwargs):
        self.grid = kwargs

    def extract(self, data, variance, **kwargs):
        nslice = data.shape[0]
        return {
            "amplitude": np.arange(nslice, dtype=float) + 10.0,
            "amplitude_variance": np.arange(nslice, dtype=float) + 1.0,
        }

    def evaluate_multi(self, wavelength, **kwargs):
        components = np.zeros((len(wavelength), 2, 15, 15))
        components[:, 0] = 0.25
        return components


def _write_auxiliary_script_inputs(tmp_path):
    from scene_model._compat.snifs_io import spectrum

    cube_path = tmp_path / "cube.fits"
    data = np.arange(12.0).reshape(3, 2, 2)
    header = fits.Header()
    header["CRVAL1"] = -0.43
    header["CRVAL2"] = -0.43
    header["CRVAL3"] = 5000.0
    header["CDELT1"] = 0.43
    header["CDELT2"] = 0.43
    header["CDELT3"] = 2.0
    fits.HDUList([
        fits.PrimaryHDU(data, header=header),
        fits.ImageHDU(np.ones_like(data), name="VARIANCE"),
    ]).writeto(cube_path)

    spectrum_path = tmp_path / "spectrum.fits"
    spectrum(
        data=np.array([2.0, 3.0, 4.0]),
        var=np.array([0.2, 0.3, 0.4]),
        start=5000.0,
        step=2.0,
    ).WR_fits_file(spectrum_path, header_list=[
        ("ES_PSF", "classic"),
        ("ES_SDEG", -1),
        ("ES_SPXSZ", 0.43),
        ("SEEING", 1.0),
    ])
    return cube_path, spectrum_path


def test_extract_fixed_star_successful_private_io_workflow(tmp_path, monkeypatch):
    from scene_model import snifs
    from scene_model._compat.snifs_io import spectrum

    cube_path, spectrum_path = _write_auxiliary_script_inputs(tmp_path)
    output = tmp_path / "fixed.fits"
    monkeypatch.setattr(snifs, "SnifsClassicSceneModel", _FixedModel)
    monkeypatch.setattr(
        sys, "argv",
        [
            "extract_fixed_star2", "--cube", str(cube_path),
            "--spec", str(spectrum_path), "--out", str(output),
        ],
    )

    runpy.run_path(str(SCRIPTS / "extract_fixed_star2.py"), run_name="__main__")

    result = spectrum(output)
    np.testing.assert_array_equal(result.data, [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(result.var, [1.0, 2.0, 3.0])


def test_subtract_psf_successful_private_fits3d_writer_workflow(
    tmp_path, monkeypatch
):
    from scene_model import snifs
    from scene_model._compat.snifs_io import SNIFS_cube

    cube_path, spectrum_path = _write_auxiliary_script_inputs(tmp_path)
    output = tmp_path / "subtracted.fits"
    psf_output = tmp_path / "psf.fits"
    monkeypatch.setattr(snifs, "SnifsClassicSceneModel", _FixedModel)
    monkeypatch.setattr(
        sys, "argv",
        [
            "subtract_psf2", "--ref", str(cube_path), "--out", str(output),
            "--psfname", str(psf_output), str(spectrum_path),
        ],
    )

    runpy.run_path(str(SCRIPTS / "subtract_psf2.py"), run_name="__main__")

    subtracted = SNIFS_cube(fits3d_file=output)
    point_source = SNIFS_cube(fits3d_file=psf_output)
    assert subtracted.data.shape == point_source.data.shape == (3, 4)
    expected = np.repeat(
        np.array([2.0, 3.0, 4.0])[:, None] * 0.25, 4, axis=1
    )
    np.testing.assert_allclose(point_source.data, expected)
