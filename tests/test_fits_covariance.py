"""Deterministic FITS round-trip checks for SNIFS-COV-1.0 spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from scene_model._compat import snifs_io as pySNIFS
from scene_model.snifs import write_pysnifs_spectrum


REQUIRED_METADATA = {
    "COVVERS": "SNIFS-COV-1.0",
    "COVMETH": "2STAGE-LAPLACE",
    "COVSCOPE": "SCENE+NATIVE-DIAG",
    "COVINPUT": "E3D-STAT-DIAG",
    "COVDER": "FINITE-DIFFERENCE",
    "COVNPAR": 2,
    "COVRANK": 2,
    "COVSTAT": "OK",
    "COVFORM": "LOWER",
    "COVTYPE": "TOTAL-WITHIN-SCOPE",
}


def _covariance_header():
    header = fits.Header()
    for key, value in REQUIRED_METADATA.items():
        header[key] = value
    header.add_history("Global scene parameters use the joint meta-slice fit.")
    header.add_history("Native coefficients are conditionally re-solved.")
    header.add_history("Input E3D statistical covariance is diagonal.")
    header.add_history("Upstream resampling covariance is not included.")
    return header


def _spectrum():
    spectrum = pySNIFS.spectrum(
        data=np.array([4.0, 3.0, 5.0]),
        var=np.array([0.4, 0.7, 0.9]),
        start=4100.0,
        step=2.5,
    )
    spectrum.cov = np.array([
        [0.4, 0.12, -0.03],
        [0.12, 0.7, 0.08],
        [-0.03, 0.08, 0.9],
    ])
    spectrum.var = np.diag(spectrum.cov).copy()
    return spectrum


def test_lower_triangle_and_metadata_round_trip(tmp_path):
    output = tmp_path / "spectrum.fits"
    expected = _spectrum().cov.copy()
    write_pysnifs_spectrum(
        _spectrum(), output, _covariance_header(), transactional=True
    )

    with fits.open(output) as hdus:
        assert [hdu.name for hdu in hdus] == ["PRIMARY", "VARIANCE", "COVAR"]
        for key, value in REQUIRED_METADATA.items():
            assert hdus[0].header[key] == value
        np.testing.assert_array_equal(hdus[1].data, np.diag(expected))
        np.testing.assert_array_equal(hdus[2].data, np.tril(expected))
        assert np.count_nonzero(np.triu(hdus[2].data, 1)) == 0
        lower = hdus[2].data
        reconstructed = lower + lower.T - np.diag(np.diag(lower))
        np.testing.assert_array_equal(reconstructed, expected)
        assert hdus[2].header["CRVAL1"] == 4100.0
        assert hdus[2].header["CDELT1"] == 2.5
        assert hdus[2].header["CRVAL2"] == 4100.0
        assert hdus[2].header["CDELT2"] == 2.5
        history = " ".join(hdus[0].header["HISTORY"]).lower()
        for phrase in ("meta-slice", "conditionally re-solved", "diagonal",
                       "upstream resampling covariance is not included"):
            assert phrase in history
        assert np.linalg.eigvalsh(reconstructed).min() >= -1e-14


def test_transactional_failure_preserves_existing_destination(tmp_path,
                                                              monkeypatch):
    output = tmp_path / "spectrum.fits"
    sentinel = b"preexisting-valid-product"
    output.write_bytes(sentinel)
    original = fits.HDUList.writeto

    def fail_after_partial_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected write failure")

    monkeypatch.setattr(fits.HDUList, "writeto", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        write_pysnifs_spectrum(
            _spectrum(), output, _covariance_header(), transactional=True
        )
    monkeypatch.setattr(fits.HDUList, "writeto", original)

    assert output.read_bytes() == sentinel
    assert list(tmp_path.glob(".spectrum.fits.*.tmp")) == []


def test_covariance_disabled_writer_has_no_covar_hdu(tmp_path):
    spectrum = _spectrum()
    del spectrum.cov
    output = tmp_path / "legacy.fits"
    write_pysnifs_spectrum(spectrum, output, fits.Header())

    with fits.open(output) as hdus:
        assert [hdu.name for hdu in hdus] == ["PRIMARY", "VARIANCE"]
