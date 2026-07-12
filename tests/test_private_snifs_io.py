"""Parity contracts for the minimized private SNIFS I/O implementation."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

import pySNIFS as legacy
from scene_model._compat.snifs_io import SNIFS_cube, spectrum


def _assert_spectrum_equal(left, right):
    for name in ("data", "var", "x"):
        left_value, right_value = getattr(left, name), getattr(right, name)
        if left_value is None:
            assert right_value is None
        else:
            np.testing.assert_array_equal(left_value, right_value)
    for name in ("start", "step", "len", "has_var"):
        assert getattr(left, name) == getattr(right, name)


def _assert_cube_equal(left, right):
    for name in ("data", "var", "lbda", "x", "y", "i", "j", "no"):
        left_value, right_value = getattr(left, name), getattr(right, name)
        if left_value is None:
            assert right_value is None
        else:
            np.testing.assert_array_equal(left_value, right_value)
    for name in ("from_e3d_file", "nlens", "nslice", "lstart", "lstep", "lend"):
        assert getattr(left, name) == getattr(right, name)


def _write_fits3d(path, *, variance=True, sparse=False):
    data = np.arange(36.0).reshape(4, 3, 3)
    if sparse:
        data[:, 1, 1] = np.nan
    header = fits.Header()
    header["CRVAL1"] = -0.43
    header["CRVAL2"] = -0.43
    header["CRVAL3"] = 5000.0
    header["CDELT1"] = 0.43
    header["CDELT2"] = 0.43
    header["CDELT3"] = 2.0
    hdus = [fits.PrimaryHDU(data, header=header)]
    if variance:
        hdus.append(fits.ImageHDU(data + 100.0, name="VARIANCE"))
    fits.HDUList(hdus).writeto(path)


def _write_e3d(path, *, variance=True):
    data = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    columns = [
        fits.Column(name="SPEC_ID", format="J", array=[11, 22]),
        fits.Column(name="SPEC_LEN", format="J", array=[4, 4]),
        fits.Column(name="SPEC_STA", format="J", array=[0, 0]),
        fits.Column(name="XPOS", format="E", array=[-0.43, 0.0]),
        fits.Column(name="YPOS", format="E", array=[0.0, 0.43]),
        fits.Column(name="DATA_SPE", format="4D", array=data),
    ]
    if variance:
        columns.append(fits.Column(name="STAT_SPE", format="4D", array=data + 10.0))
    table = fits.BinTableHDU.from_columns(columns, name="E3D_DATA")
    table.header["CRVALS"] = 4000.0
    table.header["CDELTS"] = 2.0
    table.header["CHANNEL"] = "B"
    group = fits.BinTableHDU.from_columns([], name="E3D_GRP")
    primary = fits.PrimaryHDU()
    primary.header["EURO3D"] = True
    fits.HDUList([primary, table, group]).writeto(path)


@pytest.mark.parametrize("with_variance", [False, True])
def test_regular_spectrum_construct_and_roundtrip_matches_legacy(tmp_path, with_variance):
    data = np.array([3.5, -2.0, 8.25])
    variance = np.array([0.4, 0.5, 0.9]) if with_variance else None
    old = legacy.spectrum(data=data, var=variance, start=4100.0, step=2.5)
    new = spectrum(data=data, var=variance, start=4100.0, step=2.5)
    _assert_spectrum_equal(old, new)

    old_path, new_path = tmp_path / "old.fits", tmp_path / "new.fits"
    header = [["OBJECT", "synthetic"]]
    old.WR_fits_file(old_path, header_list=header)
    new.WR_fits_file(new_path, header_list=header)
    _assert_spectrum_equal(legacy.spectrum(data_file=old_path), spectrum(data_file=new_path))
    assert fits.getheader(old_path)["OBJECT"] == fits.getheader(new_path)["OBJECT"]
    assert len(fits.open(old_path)) == len(fits.open(new_path))


def test_irregular_and_zero_spectrum_construction_matches_legacy():
    _assert_spectrum_equal(
        legacy.spectrum(x=[1.0, 1.7, 3.2]),
        spectrum(x=[1.0, 1.7, 3.2]),
    )
    _assert_spectrum_equal(
        legacy.spectrum(nx=4),
        spectrum(nx=4),
    )


@pytest.mark.parametrize("variance", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_fits3d_read_slice_and_roundtrip_matches_legacy(tmp_path, variance, sparse):
    source = tmp_path / "source.fits"
    _write_fits3d(source, variance=variance, sparse=sparse)
    old, new = legacy.SNIFS_cube(fits3d_file=source), SNIFS_cube(fits3d_file=source)
    _assert_cube_equal(old, new)
    np.testing.assert_array_equal(
        old.slice2d(1, coord="p"),
        new.slice2d(1, coord="p"),
    )

    old_out, new_out = tmp_path / "old_cube.fits", tmp_path / "new_cube.fits"
    old.writeto = old.WR_3d_fits
    new.writeto = new.WR_3d_fits
    old.writeto(old_out)
    new.writeto(new_out)
    _assert_cube_equal(
        legacy.SNIFS_cube(fits3d_file=old_out),
        SNIFS_cube(fits3d_file=new_out),
    )


@pytest.mark.parametrize("variance", [False, True])
def test_e3d_read_and_modern_writer_roundtrip_matches_legacy(tmp_path, monkeypatch, variance):
    source = tmp_path / "source_e3d.fits"
    _write_e3d(source, variance=variance)
    old, new = legacy.SNIFS_cube(e3d_file=source), SNIFS_cube(e3d_file=source)
    _assert_cube_equal(old, new)

    # The legacy writer requires compatibility aliases removed from Astropy;
    # the private writer uses their supported modern equivalents directly.
    monkeypatch.setattr(fits, "new_table", fits.BinTableHDU.from_columns, raising=False)
    monkeypatch.setattr(fits, "TRUE", True, raising=False)
    monkeypatch.setattr(fits, "FALSE", False, raising=False)
    old_out, new_out = tmp_path / "old_e3d.fits", tmp_path / "new_e3d.fits"
    old.writeto = old.WR_e3d_file
    new.writeto = new.WR_e3d_file
    old.writeto(old_out)
    new.writeto(new_out)
    _assert_cube_equal(
        legacy.SNIFS_cube(e3d_file=old_out),
        SNIFS_cube(e3d_file=new_out),
    )
    with fits.open(new_out) as hdus:
        assert hdus[0].header["EURO3D"]
        assert ("STAT_SPE" in hdus[1].columns.names) is variance


def test_empty_model_cube_matches_legacy_and_rejects_nonlinear_wavelengths():
    wavelengths = np.array([4000.0, 4002.0, 4004.0])
    _assert_cube_equal(legacy.SNIFS_cube(lbda=wavelengths), SNIFS_cube(lbda=wavelengths))
    with pytest.raises(ValueError, match="not linear"):
        SNIFS_cube(lbda=[4000.0, 4002.0, 4005.0])
