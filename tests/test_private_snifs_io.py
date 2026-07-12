"""Parity contracts for the minimized private SNIFS I/O implementation."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

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
def test_regular_spectrum_construct_and_roundtrip(tmp_path, with_variance):
    data = np.array([3.5, -2.0, 8.25])
    variance = np.array([0.4, 0.5, 0.9]) if with_variance else None
    original = spectrum(data=data, var=variance, start=4100.0, step=2.5)

    path = tmp_path / "spectrum.fits"
    header = [["OBJECT", "synthetic"]]
    original.WR_fits_file(path, header_list=header)
    _assert_spectrum_equal(original, spectrum(data_file=path))
    assert fits.getheader(path)["OBJECT"] == "synthetic"
    with fits.open(path) as hdus:
        assert len(hdus) == (2 if with_variance else 1)


def test_irregular_and_zero_spectrum_construction():
    irregular = spectrum(x=[1.0, 1.7, 3.2])
    np.testing.assert_array_equal(irregular.x, [1.0, 1.7, 3.2])
    np.testing.assert_array_equal(irregular.data, np.zeros(3))
    zero = spectrum(nx=4)
    np.testing.assert_array_equal(zero.data, np.zeros(4))
    assert zero.len == 4


@pytest.mark.parametrize("variance", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_fits3d_read_slice_and_roundtrip(tmp_path, variance, sparse):
    source = tmp_path / "source.fits"
    _write_fits3d(source, variance=variance, sparse=sparse)
    cube = SNIFS_cube(fits3d_file=source)
    image = cube.slice2d(1, coord="p")
    np.testing.assert_array_equal(image[cube.j, cube.i], cube.data[1])
    assert cube.nlens == (8 if sparse else 9)

    output = tmp_path / "roundtrip_cube.fits"
    cube.writeto = cube.WR_3d_fits
    cube.writeto(output)
    restored = SNIFS_cube(fits3d_file=output)
    np.testing.assert_array_equal(restored.data, cube.data)
    if cube.var is None:
        assert restored.var is None
    else:
        np.testing.assert_array_equal(restored.var, cube.var)
    np.testing.assert_array_equal(restored.lbda, cube.lbda)
    # FITS3D output is normalized onto the standard 15x15 SNIFS grid.
    assert fits.getheader(output)["CRVAL1"] == pytest.approx(-7 * cube.spxSize)
    assert fits.getheader(output)["CRVAL2"] == pytest.approx(-7 * cube.spxSize)


@pytest.mark.parametrize("variance", [False, True])
def test_e3d_read_and_modern_writer_roundtrip(tmp_path, variance):
    source = tmp_path / "source_e3d.fits"
    _write_e3d(source, variance=variance)
    cube = SNIFS_cube(e3d_file=source)

    output = tmp_path / "roundtrip_e3d.fits"
    cube.writeto = cube.WR_e3d_file
    cube.writeto(output)
    _assert_cube_equal(cube, SNIFS_cube(e3d_file=output))
    with fits.open(output) as hdus:
        assert hdus[0].header["EURO3D"]
        assert ("STAT_SPE" in hdus[1].columns.names) is variance


def test_empty_model_cube_and_rejects_nonlinear_wavelengths():
    wavelengths = np.array([4000.0, 4002.0, 4004.0])
    cube = SNIFS_cube(lbda=wavelengths)
    np.testing.assert_array_equal(cube.lbda, wavelengths)
    assert (cube.nslice, cube.lstart, cube.lstep, cube.lend) == (
        3, 4000.0, 2.0, 4004.0
    )
    with pytest.raises(ValueError, match="not linear"):
        SNIFS_cube(lbda=[4000.0, 4002.0, 4005.0])
