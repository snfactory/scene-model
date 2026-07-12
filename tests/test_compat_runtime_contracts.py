"""Black-box contracts for the SNIFS compatibility code actually in use."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from astropy.io import fits

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from compat_runtime_imports import (
    ADR,
    MPL,
    SNIFS_cube,
    altaz2hadec,
    evaluate_power_law,
    fit_power_law,
    hadec2zdpar,
    metaslice,
    spectrum,
    warning2stdout,
)


def test_metaslice_centering_thickness_and_validation():
    assert metaslice(15, 3, trim=2) == [3, 12, 3]
    assert metaslice(15, 3, trim=2, thickness=True) == [3, 12, 3]
    assert metaslice(14, 4) == [1, 13, 3]

    with pytest.raises(ValueError, match="Invalid input"):
        metaslice(0, 3)
    with pytest.raises(ValueError, match="Trimmed array would be empty"):
        metaslice(4, 2, trim=2)
    with pytest.raises(ValueError, match="Null-thickness"):
        metaslice(2, 3)


def test_coordinate_and_adr_numerical_contract():
    ha, dec = altaz2hadec(63.2, 127.5, phi=19.823056)
    assert ha == pytest.approx(-20.981074194828256, abs=1e-12)
    assert dec == pytest.approx(2.549244026029301, abs=1e-12)

    zd, parangle = hadec2zdpar(ha, dec, phi=19.823056)
    assert zd == pytest.approx(26.8, abs=1e-12)
    assert parangle == pytest.approx(-48.338243373692585, abs=1e-12)

    adr = ADR(616.0, 2.0, lref=5000.0, airmass=1.25, parangle=37.0)
    np.testing.assert_allclose(
        adr.get_scale([3500.0, 5000.0, 8000.0]),
        [-0.93785804, 0.0, 0.51489892],
        rtol=0,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        adr.refract([0.2], [-0.4], [3500.0, 5000.0, 8000.0]),
        [[-0.22331280, 0.2, 0.43240543],
         [0.16175505, -0.4, -0.70841242]],
        rtol=0,
        atol=5e-9,
    )
    assert adr.get_airmass() == pytest.approx(1.25)
    assert adr.get_parangle() == pytest.approx(37.0)


def test_fit_power_law_locks_parameters_and_residual_sign():
    x = np.linspace(0.72, 1.28, 17)
    y = evaluate_power_law([0.07, -1.3, 2.4], x)
    y += np.linspace(-0.001, 0.001, len(x))

    fitted = fit_power_law(x, y, deg=2)
    np.testing.assert_allclose(
        fitted,
        [0.07258208, -1.29846171, 2.40000606],
        rtol=0,
        atol=5e-9,
    )
    residual = evaluate_power_law(fitted, x) - y
    np.testing.assert_allclose(
        residual[[0, 8, -1]],
        [1.78720351e-05, 6.05934799e-06, -1.86455741e-05],
        rtol=0,
        atol=5e-12,
    )
    # The optimizer's convention is model minus data.
    assert abs(residual.sum()) < 2e-7


def test_plotting_colors_and_errorband_contract():
    assert [MPL.blue, MPL.red, MPL.green, MPL.orange] == [
        "#377EB8", "#E41A1C", "#4DAF4A", "#FF7F00"
    ]
    assert [MPL.purple, MPL.yellow, MPL.brown] == [
        "#984EA3", "#FFFF33", "#A65628"
    ]

    fig, ax = plt.subplots()
    assert hasattr(ax, "errorband")
    poly = ax.errorband(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([0.2, 0.3, 0.4]),
        color=MPL.green,
        alpha=0.25,
        label="uncertainty",
    )
    np.testing.assert_allclose(
        poly.get_xy()[:6],
        [[1, 4.2], [2, 5.3], [3, 6.4], [3, 5.6], [2, 4.7], [1, 3.8]],
    )
    assert poly.get_label() == "uncertainty"
    assert poly.get_alpha() == pytest.approx(0.25)
    plt.close(fig)


def test_cli_warning_formatter_writes_prefixed_warning_to_stdout(capsys):
    warning2stdout(UserWarning("check fixture"), UserWarning, "cube.py", 17)
    output = capsys.readouterr().out
    assert output == "WARNING: cube.py:17: UserWarning: check fixture\n"


def test_spectrum_construct_write_and_read_roundtrip(tmp_path):
    original = spectrum(
        data=np.array([3.5, -2.0, 8.25]),
        var=np.array([0.4, 0.5, 0.9]),
        start=4100.0,
        step=2.5,
    )
    assert original.has_var
    np.testing.assert_array_equal(original.x, [4100.0, 4102.5, 4105.0])

    path = tmp_path / "spectrum.fits"
    original.WR_fits_file(path, header_list=[["OBJECT", "synthetic"]])
    restored = spectrum(data_file=path)

    np.testing.assert_array_equal(restored.data, original.data)
    np.testing.assert_array_equal(restored.var, original.var)
    np.testing.assert_array_equal(restored.x, original.x)
    assert (restored.start, restored.step, restored.len, restored.has_var) == (
        4100.0, 2.5, 3, True
    )
    assert fits.getheader(path)["OBJECT"] == "synthetic"


def _write_synthetic_fits3d(path):
    data = np.arange(24.0).reshape(4, 2, 3)
    variance = data + 100.0
    header = fits.Header()
    header["CRVAL1"] = -0.43
    header["CRVAL2"] = -0.43
    header["CRVAL3"] = 5000.0
    header["CDELT1"] = 0.43
    header["CDELT2"] = 0.43
    header["CDELT3"] = 2.0
    fits.HDUList([
        fits.PrimaryHDU(data, header=header),
        fits.ImageHDU(variance, name="VARIANCE"),
    ]).writeto(path)
    return data, variance


def test_fits3d_cube_construct_slice_and_dynamic_writer_roundtrip(tmp_path):
    source = tmp_path / "source.fits"
    _write_synthetic_fits3d(source)
    cube = SNIFS_cube(fits3d_file=source)

    assert (cube.nslice, cube.nlens, cube.lstart, cube.lstep, cube.lend) == (
        4, 6, 5000.0, 2.0, 5006.0
    )
    np.testing.assert_array_equal(cube.lbda, [5000.0, 5002.0, 5004.0, 5006.0])
    np.testing.assert_array_equal(cube.spec(ind=0), cube.data[:, 0])
    image = cube.slice2d(1, coord="p", nx=3, ny=3)
    np.testing.assert_array_equal(image[cube.j, cube.i], cube.data[1])

    # subtract_psf2 dynamically selects the writer based on the detected cube.
    cube.writeto = cube.WR_3d_fits
    output = tmp_path / "roundtrip.fits"
    cube.writeto(output)
    restored = SNIFS_cube(fits3d_file=output)
    np.testing.assert_array_equal(restored.data, cube.data)
    np.testing.assert_array_equal(restored.var, cube.var)
    np.testing.assert_array_equal(restored.i, cube.i)
    np.testing.assert_array_equal(restored.j, cube.j)


def _write_synthetic_e3d(path):
    data = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    var = data + 10
    columns = [
        fits.Column(name="SPEC_ID", format="J", array=[11, 22]),
        fits.Column(name="SPEC_LEN", format="J", array=[4, 4]),
        fits.Column(name="SPEC_STA", format="J", array=[0, 0]),
        fits.Column(name="XPOS", format="E", array=[-0.43, 0.0]),
        fits.Column(name="YPOS", format="E", array=[0.0, 0.43]),
        fits.Column(name="DATA_SPE", format="4D", array=data),
        fits.Column(name="STAT_SPE", format="4D", array=var),
    ]
    data_hdu = fits.BinTableHDU.from_columns(columns, name="E3D_DATA")
    data_hdu.header["CRVALS"] = 4000.0
    data_hdu.header["CDELTS"] = 2.0
    data_hdu.header["CHANNEL"] = "B"
    group_hdu = fits.BinTableHDU.from_columns([], name="E3D_GRP")
    primary = fits.PrimaryHDU()
    primary.header["EURO3D"] = True
    fits.HDUList([primary, data_hdu, group_hdu]).writeto(path)
    return data, var


def test_e3d_cube_construct_and_dynamic_writer_roundtrip(tmp_path, monkeypatch):
    source = tmp_path / "source_e3d.fits"
    expected_data, expected_var = _write_synthetic_e3d(source)
    cube = SNIFS_cube(e3d_file=source)

    assert cube.from_e3d_file
    assert (cube.nslice, cube.nlens, cube.lstart, cube.lstep) == (4, 2, 4000., 2.)
    np.testing.assert_array_equal(cube.data, expected_data.T)
    np.testing.assert_array_equal(cube.var, expected_var.T)
    np.testing.assert_array_equal(cube.no, [11, 22])

    cube.writeto = cube.WR_e3d_file
    output = tmp_path / "roundtrip_e3d.fits"
    # The vendored writer still calls the removed astropy ``new_table`` API.
    # Supply its exact modern equivalent so this contract reaches and locks
    # the writer's data behavior; the minimization must remove this shim.
    monkeypatch.setattr(
        fits, "new_table", fits.BinTableHDU.from_columns, raising=False
    )
    monkeypatch.setattr(fits, "TRUE", True, raising=False)
    monkeypatch.setattr(fits, "FALSE", False, raising=False)
    cube.writeto(output)
    restored = SNIFS_cube(e3d_file=output)
    np.testing.assert_array_equal(restored.data, cube.data)
    np.testing.assert_array_equal(restored.var, cube.var)
    np.testing.assert_array_equal(restored.no, cube.no)
