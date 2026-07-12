"""Parity tests for the reduced private ToolBox runtime."""

import re
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ToolBox import Atmosphere as legacy_atmosphere
from ToolBox import MPL as legacy_plotting
from ToolBox.Arrays import metaslice as legacy_metaslice
from ToolBox.Astro import Coords as legacy_coords
from ToolBox.Misc import warning2stdout as legacy_warning2stdout
from scene_model._compat import atmosphere, plotting
from scene_model._compat.arrays import metaslice
from scene_model._compat import coords
from scene_model._compat.warnings import warning2stdout


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((141, 3), {"trim": 2}),
        ((225, 12), {"trim": 10}),
        ((225, 17), {"trim": 0}),
        ((141, 9), {"trim": 2, "thickness": True}),
        ((10, 3), {"thickness": True}),
    ],
)
def test_metaslice_matches_legacy(args, kwargs):
    assert metaslice(*args, **kwargs) == legacy_metaslice(*args, **kwargs)


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((0, 1), {}),
        ((10, 0), {}),
        ((10, 2), {"trim": -1}),
        ((10, 2), {"trim": 5}),
        ((4, 8), {}),
        ((4, 8), {"thickness": True}),
    ],
)
def test_metaslice_errors_match_legacy(args, kwargs):
    with pytest.raises(Exception) as legacy_error:
        legacy_metaslice(*args, **kwargs)
    with pytest.raises(
        type(legacy_error.value), match=re.escape(str(legacy_error.value))
    ):
        metaslice(*args, **kwargs)


@pytest.mark.parametrize(
    "x, y, deg",
    [
        (3.0, 4.0, False),
        (np.array([-2.0, 0.0, 3.0]), np.array([1.0, -4.0, 0.0]), False),
        (np.array([-2.0, 0.0, 3.0]), np.array([1.0, -4.0, 0.0]), True),
    ],
)
def test_rec2pol_matches_legacy(x, y, deg):
    actual = coords.rec2pol(x, y, deg=deg)
    expected = legacy_coords.rec2pol(x, y, deg=deg)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("deg", [True, False])
def test_coordinate_transforms_match_legacy_for_scalars(deg):
    if deg:
        alt, az, phi = 62.4, 127.3, 19.823056
    else:
        alt, az, phi = np.deg2rad([62.4, 127.3, 19.823056])

    actual_hadec = coords.altaz2hadec(alt, az, phi=phi, deg=deg)
    expected_hadec = legacy_coords.altaz2hadec(alt, az, phi=phi, deg=deg)
    np.testing.assert_array_equal(actual_hadec, expected_hadec)

    actual_zdpar = coords.hadec2zdpar(*actual_hadec, phi=phi, deg=deg)
    expected_zdpar = legacy_coords.hadec2zdpar(*expected_hadec, phi=phi, deg=deg)
    np.testing.assert_array_equal(actual_zdpar, expected_zdpar)


def test_coordinate_transforms_match_legacy_for_vectors():
    alt = np.array([35.0, 55.0, 75.0])
    az = np.array([0.0, 90.0, 270.0])
    actual_hadec = coords.altaz2hadec(alt, az)
    expected_hadec = legacy_coords.altaz2hadec(alt, az)
    np.testing.assert_array_equal(actual_hadec, expected_hadec)

    actual_zdpar = coords.hadec2zdpar(*actual_hadec)
    expected_zdpar = legacy_coords.hadec2zdpar(*expected_hadec)
    np.testing.assert_array_equal(actual_zdpar, expected_zdpar)


@pytest.mark.parametrize("temperature", [-15.0, 0.0, 2.0, np.array([-8.0, 3.0])])
def test_saturation_vapor_pressure_matches_legacy(temperature):
    expected = legacy_atmosphere.saturationVaporPressure(temperature)
    actual = atmosphere.saturation_vapor_pressure(temperature)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("humidity", [0, 25, 100])
def test_refractive_index_matches_legacy_for_scalar_and_vector(humidity):
    for wavelength in (5000.0, np.array([3300.0, 5000.0, 9200.0])):
        expected = legacy_atmosphere.refractiveIndexMEdlen(
            wavelength, P=615.0, T=4.0, RH=humidity
        )
        actual = atmosphere.refractive_index(
            wavelength, pressure=615.0, temperature=4.0, humidity=humidity
        )
        np.testing.assert_array_equal(actual, expected)


def _legacy_and_private_adr(**kwargs):
    return legacy_atmosphere.ADR(**kwargs), atmosphere.ADR(**kwargs)


def test_adr_initialization_and_parameter_forms_match_legacy():
    for params in (
        {"delta": 0.8, "theta": 0.3},
        {"airmass": 1.35, "parangle": 42.0},
        {"zd": 35.0, "parangle": -80.0},
    ):
        legacy, private = _legacy_and_private_adr(
            P=612.0, T=3.0, RH=35, lref=5100.0, **params
        )
        for attribute in ("P", "T", "RH", "lref", "nref", "isSet"):
            np.testing.assert_array_equal(
                getattr(private, attribute), getattr(legacy, attribute)
            )
        np.testing.assert_array_equal(private.delta, legacy.delta)
        np.testing.assert_array_equal(private.theta, legacy.theta)
        np.testing.assert_array_equal(private.get_airmass(), legacy.get_airmass())
        np.testing.assert_array_equal(private.get_parangle(), legacy.get_parangle())


def test_adr_scale_and_forward_backward_refraction_match_legacy():
    legacy, private = _legacy_and_private_adr(
        P=617.0, T=2.0, RH=40, lref=5000.0, zd=38.0, parangle=27.0
    )
    wavelengths = np.array([3400.0, 5000.0, 8700.0])
    np.testing.assert_array_equal(private.get_scale(wavelengths), legacy.get_scale(wavelengths))
    np.testing.assert_array_equal(private.get_scale(6400.0), legacy.get_scale(6400.0))

    x = np.array([-1.0, 0.25])
    y = np.array([0.5, 1.75])
    np.testing.assert_array_equal(
        private.refract(x, y, wavelengths, unit=0.43),
        legacy.refract(x, y, wavelengths, unit=0.43),
    )

    x_back = np.array([-1.0, 0.25, 2.0])
    y_back = np.array([0.5, 1.75, -0.4])
    np.testing.assert_array_equal(
        private.refract(x_back, y_back, wavelengths, backward=True),
        legacy.refract(x_back, y_back, wavelengths, backward=True),
    )


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"P": 500.0}, AssertionError, "Non-std pressure"),
        ({"T": 30.0}, AssertionError, "temperature"),
        ({"banana": 1.0}, ValueError, "Unknown parameter 'banana'"),
    ],
)
def test_adr_constructor_errors_match_legacy(kwargs, error_type, message):
    with pytest.raises(error_type, match=message):
        legacy_atmosphere.ADR(**kwargs)
    with pytest.raises(error_type, match=message):
        atmosphere.ADR(**kwargs)


def test_adr_runtime_errors_match_legacy():
    legacy, private = _legacy_and_private_adr()
    with pytest.raises(AssertionError, match="not yet initialized"):
        legacy.refract(0.0, 0.0, 5000.0)
    with pytest.raises(AssertionError, match="not yet initialized"):
        private.refract(0.0, 0.0, 5000.0)

    legacy.set_param(zd=30.0, parangle=10.0)
    private.set_param(zd=30.0, parangle=10.0)
    with pytest.raises(AssertionError, match="Incompatible x and y"):
        legacy.refract([0.0, 1.0], [0.0], 5000.0)
    with pytest.raises(AssertionError, match="Incompatible x and y"):
        private.refract([0.0, 1.0], [0.0], 5000.0)
    with pytest.raises(AssertionError, match="Incompatible x,y and lbda"):
        legacy.refract([0.0], [0.0], [4000.0, 5000.0], backward=True)
    with pytest.raises(AssertionError, match="Incompatible x,y and lbda"):
        private.refract([0.0], [0.0], [4000.0, 5000.0], backward=True)


def _polygon_geometry(function, dy):
    fig, ax = plt.subplots()
    x = np.array([1.0, 2.0, 4.0])
    y = np.array([3.0, 5.0, 4.0])
    polygon = function(
        ax,
        x,
        y,
        dy,
        color="#123456",
        alpha=0.25,
        label="range",
        hatch="/",
    )
    geometry = (
        polygon.get_xy().copy(),
        polygon.get_facecolor(),
        polygon.get_edgecolor(),
        polygon.get_alpha(),
        polygon.get_label(),
        polygon.get_zorder(),
        polygon.get_hatch(),
    )
    plt.close(fig)
    return geometry


@pytest.mark.parametrize(
    "dy",
    [np.array([0.1, 0.2, 0.4]), np.array([[0.1, 0.2, 0.4], [0.3, 0.5, 0.8]])],
)
def test_errorband_geometry_matches_legacy(dy):
    actual = _polygon_geometry(plotting.errorband, dy)
    expected = _polygon_geometry(legacy_plotting.errorband, dy)
    for actual_value, expected_value in zip(actual, expected):
        if isinstance(actual_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_plotting_colors_and_axes_patch_match_legacy():
    for color in ("blue", "red", "green", "orange", "purple", "yellow", "brown"):
        assert getattr(plotting, color) == getattr(legacy_plotting, color)
    assert Axes.errorband is plotting.errorband
    fig, ax = plt.subplots()
    assert ax.errorband([], [], []) is None
    with pytest.raises(ValueError, match="x and y must have equal length"):
        ax.errorband([1], [], [])
    plt.close(fig)


def test_warning2stdout_matches_legacy(capsys):
    with warnings.catch_warnings():
        legacy_warning2stdout("old", UserWarning, "fixture.py", 12)
        legacy_output = capsys.readouterr().out
        warning2stdout("old", UserWarning, "fixture.py", 12)
        private_output = capsys.readouterr().out
    assert private_output == legacy_output
    assert private_output.startswith("WARNING: fixture.py:12: UserWarning: old")
