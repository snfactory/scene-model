"""Minimal atmospheric differential-refraction model.

Derived from the public MIT-licensed extract-star ``ToolBox.Atmosphere``
implementation.  Presentation and standalone diagnostic methods that are not
reachable from scene-model are intentionally omitted.
"""

import numpy as np

RAD2DEG = 57.295779513082323
RAD2ARC = 206264.80624709636


def _saturation_vapor_pressure_over_water(temperature):
    k1 = 1.16705214528e3
    k2 = -7.24213167032e5
    k3 = -1.70738469401e1
    k4 = 1.20208247025e4
    k5 = -3.23255503223e6
    k6 = 1.49151086135e1
    k7 = -4.82326573616e3
    k8 = 4.05113405421e5
    k9 = -2.38555575678e-1
    k10 = 6.50175348448e2

    kelvin = temperature + 273.15
    x = kelvin + k9 / (kelvin - k10)
    a = x**2 + k1 * x + k2
    b = k3 * x**2 + k4 * x + k5
    c = k6 * x**2 + k7 * x + k8
    root = -b + np.sqrt(b**2 - 4 * a * c)
    return 1e6 * (2 * c / root) ** 4


def _saturation_vapor_pressure_over_ice(temperature):
    a1 = -13.928169
    a2 = 34.7078238
    theta = (temperature + 273.15) / 273.16
    exponent = a1 * (1 - theta**-1.5) + a2 * (1 - theta**-1.25)
    return 611.657 * np.exp(exponent)


def saturation_vapor_pressure(temperature=2.0):
    """Return saturation vapor pressure in Pa at Celsius temperature."""

    values = np.atleast_1d(temperature)
    return np.where(
        values >= 0,
        _saturation_vapor_pressure_over_water(values),
        _saturation_vapor_pressure_over_ice(values),
    )


def refractive_index(lbda, pressure=617.0, temperature=2.0, humidity=0):
    """Return the modified-Edlen refractive index of air."""

    a = 8342.54
    b = 2406147.0
    c = 15998.0
    d = 96095.43
    e = 0.601
    f = 0.00972
    g = 0.003661

    inverse_micron_sq = (lbda * 1e-4) ** -2
    standard = 1e-6 * (
        a + b / (130.0 - inverse_micron_sq) + c / (38.9 - inverse_micron_sq)
    )
    correction = (1.0 + 1e-6 * (e - f * temperature) * pressure) / (
        1.0 + g * temperature
    )
    index = 1.0 + pressure * standard * correction / d

    if humidity:
        vapor_pressure = humidity / 100.0 * saturation_vapor_pressure(temperature)
        index -= (
            1e-10
            * (292.75 / (temperature + 273.15))
            * (3.7345 - 0.0401 * inverse_micron_sq)
            * vapor_pressure
        )
    return index


class ADR:
    """Atmospheric differential refraction for SNIFS conditions."""

    def __init__(self, P=617.0, T=2.0, RH=0, **kwargs):
        assert 550 < P < 650 and -20 < T < 20, (
            "Non-std pressure (%.0f mbar) or temperature (%.0f°C)" % (P, T)
        )
        self.P = P
        self.T = T
        self.RH = RH
        self.set_ref(kwargs.pop("lref", 5000.0))
        self.set_param(**kwargs)

    def set_ref(self, lref=5000.0):
        """Set the reference wavelength in Angstrom."""

        self.lref = lref
        self.nref = refractive_index(
            self.lref,
            pressure=self.P,
            temperature=self.T,
            humidity=self.RH,
        )

    def set_param(self, **kwargs):
        """Set refraction magnitude and orientation parameters."""

        for key, value in kwargs.items():
            if key == "delta":
                self.delta = value
            elif key == "theta":
                self.theta = value
            elif key == "airmass":
                self.delta = np.tan(np.arccos(1.0 / value))
            elif key == "parangle":
                self.theta = value / RAD2DEG
            elif key == "zd":
                self.delta = np.tan(value / RAD2DEG)
            else:
                raise ValueError("Unknown parameter '%s'" % key)
        self.isSet = hasattr(self, "delta") and hasattr(self, "theta")

    def get_scale(self, lbda, **kwargs):
        """Return differential-refraction scale in arcseconds."""

        if kwargs:
            self.set_param(**kwargs)
        lbda = np.atleast_1d(lbda)
        index = refractive_index(
            lbda,
            pressure=self.P,
            temperature=self.T,
            humidity=self.RH,
        )
        return (index**-2 - self.nref**-2) * 0.5 * RAD2ARC

    def refract(self, x, y, lbda, backward=False, unit=1.0, **kwargs):
        """Apply forward or backward atmospheric differential refraction."""

        x0 = np.atleast_1d(x)
        y0 = np.atleast_1d(y)
        assert len(x0) == len(y0), "Incompatible x and y vectors."
        npos = len(x0)
        assert self.isSet, "ADR parameters are not yet initialized."

        displacement = self.delta * self.get_scale(lbda, **kwargs) / unit
        if backward:
            nlbda = len(np.atleast_1d(lbda))
            assert npos == nlbda, "Incompatible x,y and lbda vectors."
            xout = x0 - displacement * np.sin(self.theta)
            yout = y0 + displacement * np.cos(self.theta)
            output = np.vstack((xout, yout))
        else:
            displacement = displacement[:, np.newaxis]
            xout = x0 + displacement * np.sin(self.theta)
            yout = y0 - displacement * np.cos(self.theta)
            output = np.dstack((xout.T, yout.T)).T
        return output.squeeze()

    def get_airmass(self, delta=None):
        """Return plane-parallel airmass."""

        value = self.delta if delta is None else delta
        return 1 / np.cos(np.arctan(value))

    def get_parangle(self, theta=None):
        """Return parallactic angle in degrees."""

        return RAD2DEG * (self.theta if theta is None else theta)

