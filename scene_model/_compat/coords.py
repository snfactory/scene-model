"""Active coordinate transforms from extract-star's ``ToolBox.Coords``."""

import numpy as np

RAD2DEG = 180.0 / np.pi


def rec2pol(x, y, deg=False):
    """Convert rectangular coordinates to polar coordinates."""

    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    if deg:
        theta *= RAD2DEG
    return radius, theta


def hadec2zdpar(ha, dec, phi=19.823056, deg=True):
    """Convert hour angle/declination to zenith distance/parallactic angle."""

    if deg:
        ha = ha / RAD2DEG
        dec = dec / RAD2DEG
        phi = phi / RAD2DEG

    cha, sha = np.cos(ha), np.sin(ha)
    cdec, sdec = np.cos(dec), np.sin(dec)
    cphi, sphi = np.cos(phi), np.sin(phi)

    sz_sp = cphi * sha
    sz_cp = sphi * cdec - cphi * cha * sdec
    cz = sphi * sdec + cphi * cha * cdec

    sz, parangle = rec2pol(sz_cp, sz_sp)
    radius, zd = rec2pol(cz, sz)
    assert np.allclose(radius, 1), "Precision error"

    if deg:
        zd *= RAD2DEG
        parangle *= RAD2DEG
    return zd, parangle


def altaz2hadec(alt, az, phi=19.823056, deg=True):
    """Convert altitude/azimuth to hour angle/declination."""

    if deg:
        alt = alt / RAD2DEG
        az = az / RAD2DEG
        phi = phi / RAD2DEG

    calt, salt = np.cos(alt), np.sin(alt)
    caz, saz = np.cos(az), np.sin(az)
    cphi, sphi = np.cos(phi), np.sin(phi)

    sdec = sphi * salt + cphi * calt * caz
    cdec_cha = cphi * salt - sphi * calt * caz
    cdec_sha = -calt * saz

    cdec, ha = rec2pol(cdec_cha, cdec_sha)
    radius, dec = rec2pol(cdec, sdec)
    assert np.allclose(radius, 1), "Precision error"

    if deg:
        ha *= RAD2DEG
        dec *= RAD2DEG
    return ha, dec
