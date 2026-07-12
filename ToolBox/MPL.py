"""Small plotting subset required by :mod:`scene_model.snifs`.

The color values match the Python 3 extract-star port.  Importing this module
also installs ``Axes.errorband``, matching the historical ToolBox side effect.
"""

import numpy as np
from matplotlib.axes import Axes

from .Misc import make_method

blue = "#377EB8"
red = "#E41A1C"
green = "#4DAF4A"
orange = "#FF7F00"
purple = "#984EA3"
yellow = "#FFFF33"
brown = "#A65628"


@make_method(Axes)
def errorband(ax, x, y, dy, color="b", alpha=0.3, label="_", **kwargs):
    """Plot values with a symmetric or asymmetric filled error band."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if not len(x):
        return None

    dy = np.asarray(dy)
    if dy.ndim == 2:
        dym, dyp = dy
    else:
        dym, dyp = -dy, dy
    xp = np.concatenate((x, x[::-1]))
    yp = np.concatenate(((y + dyp), (y + dym)[::-1]))
    poly, = ax.fill(
        xp,
        yp,
        alpha=alpha,
        label=label,
        fc=kwargs.pop("fc", color),
        ec=kwargs.pop("ec", color),
        zorder=kwargs.pop("zorder", 2),
        **kwargs,
    )
    return poly
