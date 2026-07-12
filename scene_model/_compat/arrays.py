"""Minimal array helpers derived from extract-star's ``ToolBox.Arrays``."""


def metaslice(alen, nmeta, trim=0, thickness=False):
    """Return centered metaslice boundaries ``[start, stop, step]``.

    ``nmeta`` is the number of slices unless ``thickness`` is true, in which
    case it is their requested thickness.  The list return type is retained
    for compatibility with the historical SNIFS cube implementation.
    """

    if alen <= 0 or nmeta <= 0 or trim < 0:
        raise ValueError(
            "Invalid input (alen=%d>0, nmeta=%d>0, trim=%d>=0)"
            % (alen, nmeta, trim)
        )
    if alen <= 2 * trim:
        raise ValueError("Trimmed array would be empty")

    if thickness:
        istep = nmeta
        nmeta = (alen - 2 * trim) // istep
        if nmeta == 0:
            raise ValueError("Metaslice thickness is too big")
    else:
        istep = (alen - 2 * trim) // nmeta

    if istep <= 0:
        raise ValueError("Null-thickness metaslices")

    imin = trim + ((alen - 2 * trim) % nmeta) // 2
    imax = imin + nmeta * istep
    return [imin, imax, istep]

