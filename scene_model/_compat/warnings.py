"""Warning-output compatibility for the extract-star CLI."""


def warning2stdout(message, category, filename, lineno, file=None, line=None):
    """Format a warning and write it to stdout rather than stderr."""

    import sys
    import warnings

    sys.stdout.write(
        "WARNING: " + warnings.formatwarning(message, category, filename, lineno)
    )
