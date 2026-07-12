"""Installed command-line interfaces for scene-model."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``extract-star2`` command.

    ``argv`` excludes the program name, following the convention used by
    :mod:`argparse`.  Passing ``None`` preserves normal console-script
    behavior.
    """
    old_argv = sys.argv
    if argv is not None:
        sys.argv = ["extract-star2", *argv]
    try:
        runpy.run_module("scripts.extract_star2", run_name="__main__")
    finally:
        sys.argv = old_argv
