from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

from scene_model import config
from scene_model.cli import main


def test_runtime_version_matches_distribution_metadata():
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )

    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "scene_model.config.__version__"
    }
    assert config.__version__ == "0.1.0"


def test_main_accepts_explicit_argv_and_restores_process_argv(capsys):
    original = sys.argv

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    assert sys.argv is original
    output = capsys.readouterr().out
    assert "Usage:" in output
    assert "PSF model" in output
