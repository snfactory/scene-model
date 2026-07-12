from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
import numpy as np

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


def test_jax_runtime_and_cuda_profile_are_exactly_pinned():
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )

    assert metadata["project"]["dependencies"].count("jax==0.10.2") == 1
    assert metadata["project"]["optional-dependencies"]["cuda13"] == [
        "jax[cuda13]==0.10.2; sys_platform == 'linux' and "
        "platform_machine == 'x86_64'"
    ]


def test_main_accepts_explicit_argv_and_restores_process_argv(capsys):
    original = sys.argv

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    assert sys.argv is original
    output = capsys.readouterr().out
    assert "Usage:" in output
    assert "PSF model" in output


def test_usage_error_does_not_require_legacy_runtime(capsys):
    original = sys.argv

    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2
    assert sys.argv is original
    assert "No input datacube specified" in capsys.readouterr().err


def test_vendored_scene_runtime_is_importable_and_python3_compatible(tmp_path):
    from ToolBox.Arrays import rebin
    from ToolBox.Astro.Coords import ten
    from ToolBox.Misc import add_attrs
    from pySNIFS import spectrum

    assert rebin(np.arange(16).reshape(4, 4), (2, 2)).shape == (2, 2)
    assert ten("-0:23:34") == pytest.approx(-0.3927777777777778)

    @add_attrs(marker="ok")
    def decorated():
        return None

    assert decorated.marker == "ok"

    output = tmp_path / "spectrum.fits"
    spectrum(
        data=np.arange(3.0), var=np.ones(3), start=1.0, step=2.0
    ).WR_fits_file(output)
    assert output.exists()
