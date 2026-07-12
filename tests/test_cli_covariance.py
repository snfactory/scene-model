"""Small CLI contract tests; full B/R runs live in offline validation."""

from __future__ import annotations

import runpy
import sys

import pytest

from scene_model.cli import main


def test_main_forwards_covariance_flag_and_restores_argv(monkeypatch):
    original = sys.argv
    observed = {}

    def fake_run_module(name, run_name):
        observed["name"] = name
        observed["run_name"] = run_name
        observed["argv"] = tuple(sys.argv)

    monkeypatch.setattr(runpy, "run_module", fake_run_module)
    main(["-V", "--psf", "fourier", "input.fits"])

    assert observed == {
        "name": "scripts.extract_star2",
        "run_name": "__main__",
        "argv": ("extract-star2", "-V", "--psf", "fourier", "input.fits"),
    }
    assert sys.argv is original


@pytest.mark.parametrize("psf", ["classic", "fourier"])
def test_covariance_flag_is_documented_for_both_psf_modes(psf, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help", "--psf", psf])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "-V" in help_text
    assert "--covariance" in help_text
    assert "--jacobian-backend" in help_text
    assert "--jax-wavelength-batching" in help_text
    assert "--no-jax-wavelength-batching" in help_text


def test_jax_wavelength_batching_help_describes_default(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "Use locked 128-wavelength JAX batches" in output
    assert "Disable JAX wavelength batching" in output
