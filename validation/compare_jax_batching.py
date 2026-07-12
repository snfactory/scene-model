#!/usr/bin/env python3
"""Compare unbatched and locked-batch JAX results on external E3D cubes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from benchmark_jax_scene_covariance import _fit, _write_report
from scene_model.jax_scene import PRODUCTION_WAVELENGTH_BATCH


def _relative_frobenius(first, second):
    scale = max(np.linalg.norm(first), np.finfo(float).tiny)
    return float(np.linalg.norm(second - first) / scale)


def _worker(path, channel, psf):
    with redirect_stdout(sys.stderr):
        unbatched = _fit(path, psf, covariance=True, batched=False)
    flux = np.array(unbatched.point_source_spectrum.data, copy=True)
    jacobian = np.array(unbatched.covariance_diagnostics["jacobian"], copy=True)
    covariance = np.array(unbatched.point_source_spectrum.cov, copy=True)
    del unbatched
    gc.collect()

    with redirect_stdout(sys.stderr):
        batched = _fit(path, psf, covariance=True, batched=True)
    batched_flux = np.asarray(batched.point_source_spectrum.data)
    batched_jacobian = np.asarray(batched.covariance_diagnostics["jacobian"])
    batched_covariance = np.asarray(batched.point_source_spectrum.cov)

    return {
        "channel": channel,
        "psf": psf,
        "fixture": str(Path(path).resolve()),
        "wavelength_count": int(len(flux)),
        "jax_wavelength_batch": PRODUCTION_WAVELENGTH_BATCH,
        "accepted_flux_exact": bool(np.array_equal(flux, batched_flux)),
        "jacobian_maximum_absolute_difference": float(
            np.max(np.abs(batched_jacobian - jacobian))
        ),
        "jacobian_relative_frobenius": _relative_frobenius(
            jacobian, batched_jacobian
        ),
        "covariance_maximum_absolute_difference": float(
            np.max(np.abs(batched_covariance - covariance))
        ),
        "covariance_relative_frobenius": _relative_frobenius(
            covariance, batched_covariance
        ),
    }


def _run_child(script, path, channel, psf):
    completed = subprocess.run([
        sys.executable, str(script), "--worker", "--cube", str(path),
        "--channel", channel, "--psf", psf,
    ], check=False, text=True, stdout=subprocess.PIPE, stderr=sys.stderr)
    if completed.returncode:
        raise RuntimeError("Batch comparison failed for %s/%s" %
                           (channel, psf))
    return json.loads(completed.stdout)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--blue")
    parser.add_argument("--red")
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--cube", help=argparse.SUPPRESS)
    parser.add_argument("--channel", choices=("B", "R"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--psf", choices=("classic", "fourier"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(_worker(args.cube, args.channel, args.psf),
                         sort_keys=True))
        return 0
    if not all((args.blue, args.red, args.output)):
        raise ValueError("Parent requires blue, red, and output paths")
    script = Path(__file__).resolve()
    cases = [
        _run_child(script, path, channel, psf)
        for channel, path in (("B", args.blue), ("R", args.red))
        for psf in ("classic", "fourier")
    ]
    _write_report(args.output, {
        "format": "SNIFS-JAX-BATCH-COMPARISON-1.0",
        "jax_wavelength_batch": PRODUCTION_WAVELENGTH_BATCH,
        "cases": cases,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
