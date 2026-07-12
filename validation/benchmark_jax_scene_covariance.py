#!/usr/bin/env python3
"""Locked cold/warm runtime and RSS validation for JAX scene covariance.

Each channel/PSF case runs in a fresh child process.  This keeps JAX compiler
caches and ``ru_maxrss`` accounting from leaking between release-gate cases.
The external E3D inputs are identified by resolved path and SHA-256 in the
machine-readable report; science fixtures are never copied into this repo.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np

from scene_model.snifs import SnifsCubeFitter
from scene_model.jax_scene import PRODUCTION_WAVELENGTH_BATCH


COLD_RUNTIME_LIMIT = 5.0
WARM_RUNTIME_TARGET = 3.0
PEAK_RSS_LIMIT_BYTES = 4 * 1024**3


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _fit(path, psf, covariance, batched=False):
    fitter = SnifsCubeFitter(
        str(path), psf=psf, background_degree=0,
        subsampling=3, border=15, least_squares=False,
        filter_variance=False, prior_scale=0.0, verbosity=-1,
    )
    fitter.fit_metaslices_2d(num_meta_slices=12)
    fitter.fit_metaslices_3d()
    fitter.check_validity()
    fitter.extract(
        method="psf", covariance=covariance,
        jacobian_backend="jax" if covariance else "finite-difference",
        jax_wavelength_batch=(
            PRODUCTION_WAVELENGTH_BATCH if covariance and batched else None
        ),
    )
    return fitter


def _timed_fit(path, psf, covariance, batched=False):
    started = time.perf_counter()
    # Legacy scene-model reports progress with print().  Keep worker stdout a
    # single JSON document so the parent cannot accidentally parse log text.
    with redirect_stdout(sys.stderr):
        fitter = _fit(path, psf, covariance, batched=batched)
    return fitter, time.perf_counter() - started


def _validate_covariance(fitter):
    flux = np.asarray(fitter.point_source_spectrum.data)
    variance = np.asarray(fitter.point_source_spectrum.var)
    covariance = np.asarray(fitter.point_source_spectrum.cov)
    if covariance.shape != (len(flux), len(flux)):
        raise RuntimeError("Flux covariance has inconsistent axes")
    if not np.all(np.isfinite(covariance)):
        raise RuntimeError("Flux covariance is non-finite")
    if not np.array_equal(covariance, covariance.T):
        raise RuntimeError("Flux covariance is not exactly symmetric")
    if not np.array_equal(variance, np.diag(covariance)):
        raise RuntimeError("VARIANCE is not exactly diag(COVAR)")
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(eigenvalues[0]) < -1e-10 * scale:
        raise RuntimeError("Flux covariance is materially non-PSD")
    diagnostics = fitter.covariance_diagnostics
    derivatives = diagnostics["derivatives"]
    if derivatives.backend != "jax":
        raise RuntimeError("Covariance extraction did not use JAX")
    provenance = dict(derivatives.provenance)
    required = {"jax", "jaxlib", "backend", "device", "architecture", "x64"}
    if not required <= provenance.keys() or provenance["x64"] != "true":
        raise RuntimeError("JAX provenance is incomplete")
    return provenance


def _worker(path, channel, psf, batched=False):
    legacy, legacy_seconds = _timed_fit(path, psf, covariance=False)
    phase_rss = {"legacy_fit": _peak_rss_bytes()}
    legacy_flux = np.array(legacy.point_source_spectrum.data, copy=True)
    del legacy
    gc.collect()

    cold, cold_seconds = _timed_fit(
        path, psf, covariance=True, batched=batched
    )
    phase_rss["jax_cold_fit"] = _peak_rss_bytes()
    cold_flux = np.array(cold.point_source_spectrum.data, copy=True)
    cold_provenance = _validate_covariance(cold)
    phase_rss["jax_cold_validation"] = _peak_rss_bytes()
    del cold
    gc.collect()

    warm, warm_seconds = _timed_fit(
        path, psf, covariance=True, batched=batched
    )
    phase_rss["jax_warm_fit"] = _peak_rss_bytes()
    warm_flux = np.array(warm.point_source_spectrum.data, copy=True)
    warm_provenance = _validate_covariance(warm)
    phase_rss["jax_warm_validation"] = _peak_rss_bytes()
    del warm
    gc.collect()
    flux_parity = (np.array_equal(legacy_flux, cold_flux)
                   and np.array_equal(legacy_flux, warm_flux))
    provenance_parity = cold_provenance == warm_provenance
    cold_ratio = cold_seconds / legacy_seconds
    warm_ratio = warm_seconds / legacy_seconds
    peak_rss = _peak_rss_bytes()
    gates = {
        "cold_runtime": cold_ratio < COLD_RUNTIME_LIMIT,
        "peak_rss": peak_rss < PEAK_RSS_LIMIT_BYTES,
        "flux_parity": flux_parity,
        "provenance_parity": provenance_parity,
    }
    return {
        "channel": channel,
        "psf": psf,
        "jax_wavelength_batch": (
            PRODUCTION_WAVELENGTH_BATCH if batched else None
        ),
        "fixture": {
            "path": str(Path(path).resolve()),
            "sha256": _sha256(path),
            "wavelength_count": int(len(legacy_flux)),
        },
        "seconds": {
            "covariance_off": legacy_seconds,
            "jax_cold": cold_seconds,
            "jax_warm": warm_seconds,
        },
        "ratios": {
            "jax_cold_to_off": cold_ratio,
            "jax_warm_to_off": warm_ratio,
        },
        "peak_rss_bytes": peak_rss,
        "peak_rss_by_phase": phase_rss,
        "jax_provenance": cold_provenance,
        "warm_runtime_target_met": warm_ratio < WARM_RUNTIME_TARGET,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _run_child(script, path, channel, psf, batched=False):
    command = [
        sys.executable, str(script), "--worker", "--cube", str(path),
        "--channel", channel, "--psf", psf,
    ]
    if batched:
        command.append("--batched")
    completed = subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE,
        stderr=sys.stderr,
    )
    if completed.returncode:
        raise RuntimeError(
            "Benchmark child failed for %s/%s with status %d" %
            (channel, psf, completed.returncode)
        )
    return json.loads(completed.stdout)


def _write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run locked JAX covariance runtime/RSS gates"
    )
    parser.add_argument("--blue", help="External B-channel E3D fixture")
    parser.add_argument("--red", help="External R-channel E3D fixture")
    parser.add_argument("--output", help="JSON report path")
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--cube", help=argparse.SUPPRESS)
    parser.add_argument("--channel", choices=("B", "R"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--psf", choices=("classic", "fourier"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--batched", action="store_true",
                        help="Use the locked 128-wavelength JAX batch")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker:
        if not all((args.cube, args.channel, args.psf)):
            raise ValueError("Worker requires cube, channel, and PSF")
        print(json.dumps(_worker(
            args.cube, args.channel, args.psf, batched=args.batched),
                         sort_keys=True))
        return 0
    if not all((args.blue, args.red, args.output)):
        raise ValueError("Parent requires blue, red, and output paths")

    script = Path(__file__).resolve()
    cases = []
    for channel, path in (("B", args.blue), ("R", args.red)):
        for psf in ("classic", "fourier"):
            cases.append(_run_child(
                script, path, channel, psf, batched=args.batched
            ))
    payload = {
        "format": "SNIFS-JAX-SCENE-BENCHMARK-1.0",
        "limits": {
            "cold_runtime_ratio_exclusive": COLD_RUNTIME_LIMIT,
            "warm_runtime_ratio_target_exclusive": WARM_RUNTIME_TARGET,
            "peak_rss_bytes_exclusive": PEAK_RSS_LIMIT_BYTES,
        },
        "jax_wavelength_batch": (
            PRODUCTION_WAVELENGTH_BATCH if args.batched else None
        ),
        "cases": cases,
        "passed": len(cases) == 4 and all(case["passed"] for case in cases),
    }
    _write_report(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
