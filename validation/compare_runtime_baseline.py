#!/usr/bin/env python3
"""Compare two scene-model checkouts on external SNIFS B/R fixtures.

The parent launches each checkout in a fresh subprocess, compares the numeric
products, and writes one JSON report.  Fixtures remain external: the report
records their SHA-256 identities but neither their paths nor their contents.
No output spectra or intermediate files are written.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


KEY_INPUT_HEADERS = (
    "CHANNEL", "OBJECT", "AIRMASS", "PARANG", "EFFTIME", "EXPTIME",
    "PRESSURE", "TEMP", "CRVAL3", "CDELT3",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten_fit_parameters(parameters):
    """Return scalar parameter values with stable names for vector entries."""
    flattened = {}
    for name, value in sorted(parameters.items()):
        array = np.asarray(value)
        if array.ndim == 0:
            flattened[name] = float(array)
            continue
        for index in np.ndindex(array.shape):
            suffix = ",".join(str(item) for item in index)
            flattened[f"{name}[{suffix}]"] = float(array[index])
    return flattened


def _worker(checkout, cube_path, psf, jacobian_backend):
    checkout = str(Path(checkout).resolve())
    sys.path.insert(0, checkout)
    from astropy.io import fits
    from scene_model.snifs import SnifsCubeFitter

    with redirect_stdout(sys.stderr):
        fitter = SnifsCubeFitter(
            str(cube_path), psf=psf, background_degree=0,
            subsampling=3, border=15, least_squares=False,
            filter_variance=False, prior_scale=0.0, verbosity=-1,
        )
        fitter.fit_metaslices_2d(num_meta_slices=12)
        fitter.fit_metaslices_3d()
        fitter.check_validity()
        fitter.extract(
            method="psf", covariance=True,
            jacobian_backend=jacobian_backend,
        )

    try:
        input_header = fits.getheader(cube_path, "E3D_DATA")
    except (KeyError, IndexError):
        input_header = fits.getheader(cube_path)
    input_headers = {
        key: _json_scalar(input_header[key])
        for key in KEY_INPUT_HEADERS if key in input_header
    }
    model_headers = {
        key: _json_scalar(value)
        for key, value, _description
        in fitter.fit_scene_model.get_fits_header_items()
    }
    return {
        "fit_parameters": _flatten_fit_parameters(fitter.fit_parameters),
        "flux": np.asarray(fitter.point_source_spectrum.data).tolist(),
        "variance": np.asarray(fitter.point_source_spectrum.var).tolist(),
        "covariance": np.asarray(fitter.point_source_spectrum.cov).tolist(),
        "headers": {"input": input_headers, "model": model_headers},
    }


def _run_worker(script, checkout, fixture, channel, psf, backend):
    command = [
        sys.executable, str(script), "--worker",
        "--checkout", str(checkout), "--cube", str(fixture),
        "--channel", channel, "--psf", psf,
        "--jacobian-backend", backend,
    ]
    completed = subprocess.run(
        command, cwd=checkout, check=False, text=True,
        stdout=subprocess.PIPE, stderr=sys.stderr,
    )
    if completed.returncode:
        raise RuntimeError(
            "%s %s/%s worker failed with status %d" %
            (checkout, channel, psf, completed.returncode)
        )
    return json.loads(completed.stdout)


def _array_delta(baseline, candidate, rtol, atol):
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape:
        return {
            "baseline_shape": list(baseline.shape),
            "candidate_shape": list(candidate.shape),
            "exact": False, "within_tolerance": False,
            "maximum_absolute": None, "relative_frobenius": None,
        }
    difference = candidate - baseline
    denominator = max(float(np.linalg.norm(baseline)), np.finfo(float).tiny)
    return {
        "shape": list(baseline.shape),
        "exact": bool(np.array_equal(baseline, candidate)),
        "within_tolerance": bool(np.allclose(
            baseline, candidate, rtol=rtol, atol=atol, equal_nan=True
        )),
        "maximum_absolute": float(np.max(np.abs(difference), initial=0.0)),
        "relative_frobenius": float(np.linalg.norm(difference) / denominator),
    }


def _compare_case(baseline, candidate, rtol, atol):
    parameter_names = sorted(
        set(baseline["fit_parameters"]) | set(candidate["fit_parameters"])
    )
    parameter_delta = _array_delta(
        [baseline["fit_parameters"].get(name, np.nan) for name in parameter_names],
        [candidate["fit_parameters"].get(name, np.nan) for name in parameter_names],
        rtol, atol,
    )
    parameter_delta["names"] = parameter_names
    arrays = {
        name: _array_delta(baseline[name], candidate[name], rtol, atol)
        for name in ("flux", "variance", "covariance")
    }
    headers_equal = baseline["headers"] == candidate["headers"]
    passed = (
        parameter_delta["within_tolerance"]
        and all(delta["within_tolerance"] for delta in arrays.values())
        and headers_equal
    )
    return {
        "fit_parameters": parameter_delta,
        **arrays,
        "headers": {
            "exact": headers_equal,
            "baseline": baseline["headers"],
            "candidate": candidate["headers"],
        },
        "passed": bool(passed),
    }


def _atomic_json(path, payload):
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
        description="Compare runtime products from two scene-model checkouts"
    )
    parser.add_argument("--baseline-root")
    parser.add_argument("--candidate-root")
    parser.add_argument("--blue", help="External B-channel fixture")
    parser.add_argument("--red", help="External R-channel fixture")
    parser.add_argument("--output", help="JSON report destination")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--jacobian-backend", choices=("finite-difference", "jax"),
        default="jax",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkout", help=argparse.SUPPRESS)
    parser.add_argument("--cube", help=argparse.SUPPRESS)
    parser.add_argument("--channel", choices=("B", "R"), help=argparse.SUPPRESS)
    parser.add_argument("--psf", choices=("classic", "fourier"),
                        help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.worker:
        required = (args.checkout, args.cube, args.channel, args.psf)
        if not all(required):
            raise ValueError("Worker requires checkout, cube, channel, and PSF")
        result = _worker(
            args.checkout, args.cube, args.psf, args.jacobian_backend
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    required = (
        args.baseline_root, args.candidate_root,
        args.blue, args.red, args.output,
    )
    if not all(required):
        raise ValueError(
            "Parent requires baseline-root, candidate-root, blue, red, and output"
        )
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("Comparison tolerances must be non-negative")

    script = Path(__file__).resolve()
    cases = []
    for channel, fixture in (("B", args.blue), ("R", args.red)):
        fixture_identity = {
            "channel": channel,
            "sha256": _sha256(fixture),
        }
        for psf in ("classic", "fourier"):
            baseline = _run_worker(
                script, args.baseline_root, fixture, channel, psf,
                args.jacobian_backend,
            )
            candidate = _run_worker(
                script, args.candidate_root, fixture, channel, psf,
                args.jacobian_backend,
            )
            comparison = _compare_case(baseline, candidate, args.rtol, args.atol)
            cases.append({
                "fixture": fixture_identity,
                "psf": psf,
                **comparison,
            })

    payload = {
        "format": "SNIFS-RUNTIME-BASELINE-COMPARISON-1.0",
        "jacobian_backend": args.jacobian_backend,
        "tolerances": {"relative": args.rtol, "absolute": args.atol},
        "cases": cases,
        "passed": len(cases) == 4 and all(case["passed"] for case in cases),
    }
    _atomic_json(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
