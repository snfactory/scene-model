"""Benchmark the experimental JAX Jacobian against finite differences."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np

from .prototype import (
    centered_difference_jacobian,
    fixture_arguments,
    jax,
    jitted_flux,
    jitted_jacobian,
    make_fixture,
    require_jax,
)


def _timed(callable_):
    started = time.perf_counter()
    value = callable_()
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value, time.perf_counter() - started


def run(profile, nwave, repeats):
    require_jax()
    fixture = make_fixture(profile=profile, nwave=nwave)
    args = fixture_arguments(fixture)
    data_before = fixture.data.copy()
    parameters_before = fixture.parameters.copy()

    flux, flux_compile_seconds = _timed(
        lambda: jitted_flux(*args, profile=profile)
    )
    jacobian, jacobian_compile_seconds = _timed(
        lambda: jitted_jacobian(*args, profile=profile)
    )
    oracle, finite_difference_seconds = _timed(
        lambda: centered_difference_jacobian(fixture)
    )
    hot_times = []
    for _ in range(repeats):
        _, elapsed = _timed(lambda: jitted_jacobian(*args, profile=profile))
        hot_times.append(elapsed)

    jacobian = np.asarray(jacobian)
    flux = np.asarray(flux)
    difference = jacobian - oracle
    scale = np.maximum(np.abs(oracle), 1e-10)
    repeat_flux = np.asarray(jitted_flux(*args, profile=profile))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the other supported Unix targets report
    # KiB. scene-model's supported Python platforms do not include Windows.
    peak_rss_mib = peak_rss / (1024.0 ** 2 if sys.platform == "darwin" else 1024.0)
    return {
        "profile": profile,
        "nwave": nwave,
        "nparameters": len(fixture.parameters),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "flux_compile_seconds": flux_compile_seconds,
        "jacobian_compile_seconds": jacobian_compile_seconds,
        "jacobian_hot_median_seconds": float(np.median(hot_times)),
        "finite_difference_seconds": finite_difference_seconds,
        "hot_speedup_over_finite_difference": (
            finite_difference_seconds / np.median(hot_times)
        ),
        "jacobian_max_absolute_error": float(np.max(np.abs(difference))),
        "jacobian_max_relative_error": float(np.max(np.abs(difference) / scale)),
        "jacobian_relative_frobenius_error": float(
            np.linalg.norm(difference) / np.linalg.norm(oracle)
        ),
        "flux_repeat_array_equal": bool(np.array_equal(flux, repeat_flux)),
        "input_data_unchanged": bool(np.array_equal(fixture.data, data_before)),
        "input_parameters_unchanged": bool(
            np.array_equal(fixture.parameters, parameters_before)
        ),
        "peak_rss_mib": peak_rss_mib,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("classic", "fourier", "both"),
                        default="both")
    parser.add_argument("--nwave", type=int, default=779)
    parser.add_argument("--repeats", type=int, default=7)
    arguments = parser.parse_args(argv)
    profiles = ("classic", "fourier") if arguments.profile == "both" else (arguments.profile,)
    print(json.dumps([
        run(profile, arguments.nwave, arguments.repeats)
        for profile in profiles
    ], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
