#!/usr/bin/env python3
"""Checkpointed, refitted Cov-M1 Monte Carlo feasibility validation.

This is an offline scientific runner, not a CI test.  It deliberately uses an
independent empirical covariance oracle and consumes only public extraction
results plus the documented diagnostic attributes of ``SnifsCubeFitter``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scene_model.snifs import SnifsCubeFitter


ROOT_SEED = 20260712
MINIMUM_SAMPLES = 256
ESCALATED_SAMPLES = 1024
BOOTSTRAP_DRAWS = 400
BRIGHT_SCALE = 1.0
FAINT_SCALE = 0.1
BAND_FRACTIONS = ((0.10, 0.30), (0.40, 0.60), (0.70, 0.90))
PAIR_FRACTIONS = ((0.25, 0.25, 1), (0.50, 0.50, 5),
                  (0.75, 0.75, 1), (0.20, 0.80, 0),
                  (0.35, 0.65, 0))
BOUNDARIES = {
    "pull_mean": (-0.1, 0.1),
    "pull_rms": (0.9, 1.1),
    "variance_ratio_median": (0.9, 1.1),
    "band_ratio_0": (0.9, 1.1),
    "band_ratio_1": (0.9, 1.1),
    "band_ratio_2": (0.9, 1.1),
}


@dataclass(frozen=True)
class Scenario:
    index: int
    channel: str
    psf: str
    level: str
    source_scale: float
    cube_path: str

    @property
    def name(self):
        return "%s-%s-%s" % (self.channel, self.psf, self.level)


def realization_seed(root_seed, scenario_index, realization_index):
    """Return a schedule-stable seed independent of checkpoint boundaries."""
    sequence = np.random.SeedSequence(
        [int(root_seed), int(scenario_index), int(realization_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def current_peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and most BSDs report KiB.
    if platform.system() == "Darwin":
        return int(value)
    return int(value * 1024)


def _fit(cube_path, psf, covariance):
    fitter = SnifsCubeFitter(
        str(cube_path), psf=psf, background_degree=0,
        subsampling=3, border=15, least_squares=False,
        filter_variance=False, prior_scale=0.0, verbosity=-1,
    )
    fitter.fit_metaslices_2d(num_meta_slices=12)
    fitter.fit_metaslices_3d()
    fitter.check_validity()
    fitter.extract(method="psf", covariance=covariance)
    return fitter


def _nominal_truth(cube_path, psf):
    start = time.perf_counter()
    fitter = _fit(cube_path, psf, covariance=False)
    legacy_seconds = time.perf_counter() - start
    table = fitter.extraction
    coefficient_names = tuple(
        name for name in table.colnames if not name.endswith("_variance")
    )
    model_data = np.empty_like(fitter.cube.data, dtype=float)
    for wavelength_index, wavelength in enumerate(fitter.cube.lbda):
        parameters = {"wavelength": wavelength}
        parameters.update({
            name: float(table[name][wavelength_index])
            for name in coefficient_names
        })
        image = fitter.fit_scene_model.evaluate(**parameters)
        model_data[wavelength_index] = image[fitter.cube.i, fitter.cube.j]
    return fitter, model_data, legacy_seconds


def _scaled_truth(fitter, nominal_model, scale):
    table = fitter.extraction
    model_data = np.empty_like(nominal_model)
    truth_flux = np.asarray(table["amplitude"], dtype=float) * scale
    coefficient_names = tuple(
        name for name in table.colnames if not name.endswith("_variance")
    )
    for wavelength_index, wavelength in enumerate(fitter.cube.lbda):
        parameters = {"wavelength": wavelength}
        parameters.update({
            name: float(table[name][wavelength_index])
            for name in coefficient_names
        })
        parameters["amplitude"] *= scale
        image = fitter.fit_scene_model.evaluate(**parameters)
        model_data[wavelength_index] = image[fitter.cube.i, fitter.cube.j]
    return truth_flux, model_data


def _write_realization(template_cube, data, path):
    original = template_cube.data
    try:
        template_cube.data = np.asarray(data, dtype=float)
        template_cube.WR_e3d_file(str(path))
    finally:
        template_cube.data = original


def _band_weights(length):
    result = []
    for low_fraction, high_fraction in BAND_FRACTIONS:
        low = int(np.floor(low_fraction * length))
        high = max(low + 1, int(np.ceil(high_fraction * length)))
        weights = np.zeros(length)
        weights[low:high] = 1.0 / (high - low)
        result.append(weights)
    return np.asarray(result)


def _correlation_pairs(length):
    pairs = []
    for left_fraction, right_fraction, lag in PAIR_FRACTIONS:
        left = min(length - 1, int(round(left_fraction * (length - 1))))
        if lag:
            right = min(length - 1, left + lag)
        else:
            right = min(length - 1,
                        int(round(right_fraction * (length - 1))))
        pairs.append((left, right))
    return tuple(pairs)


def _empty_checkpoint(truth_flux, scenario, root_seed):
    length = len(truth_flux)
    pairs = _correlation_pairs(length)
    return {
        "scenario_json": json.dumps(asdict(scenario), sort_keys=True),
        "root_seed": int(root_seed),
        "truth_flux": np.asarray(truth_flux),
        "fluxes": np.empty((0, length)),
        "predicted_diagonals": np.empty((0, length)),
        "predicted_covariance_sum": np.zeros((length, length)),
        "predicted_band_variances": np.empty((0, len(BAND_FRACTIONS))),
        "predicted_pair_covariances": np.empty((0, len(pairs))),
        "runtimes": np.empty(0),
        "peak_rss_bytes": np.empty(0, dtype=np.int64),
        "ranks": np.empty(0, dtype=np.int64),
        "maximum_derivative_stability": np.empty(0),
        "flux_parity": np.empty(0, dtype=bool),
        "seeds": np.empty(0, dtype=np.uint64),
    }


def _save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **state)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path, truth_flux, scenario, root_seed):
    if not path.exists():
        return _empty_checkpoint(truth_flux, scenario, root_seed)
    with np.load(path, allow_pickle=False) as saved:
        state = {name: saved[name] for name in saved.files}
    saved_scenario = str(state["scenario_json"])
    expected_scenario = json.dumps(asdict(scenario), sort_keys=True)
    if saved_scenario != expected_scenario:
        raise RuntimeError("Checkpoint scenario does not match requested run")
    if int(state["root_seed"]) != root_seed:
        raise RuntimeError("Checkpoint root seed does not match requested run")
    np.testing.assert_array_equal(state["truth_flux"], truth_flux)
    return state


def _append(state, key, value):
    value = np.asarray(value)
    if state[key].ndim == 1:
        state[key] = np.append(state[key], value.reshape(1))
    else:
        state[key] = np.concatenate((state[key], value[np.newaxis]), axis=0)


def _run_one_realization(scenario, nominal_fitter, truth_model, truth_flux,
                         root_seed, realization_index, scratch):
    seed = realization_seed(root_seed, scenario.index, realization_index)
    rng = np.random.default_rng(seed)
    variance = np.asarray(nominal_fitter.cube.var, dtype=float)
    noise = rng.normal(size=variance.shape) * np.sqrt(variance)
    noisy_data = truth_model + noise
    realization_path = scratch / ("%s-%04d.fits" %
                                    (scenario.name, realization_index))
    _write_realization(nominal_fitter.cube, noisy_data, realization_path)
    start = time.perf_counter()
    try:
        fitted = _fit(realization_path, scenario.psf, covariance=False)
        flux_without = np.asarray(fitted.point_source_spectrum.data).copy()
        fitted.extract(method="psf", covariance=True)
        runtime = time.perf_counter() - start
    finally:
        realization_path.unlink(missing_ok=True)
    flux = np.asarray(fitted.point_source_spectrum.data).copy()
    covariance = np.asarray(fitted.point_source_spectrum.cov).copy()
    factor = fitted.covariance_diagnostics["factorization"]
    derivatives = fitted.covariance_diagnostics["derivatives"]
    return {
        "seed": seed,
        "flux": flux,
        "covariance": covariance,
        "runtime": runtime,
        "rss": current_peak_rss_bytes(),
        "rank": factor.rank,
        "stability": max(derivatives.stability, default=0.0),
        "parity": np.array_equal(flux_without, flux),
    }


def _record(state, result, band_weights, pairs):
    covariance = result["covariance"]
    _append(state, "fluxes", result["flux"])
    _append(state, "predicted_diagonals", np.diag(covariance))
    state["predicted_covariance_sum"] += covariance
    _append(state, "predicted_band_variances", np.array([
        weights @ covariance @ weights for weights in band_weights
    ]))
    _append(state, "predicted_pair_covariances", np.array([
        covariance[left, right] for left, right in pairs
    ]))
    for key, value in (
        ("runtimes", result["runtime"]),
        ("peak_rss_bytes", result["rss"]),
        ("ranks", result["rank"]),
        ("maximum_derivative_stability", result["stability"]),
        ("flux_parity", result["parity"]),
        ("seeds", result["seed"]),
    ):
        _append(state, key, value)


def _point_metrics(fluxes, predicted_diagonals, predicted_band_variances,
                   truth_flux, band_weights):
    residuals = fluxes - truth_flux
    pulls = residuals / np.sqrt(predicted_diagonals)
    empirical_variance = np.var(fluxes, axis=0, ddof=1)
    predicted_variance = np.mean(predicted_diagonals, axis=0)
    ratios = empirical_variance / predicted_variance
    metrics = {
        "pull_mean": float(np.mean(pulls)),
        "pull_rms": float(np.sqrt(np.mean(pulls * pulls))),
        "variance_ratio_median": float(np.median(ratios)),
    }
    for index, weights in enumerate(band_weights):
        empirical = np.var(fluxes @ weights, ddof=1)
        predicted = np.mean(predicted_band_variances[:, index])
        metrics["band_ratio_%d" % index] = float(empirical / predicted)
    return metrics


def _bootstrap_intervals(state, band_weights, root_seed, scenario_index,
                         draws=BOOTSTRAP_DRAWS):
    sample_count = len(state["fluxes"])
    rng = np.random.default_rng(
        np.random.SeedSequence([root_seed, scenario_index, 0xC0B1])
    )
    values = {key: [] for key in BOUNDARIES}
    for _ in range(draws):
        indices = rng.integers(0, sample_count, sample_count)
        metrics = _point_metrics(
            state["fluxes"][indices], state["predicted_diagonals"][indices],
            state["predicted_band_variances"][indices],
            state["truth_flux"], band_weights,
        )
        for key in values:
            values[key].append(metrics[key])
    return {
        key: [float(value) for value in np.percentile(samples, [2.5, 97.5])]
        for key, samples in values.items()
    }


def _overlaps_boundary(interval, accepted):
    low, high = interval
    accepted_low, accepted_high = accepted
    return low <= accepted_low <= high or low <= accepted_high <= high


def _correlation_metrics(state, pairs):
    fluxes = state["fluxes"]
    sample_count = len(fluxes)
    predicted_covariance = (state["predicted_covariance_sum"] / sample_count)
    results = []
    for pair_index, (left, right) in enumerate(pairs):
        empirical = float(np.corrcoef(fluxes[:, left], fluxes[:, right])[0, 1])
        predicted = float(
            predicted_covariance[left, right]
            / np.sqrt(predicted_covariance[left, left]
                      * predicted_covariance[right, right])
        )
        # Fisher-z 95% interval for the empirical correlation.
        clipped = np.clip(empirical, -0.999999, 0.999999)
        half_width = 1.96 / np.sqrt(max(1, sample_count - 3))
        interval = np.tanh(np.arctanh(clipped) + np.array([-half_width,
                                                           half_width]))
        results.append({
            "indices": [left, right],
            "kind": "nearby" if abs(right - left) <= 5 else "long-range",
            "empirical": empirical,
            "predicted": predicted,
            "empirical_95_ci": interval.tolist(),
            "consistent": bool(interval[0] <= predicted <= interval[1]),
            "mean_predicted_covariance": float(np.mean(
                state["predicted_pair_covariances"][:, pair_index]
            )),
        })
    return results


def _summarize(state, scenario, band_weights, pairs, legacy_seconds,
               root_seed):
    count = len(state["fluxes"])
    metrics = _point_metrics(
        state["fluxes"], state["predicted_diagonals"],
        state["predicted_band_variances"], state["truth_flux"], band_weights,
    )
    intervals = _bootstrap_intervals(
        state, band_weights, root_seed, scenario.index
    )
    correlations = _correlation_metrics(state, pairs)
    gates = {
        key: bool(low < metrics[key] < high)
        for key, (low, high) in BOUNDARIES.items()
    }
    gates.update({
        "correlations": all(item["consistent"] for item in correlations),
        "peak_rss": int(np.max(state["peak_rss_bytes"])) < 4 * 1024**3,
        "runtime": (float(np.median(state["runtimes"]))
                    < 5.0 * legacy_seconds),
        "flux_parity": bool(np.all(state["flux_parity"])),
        "derivative_stability": (
            float(np.max(state["maximum_derivative_stability"])) <= 5e-3
        ),
    })
    overlap = {
        key: _overlaps_boundary(intervals[key], BOUNDARIES[key])
        for key in BOUNDARIES
    }
    return {
        "scenario": asdict(scenario),
        "sample_count": count,
        "root_seed": root_seed,
        "first_seed": int(state["seeds"][0]),
        "last_seed": int(state["seeds"][-1]),
        "metrics": metrics,
        "bootstrap_95_ci": intervals,
        "boundary_overlap": overlap,
        "correlations": correlations,
        "runtime_seconds": {
            "legacy_reference": legacy_seconds,
            "median_covariance_enabled": float(np.median(state["runtimes"])),
            "maximum_covariance_enabled": float(np.max(state["runtimes"])),
            "median_ratio": float(np.median(state["runtimes"])
                                  / legacy_seconds),
        },
        "peak_rss_bytes": int(np.max(state["peak_rss_bytes"])),
        "covariance_rank": {
            "minimum": int(np.min(state["ranks"])),
            "maximum": int(np.max(state["ranks"])),
        },
        "maximum_derivative_stability": float(np.max(
            state["maximum_derivative_stability"]
        )),
        "flux_array_equal_all": bool(np.all(state["flux_parity"])),
        "gates": gates,
        "passed": all(gates.values()),
        "scope": {
            "input_cube_covariance": "E3D statistical diagonal only",
            "upstream_resampling_covariance": "not included",
            "approximation": "two-stage meta/native Laplace",
        },
    }


def _scenarios(blue, red):
    result = []
    index = 0
    for channel, cube in (("B", blue), ("R", red)):
        for psf in ("classic", "fourier"):
            for level, scale in (("bright", BRIGHT_SCALE),
                                 ("faint", FAINT_SCALE)):
                result.append(Scenario(index, channel, psf, level, scale,
                                       str(Path(cube).resolve())))
                index += 1
    return result


def _write_reports(output, reports):
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "SNIFS-COV-M1-VALIDATION-1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline_science_ensemble": True,
        "scenario_count": len(reports),
        "passed": len(reports) == 8 and all(item["passed"] for item in reports),
        "scenarios": reports,
    }
    (output / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Cov-M1 scene covariance validation",
        "",
        "This is the offline refitted ensemble. Input cube covariance is ",
        "limited to the E3D statistical diagonal; upstream resampling ",
        "covariance is not included.",
        "",
        "| Scenario | N | Pull mean | Pull RMS | Median var ratio | "
        "Runtime ratio | Peak RSS GiB | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for report in reports:
        metrics = report["metrics"]
        lines.append(
            "| {name} | {count} | {mean:.4f} | {rms:.4f} | {ratio:.4f} | "
            "{runtime:.2f} | {rss:.2f} | {result} |".format(
                name=report["scenario"]["channel"] + "-"
                + report["scenario"]["psf"] + "-"
                + report["scenario"]["level"],
                count=report["sample_count"], mean=metrics["pull_mean"],
                rms=metrics["pull_rms"],
                ratio=metrics["variance_ratio_median"],
                runtime=report["runtime_seconds"]["median_ratio"],
                rss=report["peak_rss_bytes"] / 1024**3,
                result="PASS" if report["passed"] else "FAIL",
            )
        )
    lines.extend(["", "Overall: **%s**" %
                  ("PASS" if payload["passed"] else "FAIL"), ""])
    (output / "report.md").write_text("\n".join(lines))


def run(args):
    if args.minimum_samples < MINIMUM_SAMPLES:
        raise ValueError("Science ensemble requires at least 256 samples")
    if args.escalated_samples < ESCALATED_SAMPLES:
        raise ValueError("Boundary escalation requires at least 1024 samples")
    scenarios = _scenarios(args.blue, args.red)
    output = Path(args.output)
    checkpoint_directory = output / "checkpoints"
    reports = []
    nominal_cache = {}
    scratch_root = output / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        cache_key = (scenario.channel, scenario.psf)
        if cache_key not in nominal_cache:
            nominal_cache[cache_key] = _nominal_truth(
                scenario.cube_path, scenario.psf
            )
        nominal_fitter, nominal_model, legacy_seconds = nominal_cache[cache_key]
        truth_flux, truth_model = _scaled_truth(
            nominal_fitter, nominal_model, scenario.source_scale
        )
        bands = _band_weights(len(truth_flux))
        pairs = _correlation_pairs(len(truth_flux))
        checkpoint = checkpoint_directory / (scenario.name + ".npz")
        state = _load_checkpoint(
            checkpoint, truth_flux, scenario, args.root_seed
        )

        target = args.minimum_samples
        while True:
            with tempfile.TemporaryDirectory(
                    prefix=scenario.name + "-", dir=scratch_root) as scratch:
                scratch = Path(scratch)
                while len(state["fluxes"]) < target:
                    index = len(state["fluxes"])
                    result = _run_one_realization(
                        scenario, nominal_fitter, truth_model, truth_flux,
                        args.root_seed, index, scratch,
                    )
                    _record(state, result, bands, pairs)
                    if (len(state["fluxes"]) % args.checkpoint_every == 0
                            or len(state["fluxes"]) == target):
                        _save_checkpoint(checkpoint, state)
            report = _summarize(
                state, scenario, bands, pairs, legacy_seconds, args.root_seed
            )
            if (target < args.escalated_samples
                    and any(report["boundary_overlap"].values())):
                target = args.escalated_samples
                continue
            break
        report["checkpoint_bytes"] = checkpoint.stat().st_size
        report["dense_covar_output_bytes"] = len(truth_flux)**2 * 8
        reports.append(report)
        _write_reports(output, reports)

    return 0 if all(report["passed"] for report in reports) else 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the offline refitted Cov-M1 science ensemble"
    )
    parser.add_argument("--blue", required=True, help="B-channel E3D fixture")
    parser.add_argument("--red", required=True, help="R-channel E3D fixture")
    parser.add_argument("--output", required=True, help="Result directory")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument("--minimum-samples", type=int,
                        default=MINIMUM_SAMPLES)
    parser.add_argument("--escalated-samples", type=int,
                        default=ESCALATED_SAMPLES)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
