# Production JAX scene-covariance validation

Current real-cube performance, batching tradeoffs, numerical differences, and
remaining release gates are summarized in
[`JAX_SCENE_PRODUCTION_REPORT.md`](JAX_SCENE_PRODUCTION_REPORT.md).

Validation has two locked stages.  The runtime/RSS benchmark runs each B/R and
classic/Fourier case in a fresh process.  The statistical runner then executes
the eight-scenario refitted ensemble.  Both tools accept external E3D paths
and record each resolved path and SHA-256 in their JSON output; fixture data is
not copied into this repository.

## Runtime and memory

```bash
python validation/benchmark_jax_scene_covariance.py \
  --blue /path/to/e3d_target_B.fits \
  --red /path/to/e3d_target_R.fits \
  --output validation/results/jax_benchmark.json
```

The benchmark command defaults to the unbatched comparison mode; add
`--batched` to benchmark the production-default locked 128-wavelength mode.
Compare batched and unbatched Jacobians/covariances directly with:

```bash
python validation/compare_jax_batching.py \
  --blue /path/to/e3d_target_B.fits \
  --red /path/to/e3d_target_R.fits \
  --output validation/results/jax_batch_comparison.json
```

For every channel/PSF case, covariance-enabled cold runtime (including JAX
compilation) must be less than five times covariance-off runtime and peak RSS
must be below 4 GiB.  A warm runtime below three times covariance-off is
reported as the optimization target, not used to weaken or replace the cold
gate.  Exact flux equality, exact variance/covariance-diagonal equality,
finite symmetry, relative PSD tolerance `1e-10`, and complete JAX provenance
are also required.

## Refitted coverage ensemble

`validate_scene_covariance.py` is the locked scientific validation runner.  It
is intentionally excluded from the normal test suite: the minimum ensemble is
eight scenarios times 256 complete meta-slice refits.

The scenarios are the Cartesian product of B/R channel, classic/Fourier PSF,
and bright/faint source scale.  Each realization draws independent Gaussian
noise from the input E3D `STAT_SPE` diagonal, reruns the global meta fit, and
re-extracts native flux with and without covariance.  Holding `STAT_SPE` fixed
when the source is scaled makes this a validation of the declared conditional
diagonal-noise model, not a physical source-dependent Poisson simulation.

Run from an installed scene-model checkout:

```bash
python validation/validate_scene_covariance.py \
  --blue /path/to/e3d_target_B.fits \
  --red /path/to/e3d_target_R.fits \
  --output validation/results/cov_m1 \
  --jacobian-backend jax
```

The runner rejects requested minima below 256.  At 256 samples it bootstraps
the gated statistics.  If a 95% interval overlaps an acceptance boundary, the
affected scenario automatically continues to 1024 samples.  Checkpoints are
written atomically and are safe to resume with the same inputs and root seed.

The locked statistical gates are inclusive pull mean `[-0.1, 0.1]`, pull RMS
`[0.9, 1.1]`, and median empirical/predicted variance ratio `[0.9, 1.1]`.
Synthetic-band ratios and selected nearby/long-range correlations remain in
the report as scientific diagnostics, but are not additional release gates.

The JSON and Markdown reports record seeds, sample counts, metrics, runtime,
peak RSS, output/checkpoint sizes, covariance ranks, exact flux parity, JAX
backend/provenance, and the explicit absence of upstream cube covariance.
Runtime and RSS fields in the ensemble are diagnostic; their release gates
belong to the fresh-process benchmark above.
