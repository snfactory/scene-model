# Cov-M1 offline scene validation

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
  --output validation/results/cov_m1
```

The runner rejects requested minima below 256.  At 256 samples it bootstraps
the gated statistics.  If a 95% interval overlaps an acceptance boundary, the
affected scenario automatically continues to 1024 samples.  Checkpoints are
written atomically and are safe to resume with the same inputs and root seed.

The JSON and Markdown reports record seeds, sample counts, algebraic metrics,
runtime, peak RSS, output/checkpoint sizes, covariance ranks, derivative
stability, exact flux parity, and the explicit absence of upstream cube
covariance.
