# JAX scene-covariance production report

## Review status

This report records the current `jax-fourier-psf` implementation and the
locked real-cube benchmark run on 2026-07-12.  It is intended for SNIFS-team
review.  It does not by itself authorize a production release or establish
statistical coverage.

The tested covariance product is the two-stage approximation

$$
C_a = D_{\mathrm{cond}} + J C_p J^T.
$$

The accepted NumPy extraction remains authoritative for point flux.  JAX is
used only to calculate the response Jacobian $J$ for the analytic
Gaussian/Moffat and Fourier-domain PSFs.  Input Euro3D covariance is treated
as diagonal; upstream cube covariance is not included in this report.

## What the runtime ratios mean

Every ratio uses a complete covariance-disabled `SnifsCubeFitter` run as its
denominator:

```text
full meta-slice fit + native extraction + JAX scene covariance
----------------------------------------------------------------
full meta-slice fit + ordinary covariance-disabled extraction
```

The denominator still produces the existing per-wavelength conditional
variance.  It does not propagate global scene-parameter covariance.  These
ratios are therefore total pipeline overhead ratios, not isolated Jacobian
kernel speedups.

The earlier finite-difference covariance implementation required about
8.1--11.7 times the covariance-disabled runtime.  The production JAX path
reduces the observed total warm ratio to 1.37--1.95 unbatched and 1.39--1.94
with optional wavelength batching.

## Locked benchmark configuration

- Platform: macOS arm64 CPU.
- JAX and jaxlib: 0.10.2, float64 enabled.
- Inputs: exposure `25_056_084_003_17`, B and R Euro3D cubes.
- Native wavelength counts: 779 B, 1,726 R.
- PSFs: analytic Gaussian/Moffat and Fourier-domain.
- Model grid: subsampling 3, border 15.
- Cold runtime gate: less than 5 times covariance-disabled runtime.
- Warm runtime target: less than 3 times covariance-disabled runtime.
- Peak RSS gate: less than 4 GiB.
- Numerical gates: exact accepted-flux preservation, covariance/variance
  consistency, finite symmetry, relative PSD tolerance `1e-10`, and complete
  JAX provenance.

Each channel/PSF case runs in a fresh process.  Within that process, the
covariance-disabled, cold-JAX, and warm-JAX fitters are released between runs;
the JAX executable cache is intentionally retained for the warm measurement.

## Unbatched results

Unbatched differentiation evaluates all native wavelengths simultaneously.
It is the default because it preserves the originally validated JAX execution
shape.  All numerical, provenance, and runtime gates passed.  The 4 GiB RSS
gate failed.

| Channel | PSF | Off (s) | Cold JAX (s) | Warm JAX (s) | Cold/off | Warm/off | Peak RSS | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B | Gaussian/Moffat | 10.53 | 16.61 | 15.58 | 1.58 | 1.48 | 5.75 GB | RSS fail |
| B | Fourier-domain | 7.66 | 11.60 | 10.49 | 1.51 | 1.37 | 5.08 GB | RSS fail |
| R | Gaussian/Moffat | 12.69 | 25.84 | 24.72 | 2.04 | 1.95 | 10.93 GB | RSS fail |
| R | Fourier-domain | 9.44 | 17.68 | 17.05 | 1.87 | 1.81 | 9.47 GB | RSS fail |

The high-water measurements occur during JAX extraction, before the dense
flux-covariance validation eigensolve.  The covariance-disabled R
Gaussian/Moffat run peaked at approximately 0.51 GB, while its unbatched JAX
run reached approximately 10.9 GB.

## Optional 128-wavelength batching

The opt-in mode evaluates independent wavelength slices in locked batches of
128.  It is enabled with:

```text
extract-star2 -V --jacobian-backend jax --jax-wavelength-batching ...
```

The Python API uses `jax_wavelength_batch=128`.  Other sizes are rejected so
the production configuration cannot drift silently.  Enabling the mode emits
a warning explaining its numerical effect.

| Channel | PSF | Off (s) | Cold JAX (s) | Warm JAX (s) | Cold/off | Warm/off | Peak RSS | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B | Gaussian/Moffat | 11.21 | 17.65 | 16.28 | 1.57 | 1.45 | 1.59 GB | pass |
| B | Fourier-domain | 7.80 | 12.73 | 10.85 | 1.63 | 1.39 | 1.51 GB | pass |
| R | Gaussian/Moffat | 12.95 | 26.31 | 25.11 | 2.03 | 1.94 | 2.05 GB | pass |
| R | Fourier-domain | 9.69 | 18.93 | 16.98 | 1.95 | 1.75 | 1.89 GB | pass |

Relative to the unbatched measurements, observed cold JAX wall time changed by
+6.3%, +9.8%, +1.8%, and +7.1% in the table order.  Warm JAX wall time changed
by +4.5%, +3.4%, +1.6%, and -0.4%.  These are separate wall-clock runs rather
than paired microbenchmarks, so small differences include ordinary run noise.
Peak RSS decreased by approximately 70--81%.

All four batched cases passed the locked runtime, RSS, flux-parity, covariance,
and provenance gates.

## Numerical effect of batching

Native coefficient solves have no mathematical coupling between wavelengths,
so batching does not change the statistical model.  It does change the leading
array shape compiled by XLA.  XLA may consequently select slightly different
floating-point schedules or FFT plans.

Synthetic production-equation comparisons measured:

| PSF | Flux max abs. change | Jacobian max abs. change | Jacobian relative Frobenius | $JJ^T$ relative Frobenius |
|---|---:|---:|---:|---:|
| Gaussian/Moffat | 2.66e-15 | 8.88e-16 | 2.84e-16 | 2.72e-16 |
| Fourier-domain | 7.11e-15 | 1.95e-14 | 1.57e-15 | 2.53e-15 |

The point-source flux written to the product does not acquire these changes:
the accepted NumPy flux remains authoritative.  Only the JAX surrogate and
the propagated scene-covariance term can change at this roundoff level.

The same comparison was then run on the locked real B/R cubes, fitting each
PSF once unbatched and once with the 128-wavelength option:

| Channel | PSF | Accepted flux exact | Jacobian max abs. change | Jacobian relative Frobenius | Covariance max abs. change | Covariance relative Frobenius |
|---|---|---|---:|---:|---:|---:|
| B | Gaussian/Moffat | yes | 0 | 0 | 0 | 0 |
| B | Fourier-domain | yes | 0 | 0 | 0 | 0 |
| R | Gaussian/Moffat | yes | 0 | 0 | 0 | 0 |
| R | Fourier-domain | yes | 0 | 0 | 0 | 0 |

Thus batching happened to be bit-identical for the tested production cubes,
including the complete propagated covariance.  The synthetic tests prove that
bit identity is not guaranteed for every wavelength shape, so the user-facing
roundoff warning and explicit opt-in remain necessary.

Unbatched JAX remains the default.  Batching is an explicit operational choice
for installations that prefer lower memory use and accept the documented
roundoff-level covariance difference.

## Current conclusion and remaining gates

The exact Fourier-domain JAX map is now wired into real covariance extraction.
Both PSFs preserve accepted flux and meet the locked total-runtime targets.
The optional 128-wavelength mode also meets the 4 GiB RSS gate.

Production readiness is not yet established.  The following remain required:

1. Run the locked eight-scenario bright/faint refitted Monte Carlo ensemble,
   with at least 256 realizations per scenario and automatic escalation where
   confidence intervals touch a boundary.
2. Review pull means, pull RMS, empirical/predicted variance ratios, and
   synthetic-band correlation diagnostics.
3. Resolve the separate vendored-code provenance and redistribution audit
   before any public release.
4. Merge this feature only through a reviewed pull request into
   `robust-flux-cov`; do not merge it directly.
