# Cov-M1 scene covariance validation report

## Decision

The algebraic and real-cube execution checks pass, but the Cov-M1 feasibility
gate does not pass. Covariance-enabled extraction exceeded the required
five-times runtime limit in all four measured B/R classic/Fourier runs. The
locked refitted Monte Carlo ensemble has not been run, so statistical coverage
is not established.

Per the Cov-M1 decision rule, upstream cube-covariance implementation stops
here. The next investigation should target a batched or differentiable native
variable-projection Jacobian (a focused JAX prototype is a candidate) before
re-running the locked ensemble.

## Scope

The product is

$$
C_a = D_{\mathrm{cond}} + J C_p J^T
    = D_{\mathrm{cond}} + U U^T.
$$

It treats input E3D statistical covariance as diagonal. It includes the
marginal global scene-parameter block from the complete meta-slice covariance
and native conditional source-amplitude variance. It does not include upstream
cube resampling covariance or other detector-to-flux covariance.

## Automated checks

Command:

```text
MPLCONFIGDIR=/tmp/mpl-cache PYTHONPATH=. python -m pytest -q
```

Result: `28 passed`.

The suite covers:

- the complete native coefficient covariance against a dense normal-matrix
  inverse, including source/background covariance;
- exact-name marginal global-block selection under reordered names;
- analytic variable-projection derivatives, centered differences, and
  second-order one-sided differences at bounds;
- the `1e-10` negative-eigenvalue policy and retained factor rank;
- `U U^T` against direct `J C_p J^T`;
- exact covariance-off/on flux parity guards;
- lower-triangle FITS reconstruction, metadata, variance-diagonal equality,
  PSD checks, and injected transactional write failure; and
- installed CLI covariance flag behavior for classic and Fourier PSFs.

## Real-cube execution matrix

Inputs were the checked-in B and R E3D fixtures for exposure
`25_056_084_003_17`, with subsampling 3 and diagonal input `STAT_SPE`.

| Channel | PSF | Samples | Flux equal | Variance equal to covariance diagonal | Minimum total eigenvalue | Scene rank | Output size | Approx. on/off wall ratio |
|---|---|---:|---|---|---:|---:|---:|---:|
| B | classic | 779 | yes | yes | 6263.27 | 9 | 4.67 MiB | 8.1 |
| B | Fourier | 779 | yes | yes | 6908.78 | 8 | 4.67 MiB | 8.5 |
| R | classic | 1726 | yes | yes | 11408.91 | 9 | 22.79 MiB | 11.6 |
| R | Fourier | 1726 | yes | yes | 12713.17 | 8 | 22.79 MiB | 11.7 |

The ratios are direct observed wall-time comparisons from this development
machine, not benchmark-harness medians. They are sufficiently above 5 to fail
the locked runtime gate without ambiguity.

No parameter-covariance factorization required eigenvalue clipping. The
largest derivative column stability diagnostic was `1.13e-5`, below the
required `5e-3`. Scene covariance condition numbers were approximately 18,468
(B classic), 123 (B Fourier), 1,852 (R classic), and 92 (R Fourier).

## Gates not run

- Eight-scenario bright/faint refitted Monte Carlo, minimum 256 realizations
  per scenario: **NOT RUN**.
- Pull mean/RMS and empirical/predicted variance coverage: **NOT RUN**.
- Synthetic-band variance and near/long correlation coverage: **NOT RUN**.
- Peak RSS under the locked harness: **NOT RUN**.

The checkpointed runner is `validation/validate_scene_covariance.py`. It locks
the eight scenarios, deterministic seeds, bootstrap intervals, automatic 1024
sample escalation, runtime/RSS accounting, exact flux parity, and explicit
diagonal-input scope.
