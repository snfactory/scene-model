# Results and recommendation

Initial run: 2026-07-12, CPU backend, JAX/jaxlib 0.7.2, x64 enabled.

Migration rerun: 2026-07-12, macOS arm64 CPU, JAX/jaxlib 0.10.2, x64 enabled.
The locked 0.10.2 profile passed all 36 scene-model and experiment tests. The
779-slice classic and Fourier relative Frobenius errors were 8.15e-11 and
8.33e-10. Hot Jacobians were 2.05x and 2.01x faster than the two-point finite
difference oracle, with peak RSS of 510 MiB and 569 MiB. The production
three-step stability policy needs roughly three times the oracle work.

## 779-slice isolated-process results

| Profile | JAX compile + first Jacobian | Hot Jacobian median | 2-point FD | Hot speedup | Relative Frobenius error | Max absolute error | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| classic | 0.246 s | 0.0141 s | 0.0284 s | 2.02x | 8.21e-11 | 1.35e-7 | 468 MiB |
| Fourier | 0.356 s | 0.0244 s | 0.0569 s | 2.33x | 7.54e-10 | 2.09e-6 | 513 MiB |

Cov-M1 evaluates centered/one-sided derivatives at three step sizes. The
two-point oracle above performs only two flux evaluations per parameter, so
its cost is about one third of the successful production stability policy
before retries. Relative to that policy, the representative hot JAX advantage
is therefore approximately 6--7x. The comparison deliberately does not count
the existing mutable Python element-graph and extraction overhead, which JAX
would eliminate from repeated perturbations.

At 1,726 slices, hot medians were 0.0315 s classic and 0.0549 s Fourier.
Peak RSS while compiling and retaining both profile/shape executables in one
process was 1,221 MiB, below the 4 GiB feasibility gate. Isolated 779-slice
peak RSS was 468--513 MiB.

The flux function was bitwise repeatable within a process. Fixture data and
parameter arrays remained bitwise unchanged. Jacobian state has shape
`(nwave, nparameter)` and did not allocate a wavelength-squared array.

Commands:

```text
python -m pytest -q experiments/jax_jacobian/test_prototype.py
python -m experiments.jax_jacobian.benchmark --profile classic --nwave 779 --repeats 15
python -m experiments.jax_jacobian.benchmark --profile fourier --nwave 779 --repeats 15
python -m experiments.jax_jacobian.benchmark --profile both --nwave 1726 --repeats 9
python -m pytest -q
```

Results: 4 experiment tests passed; 28 production tests passed.

## Scope and unrun gates

These profiles reproduce the relevant operation classes, not every production
PSF equation. They demonstrate x64 differentiation through real profiles,
complex FFT/IFFT, normalization, weighted normal equations, and batched linear
solves. A real B cube benchmark was **NOT RUN** because using a representative
profile on accepted native data would not test flux/Jacobian parity. That gate
requires a pure-array port of the exact classic PSF basis first. B classic
must remain the first real-cube target. The refitted Monte Carlo was **NOT
RUN**, as required.

Production flux equality is **NOT ESTABLISHED** by this experiment. Any exact
PSF port must compare its baseline amplitude against the accepted NumPy
extraction with `np.array_equal`; covariance mode cannot substitute a JAX
point estimate if it differs. One safe integration is to keep the accepted
NumPy flux and use JAX only for its Jacobian after a strict baseline parity
check.

## Recommendation

**GO for promoting forward-mode JAX 0.10.2 through one bounded exact-classic
port first.** Scene-model now pins that version as a production dependency,
with a separate CUDA 13 installation profile. The representative experiment
shows enough speed and memory margin to address the Cov-M1 runtime failure,
but it does not establish production flux or Jacobian parity. The production
work must export fixed arrays and masks from `SnifsCubeFitter`, port the exact
classic basis, and benchmark B classic end-to-end. Proceed to Fourier only
after exact classic flux and Jacobian parity pass.

Numba is a weaker next choice: it is unavailable here, does not supply the
needed automatic differentiation, and its support for this FFT-heavy array
path would still require hand derivatives. Fully analytic/implicit
derivatives could ultimately be fastest and avoid a large dependency, but the
classic derived-parameter and Fourier/ADR derivative surface is substantially
larger and riskier. If exact JAX baseline parity or memory fails on B classic,
prefer analytic source-basis derivatives plus the existing implicit
variable-projection formula over a Numba-only rewrite.
