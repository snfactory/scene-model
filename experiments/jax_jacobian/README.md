# JAX native variable-projection Jacobian experiment

This experiment isolates the array-valued core of Cov-M1 differentiation:

1. evaluate one point-source basis per wavelength from global parameters;
2. combine it with fixed background bases;
3. analytically solve the weighted normal equations at every wavelength; and
4. differentiate the resulting source amplitudes with respect to the global
   parameters.

The classic representative evaluates a normalized real-space
Gaussian/Moffat mixture. The Fourier representative evaluates a normalized
Gaussian/exponential-power mixture with `jax.numpy.fft.ifft2`. Both use a
15-by-15 native slice and source plus constant/x/y linear coefficients. They
are not replacements for the production PSFs; they test whether JAX supports
the operations and scaling needed by the production seam.

Run from the repository root:

```bash
python -m experiments.jax_jacobian.benchmark --profile both --nwave 779
pytest -q experiments/jax_jacobian/test_prototype.py
```

Scene-model now pins JAX 0.10.2 as a production dependency, with a separate
CUDA 13 installation profile for supported Linux x86_64 deployments. This
experiment remains isolated from production code: it exercises representative
operation classes and is not an implementation of either production PSF.

## Integration obstacles

- `SceneModel.evaluate` is a mutable Python element graph using dictionaries,
  object caches, and dynamic component lists; it cannot be directly traced.
- Native extraction creates Astropy tables, processes masks in Python, and
  uses NumPy/SciPy operations. Those boundaries must remain outside JIT.
- The production classic and Fourier element formulas must be ported to pure
  `jax.numpy` functions. JAX FFT and batched `linalg.solve` themselves work.
- Wavelengths with different masks have different design shapes. Production
  integration needs a fixed 225-pixel representation with zero weights for
  masked pixels, preserving the existing accepted mask exactly.
- Default JAX is float32; covariance parity requires enabling x64 before any
  arrays are created.
- Compilation cost must be amortized within an extraction. Cache keys must
  include channel/profile, wavelength count, grid/subsampling, coefficient
  count, and mask representation.
- Forward-mode automatic differentiation is required here: there are only
  5--9 global inputs and hundreds of wavelength outputs. Reverse mode scales
  poorly for this Jacobian shape.
