# Imperium MLEbench runtime

`pyproject.toml` is the maintained dependency policy. `uv.lock` is the exact Python lock
used by workers, evaluated solutions, reference solutions, QC, and client reproduction.
`sources.toml` records the external snapshots reconciled into the policy.

The environment intentionally uses the current normal-PyPI PyTorch/TorchVision pair rather
than reproducing Kaggle's entire Colab-derived GPU image. Kaggle v170 is the package-feature
baseline; hosted-notebook, duplicate GPU-framework, TPU, RAPIDS, and cloud SDK packages are
excluded unless a later task supplies a concrete need and the resulting environment passes
the full release tests.

Update the lock from this directory:

```bash
uv lock
uv sync --frozen --no-dev
```

The lock is necessary but not sufficient for release. Build the immutable runtime container,
then run CPU, RTX 6000 Ada (`sm_89`), RTX PRO 6000 Blackwell (`sm_120`), DALI, decoder,
offline-weight-cache, and cold-reference smoke tests before recording its digest in the
pipeline release.

The checked-in build and both local GPU tests run with:

```bash
./build_and_test.sh
```

The build uses CUDA 13.0.2/cuDNN on Ubuntu 24.04, a uv-managed CPython 3.11.15 runtime, the
frozen lock, and only the declared system libraries. `weights-manifest.json` is intentionally
empty until an approved, licensed offline weight artifact is added with its exact hash.
