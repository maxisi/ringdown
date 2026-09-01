# ringdown

[![PyPI version](https://badge.fury.io/py/ringdown.svg)](https://badge.fury.io/py/ringdown)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxisi/ringdown/HEAD)
[![DOI](https://zenodo.org/badge/368680640.svg)](https://zenodo.org/badge/latestdoi/368680640)
![pytest](https://github.com/maxisi/ringdown/actions/workflows/python-app.yml/badge.svg?event=push)
[![Documentation Status](https://readthedocs.org/projects/ringdown/badge/?version=latest)](https://ringdown.readthedocs.io/en/latest/?badge=latest)

Bayesian analysis of black hole ringdowns.  The original paper that inspired this code package is [Isi, et al. (2019)](https://arxiv.org/abs/1905.00869); a full description of the code and method can be found in [Isi & Farr (2021)](https://arxiv.org/abs/2107.05609).

## Installation

Requires Python 3.12–3.14. Python 3.13 is recommended (`qnm` emits `SyntaxWarning`s on 3.14). Intel Macs are not supported (JAX no longer ships x86_64 macOS wheels).

This package is pip installable:

```shell
pip install ringdown
```

For the latest and greatest version, you can install directly from the git repo:

```shell
pip install git+https://github.com/maxisi/ringdown.git
```

### GPU support

The default install provides a CPU build of JAX. To run on NVIDIA GPUs, upgrade JAX after installing `ringdown`:

```shell
pip install --upgrade "jax[cuda12]==0.11.1"
```

See the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html) for CUDA version details and other accelerators.

### Complete Environments

The recommended way to set up a complete environment (including Jupyter and development tools) is with [uv](https://docs.astral.sh/uv/) from a clone of this repository:

```shell
uv sync
```

You can also easily install all optional dependencies:

```shell
uv sync --all-extras
```

Alternatively, you can use [conda](https://docs.conda.io/en/latest/) with `environment.yml`:

```shell
conda env create -f environment.yml
conda activate ringdown
```

The `environment.yml` file enables running `ringdown` in JupyterHub services like [MyBinder](https://mybinder.org/) by pointing MyBinder at this repository or clicking the button at the top of this README.

## Examples and tips

See the [example gallery](https://ringdown.readthedocs.io/en/latest/gallery.html) in the docs for several examples. You can download the Jupyter notebooks featured in the docs from the `docs/examples`.

### Performance notes

_ringdown_ configures jax and numpyro for you through a single call. To run on a CPU with four host devices (so chains can sample in parallel) and double precision, do the following at the top of your script:
```python
import ringdown as rd
rd.setup()
```

To run on a GPU with single precision you can instead do:
```python
import ringdown as rd
rd.setup(platform='gpu')
```

All of these settings freeze as soon as jax initializes its backends, so call `rd.setup()` right after the import, before any jax operation. Importing _ringdown_ also caps BLAS/OpenMP threading (`OMP_NUM_THREADS=1`, unless already set) so parallel chains do not oversubscribe the machine; see the [configuration docs](https://ringdown.readthedocs.io/en/latest/configuration.html) for the knobs and manual equivalents.

You will see significant performance enhancements when running on a GPU with 32-bit precision. If you have multiple GPUs, `numpyro` can use them in parallel to run different chains, just as with CPUs. Sampling one chain for a GW150914-like system takes O(s) on an Nvidia A100 GPU.

⚠️ _Caveat emptor:_ depending on the autocovariance function (ACF), using `float32` can cause numerical problems when computing the likelihood; _ringdown_ will automatically rescale the strain in an attempt to prevent this, but you should use this feature at your own risk.

## Citations

We ask that scientific users of this code cite the corresponding Zenodo entry (see blue DOI badge above), as well as [Isi & Farr (2021)](https://arxiv.org/abs/2107.05609):

```bibtex
@article{Isi:2021iql,
    author = "Isi, Maximiliano and Farr, Will M.",
    title = "{Analyzing black-hole ringdowns}",
    eprint = "2107.05609",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "LIGO-P2100227",
    month = "7",
    year = "2021"
}
```
