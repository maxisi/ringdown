# ringdown

[![PyPI version](https://badge.fury.io/py/ringdown.svg)](https://badge.fury.io/py/ringdown)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxisi/ringdown/HEAD)
[![DOI](https://zenodo.org/badge/368680640.svg)](https://zenodo.org/badge/latestdoi/368680640)
![pytest](https://github.com/maxisi/ringdown/actions/workflows/python-app.yml/badge.svg?event=push)
[![Documentation Status](https://readthedocs.org/projects/ringdown/badge/?version=latest)](https://ringdown.readthedocs.io/en/latest/?badge=latest)

Bayesian analysis of black hole ringdowns.  The original paper that inspired this code package is [Isi, et al. (2019)](https://arxiv.org/abs/1905.00869); a full description of the code and method can be found in [Isi & Farr (2021)](https://arxiv.org/abs/2107.05609).

## Installation

This package is pip installable:

```shell
pip install ringdown
```

For the latest and greatest version, you can install directly from the git repo:

```shell
pip install git+https://github.com/maxisi/ringdown.git
```

### Complete Environments

A complete [conda](https://docs.conda.io/en/latest/) environment that includes all the prerequisites (and more!) to install `ringdown` can be found in  `environment.yml` in the current directory:

```shell
conda env create -f environment.yml
conda activate ringdown
pip install ringdown
```

will leave the shell in an environment that includes `jupyterlab` ready to explore the `ringdown` package.  

The `environment.yml` file enables running `ringdown` in JupyterHub services like [MyBinder](https://mybinder.org/) by pointing MyBinder at this repository or clicking the button at the top of this README.

## Examples and tips

See the [example gallery](https://ringdown.readthedocs.io/en/latest/gallery.html) in the docs for several examples. You can download the Jupyter notebooks featured in the docs from the `docs/examples`.

### Performance notes

In order to run Jax on a CPU with four cores and use double precision, you can do the following at the top of your script:
```python
# disable numpy multithreading to avoid conflicts
# with jax multiprocessing in numpyro
import os
os.environ["OMP_NUM_THREADS"] = "1"

# import jax and set it up to use double precision
from jax import config
config.update("jax_enable_x64", True)

# import numpyro and set it up to use 4 CPU devices
import numpyro
numpyro.set_host_device_count(4)
numpyro.set_platform('cpu')
```

To run on a GPU with single precision you can instead do:
```python
# import jax and set it up to use double precision
from jax import config
config.update("jax_enable_x64", False)

# import numpyro and set it up to use 4 CPU devices
import numpyro
numpyro.set_platform('gpu')
```

You will see significant performance enhancements when running ona GPU with 32-bit precision. If you have multiple GPUs, `numpyro` can use them in parallel to run different chains, just as with CPUs. Sampling one chain for a GW150914-like system takes O(s) on an Nvidia A100 GPU.

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
