# ringdown

[![PyPI version](https://badge.fury.io/py/ringdown.svg)](https://badge.fury.io/py/ringdown)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxisi/ringdown/HEAD)
[![DOI](https://zenodo.org/badge/368680640.svg)](https://zenodo.org/badge/latestdoi/368680640)      

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

The `environment.yml` file enables running `ringdown` in JupyterHub services like [MyBinder](https://mybinder.org/) by pointing MyBinder at this repository or clicking the button at the top of this README.  (Don't forget to `pip install ringdown` after the binder activates!)

## Examples of Use

See the `examples` directory for Jupyter notebooks that give examples of using the package.  In particular, `examples/GW150914.ipynb` demonstrates an analysis of the ringdown in GW150914 and uses the fundamental (2,2) mode and first overtone to constrain the Kerr-ness of the post-merger spacetime, much like [Isi, et al. (2019)](https://arxiv.org/abs/1905.00869).

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
