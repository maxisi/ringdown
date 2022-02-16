# ringdown

[![PyPI version](https://badge.fury.io/py/ringdown.svg)](https://badge.fury.io/py/ringdown)
[![DOI](https://zenodo.org/badge/368680640.svg)](https://zenodo.org/badge/latestdoi/368680640)
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

We no longer advocate for the use of a Conda environment for this package (pystan does not play nice with Conda).  Instead, you can create a virtualenv if you want to isolate the installation from your system Python:

```shell
python -m venv /path/to/virtual/env
```

(a common location is apparently `$HOME/.venv/ringdown`).

You can then activate the venv with 

```shell
source /path/to/virtual/env/bin/activate
```

and then install this package and its dependencies with 

```shell
python -m pip install ringdown
```

or, for development / hacking

```shell
python -m pip install -e /path/to/ringdown/project/root
```

You may find the additional packages in `requirements-notebook.txt` useful if you want to run the examples in a notebook or work with jupyter notebooks and this package:

```shell
python -m pip install -r requirements-notebook.txt
```

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
