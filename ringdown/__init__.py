__all__ = []

import os

# Cap BLAS/OpenMP threading before numpy/jax load it (the setting is frozen
# at library load): one thread per chain is the right default when running
# parallel chains. Export OMP_NUM_THREADS yourself, or call
# ringdown.setup(num_threads=...), to override. The imports below must stay
# after this line, hence the noqa markers.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .data import *  # noqa: E402
from .fit import *  # noqa: E402
from .result import *  # noqa: E402
from .waveforms import *  # noqa: E402
from .imr import IMRResult  # noqa: E402
from . import qnms  # noqa: E402
from . import model  # noqa: E402
from . import utils  # noqa: E402
from ._setup import setup  # noqa: E402

from importlib.metadata import version  # noqa: E402
__version__ = version("ringdown")

# ############################################################################
# rcParams

# # make plots fit the LaTex column size but rescale them for ease of display
# scale_factor = 2

# # Get columnsize from LaTeX using \showthe\columnwidth
# fig_width_pt = scale_factor*246.0
# # Convert pts to inches
# inches_per_pt = 1.0/72.27
# # Golden ratio
# fig_ratio = (np.sqrt(5)-1.0)/2.0
# fig_width = fig_width_pt*inches_per_pt
# fig_height =fig_width*fig_ratio

# figsize_column = (fig_width, fig_height)
# figsize_square = (fig_width, fig_width)

# fig_width_page = scale_factor*inches_per_pt*508.87
# figsize_page = (fig_width_page, fig_height)

# rcParams = {'figure.figsize': figsize_column}

# # LaTex text font sizse in points (rescaled as above)
# fs = scale_factor*9
# fs_label = 0.8*fs
# rcParams['axes.labelsize'] = fs
# rcParams['legend.fontsize'] = fs
# rcParams['xtick.labelsize'] = fs_label
# rcParams['ytick.labelsize'] = fs_label
# rcParams["text.usetex"] = "true"
