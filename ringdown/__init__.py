from .data import *
from .fit import *
from .kde_contour import *
from pylab import *
from . import qnms
from . import data
from . import fit

# ############################################################################
# rcParams

# make plots fit the LaTex column size but rescale them for ease of display
scale_factor = 2

# Get columnsize from LaTeX using \showthe\columnwidth
fig_width_pt = scale_factor*246.0
# Convert pts to inches
inches_per_pt = 1.0/72.27               
# Golden ratio
fig_ratio = (np.sqrt(5)-1.0)/2.0
fig_width = fig_width_pt*inches_per_pt
fig_height =fig_width*fig_ratio

figsize_column = (fig_width, fig_height)
figsize_square = (fig_width, fig_width)

fig_width_page = scale_factor*inches_per_pt*508.87
figsize_page = (fig_width_page, fig_height)

rcParams = {'figure.figsize': figsize_column}

# LaTex text font sizse in points (rescaled as above)
fs = scale_factor*9
fs_label = 0.8*fs
rcParams['axes.labelsize'] = fs
rcParams['legend.fontsize'] = fs
rcParams['xtick.labelsize'] = fs_label
rcParams['ytick.labelsize'] = fs_label
rcParams["text.usetex"] = "true"


