from pylab import *
import scipy.stats as ss
import seaborn as sns

# The following routine, Bounded_2d_kde, was copied from
# https://git.ligo.org/publications/gw190412/gw190412-discovery/-/blob/851f91431b7c36e7ea66fa47e8516f2aef9d7daf/scripts/bounded_2d_kde.py
class Bounded_2d_kde(ss.gaussian_kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None,
                 *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(Bounded_2d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(Bounded_2d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x, y])
        if self.xhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x, y])
        if self.ylow is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.ylow - y])
        if self.yhigh is not None:
            pdf += super(Bounded_2d_kde, self).evaluate([x, 2*self.yhigh - y])
        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x,
                                                             2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xlow - x,
                                                             2*self.yhigh - y])
        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x,
                                                             2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(Bounded_2d_kde, self).evaluate([2*self.xhigh - x,
                                                             2*self.yhigh - y])
        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results


# ############################################################################
# PLOTTING

def kdeplot_2d_clevels(xs, ys, levels=11, **kwargs):
    """ Plot contours at specified credible levels.

    Arguments
    ---------
    xs: array
        samples of the first variable.
    ys: array
        samples of the second variable, drawn jointly with `xs`.
    levels: float, array
        if float, interpreted as number of credible levels to be equally 
        spaced between (0, 1); if array, interpreted as list of credible
        levels.
    xlow: float
        lower bound for abscissa passed to Bounded_2d_kde (optional).
    xigh: float
        upper bound for abscissa passed to Bounded_2d_kde (optional).
    ylow: float
        lower bound for ordinate passed to Bounded_2d_kde (optional).
    yhigh: float
        upper bound for ordinate passed to Bounded_2d_kde (optional).
    ax: Axes
        matplotlib axes on which to plot (optional).
    kwargs:
        additional arguments passed to plt.contour().
    """
    try:
        xs = xs.values.astype(float)
        ys = ys.values.astype(float)
    except AttributeError:
        pass
    
    if all(~isfinite(xs)) or all(~isfinite(ys)):
        return None
    try:
        len(levels)
        f = 1 - np.array(levels)
    except TypeError:
        f = linspace(0, 1, levels+2)[1:-1]
    if kwargs.get('auto_bound', False):
        kwargs['xlow'] = min(xs)
        kwargs['xhigh'] = max(xs)
        kwargs['ylow'] = min(ys)
        kwargs['yhigh'] = max(ys)
    kde_kws = {k: kwargs.pop(k, None) for k in ['xlow', 'xhigh', 'ylow', 'yhigh']}
    k = Bounded_2d_kde(np.column_stack((xs, ys)), **kde_kws)
    size = max(10*(len(f)+2), 500)
    c = np.random.choice(len(xs), size=size)
    p = k(np.column_stack((xs[c], ys[c])))
    i = argsort(p)
    l = array([p[i[int(round(ff*len(i)))]] for ff in f])

    Dx = np.percentile(xs, 99) - np.percentile(xs, 1)
    Dy = np.percentile(ys, 99) - np.percentile(ys, 1)

    x = linspace(np.percentile(xs, 1)-0.1*Dx, np.percentile(xs, 99)+0.1*Dx, 128)
    y = linspace(np.percentile(ys, 1)-0.1*Dy, np.percentile(ys, 99)+0.1*Dy, 128)

    XS, YS = meshgrid(x, y, indexing='ij')
    ZS = k(np.column_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)

    ax = kwargs.pop('ax', gca())
    kwargs['colors'] = kwargs.get('colors', [kwargs.pop('color', None),])

    ax.contour(XS, YS, ZS, levels=l, **kwargs)


class Bounded_1d_kde(ss.gaussian_kde):
    """ Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain.

    Authorship: Ben Farr, LIGO
    """

    def __init__(self, pts, xlow=None, xhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.
        """
        pts = np.atleast_1d(pts)

        assert pts.ndim == 1, 'Bounded_1d_kde can only be one-dimensional'

        super(Bounded_1d_kde, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_1d(pts)
        assert pts.ndim == 1, 'points must be one-dimensional'

        x = pts.T
        pdf = super(Bounded_1d_kde, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(Bounded_1d_kde, self).evaluate(2*self.xlow - x)

        if self.xhigh is not None:
            pdf += super(Bounded_1d_kde, self).evaluate(2*self.xhigh - x)

        return pdf

    def __call__(self, pts):
        pts = np.atleast_1d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts > self.xhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results
