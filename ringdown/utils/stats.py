import numpy as np
from scipy.stats import gaussian_kde, norm
from .kde_contour import Bounded_1d_kde


def compute_hpd(samples, p=0.68, out="both", sorted=False):
    # NOTE: this function does the same thing as arviz.hdi
    # compute the minimum-width p%-credible interval
    if sorted:
        sorted_samples = samples
    else:
        sorted_samples = np.sort(samples)
    n = len(samples)
    # number of samples that will be enclosed in p%-credible interval
    n_in = int(np.floor(p*n))
    # number of samples that will be outside p%-credible interval
    # this is also how many locations there are for the first sample in the
    # interval, since we can slide the `n_in` long window over `n_out` slots
    n_out = n - n_in
    # compute p%-credible interval widths for different starting locations
    # the first term `sorted_samples[n_in]` is the lowest-possible location
    # of the high-end of the CI; the second term `sorted_samples[n_out]` is
    # the highest-possible location of the low_end of the CI; others are in
    # between with steps of 1
    widths = sorted_samples[n_in-1:-1] - sorted_samples[0:n_out]
    # find location of first sample in tightest CI
    i = np.argmin(widths)
    if out.lower() == "both":
        return sorted_samples[i], sorted_samples[i+n_in]
    elif out.lower() == "high":
        return sorted_samples[i+n_in]
    elif out.lower() == "low":
        return sorted_samples[i]
    else:
        raise ValueError("Invalid output type.")


def q_of_zero(xs, q_min=0.01, q_tol=0.005) -> float:
    """Compute the quantile of zero for a given set of nonnegative samples.

    This function computes the quantile of zero for a given set of nonnegative
    samples. The quantile of zero is the smallest value of q such that the
    q-credible HPD of the samples includes zero.

    The algorithm works through bisection by progressively halving the
    interval between q_min and 1, and checking if the HPD of the samples at
    the mid-point includes zero. If it does, the upper bound is shrunk to the
    mid-point; otherwise, the lower bound is shrunk to the mid-point. The 
    algorithm continues until the interval is smaller than a given tolerance.

    Arguments
    ---------
    xs : array_like
        The set of nonnegative samples.
    q_min : float, optional
        The lower bound of the quantile. The default is 0.01.
    q_tol : float, optional
        The tolerance for the quantile. The default is 0.005.

    Returns
    -------
    q: float
        The quantile of zero.
    """
    xs = np.sort(xs)
    xmin = xs[0]

    if xmin < 0:
        raise ValueError("The smallest sample is less than zero.")

    if compute_hpd(xs, q_min, sorted=True)[0] < xmin:
        # if the lower bound of the q_min credible interval is less than the
        # smallest sample, then the peak is against an edge so set the 
        # quantile of the origin to zero
        return 0.0

    # start from the fact that the quantile of zero must be between q_min
    # and q_max = 1, and progressively halve the interval following a 
    # root-finding algorithm
    q_max = 1.0
    while q_max - q_min > q_tol:
        # find the lower edge of the HPD corresponding to the average CL
        q_mid = 0.5*(q_min + q_max)
        l, _ = compute_hpd(xs, q_mid, sorted=True)

        if l == xmin:
            # this HPD hits the left edge, so we can shrink the upper bound
            q_max = q_mid
        else:
            # this HPD does not hit that edge, so we shrink the lower bound
            q_min = q_mid
    # once we are done, we have zeroed-in onto the tightest q that is bounded
    # by zero from below; since we have finite tolerance, return the mean
    # of the two (tight) bounds
    return 0.5*(q_min + q_max)


def hpd_zero_p_value(xs, min=None, max=None):
    if min is not None or max is not None:
        kde = Bounded_1d_kde(xs, x_min=min, x_max=max)
    else:
        kde = gaussian_kde(xs)
    p0 = kde(0)
    pxs = kde(xs)
    n = np.sum(pxs < p0)
    return n/len(xs)


def z_score(credible_level, two_tailed=False):
    # Compute the z-score
    alpha = 1 - credible_level  # Tail probability
    if two_tailed:
        alpha /= 2
    return norm.ppf(1 - alpha)

def get_hpd_from_samples(xs, q=0.68, min=None, max=None, n_grid=1000):
    if min is not None or max is not None:
        kde = Bounded_1d_kde(xs, x_min=min, x_max=max)
    else:
        kde = gaussian_kde(xs)
    if min is None:
        min = xs.min()
    if max is None:
        max = xs.max()
    grid = np.linspace(min, max, n_grid)
    density = kde(grid)
    density_sorted = np.sort(density)[::-1]
    density_threshold = density_sorted[int(q * len(density_sorted))]
    hpd_mask = density >= density_threshold
    return grid[hpd_mask][0], grid[hpd_mask][-1]
    

def get_hpd_from_grid(x, q=0.68, A_min=0):
    xs = x.sort_values(ascending=False)
    d = x.index[1] - x.index[0]
    lebesgue_integral = np.cumsum(xs)*d
    # normalize it
    lebesgue_integral /= max(lebesgue_integral)
    l, h = np.sort(np.abs(lebesgue_integral-q).sort_values().index.values[:2])
    if np.abs(l-h) < 2*d:
        # the interval has effectively zero width
        # assume that's because we've hit an edge on the LHS and set
        # that lower value to the minimum
        l = A_min
    # print(q, lebesgue_integral[l], lebesgue_integral[h])
    return l, h


def get_quantile_from_grid(x, q=0.5, return_q=False):
    xs = x.sort_index(ascending=True)
    d = x.index[1] - x.index[0]
    cdf = np.cumsum(xs)*d
    cdf /= max(cdf)
    i = (np.abs(cdf - q)).idxmin()
    if np.abs(cdf[i] - q) > 0.01:
        print(f"WARNING: quantile error greater than 1% ({cdf[i]})")
    if return_q:
        return i, cdf[i]
    else:
        return i


def get_sym_from_grid(x, q=0.68):
    xs = x.sort_index(ascending=True)
    d = x.index[1] - x.index[0]
    # integrate from left and right until reaching half of the unenclosed prob
    p = (1 - q)/2
    cdf = np.cumsum(xs)*d
    cdf /= max(cdf)
    l = (np.abs(cdf - p)).idxmin()
    h = (np.abs(1 - cdf - p)).idxmin()
    return l, h
