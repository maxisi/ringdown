import numpy as np
from scipy.stats import gaussian_kde, norm
from .kde_contour import Bounded_1d_kde
import logging


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


def compute_hpd_with_kde(xs, q=0.68, min=None, max=None, grid=1000):
    if min is not None or max is not None:
        kde = Bounded_1d_kde(xs, x_min=min, x_max=max)
    else:
        kde = gaussian_kde(xs)
    if min is None:
        min = xs.min()
    if max is None:
        max = xs.max()

    # evaluate the KDE over a grid
    if isinstance(grid, int):
        grid = np.linspace(min, max, grid)
    density = kde(grid)

    # Compute threshold to find the highest density region
    sorted_indices = np.argsort(density)[::-1]
    density_cumsum = np.cumsum(density[sorted_indices])
    idx = np.searchsorted(density_cumsum, q * density_cumsum[-1])
    density_threshold = density[sorted_indices][idx]

    # Identify HPD region
    hpd_mask = density >= density_threshold
    # Get min and max bounds
    hpd_bounds = grid[np.flatnonzero(hpd_mask)[[0, -1]]]

    return hpd_bounds[0], hpd_bounds[1]


def compute_hpd_with_kde_multimodal(samples, q=0.68, n_grid=1000):
    """
    Compute disjoint HPD intervals using KDE for a (potentially
    multimodal) distribution.

    Parameters:
      samples : array-like
          Posterior samples.
      p : float
          Desired probability mass (e.g., 0.95 for a 95% HPD region).
      n_grid : int
          Number of points in the grid for density evaluation.

    Returns:
      intervals : list of tuples
          List of intervals [(low1, high1), (low2, high2), ...] that together
          cover at least p probability mass.
    """
    # Compute the KDE over a grid spanning the sample range.
    kde = gaussian_kde(samples)
    x_min, x_max = samples.min(), samples.max()
    grid = np.linspace(x_min, x_max, n_grid)
    density = kde(grid)

    # Compute grid spacing.
    dx = grid[1] - grid[0]

    # Sort density values in descending order.
    sort_idx = np.argsort(density)[::-1]
    density_sorted = density[sort_idx]

    # Compute cumulative probability mass over the sorted grid points.
    cumulative_prob = np.cumsum(density_sorted) * dx
    # Find the index where cumulative probability reaches p.
    idx_threshold = np.searchsorted(cumulative_prob, q)
    if idx_threshold >= len(density_sorted):
        idx_threshold = len(density_sorted) - 1
    threshold = density_sorted[idx_threshold]

    # Identify grid points where the density exceeds the threshold.
    mask = density >= threshold
    indices = np.nonzero(mask)[0]
    if len(indices) == 0:
        return []

    # Group contiguous indices using a vectorized approach.
    # Compute the difference between consecutive indices.
    diff = np.diff(indices)
    # Where the difference is greater than 1, there is a break.
    split_indices = np.where(diff > 1)[0] + 1
    groups = np.split(indices, split_indices)

    # Build the list of intervals from the groups.
    intervals = [(grid[group[0]], grid[group[-1]]) for group in groups]

    return intervals


def q_of_zero_old(xs, q_min=0.01, q_tol=0.001, kde=False, xmin=None,
                  **kws) -> float:
    """Compute the quantile of zero for a given set of nonnegative samples with
    a unimodal distribution.

    This function computes the quantile of zero for a given set of nonnegative
    samples. The quantile of zero is the smallest value of q such that the
    q-credible HPD of the samples includes zero.

    The algorithm works through bisection by progressively halving the
    interval between q_min and 1, and checking if the HPD of the samples at
    the mid-point includes zero. If it does, the upper bound is shrunk to the
    mid-point; otherwise, the lower bound is shrunk to the mid-point. The
    algorithm continues until the interval is smaller than a given tolerance.

    It is different from the CL computation because it does not max out
    at 1/len(xs).

    Arguments
    ---------
    xs : array_like
        The set of nonnegative samples.
    q_min : float, optional
        The lower bound of the quantile. The default is 0.01.
    q_tol : float, optional
        The tolerance for the quantile. The default is 0.001.

    Returns
    -------
    q: float
        The quantile of zero.
    """
    logging.warning("This function is deprecated. Use `quantile_at_value`.")

    xs = np.sort(xs)
    if xmin is None:
        xmin = xs[0]

    if xmin < 0:
        raise ValueError("The smallest sample is less than zero.")

    if kde:
        hpd = compute_hpd_with_kde
        # make sure we are using the bounded KDE or this will not work
        # well if the posterior is close to zero
        kws['min'] = kws.get('min', 0)
        xmin = 0
    else:
        kws["sorted"] = True
        hpd = compute_hpd

    if hpd(xs, q_min, **kws)[0] < xmin:
        # if the lower bound of the q_min credible interval is below the
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
        low, _ = hpd(xs, q_min, **kws)

        if low == xmin:
            # this HPD hits the left edge, so we can shrink the upper bound
            q_max = q_mid
        else:
            # this HPD does not hit that edge, so we shrink the lower bound
            q_min = q_mid
    # once we are done, we have zeroed-in onto the tightest q that is bounded
    # by zero from below; since we have finite tolerance, return the mean
    # of the two (tight) bounds
    return 0.5*(q_min + q_max)


def quantile_at_value(xs, target=0, min=None, max=None, z_score=False,
                      interpolate=False, silent=False):
    if min is not None or max is not None:
        kde = Bounded_1d_kde(xs, x_min=min, x_max=max)
    else:
        kde = gaussian_kde(xs)
    p0 = kde(target)
    pxs = kde(xs)

    if interpolate:
        # sort densities
        sorted_pxs = np.sort(pxs)
        # form backwards grid to agree with q definition above
        qs = np.linspace(1, 0, len(sorted_pxs))
        # Interpolate handling beyond bounds
        q = np.interp(p0, sorted_pxs, qs, left=1, right=0)[0]
    else:
        n = np.sum(pxs > p0)
        q = n/len(xs)

    if q == 1 and not silent:
        logging.warning("CL maxed out at 1/N")

    if z_score:
        return z_score(q)
    return q


def z_score(credible_level, two_tailed=False):
    # Compute the z-score
    alpha = 1 - credible_level  # Tail probability
    if two_tailed:
        alpha /= 2
    return norm.ppf(1 - alpha)
