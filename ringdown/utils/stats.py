import numpy as np
from scipy.stats import gaussian_kde, norm
from .kde_contour import Bounded_1d_kde


def compute_hpd(samples, p=0.68, out="both"):
    # compute the minimum-width p%-credible interval
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


# def hpd_interval(xs, q):
#     xs = np.sort(xs)
#     N = len(xs)
#     n = int(round(q*N))

#     intmin = -1
#     length = np.inf
#     for i in range(0, N-n+1):
#         l = xs[i+n-1]-xs[i]
#         if l < length:
#             intmin = i
#             length = l
#     return xs[intmin], xs[intmin+n-1]


def q_of_zero(xs):
    """Compute the quantile of zero for a given set of samples
    """
    xs = np.sort(xs)
    xmin = np.min(xs)

    qshort = 0.01
    if compute_hpd(xs, qshort)[0] < xmin:
        return 0.0
    qlong = 1.0
    while qlong - qshort > 0.005:
        qmid = 0.5*(qshort + qlong)
        l, _ = compute_hpd(xs, qmid)

        if l == xmin:
            qlong = qmid
        else:
            qshort = qmid
    return 0.5*(qshort + qlong)


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
