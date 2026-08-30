"""Small smoothing helpers for test comparisons."""


def moving_average(x, w):
    """Return the moving average of sequence x with window length w."""
    out = []
    for i in range(len(x) - w):
        out.append(sum(x[i:i + w]) / w)
    return out
