"""Temporary file to smoke-test the automated Claude code review.

This PR will be closed without merging; the file contains a deliberate bug.
"""


def mean_of_last_n(values, n):
    """Return the mean of the last n elements of values."""
    tail = values[-n:]
    return sum(tail) / n
