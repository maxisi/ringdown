import jax.numpy as jnp

# =============================================================================
# OPTIMIZATION HELPERS (FFT / Gohberg-Semencul)
# =============================================================================


def next_fast_len(target):
    """Find the next power of 2 for efficient FFT."""
    return 1 << (int(target) - 1).bit_length()


def apply_matrix_fft_precomputed(x, filter_fft, N, n_fft):
    """
    Computes A @ x using precomputed FFT of the filter.
    A is Lower Triangular Toeplitz.
    """
    X_f = jnp.fft.rfft(x, n=n_fft)
    out_f = X_f * filter_fft
    out = jnp.fft.irfft(out_f, n=n_fft)
    return out[:N]


def apply_cinv_gs_fast(vector, fft_a, fft_b, n_fft, sigma):
    """
    Computes x^T C^{-1} y using Gohberg-Semencul with precomputed FFTs.
    Returns C^{-1} @ vector.
    """
    N = vector.shape[0]

    # Helper to apply A @ v (Lower Triangular Toeplitz)
    def apply_A(v):
        return apply_matrix_fft_precomputed(v, fft_a, N, n_fft)

    # Helper to apply A.T @ v = Flip( A @ Flip(v) )
    def apply_At(v):
        return jnp.flip(apply_A(jnp.flip(v)))

    # Helper to apply B @ v
    def apply_B(v):
        return apply_matrix_fft_precomputed(v, fft_b, N, n_fft)

    # Helper to apply B.T @ v
    def apply_Bt(v):
        return jnp.flip(apply_B(jnp.flip(v)))

    # GS Formula: C^{-1} r = (1/sigma^2) * [ A(A.T r) - B(B.T r) ]

    # Term 1: A @ (A.T @ vector)
    t1 = apply_A(apply_At(vector))

    # Term 2: B @ (B.T @ vector)
    t2 = apply_B(apply_Bt(vector))

    return (t1 - t2) / (sigma**2)
