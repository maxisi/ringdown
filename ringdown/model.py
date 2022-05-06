__all__ = ['mchi_model', 'mchi_aligned_model']

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist

import numpy as np

def rd(ts, f, gamma, Apx, Apy, Acx, Acy, Fp, Fc):
    """Generate a ringdown waveform as it appears in a detector.
    
    Arguments
    ---------
    
    ts : array_like
        The times at which the ringdown waveform should be evaluated.  
    f : real
        The frequency.
    gamma : real
        The damping rate.
    Apx : real
        The amplitude of the "plus" cosine-like quadrature.
    Apy : real
        The amplitude of the "plus" sine-like quadrature.
    Acx : real
        The amplitude of the "cross" cosine-like quadrature.
    Acy : real
        The amplitude of the "cross" sine-like quadrature.
    Fp : real
        The coefficient of the "plus" polarization in the detector.
    Fc : real
        The coefficient of the "cross" term in the detector.

    Returns
    -------

    Array of the ringdown waveform in the detector.
    """
    ct = jnp.cos(2*np.pi*f*ts)
    st = jnp.sin(2*np.pi*f*ts)
    decay = jnp.exp(-gamma*ts)
    p = decay*(Apx*ct + Apy*st)
    c = decay*(Acx*ct + Acy*st)

    return Fp*p + Fc*c

def chi_factors(chi, coeffs):
    log1mc = jnp.log1p(-chi)
    log1mc2 = log1mc*log1mc
    log1mc3 = log1mc2*log1mc
    log1mc4 = log1mc2*log1mc2
    v = jnp.stack([chi, jnp.asarray(1.0), log1mc, log1mc2, log1mc3, log1mc4])

    return jnp.dot(coeffs, v)

def get_snr(h, d, L):
    wh = jsl.solve_triangular(L, h, lower=True)
    wd = jsl.solve_triangular(L, h, lower=True)

    return jnp.dot(wh, wd) / jnp.sqrt(jnp.dot(wh, wh))

def compute_h_det_mode(ts, Fps, Fcs, fs, gammas, Apxs, Apys, Acxs, Acys):
    ndet = ts.shape[0]
    nmode = fs.shape[0]
    nsamp = ts[0].shape[0]

    ts = jnp.asarray(ts).reshape((ndet, 1, nsamp))
    Fps = jnp.asarray(Fps).reshape((ndet, 1, 1))
    Fcs = jnp.asarray(Fcs).reshape((ndet, 1, 1))
    fs = jnp.asarray(fs).reshape((1, nmode, 1))
    gammas = jnp.asarray(gammas).reshape((1, nmode, 1))
    Apxs = jnp.asarray(Apxs).reshape((1, nmode, 1))
    Apys = jnp.asarray(Apys).reshape((1, nmode, 1))
    Acxs = jnp.asarray(Acxs).reshape((1, nmode, 1))
    Acys = jnp.asarray(Acys).reshape((1, nmode, 1))

    return rd(ts, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)

def mchi_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, **kwargs):
    M_min = kwargs.pop("M_min", 35.0)
    M_max = kwargs.pop("M_max", 140.0)
    chi_min = kwargs.pop("chi_min", 0.0)
    chi_max = kwargs.pop("chi_max", 0.99)
    A_scale = kwargs.pop("A_scale", 1e-21)
    df_max = kwargs.pop("df_max", 0.5)
    dtau_max = kwargs.pop("dtau_max", 0.5)
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)

    assert not (flat_A and flat_A_ellip), "at most one of `flat_A` and `flat_A_ellip` can be `True`"

    ndet = len(t0)
    nmode = f_coeffs.shape[0]

    fref = 2985.668287014743
    mref = 68.0

    times = jnp.asarray(times - np.array(t0)[:,np.newaxis])
    Ls = jnp.asarray(Ls)
    strains = jnp.asarray(strains)

    perturb_f = jnp.asarray(perturb_f)
    perturb_tau = jnp.asarray(perturb_tau)

    M = numpyro.sample('M', dist.Uniform(M_min, M_max))
    chi = numpyro.sample("chi", dist.Uniform(chi_min, chi_max))

    Apx_unit = numpyro.sample("Apx_unit", dist.ImproperUniform(dist.constraints.real, (), event_shape=(nmode,)))
    Apy_unit = numpyro.sample("Apy_unit", dist.ImproperUniform(dist.constraints.real, (), event_shape=(nmode,)))
    Acx_unit = numpyro.sample("Acx_unit", dist.ImproperUniform(dist.constraints.real, (), event_shape=(nmode,)))
    Acy_unit = numpyro.sample("Acy_unit", dist.ImproperUniform(dist.constraints.real, (), event_shape=(nmode,)))

    df = numpyro.sample("df", dist.Uniform(-df_max, df_max), sample_shape=(nmode,))
    dtau = numpyro.sample("dtau", dist.Uniform(-dtau_max, dtau_max), sample_shape=(nmode,))

    Apx = numpyro.deterministic("Apx", A_scale*Apx_unit)
    Apy = numpyro.deterministic("Apy", A_scale*Apy_unit)
    Acx = numpyro.deterministic("Acx", A_scale*Acx_unit)
    Acy = numpyro.deterministic("Acy", A_scale*Acy_unit)

    A = numpyro.deterministic("A", A_scale*0.5*(jnp.sqrt(jnp.square(Acy_unit + Apx_unit) + jnp.square(Acx_unit - Apy_unit)) + jnp.sqrt(jnp.square(Acy_unit - Apx_unit) + jnp.square(Acx_unit + Apy_unit))))
    ellip = numpyro.deterministic("ellip", (jnp.sqrt(jnp.square(Acy_unit + Apx_unit) + jnp.square(Acx_unit - Apy_unit)) - jnp.sqrt(jnp.square(Acy_unit - Apx_unit) + jnp.square(Acx_unit + Apy_unit))) / (jnp.sqrt(jnp.square(Acy_unit + Apx_unit) + jnp.square(Acx_unit - Apy_unit)) + jnp.sqrt(jnp.square(Acy_unit - Apx_unit) + jnp.square(Acx_unit + Apy_unit))))

    f0 = fref*mref/M
    f = numpyro.deterministic("f", f0*chi_factors(chi, f_coeffs) * jnp.exp(df * perturb_f))
    gamma = numpyro.deterministic("gamma", f0*chi_factors(chi, g_coeffs) * jnp.exp(-dtau * perturb_tau))
    tau = numpyro.deterministic("tau", 1/gamma)
    Q = numpyro.deterministic("Q", np.pi * f * tau)
    phiR = numpyro.deterministic("phiR", jnp.arctan2(-Acx + Apy, Acy + Apx))
    phiL = numpyro.deterministic("phiL", jnp.arctan2(-Acx - Apy, -Acy + Apx))
    theta = numpyro.deterministic("theta", -0.5*(phiR + phiL))
    phi = numpyro.deterministic("phi", 0.5*(phiR - phiL))

    h_det_mode = numpyro.deterministic("h_det_mode", compute_h_det_mode(times, Fps, Fcs, f, gamma, Apx, Apy, Acx, Acy))
    h_det = numpyro.deterministic("h_det", jnp.sum(h_det_mode, axis=1))

    # Priors:

    # Flat in M-chi already

    # Amplitude prior
    if flat_A:
        numpyro.factor("flat_A_prior", -3*jnp.sum(jnp.log(A)))
    elif flat_A_ellip:
        numpyro.factor("flat_A_ellip_prior", jnp.sum(-3*jnp.log(A) - jnp.log1m(jnp.square(ellip))))
    else:
        numpyro.factor("gaussian_A_quadratures_prior", -0.5*jnp.sum(jnp.square(Apx_unit) + jnp.square(Apy_unit) + jnp.square(Acx_unit) + jnp.square(Acy_unit)))

    # Flat prior on the delta-fs and delta-taus

    # Likelihood:
    numpyro.sample('strain', dist.MultivariateNormal(loc=h_det, scale_tril=Ls), obs=strains)
        
def mchi_aligned_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, **kwargs):
    M_min = kwargs.pop("M_min", 35.0)
    M_max = kwargs.pop("M_max", 140.0)
    chi_min = kwargs.pop("chi_min", 0.0)
    chi_max = kwargs.pop("chi_max", 0.99)
    cosi_min = kwargs.pop("cosi_min", -1.0)
    cosi_max = kwargs.pop("cosi_max", 1.0)
    A_scale = kwargs.pop("A_scale", 1e-21)
    df_max = kwargs.pop("df_max", 0.5)
    dtau_max = kwargs.pop("dtau_max", 0.5)
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    
    ndet = len(t0)
    nmode = f_coeffs.shape[0]

    fref = 2985.668287014743
    mref = 68.0

    times = jnp.asarray(times - np.array(t0)[:,np.newaxis])
    Ls = jnp.asarray(Ls)
    strains = jnp.asarray(strains)

    perturb_f = jnp.asarray(perturb_f)
    perturb_tau = jnp.asarray(perturb_tau)

    M = numpyro.sample("M", dist.Uniform(M_min, M_max))
    chi = numpyro.sample("chi", dist.Uniform(chi_min, chi_max))

    cosi = numpyro.sample("cosi", dist.Uniform(cosi_min, cosi_max))

    Ax_unit = numpyro.sample("Ax_unit", dist.ImproperUniform(dist.constraints.real, (), event_shape=(nmode,)))
    Ay_unit = numpyro.sample("Ay_unit", dist.ImproperUniform(dist.constraints.real, (), event_shape=(nmode,)))

    df = numpyro.sample("df", dist.Uniform(-df_max, df_max), sample_shape=(nmode,))
    dtau = numpyro.sample("dtau", dist.Uniform(-dtau_max, dtau_max), sample_shape=(nmode,))

    A = numpyro.deterministic("A", A_scale*jnp.sqrt(jnp.square(Ax_unit)+jnp.square(Ay_unit)))
    phi = numpyro.deterministic("phi", jnp.arctan2(Ay_unit, Ax_unit))

    f0 = fref*mref/M
    f = numpyro.deterministic('f', f0*chi_factors(chi, f_coeffs)*jnp.exp(df * perturb_f))
    gamma = numpyro.deterministic('gamma', f0*chi_factors(chi, g_coeffs)*jnp.exp(-dtau * perturb_tau))
    tau = numpyro.deterministic('tau', 1/gamma)
    Q = numpyro.deterministic('Q', np.pi*f*tau)
    Ap = numpyro.deterministic('Ap', (1 + jnp.square(cosi))*A)
    Ac = numpyro.deterministic('Ac', 2*cosi*A)
    ellip = numpyro.deterministic('ellip', Ac/Ap)

    Apx = (1 + jnp.square(cosi))*A*jnp.cos(phi)
    Apy = (1 + jnp.square(cosi))*A*jnp.sin(phi)
    Acx = -2*cosi*A*jnp.sin(phi)
    Acy = 2*cosi*A*jnp.cos(phi)

    h_det_mode = numpyro.deterministic("h_det_mode", compute_h_det_mode(times, Fps, Fcs, f, gamma, Apx, Apy, Acx, Acy))
    h_det = numpyro.deterministic("h_det", jnp.sum(h_det_mode, axis=1))

    # Priors:

    # Flat in M-chi already

    # Amplitude prior
    if flat_A:
        numpyro.factor("flat_A_prior", -jnp.sum(jnp.log(A)))
    else:
        numpyro.factor("gaussian_A_quadratures_prior", -0.5*jnp.sum(jnp.square(Ax_unit) + jnp.square(Ay_unit)))

    # Flat prior on the delta-fs and delta-taus

    # Likelihood
    numpyro.sample('likelihood', dist.MultivariateNormal(loc=h_det, scale_tril=Ls), obs=strains)


