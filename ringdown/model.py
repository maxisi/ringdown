__all__ = ['make_model']

from dataclasses import dataclass
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
from . import qnms
import warnings

# reference frequency and mass values to translate linearly between the two
FREF = 2985.668287014743
MREF = 68.0

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

def rd_design_matrix(t0s, ts, f, gamma, Fp, Fc, Ascales):
    ts = jnp.as_tensor(ts)
    nifo, nt = ts.shape

    nmode = f.shape[0]

    t0s = jnp.reshape(t0s, (nifo, 1, 1))
    ts = jnp.reshape(ts, (nifo, 1, nt))
    f = jnp.reshape(f, (1, nmode, 1))
    gamma = jnp.reshape(gamma, (1, nmode, 1))
    Fp = jnp.reshape(Fp, (nifo, 1, 1))
    Fc = jnp.reshape(Fc, (nifo, 1, 1))
    Ascales = jnp.reshape(Ascales, (1, nmode, 1))

    t = ts - t0s

    ct = jnp.cos(2*np.pi*f*t)
    st = jnp.sin(2*np.pi*f*t)
    decay = jnp.exp(-gamma*t)
    return jnp.concatenate((Ascales*Fp*decay*ct, Ascales*Fp*decay*st, Ascales*Fc*decay*ct, Ascales*Fc*decay*st), axis=1)

def chi_factors(chi, coeffs):
    log1mc = jnp.log1p(-chi)
    log1mc2 = log1mc*log1mc
    log1mc3 = log1mc2*log1mc
    log1mc4 = log1mc2*log1mc2
    v = jnp.stack([chi, jnp.array([1.0]), log1mc, log1mc2,
                   log1mc3, log1mc4])
    return jnp.dot(coeffs, v)

def compute_h_det_mode(t0s, ts, Fps, Fcs, fs, gammas, Apxs, Apys, Acxs, Acys):
    ndet = len(t0s)
    nmode = fs.shape[0]
    nsamp = ts[0].shape[0]

    t0s = jnp.array(t0s).reshape((ndet, 1, 1))
    ts = jnp.array(ts).reshape((ndet, 1, nsamp))
    Fps = jnp.array(Fps).reshape((ndet, 1, 1))
    Fcs = jnp.array(Fcs).reshape((ndet, 1, 1))
    fs = jnp.array(fs).reshape((1, nmode, 1))
    gammas = jnp.array(gammas).reshape((1, nmode, 1))
    Apxs = jnp.array(Apxs).reshape((1, nmode, 1))
    Apys = jnp.array(Apys).reshape((1, nmode, 1))
    Acxs = jnp.array(Acxs).reshape((1, nmode, 1))
    Acys = jnp.array(Acys).reshape((1, nmode, 1))

    return rd(ts - t0s, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)

def a_from_quadratures(Apx, Apy, Acx, Acy):
    A = 0.5*(jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy)) +
             jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy)))
    return A

def ellip_from_quadratures(Apx, Apy, Acx, Acy):
    A = a_from_quadratures(Apx, Apy, Acx, Acy)
    e = 0.5*(jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy)) -
             jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy))) / A
    return e

def Aellip_from_quadratures(Apx, Apy, Acx, Acy):
    # should be slightly cheaper than calling the two functions separately
    term1 = jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy))
    term2 = jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy))
    A = 0.5*(term1 + term2)
    e = 0.5*(term1 - term2) / A
    return A, e

def phiR_from_quadratures(Apx, Apy, Acx, Acy):
    return jnp.arctan2(-Acx + Apy, Acy + Apx)

def phiL_from_quadratures(Apx, Apy, Acx, Acy):
    return jnp.arctan2(-Acx - Apy, -Acy + Apx)

def flat_A_quadratures_prior(Apx_unit, Apy_unit, Acx_unit, Acy_unit):
    return 

def get_quad_derived_quantities(design_matrices, quads, a_scale, store_h_det, store_h_det_mode, compute_h_det=False):
    nifo, nmodes, ntimes = design_matrices.shape
    apx = numpyro.deterministic('apx', quads[:nmodes] * a_scale)
    apy = numpyro.deterministic('apy', quads[nmodes:2*nmodes] * a_scale)
    acx = numpyro.deterministic('acx', quads[2*nmodes:3*nmodes] * a_scale)
    acy = numpyro.deterministic('acy', quads[3*nmodes:] * a_scale)

    a = numpyro.deterministic('a', a_from_quadratures(apx, apy, acx, acy))
    ellip = numpyro.deterministic('ellip', ellip_from_quadratures(apx, apy, acx, acy))
    phi_r = numpyro.deterministic('phi_r', phiR_from_quadratures(apx, apy, acx, acy))
    phi_l = numpyro.deterministic('phi_l', phiL_from_quadratures(apx, apy, acx, acy))
    theta = numpyro.deterministic('theta', -0.5*(phi_r + phi_l))
    phi = numpyro.deterministic('phi', 0.5*(phi_r - phi_l))

    if compute_h_det or store_h_det_mode or store_h_det:
        h_det_mode = jnp.zeros((nifo, nmodes, ntimes))
        hh = design_matrices * quads[jnp.newaxis,:,jnp.newaxis]

        for i in range(nmodes):
            h_det_mode = h_det_mode.at[:,i,:].set(jnp.sum(hh[:,i:nmodes:,:], axis=1))
        h_det = jnp.sum(h_det_mode, axis=1)

        if store_h_det_mode:
            _ = numpyro.deterministic('h_det_mode', h_det_mode)
        if store_h_det:
            _ = numpyro.deterministic('h_det', h_det)
        
        return a, h_det

def make_model(modes : int | list[(int, int, int, int)], 
               marginalized : bool = True, 
               a_scale_max : float = None, 
               m_min : float | None = None, m_max : float | None = None,
               chi_min : float = 0.0, chi_max : float = 0.99,
               df_min : None | list[None | float] = None, df_max : None | list[None | float] = None,
               dg_min : None | list[None | float] = None, dg_max : None | list[None | float] = None,
               f_min : None | list[float] = None, f_max : None | list[float] = None,
               g_min : None | list[float] = None, g_max : None | list[float] = None,
               flat_amplitude_prior : bool = False,
               modes_ordered_by_frequency : bool = False,
               prior : bool = False,               
               predictive : bool = False, store_h_det : bool = True, store_h_det_mode : bool = True,
               **kwargs):
    """
    Arguments
    ---------
    modes : int or list[tuple]
        If integer, the number of `f`, `tau` modes to use.  If list of tuples,
        each entry should be of the form `(p, s, ell, m)`, where `p` is `1` for
        prograde `-1` for retrograde; `s` is the spin weight (`-2` for the usual
        GW modes); and `ell` and `m` refer to the usual angular quantum numbers.
    """

    def model(t0s, times, strains, ls, fps, fcs):

        # Here is where the particular model choice is made:
        #
        # If modes is an int, then we use a model with f-gamma (i.e. we don't
        # impose any GR constraint on the frequencies / damping rates)
        #
        # If modes is a list, then we use a model with GR-imposed frequencies
        # and damping rates (possibly with deviations).  If you have a
        # beyond-Kerr model, this is where you would put your logic to implement
        # it.
        if isinstance(modes, int):
            if modes_ordered_by_frequency:
                f = numpyro.sample('f', dist.Uniform(f_min, f_max, support=dist.constraints.ordered_vector), sample_shape=(modes,))
                g = numpyro.sample('g', dist.Uniform(g_min, g_max), sample_shape=(modes,))
            else:
                f = numpyro.sample('f', dist.Uniform(f_min, f_max), sample_shape=(modes,))
                g = numpyro.sample('g', dist.Uniform(g_min, g_max, support=dist.constraints.ordered_vector), sample_shape=(modes,))
        elif isinstance(modes, list):
            fcs = []
            gcs = []
            for mode in modes:
                c = qnms.KerrMode(mode).coefficients
                fcs.append(c[0])
                gcs.append(c[1])
            fcs = jnp.array(fcs)
            gcs = jnp.array(gcs)

            m = numpyro.sample('m', dist.Uniform(m_min, m_max))
            chi = numpyro.sample('chi', dist.Uniform(chi_min, chi_max))

            f0 = FREF*MREF/m
            f_gr = f0*chi_factors(chi, fcs)
            g_gr = f0*chi_factors(chi, gcs)

            if df_min is None or df_max is None:
                f = numpyro.deterministic('f', f_gr)
            else:
                df_unit = numpyro.sample('df_unit', dist.Uniform(0, 1), sample_shape=(len(modes),))
                df_min = jnp.array([0.0 if x is None else x for x in df_min])
                df_max = jnp.array([0.0 if x is None else x for x in df_max])
                
                df = numpyro.deterministic('df', df_unit*(df_max - df_min) + df_min)
                f = numpyro.deterministic('f', f_gr * jnp.exp(df))

            if dg_min is None or dg_max is None:
                g = numpyro.deterministic('g', g_gr)
            else:
                dg_unit = numpyro.sample('dg_unit', dist.Uniform(0, 1), sample_shape=(len(modes),))
                dg_min = jnp.array([0.0 if x is None else x for x in dg_min])
                dg_max = jnp.array([0.0 if x is None else x for x in dg_max])
                
                dg = numpyro.deterministic('dg', dg_unit*(dg_max - dg_min) + dg_min)
                g = numpyro.deterministic('g', g_gr * jnp.exp(dg))
        # At this point the frequencies `f` and damping rates `g` of the various
        # modes should be established, and we can proceed with the rest of the
        # model.

        tau = numpyro.deterministic('tau', 1/g)
        omega = numpyro.deterministic('omega', 2*np.pi*f)
        quality = numpyro.deterministic('quality', np.pi*f*tau)

        if marginalized:
            a_scale = numpyro.sample('a_scale', dist.Uniform(0, a_scale_max), sample_shape=(len(modes),))
            design_matrices = rd_design_matrix(t0s, times, f, g, fps, fcs, a_scale)

            mu = jnp.zeros(4*len(modes))
            lambda_inv = jnp.eye(4*len(modes))
            lambda_inv_chol = jnp.eye(4*len(modes))
            if not prior:
                for i in range(len(t0s)):
                    mm = design_matrices[i,:,:].T # (ndet, 4*nmode, ntime) => (i, ntime, 4*nmode)

                    a_inv = lambda_inv + jnp.dot(mm.T, jsp.linalg.cho_solve((ls[i], True), mm))
                    a_inv_chol = jsp.linalg.cholesky(a_inv)

                    a = jsp.linalg.cho_solve((a_inv_chol, True), jnp.dot(lambda_inv, mu) + jnp.dot(mm.T, jsp.linalg.cho_solve((ls[i], True), strains[i])))

                    b = jnp.dot(mm, mu)

                    blogsqrtdet = jnp.sum(jnp.log(jnp.diag(ls[i]))) - jnp.sum(jnp.log(jnp.diag(lambda_inv_chol))) + jnp.sum(jnp.log(jnp.diag(a_inv_chol)))

                    r = strains[i] - b
                    cinv_r = jsp.linalg.cho_solve((ls[i], True), r)
                    mamtcinv_r = jnp.dot(mm, jsp.linalg.cho_solve((a_inv_chol, True), jnp.dot(mm.T, cinv_r)))
                    cinvmamtcinv_r = jsp.linalg.cho_solve((ls[i], True), mamtcinv_r)
                    logl = -0.5*jnp.dot(r, cinv_r - cinvmamtcinv_r) - blogsqrtdet

                    numpyro.factor(f'strain_logl_{i}', logl)

                    mu = a
                    lambda_inv = a_inv
                    lambda_inv_chol = a_inv_chol
                
            if predictive:
                # Generate the actual quadrature amplitudes 

                # Lambda_inv_chol.T: Lambda_inv = Lambda_inv_chol * Lambda_inv_chol.T,
                # so Lambda = (Lambda_inv_chol.T)^{-1} Lambda_inv_chol^{-1} To achieve
                # the desired covariance, we can *right multiply* iid N(0,1) variables
                # by Lambda_inv_chol^{-1}, so that y = x Lambda_inv_chol^{-1} has
                # covariance < y^T y > = (Lambda_inv_chol^{-1}).T < x^T x >
                # Lambda_inv_chol^{-1} = (Lambda_inv_chol^{-1}).T I Lambda_inv_chol^{-1}
                # = Lambda.
                apx_unit = numpyro.sample('apx_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
                apy_unit = numpyro.sample('apy_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
                acx_unit = numpyro.sample('acx_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
                acy_unit = numpyro.sample('acy_unit', dist.Normal(0, 1), sample_shape=(len(modes),))

                quads = mu + jsp.linalg.solve(lambda_inv_chol.T, jnp.concatenate((apx_unit, apy_unit, acx_unit, acy_unit)))

                get_quad_derived_quantities(design_matrices, quads, a_scale, store_h_det, store_h_det_mode)
        else:
            design_matrices = rd_design_matrix(t0s, times, f, g, fps, fcs, a_scale_max)
            apx_unit = numpyro.sample('apx_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
            apy_unit = numpyro.sample('apy_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
            acx_unit = numpyro.sample('acx_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
            acy_unit = numpyro.sample('acy_unit', dist.Normal(0, 1), sample_shape=(len(modes),))

            quads = jnp.concatenate((apx_unit, apy_unit, acx_unit, acy_unit))
            a, h_det = get_quad_derived_quantities(design_matrices, quads, a_scale_max, 
                                                   store_h_det, store_h_det_mode, compute_h_det=(not prior))
            
            if flat_amplitude_prior:
                # We need a Jacobian that is A^-3
                numpyro.factor('flat_A_prior', -3*jnp.sum(jnp.log(a)) + \
                               0.5*jnp.sum((jnp.square(apx_unit) + jnp.square(apy_unit) + \
                               jnp.square(acx_unit) + jnp.square(acy_unit))))
                
                if prior:
                    raise ValueError('you did not want to impose a flat amplitude prior without a likelihood')

            if not prior:
                for i, strain in enumerate(strains):
                    numpyro.sample(f'strain_logl_{i}', dist.MultivariateNormal(h_det[i,:], scale_tril=ls[i]), obs=strain)

    return model