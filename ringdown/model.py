__all__ = ['make_model']

from dataclasses import dataclass
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
from . import qnms

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

def flat_A_quadratures_prior(Apx_unit, Apy_unit, Acx_unit, Acy_unit,flat_A):
    return 0.5*jnp.sum((jnp.square(Apx_unit) + jnp.square(Apy_unit) +
                        jnp.square(Acx_unit) + jnp.square(Acy_unit))*flat_A)

def get_quad_derived_quantities(design_matrices, quads, a_scale, predict_h_det, predict_h_det_mode):
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

    h_det_mode = jnp.zeros((nifo, nmodes, ntimes))
    hh = design_matrices * quads[jnp.newaxis,:,jnp.newaxis]

    for i in range(nmodes):
        h_det_mode = h_det_mode.at[:,i,:].set(jnp.sum(hh[:,i:nmodes:,:], axis=1))
    h_det = jnp.sum(h_det_mode, axis=1)

    if predict_h_det_mode:
        _ = numpyro.deterministic('h_det_mode', h_det_mode)
    if predict_h_det:
        _ = numpyro.deterministic('h_det', h_det)
    
    return h_det



def make_model(modes : int | list[(int, int, int, int)], 
               marginalized : bool =True, 
               a_scale_max : float =None, 
               m_min : float | None = None, m_max : float | None = None,
               chi_min : float = 0.0, chi_max : float = 0.99,
               df_min : None | list[None | float] = None, df_max : None | list[None | float] = None,
               dg_min : None | list[None | float] = None, dg_max : None | list[None | float] = None,
               f_min : None | list[float] = None, f_max : None | list[float] = None,
               g_min : None | list[float] = None, g_max : None | list[float] = None,
               prior : bool = False,               
               predictive : bool = False, predict_h_det : bool = True, predict_h_det_mode : bool = True,
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
        if isinstance(modes, int):
            f = numpyro.sample('f', dist.Uniform(f_min, f_max), sample_shape=(modes,))
            g = numpyro.sample('g', dist.Uniform(g_min, g_max), sample_shape=(modes,))

            tau = numpyro.deterministic('tau', 1/g)
            omega = numpyro.deterministic('omega', 2*np.pi*f)
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

                    _ = get_quad_derived_quantities(design_matrices, quads, a_scale, predict_h_det, predict_h_det_mode)
        else:
            design_matrices = rd_design_matrix(t0s, times, f, g, fps, fcs, a_scale_max)
            apx_unit = numpyro.sample('apx_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
            apy_unit = numpyro.sample('apy_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
            acx_unit = numpyro.sample('acx_unit', dist.Normal(0, 1), sample_shape=(len(modes),))
            acy_unit = numpyro.sample('acy_unit', dist.Normal(0, 1), sample_shape=(len(modes),))

            quads = jnp.concatenate((apx_unit, apy_unit, acx_unit, acy_unit))
            
            ## Incomplete!!  call get_quad_derived_quantities




def make_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale_max = kwargs.pop("A_scale_max")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)
    prior_run = kwargs.pop('prior_run', False)

    nmode = f_coeffs.shape[0]

    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    if not np.isscalar(df_min) and not np.isscalar(df_max):
        if len(df_min)!=len(df_max):
            raise ValueError("df_min, df_max must be scalar or arrays of length equal to the number of modes")
        for el in np.arange(len(df_min)):
            if df_min[el]==df_max[el]:
                raise ValueError("df_min and df_max must not be equal for any given mode")

    if not np.isscalar(dtau_min) and not np.isscalar(dtau_max):
        if len(dtau_min)!=len(dtau_max):
            raise ValueError("dtau_min, dtau_max must be scalar or arrays of length equal to the number of modes")
        for el in np.arange(len(dtau_min)):
            if dtau_min[el]==dtau_max[el]:
                raise ValueError("dtau_min and dtau_max must not be equal for any given mode")

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    Llogdet = np.array([np.sum(np.log(np.diag(L))) for L in Ls])

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        df = pm.Uniform("df", df_min, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=['mode'])

        # log_A_scale = pm.Uniform('log_A_scale', np.log(A_scale_min), np.log(A_scale_max), dims=['mode'])
        # A_scale = pm.Deterministic('A_scale', at.exp(log_A_scale))
        A_scale = pm.Uniform('A_scale', 0, A_scale_max, dims=['mode'])

        Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
        Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
        Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
        Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic("f",
            f0*chi_factors(chi, f_coeffs)*at.exp(df*perturb_f),
            dims=['mode'])
        gamma = pm.Deterministic("gamma",
            f0*chi_factors(chi, g_coeffs)*at.exp(-dtau*perturb_tau),
            dims=['mode'])
        tau = pm.Deterministic("tau", 1/gamma, dims=['mode'])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=['mode'])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential('f_min_cut', at.sum(at.where(f < f_min, np.NINF, 0.0)))
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential('f_max_cut', at.sum(at.where(f > f_max, np.NINF, 0.0)))

        # Priors:

        # Flat in M-chi already

        # Flat prior on the delta-fs and delta-taus

        design_matrices = rd_design_matrix(t0, times, f, gamma, Fps, Fcs, A_scale)

        mu = at.zeros(4*nmode)
        Lambda_inv = at.eye(4*nmode)
        Lambda_inv_chol = at.eye(4*nmode)
        # Likelihood:
        if not prior_run:
            for i in range(ndet):
                MM = design_matrices[i, :, :].T # (ndet, 4*nmode, ntime) => (i, ntime, 4*nmode)

                A_inv = Lambda_inv + at.dot(MM.T, _atl_cho_solve((Ls[i], True), MM))
                A_inv_chol = atl.cholesky(A_inv)

                a = _atl_cho_solve((A_inv_chol, True), at.dot(Lambda_inv, mu) + at.dot(MM.T, _atl_cho_solve((Ls[i], True), strains[i])))

                b = at.dot(MM, mu)
            
                Blogsqrtdet = Llogdet[i] - at.sum(at.log(at.diag(Lambda_inv_chol))) + at.sum(at.log(at.diag(A_inv_chol)))

                r = strains[i] - b
                Cinv_r = _atl_cho_solve((Ls[i], True), r)
                MAMTCinv_r = at.dot(MM, _atl_cho_solve((A_inv_chol, True), at.dot(MM.T, Cinv_r)))
                CinvMAMTCinv_r = _atl_cho_solve((Ls[i], True), MAMTCinv_r)
                logl = -0.5*at.dot(r, Cinv_r - CinvMAMTCinv_r) - Blogsqrtdet

                key = ifos[i]
                if isinstance(key, bytes):
                 # Don't want byte strings in our names!
                    key = key.decode('utf-8')

                pm.Potential(f'strain_{key}', logl)

                mu = a
                Lambda_inv = A_inv
                Lambda_inv_chol = A_inv_chol
        else:
            # We're done.  There is no likelihood.
            pass
        
        # Lambda_inv_chol.T: Lambda_inv = Lambda_inv_chol * Lambda_inv_chol.T,
        # so Lambda = (Lambda_inv_chol.T)^{-1} Lambda_inv_chol^{-1} To achieve
        # the desired covariance, we can *right multiply* iid N(0,1) variables
        # by Lambda_inv_chol^{-1}, so that y = x Lambda_inv_chol^{-1} has
        # covariance < y^T y > = (Lambda_inv_chol^{-1}).T < x^T x >
        # Lambda_inv_chol^{-1} = (Lambda_inv_chol^{-1}).T I Lambda_inv_chol^{-1}
        # = Lambda.
        theta = mu + atl.solve(Lambda_inv_chol.T, at.concatenate((Apx_unit, Apy_unit, Acx_unit, Acy_unit)))

        Apx = pm.Deterministic("Apx", theta[:nmode] * A_scale, dims=['mode'])
        Apy = pm.Deterministic("Apy", theta[nmode:2*nmode] * A_scale, dims=['mode'])
        Acx = pm.Deterministic("Acx", theta[2*nmode:3*nmode] * A_scale, dims=['mode'])
        Acy = pm.Deterministic("Acy", theta[3*nmode:] * A_scale, dims=['mode'])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy),
                             dims=['mode'])
        ellip = pm.Deterministic("ellip",
            ellip_from_quadratures(Apx, Apy, Acx, Acy),
            dims=['mode'])
        
        phiR = pm.Deterministic("phiR",
             phiR_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        phiL = pm.Deterministic("phiL",
             phiL_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL), dims=['mode'])
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL), dims=['mode'])

        h_det_mode = pm.Deterministic("h_det_mode",
                compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                                   Apx, Apy, Acx, Acy),
                dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        return model


def make_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale = kwargs.pop("A_scale")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    flat_A_ellip = kwargs.pop("flat_A_ellip", False)
    f_min = kwargs.pop('f_min', None)
    f_max = kwargs.pop('f_max', None)
    prior_run = kwargs.pop('prior_run', False)

    nmode = f_coeffs.shape[0]

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A,nmode)
    if np.isscalar(flat_A_ellip):
        flat_A_ellip = np.repeat(flat_A_ellip,nmode)
    elif len(flat_A)!=nmode:
        raise ValueError("flat_A must either be a scalar or array of length equal to the number of modes")
    elif len(flat_A_ellip)!=nmode:
        raise ValueError("flat_A_ellip must either be a scalar or array of length equal to the number of modes") 

    if any(flat_A) and any(flat_A_ellip):
        raise ValueError("at most one of `flat_A` and `flat_A_ellip` can have an element that is " "`True`")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")

    if not np.isscalar(df_min) and not np.isscalar(df_max):
        if len(df_min)!=len(df_max):
            raise ValueError("df_min, df_max must be scalar or arrays of length equal to the number of modes")
        for el in np.arange(len(df_min)):
            if df_min[el]==df_max[el]:
                raise ValueError("df_min and df_max must not be equal for any given mode")

    if not np.isscalar(dtau_min) and not np.isscalar(dtau_max):
        if len(dtau_min)!=len(dtau_max):
            raise ValueError("dtau_min, dtau_max must be scalar or arrays of length equal to the number of modes")
        for el in np.arange(len(dtau_min)):
            if dtau_min[el]==dtau_max[el]:
                raise ValueError("dtau_min and dtau_max must not be equal for any given mode")

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        Apx_unit = pm.Normal("Apx_unit", dims=['mode'])
        Apy_unit = pm.Normal("Apy_unit", dims=['mode'])
        Acx_unit = pm.Normal("Acx_unit", dims=['mode'])
        Acy_unit = pm.Normal("Acy_unit", dims=['mode'])

        df = pm.Uniform("df", df_min, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=['mode'])

        Apx = pm.Deterministic("Apx", A_scale*Apx_unit, dims=['mode'])
        Apy = pm.Deterministic("Apy", A_scale*Apy_unit, dims=['mode'])
        Acx = pm.Deterministic("Acx", A_scale*Acx_unit, dims=['mode'])
        Acy = pm.Deterministic("Acy", A_scale*Acy_unit, dims=['mode'])

        A = pm.Deterministic("A", a_from_quadratures(Apx, Apy, Acx, Acy),
                             dims=['mode'])
        ellip = pm.Deterministic("ellip",
            ellip_from_quadratures(Apx, Apy, Acx, Acy),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic("f",
            f0*chi_factors(chi, f_coeffs)*at.exp(df*perturb_f),
            dims=['mode'])
        gamma = pm.Deterministic("gamma",
            f0*chi_factors(chi, g_coeffs)*at.exp(-dtau*perturb_tau),
            dims=['mode'])
        tau = pm.Deterministic("tau", 1/gamma, dims=['mode'])
        Q = pm.Deterministic("Q", np.pi * f * tau, dims=['mode'])
        phiR = pm.Deterministic("phiR",
             phiR_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        phiL = pm.Deterministic("phiL",
             phiL_from_quadratures(Apx, Apy, Acx, Acy),
             dims=['mode'])
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL), dims=['mode'])
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL), dims=['mode'])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential('f_min_cut', at.sum(at.where(f < f_min, np.NINF, 0.0)))
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential('f_max_cut', at.sum(at.where(f > f_max, np.NINF, 0.0)))

        h_det_mode = pm.Deterministic("h_det_mode",
                compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                                   Apx, Apy, Acx, Acy),
                dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if any(flat_A):
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit,flat_A))
            # bring us to flat-in-A prior
            pm.Potential("flat_A_prior", -3*at.sum(at.log(A)*flat_A))
        elif any(flat_A_ellip):
            # bring us back to flat-in-quadratures
            pm.Potential("flat_A_quadratures_prior",
                         flat_A_quadratures_prior(Apx_unit, Apy_unit,
                                                  Acx_unit, Acy_unit,flat_A_ellip))
            # bring us to flat-in-A and flat-in-ellip prior
            pm.Potential("flat_A_ellip_prior", 
                         at.sum((-3*at.log(A) - at.log1m(at.square(ellip)))*flat_A_ellip))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood:
        if not prior_run:
            for i in range(ndet):
                key = ifos[i]
                if isinstance(key, bytes):
                 # Don't want byte strings in our names!
                    key = key.decode('utf-8')
                _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
                            observed=strains[i], dims=['time_index'])
        else:
            print("Sampling prior")
            samp_prior_cond = pm.Potential('A_prior', at.sum(at.where(A > (10*A_scale or 1e-19), np.NINF, 0.0))) #this condition is to bound flat priors just for sampling from the prior

        
        return model
        
def make_mchi_aligned_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs,
                            g_coeffs, **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    cosi_min = kwargs.pop("cosi_min")
    cosi_max = kwargs.pop("cosi_max")
    A_scale = kwargs.pop("A_scale")
    df_min = kwargs.pop("df_min")
    df_max = kwargs.pop("df_max")
    dtau_min = kwargs.pop("dtau_min")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    f_min = kwargs.pop('f_min', 0.0)
    f_max = kwargs.pop('f_max', np.inf)
    nmode = f_coeffs.shape[0]
    prior_run = kwargs.pop('prior_run',False)

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A,nmode)
    elif len(flat_A)!=nmode:
        raise ValueError("flat_A must either be a scalar or array of length equal to the number of modes")

    if (cosi_min < -1) or (cosi_max > 1):
        raise ValueError("cosi boundaries must be contained in [-1, 1]")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")
    
    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        cosi = pm.Uniform("cosi", cosi_min, cosi_max)

        Ax_unit = pm.Normal("Ax_unit", dims=['mode'])
        Ay_unit = pm.Normal("Ay_unit", dims=['mode'])

        df = pm.Uniform("df", df_min, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=['mode'])

        A = pm.Deterministic("A",
            A_scale*at.sqrt(at.square(Ax_unit)+at.square(Ay_unit)),
            dims=['mode'])
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit),
            dims=['mode'])

        f0 = FREF*MREF/M
        f = pm.Deterministic('f',
            f0*chi_factors(chi, f_coeffs)*at.exp(df * perturb_f),
            dims=['mode'])
        gamma = pm.Deterministic('gamma',
             f0*chi_factors(chi, g_coeffs)*at.exp(-dtau * perturb_tau),
             dims=['mode'])
        tau = pm.Deterministic('tau', 1/gamma, dims=['mode'])
        Q = pm.Deterministic('Q', np.pi*f*tau, dims=['mode'])
        Ap = pm.Deterministic('Ap', (1 + at.square(cosi))*A, dims=['mode'])
        Ac = pm.Deterministic('Ac', 2*cosi*A, dims=['mode'])
        ellip = pm.Deterministic('ellip', Ac/Ap, dims=['mode'])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential('f_min_cut', at.sum(at.where(f < f_min, np.NINF, 0.0)))
            print("Running with f_min_cut on modes:",f_min)
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential('f_max_cut', at.sum(at.where(f > f_max, np.NINF, 0.0)))
            print("Running with f_max_cut on modes:",f_max)


        Apx = (1 + at.square(cosi))*A*at.cos(phi)
        Apy = (1 + at.square(cosi))*A*at.sin(phi)
        Acx = -2*cosi*A*at.sin(phi)
        Acy = 2*cosi*A*at.cos(phi)

        h_det_mode = pm.Deterministic("h_det_mode",
            compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                               Apx, Apy, Acx, Acy),
            dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if any(flat_A):
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum((at.square(Ax_unit) + at.square(Ay_unit))*flat_A))
            # now to flat in A
            pm.Potential("flat_A_prior", -at.sum(at.log(A)*flat_A))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood
        if not prior_run:
            for i in range(ndet):
                key = ifos[i]
                if isinstance(key, bytes):
                    # Don't want byte strings in our names!
                    key = key.decode('utf-8')
                _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
                            observed=strains[i], dims=['time_index'])
        else:
            print("Sampling prior")
            samp_prior_cond = pm.Potential('A_prior', at.sum(at.where(A > (10*A_scale or 1e-19), np.NINF, 0.0))) #this condition is to bound flat priors just for sampling from the prior
        
        return model



def logit(p):
    return np.log(p) - np.log1p(-p)

def make_ftau_model(t0, times, strains, Ls, **kwargs):
    f_min = kwargs.pop("f_min")
    f_max = kwargs.pop("f_max")
    gamma_min = kwargs.pop("gamma_min")
    gamma_max = kwargs.pop("gamma_max")
    A_scale = kwargs.pop("A_scale")
    flat_A = kwargs.pop("flat_A", True)
    nmode = kwargs.pop("nmode", 1)
    prior_run = kwargs.pop('prior_run', False)

    if np.isscalar(flat_A):
        flat_A = np.repeat(flat_A,nmode)
    elif len(flat_A)!=nmode:
        raise ValueError("flat_A must either be a scalar or array of length equal to the number of modes")

    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))

    coords = {
        'ifo': ifos,
        'mode': modes,
        'time_index': np.arange(nt)
    }

    with pm.Model(coords=coords) as model:
        pm.ConstantData('times', times, dims=['ifo', 'time_index'])
        pm.ConstantData('t0', t0, dims=['ifo'])
        pm.ConstantData('L', Ls, dims=['ifo', 'time_index', 'time_index'])

        f = pm.Uniform("f", f_min, f_max, dims=['mode'])
        gamma = pm.Uniform('gamma', gamma_min, gamma_max, dims=['mode'],
                           transform=pm.distributions.transforms.multivariate_ordered)

        Ax_unit = pm.Normal("Ax_unit", dims=['mode'])
        Ay_unit = pm.Normal("Ay_unit", dims=['mode'])

        A = pm.Deterministic("A",
            A_scale*at.sqrt(at.square(Ax_unit)+at.square(Ay_unit)),
            dims=['mode'])
        phi = pm.Deterministic("phi", at.arctan2(Ay_unit, Ax_unit),
                               dims=['mode'])

        tau = pm.Deterministic('tau', 1/gamma, dims=['mode'])
        Q = pm.Deterministic('Q', np.pi*f*tau, dims=['mode'])

        Apx = A*at.cos(phi)
        Apy = A*at.sin(phi)

        h_det_mode = pm.Deterministic("h_det_mode",
            compute_h_det_mode(t0, times, np.ones(ndet), np.zeros(ndet),
                               f, gamma, Apx, Apy, np.zeros(nmode),
                               np.zeros(nmode)),
            dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if any(flat_A):
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum((at.square(Ax_unit) + at.square(Ay_unit))*flat_A))
            pm.Potential("flat_A_prior", -at.sum(at.log(A)*flat_A))

        # Flat prior on the delta-fs and delta-taus

        # Likelihood
        if not prior_run:
            for i in range(ndet):
                key = ifos[i]
                if isinstance(key, bytes):
                # Don't want byte strings in our names!
                   key = key.decode('utf-8')
                _ = pm.MvNormal(f"strain_{key}", mu=h_det[i,:], chol=Ls[i],
                            observed=strains[i], dims=['time_index'])
        else:
            print("Sampling prior")
            samp_prior_cond = pm.Potential('A_prior', at.sum(at.where(A > (10*A_scale or 1e-19), np.NINF, 0.0))) #this condition is to bound flat priors just for sampling from the prior
        
        return model


