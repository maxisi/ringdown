__all__ = ['make_mchi_model', 'make_mchi_aligned_model', 'make_ftau_model']

import aesara.tensor as at
import aesara.tensor.slinalg as atl
import numpy as np
import pymc as pm

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
    ct = at.cos(2*np.pi*f*ts)
    st = at.sin(2*np.pi*f*ts)
    decay = at.exp(-gamma*ts)
    p = decay*(Apx*ct + Apy*st)
    c = decay*(Acx*ct + Acy*st)
    return Fp*p + Fc*c

def chi_factors(chi, coeffs):
    log1mc = at.log1p(-chi)
    log1mc2 = log1mc*log1mc
    log1mc3 = log1mc2*log1mc
    log1mc4 = log1mc2*log1mc2
    v = at.stack([chi, at.as_tensor_variable(1.0), log1mc, log1mc2,
                  log1mc3, log1mc4])
    return at.dot(coeffs, v)

def get_snr(h, d, L):
    wh = atl.solve_lower_triangular(L, h)
    wd = atl.solve_lower_triangular(L, h)
    return at.dot(wh, wd) / at.sqrt(at.dot(wh, wh))

def compute_h_det_mode(t0s, ts, Fps, Fcs, fs, gammas, Apxs, Apys, Acxs, Acys):
    ndet = len(t0s)
    nmode = fs.shape[0]
    nsamp = ts[0].shape[0]

    t0s = at.as_tensor_variable(t0s).reshape((ndet, 1, 1))
    ts = at.as_tensor_variable(ts).reshape((ndet, 1, nsamp))
    Fps = at.as_tensor_variable(Fps).reshape((ndet, 1, 1))
    Fcs = at.as_tensor_variable(Fcs).reshape((ndet, 1, 1))
    fs = at.as_tensor_variable(fs).reshape((1, nmode, 1))
    gammas = at.as_tensor_variable(gammas).reshape((1, nmode, 1))
    Apxs = at.as_tensor_variable(Apxs).reshape((1, nmode, 1))
    Apys = at.as_tensor_variable(Apys).reshape((1, nmode, 1))
    Acxs = at.as_tensor_variable(Acxs).reshape((1, nmode, 1))
    Acys = at.as_tensor_variable(Acys).reshape((1, nmode, 1))

    return rd(ts - t0s, fs, gammas, Apxs, Apys, Acxs, Acys, Fps, Fcs)

def a_from_quadratures(Apx, Apy, Acx, Acy):
    A = 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) +
             at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy)))
    return A

def ellip_from_quadratures(Apx, Apy, Acx, Acy):
    A = a_from_quadratures(Apx, Apy, Acx, Acy)
    e = 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) -
             at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))) / A
    return e

def Aellip_from_quadratures(Apx, Apy, Acx, Acy):
    # should be slightly cheaper than calling the two functions separately
    term1 = at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy))
    term2 = at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))
    A = 0.5*(term1 + term2)
    e = 0.5*(term1 - term2) / A
    return A, e

def phiR_from_quadratures(Apx, Apy, Acx, Acy):
    return at.arctan2(-Acx + Apy, Acy + Apx)

def phiL_from_quadratures(Apx, Apy, Acx, Acy):
    return at.arctan2(-Acx - Apy, -Acy + Apx)

def flat_A_quadratures_prior(Apx_unit, Apy_unit, Acx_unit, Acy_unit,flat_A):
    return 0.5*at.sum((at.square(Apx_unit) + at.square(Apy_unit) +
                      at.square(Acx_unit) + at.square(Acy_unit))*flat_A)

def make_mchi_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs,
                    **kwargs):
    M_min = kwargs.pop("M_min")
    M_max = kwargs.pop("M_max")
    chi_min = kwargs.pop("chi_min")
    chi_max = kwargs.pop("chi_max")
    A_scale = kwargs.pop("A_scale")
    df_max = kwargs.pop("df_max")
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

        df = pm.Uniform("df", -df_max, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", -dtau_max, dtau_max, dims=['mode'])

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
    df_max = kwargs.pop("df_max")
    dtau_max = kwargs.pop("dtau_max")
    perturb_f = kwargs.pop("perturb_f", 0)
    perturb_tau = kwargs.pop("perturb_tau", 0)
    flat_A = kwargs.pop("flat_A", True)
    f_min = kwargs.pop('f_min', 0.0)
    f_max = kwargs.pop('f_max', np.inf)
    nmode = f_coeffs.shape[0]
    prior_run = kwargs.pop('prior_run',False)


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

        df = pm.Uniform("df", -df_max, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", -dtau_max, dtau_max, dims=['mode'])

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
        if flat_A:
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum(at.square(Ax_unit) + at.square(Ay_unit)))
            # now to flat in A
            pm.Potential("flat_A_prior", -at.sum(at.log(A)))

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
                           transform=pm.distributions.transforms.ordered)

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
        if flat_A:
            # first bring us to flat in quadratures
            pm.Potential("flat_A_quadratures_prior",
                         0.5*at.sum(at.square(Ax_unit) + at.square(Ay_unit)))
            pm.Potential("flat_A_prior", -at.sum(at.log(A)))

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


