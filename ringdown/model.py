__all__ = ['make_mchi_model', 'make_mchi_aligned_model', 'make_mchi_aligned_marginalized_model', 'make_ftau_model',
           'sp_w_ylm_p', 'sp_w_ylm_c']

import pytensor.tensor as at
import pytensor.tensor.slinalg as atl
import numpy as np
import pymc as pm
from scipy.special import factorial

# reference frequency and mass values to translate linearly between the two
FREF = 2985.668287014743
MREF = 68.0

def _atl_cho_solve(L_and_lower, b):
    """Replacement for `aesara.tensor.slinalg.cho_solve` that enables backprop using two `solve_triangular`.
    
    Assumes `L` is lower triangular, and solves for `x` where :math:`L L^T x = b`.
    """
    L, lower = L_and_lower

    y = atl.solve_triangular(L, b, lower=lower)
    return atl.solve_triangular(L.T, y, lower=(not lower))

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

def rd_design_matrix(t0s, ts, f, gamma, Fp, Fc, Ascales):
    ts = at.as_tensor(ts)
    nifo, nt = ts.shape

    nmode = f.shape[0]

    t0s = at.reshape(t0s, (nifo, 1, 1))
    ts = at.reshape(ts, (nifo, 1, nt))
    f = at.reshape(f, (1, nmode, 1))
    gamma = at.reshape(gamma, (1, nmode, 1))
    Fp = at.reshape(Fp, (nifo, 1, 1))
    Fc = at.reshape(Fc, (nifo, 1, 1))
    Ascales = at.reshape(Ascales, (1, nmode, 1))

    t = ts - t0s

    ct = at.cos(2*np.pi*f*t)
    st = at.sin(2*np.pi*f*t)
    decay = at.exp(-gamma*t)
    return at.concatenate((Ascales*Fp*decay*ct, Ascales*Fp*decay*st, Ascales*Fc*decay*ct, Ascales*Fc*decay*st), axis=1)

def rd_single_det_design_matrix(t0s, ts, f, gamma, A):
    ts = at.as_tensor(ts)
    nifo, nt = ts.shape

    nmode = f.shape[0]

    t0s = at.reshape(t0s, (nifo, 1, 1))
    ts = at.reshape(ts, (nifo, 1, nt))
    f = at.reshape(f, (1, nmode, 1))
    gamma = at.reshape(gamma, (1, nmode, 1))
    A = at.reshape(A, (1, nmode, 1))

    t = ts - t0s

    ct = at.cos(2*np.pi*f*t)
    st = at.sin(2*np.pi*f*t)
    decay = at.exp(-gamma*t)
    
    return at.concatenate((A*decay*ct, A*decay*st), axis=1)

def tens_factorial(j):
    arr = j.eval()
    fact = factorial(arr)
    tens_fact = at.as_tensor_variable(fact)
    return tens_fact

def tens_binom(n, k):
    """Returns binomial coefficient, `n choose k`.
    
    Arguments
    ---------
    n: int
      number of possibilities
    k: int
      number of unordered outcomes to choose
    """

    n_arr = n.eval()
    k_arr = k.eval()
    zer = np.zeros(len(k_arr))
    one = np.ones(len(k_arr))
    
    ans = np.zeros(len(k_arr))

    for idx in range(len(ans)):
        n = n_arr[idx]
        k = k_arr[idx]
        if 0 <= k < n:
            ans[idx] = factorial(int(n)) / factorial(int(k)) / factorial(int(n)-int(k))
    
    eidx = np.argwhere(k_arr == n_arr)
    for idx in eidx:
        ans[idx] = 1

    zidx = np.argwhere(k_arr.any() < 0)
    for idx in zidx:
        ans[idx] = 0
        
    return at.as_tensor_variable(ans)

def sp_w_ylm(es, el, em):
    
    ## with phi = 0
    ## with theta = arccos(cosi)
    
    r = el - es
    
    def get_rs(r):
        ### getting each r index for summation term
        ### use the largest r in the vector---no harm as tens_binom(el-es, r) will return zero
        leng = len(r.eval())
        coll = []
        for i in range(int(max(r.eval()))+1):
            rs = at.as_tensor_variable(np.zeros(leng))
            v = rs + i
            coll.append(v)
        return coll
    
    rs = get_rs(r)
    
    def sin_th_2(cosi):
        return at.sqrt((1-cosi)/2)
    
    def cos_th_2(cosi):
        return at.sqrt((1+cosi)/2)
    
    def cot_th_2(cosi):
        return cos_th_2(cosi)/sin_th_2(cosi)

    ## normalization constant is sqrt(1/0.159) ##
    
    return lambda cosi: (-1)**(el+em-es) * at.sqrt(tens_factorial(el+em)*
                                                   tens_factorial(el-em)*
                                                   ((2*el)+1)/
                                                   (4*np.pi)/
                                                   tens_factorial(el+es)/
                                                   tens_factorial(el-es)) * (sin_th_2(cosi))**(2*el) * np.sum([(-1)**n * tens_binom(el-es, n) * tens_binom(el+es, n+es-em) * (cot_th_2(cosi))**((2*n)+es-em) for n in rs]) * np.sqrt(1/0.159)

def sp_w_ylm_p(cosi, l, m):
    return (sp_w_ylm(-2, l, m)(cosi) + sp_w_ylm(-2, l, m)(-cosi)) * np.sqrt(5/np.pi) # multiply by constant sqrt(5/pi) to match Max's original version

def sp_w_ylm_c(cosi, l, m):
    return (sp_w_ylm(-2, l, m)(cosi) - sp_w_ylm(-2, l, m)(-cosi)) * np.sqrt(5/np.pi)

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
    l = kwargs.pop('ls', np.zeros(nmode))
    m = kwargs.pop('ms', np.zeros(nmode))

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
        
        Ap = pm.Deterministic('Ap', sp_w_ylm_p(cosi, at.as_tensor_variable(l), at.as_tensor_variable(m))*A, dims=['mode'])
        Ac = pm.Deterministic('Ac', sp_w_ylm_c(cosi, at.as_tensor_variable(l), at.as_tensor_variable(m))*A, dims=['mode'])

        x = at.abs(sp_w_ylm(-2, at.as_tensor_variable(l), at.as_tensor_variable(m))(cosi))
        y = at.abs(sp_w_ylm(-2, at.as_tensor_variable(l), at.as_tensor_variable(m))(-1*cosi))
        ellip = pm.Deterministic('ellip', (x - y)/(x + y), dims=['mode'])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential('f_min_cut', at.sum(at.where(f < f_min, np.NINF, 0.0)))
            print("Running with f_min_cut on modes:",f_min)
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential('f_max_cut', at.sum(at.where(f > f_max, np.NINF, 0.0)))
            print("Running with f_max_cut on modes:",f_max)

        Apx = Ap*at.cos(phi)
        Apy = Ap*at.sin(phi)
        Acx = -1*Ac*at.sin(phi)
        Acy = Ac*at.cos(phi)

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

def make_mchi_marginalized_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs,
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

def make_mchi_aligned_marginalized_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs,
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
    f_min = kwargs.pop('f_min', 0.0)
    f_max = kwargs.pop('f_max', np.inf)
    nmode = f_coeffs.shape[0]
    prior_run = kwargs.pop('prior_run',False)

    if (cosi_min < -1) or (cosi_max > 1):
        raise ValueError("cosi boundaries must be contained in (-1, 1)")
    if (chi_min < 0) or (chi_max > 1):
        raise ValueError("chi boundaries must be contained in [0, 1)")
    
    ndet = len(t0)
    nt = len(times[0])

    ifos = kwargs.pop('ifos', np.arange(ndet))
    modes = kwargs.pop('modes', np.arange(nmode))
    l = at.as_tensor_variable(kwargs.pop('ls', np.zeros(nmode)))
    m = at.as_tensor_variable(kwargs.pop('ms', np.zeros(nmode)))
    
    Llogdet = np.array([np.sum(np.log(np.diag(L))) for L in Ls])

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

        cosi = pm.Uniform("cosi", cosi_min, cosi_max) if len(ifos) > 1 else pm.ConstantData("cosi", (cosi_min+cosi_max)/2) ### fixing cosi for single-detector fits
        
        A_scale = pm.Uniform('A_scale', 0, A_scale, dims=['mode'])

        Ax_unit = pm.Normal("Ax_unit", dims=['mode'])
        Ay_unit = pm.Normal("Ay_unit", dims=['mode'])

        df = pm.Uniform("df", df_min, df_max, dims=['mode'])
        dtau = pm.Uniform("dtau", dtau_min, dtau_max, dims=['mode'])
        
        f0 = FREF*MREF/M
        
        f = pm.Deterministic('f',
            f0*chi_factors(chi, f_coeffs)*at.exp(df * perturb_f),
            dims=['mode'])
        gamma = pm.Deterministic('gamma',
             f0*chi_factors(chi, g_coeffs)*at.exp(-dtau * perturb_tau),
             dims=['mode'])
        tau = pm.Deterministic('tau', 1/gamma, dims=['mode'])
        Q = pm.Deterministic('Q', np.pi*f*tau, dims=['mode'])

        Yp = at.reshape(sp_w_ylm_p(cosi, l, m), (1, nmode, 1))
        Yc = at.reshape(sp_w_ylm_c(cosi, l, m), (1, nmode, 1))
        
        if len(ifos) == 1: ### combine Fs, Ys, As into one big amplitude param since one of the polarizations is zero (not detected by single det)
            design_matrices = rd_single_det_design_matrix(t0, times, f, gamma, A_scale)

        else:        
            old_dms = rd_design_matrix(t0, times, f, gamma, Fps, Fcs, A_scale)
            design_matrices = at.concatenate(((Yp * old_dms[:,:nmode,:]) + (Yc * old_dms[:,2*nmode:3*nmode,:]), 
                                              (Yp * old_dms[:,nmode:2*nmode,:]) + (Yc * old_dms[:,3*nmode:,:])), axis=1)

        mu = at.zeros(2*nmode)
        Lambda_inv = at.eye(2*nmode)
        Lambda_inv_chol = at.eye(2*nmode)
        
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
        
        theta = mu + atl.solve(Lambda_inv_chol.T, at.concatenate((Ax_unit, Ay_unit)))

        Ax_unit_marg = theta[:nmode]
        Ay_unit_marg = theta[nmode:2*nmode]

        phi = pm.Deterministic("phi", at.arctan2(Ay_unit_marg, Ax_unit_marg), dims=['mode'])
        
        A = pm.Deterministic("A",
            A_scale*at.sqrt(at.square(theta[:nmode])+at.square(theta[nmode:2*nmode])),
            dims=['mode']) ### Note: if fitting to a single-detector event, this A parameter will include the antenna pattern and spin-weighted spherical harmonic of the detected polarization (for a fixed inclination)
        
        Ap = pm.Deterministic('Ap', at.reshape(Yp, (nmode, ))*A, dims=['mode'])
        Ac = pm.Deterministic('Ac', at.reshape(Yc, (nmode, ))*A, dims=['mode'])
        
        x = at.abs(sp_w_ylm(-2, at.as_tensor_variable(l), at.as_tensor_variable(m))(cosi))
        y = at.abs(sp_w_ylm(-2, at.as_tensor_variable(l), at.as_tensor_variable(m))(-1*cosi))
        ellip = pm.Deterministic('ellip', (x - y)/(x + y), dims=['mode'])

        # Check limits on f
        if not np.isscalar(f_min) or not f_min == 0.0:
            _ = pm.Potential('f_min_cut', at.sum(at.where(f < f_min, np.NINF, 0.0)))
            print("Running with f_min_cut on modes:",f_min)
        if not np.isscalar(f_max) or not f_max == np.inf:
            _ = pm.Potential('f_max_cut', at.sum(at.where(f > f_max, np.NINF, 0.0)))
            print("Running with f_max_cut on modes:",f_max)

        Apx = Ap*at.cos(phi)
        Apy = Ap*at.sin(phi)
        Acx = -1*Ac*at.sin(phi)
        Acy = Ac*at.cos(phi)

        h_det_mode = pm.Deterministic("h_det_mode",
            compute_h_det_mode(t0, times, Fps, Fcs, f, gamma,
                               Apx, Apy, Acx, Acy),
            dims=['ifo', 'mode', 'time_index'])
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1),
                                 dims=['ifo', 'time_index'])
        
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


