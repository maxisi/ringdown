__all__ = ['make_mchi_model']

import aesara.tensor as at
import aesara.tensor.slinalg as atl
import numpy as np
import pymc as pm

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
    v = at.stack([chi, at.as_tensor_variable(1.0), log1mc, log1mc2, log1mc3, log1mc4])

    return at.dot(coeffs, v)

def get_snr(h, d, L):
    wh = atl.solve_lower_triangular(L, h)
    wd = atl.solve_lower_triangular(L, h)

    return at.dot(wh, wd) / at.sqrt(at.dot(wh, wh))

def make_mchi_model(t0, times, strains, Ls, Fps, Fcs, f_coeffs, g_coeffs, 
                    M_min=10.0, M_max=500.0,
                    chi_min=0.01, chi_max=0.99,
                    A_scale=1e-21,
                    df_max=0.5, dtau_max=0.5,
                    perturb_f=0, perturb_tau=0,
                    flat_A=True, flat_A_ellip=False, **kwargs):
    ndet = len(t0)
    nmode = f_coeffs.shape[0]

    fref = 2985.668287014743
    mref = 68.0

    with pm.Model() as model:
        M = pm.Uniform("M", M_min, M_max)
        chi = pm.Uniform("chi", chi_min, chi_max)

        Apx_unit = pm.Flat("Apx_unit", shape=(nmode,))
        Apy_unit = pm.Flat("Apy_unit", shape=(nmode,))
        Acx_unit = pm.Flat("Acx_unit", shape=(nmode,))
        Acy_unit = pm.Flat("Acy_unit", shape=(nmode,))

        df = pm.Uniform("df", -df_max, df_max, shape=(nmode,))
        dtau = pm.Uniform("dtau", -dtau_max, dtau_max, shape=(nmode,))

        Apx = pm.Deterministic("Apx", A_scale*Apx_unit)
        Apy = pm.Deterministic("Apy", A_scale*Apy_unit)
        Acx = pm.Deterministic("Acx", A_scale*Acx_unit)
        Acy = pm.Deterministic("Acy", A_scale*Acy_unit)

        A = pm.Deterministic("A", 0.5*(at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) + at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))))
        ellip = pm.Deterministic("ellip", (at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) - at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))) / (at.sqrt(at.square(Acy + Apx) + at.square(Acx - Apy)) + at.sqrt(at.square(Acy - Apx) + at.square(Acx + Apy))))

        f0 = fref*mref/M
        f = pm.Deterministic("f", f0*chi_factors(chi, f_coeffs) * at.exp(df * perturb_f))
        gamma = pm.Deterministic("gamma", f0*chi_factors(chi, g_coeffs) * at.exp(-dtau * perturb_tau))
        tau = pm.Deterministic("tau", 1/gamma)
        Q = pm.Deterministic("Q", np.pi * f * tau)
        phiR = pm.Deterministic("phiR", at.arctan2(-Acx + Apy, Acy + Apx))
        phiL = pm.Deterministic("phiL", at.arctan2(-Acx - Apy, -Acy + Apx))
        theta = pm.Deterministic("theta", -0.5*(phiR + phiL))
        phi = pm.Deterministic("phi", 0.5*(phiR - phiL))

        hdms = []
        for i in range(ndet):
            hdms_row = []
            for j in range(nmode):
                hdms_row.append(rd(times[i]-t0[i], f[j], gamma[j], Apx[j], Apy[j], Acx[j], Acy[j], Fps[i], Fcs[i]))
            hdms.append(hdms_row)
        h_det_mode = pm.Deterministic("h_det_mode", at.stack([at.stack(row) for row in hdms]))
        h_det = pm.Deterministic("h_det", at.sum(h_det_mode, axis=1))

        # Priors:

        # Flat in M-chi already

        # Amplitude prior
        if flat_A:
            pm.Potential("flat_A_prior", -3*at.sum(at.log(A)))
        elif flat_A_ellip:
            pm.Potential("flat_A_ellip_prior", at.sum(-3*at.log(A) - at.log1m(at.square(ellip))))
        else:
            pm.Potential("gaussian_A_quadratures_prior", -0.5*at.sum(at.square(Apx_unit) + at.square(Apy_unit) + at.square(Acx_unit) + at.square(Acy_unit)))

        # Flat prior on the delta-fs and delta-taus

        for i in range(ndet):
            _ = pm.MvNormal(f"likelihood_detector_{i}", mu=h_det[i,:], chol=Ls[i], observed=strains[i])
        
        return model
        
