__all__ = ['make_model', 'get_arviz']

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro
import numpyro.distributions as dist
from . import qnms
from .indexing import ModeIndexList
from .result import Result
import arviz as az
import warnings
from arviz.data.base import dict_to_dataset
import logging


def rd_design_matrix(ts, f, gamma, Fp, Fc, Ascales):
    ts = jnp.array(ts)
    nifo, nt = ts.shape

    nmode = f.shape[0]

    ts = jnp.reshape(ts, (nifo, 1, nt))
    f = jnp.reshape(f, (1, nmode, 1))
    gamma = jnp.reshape(gamma, (1, nmode, 1))
    Fp = jnp.reshape(Fp, (nifo, 1, 1))
    Fc = jnp.reshape(Fc, (nifo, 1, 1))
    Ascales = jnp.reshape(Ascales, (1, nmode, 1))

    ct = jnp.cos(2*np.pi*f*ts)
    st = jnp.sin(2*np.pi*f*ts)
    decay = jnp.exp(-gamma*ts)
    return jnp.concatenate((Ascales*Fp*decay*ct, Ascales*Fp*decay*st,
                            Ascales*Fc*decay*ct, Ascales*Fc*decay*st), 
                            axis=1)

def chi_factors(chi, coeffs):
    log1mc = jnp.log1p(-chi)
    log1mc2 = log1mc*log1mc
    log1mc3 = log1mc2*log1mc
    log1mc4 = log1mc2*log1mc2
    v = jnp.stack([chi, jnp.ones_like(chi), log1mc, log1mc2,
                   log1mc3, log1mc4])
    return jnp.dot(coeffs, v)

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

def get_quad_derived_quantities(design_matrices, quads, a_scale, store_h_det, 
                                store_h_det_mode, compute_h_det=False):
    nifo, nmodes4, ntimes = design_matrices.shape
    nmodes = nmodes4 // 4

    apx_unit = quads[:nmodes]
    apy_unit = quads[nmodes:2*nmodes]
    acx_unit = quads[2*nmodes:3*nmodes]
    acy_unit = quads[3*nmodes:]

    apx = numpyro.deterministic('apx', apx_unit * a_scale)
    apy = numpyro.deterministic('apy', apy_unit * a_scale)
    acx = numpyro.deterministic('acx', acx_unit * a_scale)
    acy = numpyro.deterministic('acy', acy_unit * a_scale)

    a_norm, e = Aellip_from_quadratures(apx_unit, apy_unit, acx_unit, acy_unit)
    a = numpyro.deterministic('a', a_scale * a_norm)
    ellip = numpyro.deterministic('ellip', e)
    phi_r = numpyro.deterministic('phi_r', phiR_from_quadratures(apx_unit, apy_unit, acx_unit, acy_unit))
    phi_l = numpyro.deterministic('phi_l', phiL_from_quadratures(apx_unit, apy_unit, acx_unit, acy_unit))
    theta = numpyro.deterministic('theta', -0.5*(phi_r + phi_l))
    phi = numpyro.deterministic('phi', 0.5*(phi_r - phi_l))

    if compute_h_det or store_h_det_mode or store_h_det:
        h_det_mode = jnp.zeros((nifo, nmodes, ntimes))
        hh = design_matrices * quads[jnp.newaxis,:,jnp.newaxis]

        for i in range(nmodes):
            h_det_mode = h_det_mode.at[:,i,:].set(jnp.sum(hh[:,i::nmodes,:], axis=1))
        h_det = jnp.sum(h_det_mode, axis=1)

        if store_h_det_mode:
            _ = numpyro.deterministic('h_det_mode', h_det_mode)
        if store_h_det:
            _ = numpyro.deterministic('h_det', h_det)
        
        return a, h_det

def make_model(modes : int | list[(int, int, int, int)], 
               a_scale_max : float,
               marginalized : bool = True, 
               m_min : float | None = None,
               m_max : float | None = None,
               chi_min : float = 0.0,
               chi_max : float = 0.99,
               df_min : None | float | list[None | float] = None,
               df_max : None | float | list[None | float] = None,
               dg_min : None | float | list[None | float] = None,
               dg_max : None | float | list[None | float] = None,
               f_min : None | float | list[float] = None,
               f_max : None | float | list[float] = None,
               g_min : None | float | list[float] = None,
               g_max : None | float | list[float] = None,
               flat_amplitude_prior : bool = False,
               mode_ordering : None | str = None,
               prior : bool = False,               
               predictive : bool = True, 
               store_h_det : bool = True, 
               store_h_det_mode : bool = True):
    """
    Arguments
    ---------
    modes : int or list[tuple]
        If integer, the number of damped sinusoids to use.  If list of tuples,
        each entry should be of the form `(p, s, ell, m)`, where `p` is `1` for
        prograde `-1` for retrograde; `s` is the spin weight (`-2` for the usual
        GW modes); and `ell` and `m` refer to the usual angular quantum numbers.

    a_scale_max : float
        The maximum value of the amplitude scale parameter. This is used to
        define the prior on the amplitude scale parameter.

    marginalized : bool
        Whether or not to marginalize over the quadrature amplitudes
        analytically.

    m_min : float
        The minimum mass of the black hole in solar masses.

    m_max : float
        The maximum mass of the black hole in solar masses.

    chi_min : float
        The minimum dimensionless spin of the black hole.

    chi_max : float
        The maximum dimensionless spin of the black hole.

    df_min : None or float or list[None or float]
        The minimum fractional deviation from the GR frequency.  If a list, then
        it should have the same length as `modes`.

    df_max : None or float or list[None or float]
        The maximum fractional deviation from the GR frequency.  If a list, then
        it should have the same length as `modes`.

    dg_min : None or float or list[None or float]
        The minimum fractional deviation from the GR damping rate.  If a list,
        then it should have the same length as `modes`.

    dg_max : None or float or list[None or float]
        The maximum fractional deviation from the GR damping rate.  If a list,
        then it should have the same length as `modes`.

    flat_amplitude_prior : bool
        Whether or not to impose a flat prior on the amplitude scale parameter.
        This is only relevant if `marginalized` is `False`.

    mode_ordering : None or str
        Relevant to the case where `modes` is an integer, and the model consists
        of arbitrary damped sinusoids.  If `None`, then the frequencies and
        damping rates are only constrained by the bounds `f_min`, `f_max`,
        `g_min`, and `g_max`.  If `'f'`, then the frequencies are constrained to
        be in increasing order; if `'g'`, then the damping rates are constrained
        to be in increasing order.

    prior : bool
        Whether or not to compute the likelihood.  If `True`, then the
        likelihood is not computed, and the model is used for prior predictive
        sampling.

    predictive : bool
        Whether to generate the quadrature amplitudes when `marginalized=True`.

    store_h_det : bool
        Whether to store the detector-frame waveform in the model.

    store_h_det_mode : bool
        Whether to store the mode-by-mode detector-frame waveform in the model.

    Returns
    -------
    model : function
        A model function that can be used with `numpyro` to sample from the
        posterior distribution of the ringdown parameters.
    """

    n_modes = modes if isinstance(modes, int) else len(modes)

    # check arguments for free damped sinusoid fits
    if mode_ordering is not None:
        if not isinstance(modes, int):
            raise ValueError('mode_ordering is only relevant if modes is an int')
        if mode_ordering not in ['f', 'g']:
            raise ValueError('mode_ordering must be None, "f", or "g"')
        elif mode_ordering == 'f':
            if not np.isscalar(f_min) or not np.isscalar(f_max):
                raise ValueError('mode_ordering is "f" but f_min and/or f_max are not scalars')
        elif mode_ordering == 'g':
            if not np.isscalar(g_min) or not np.isscalar(g_max):
                raise ValueError('mode_ordering is "g" but g_min and/or g_max are not scalars')
    elif isinstance(modes, int):
        # for sampling below, we want to make sure all of these are arrays
        if np.isscalar(f_min):
            f_min = [f_min]*n_modes
        if np.isscalar(f_max):
            f_max = [f_max]*n_modes
        if np.isscalar(g_min):
            g_min = [g_min]*n_modes
        if np.isscalar(g_max):
            g_max = [g_max]*n_modes
        # turn lists into jnp arrays
        if not np.isscalar(f_min) or not np.isscalar(f_max):
            f_min = jnp.array(f_min)
            f_max = jnp.array(f_max)
        if not np.isscalar(g_min) or not np.isscalar(g_max):
            g_min = jnp.array(g_min)
            g_max = jnp.array(g_max)
        for a in [f_min, f_max, g_min, g_max]:
            if a.shape[0] != n_modes:
                raise ValueError('f_min, f_max, g_min, and g_max must have '
                                 'the same length as modes')
            
    # if df_min and df_max are floats, then make it an array
    if df_min is not None and np.isscalar(df_min):
        df_min = [df_min]*n_modes
    if df_max is not None and np.isscalar(df_max):
        df_max = [df_max]*n_modes
    if dg_min is not None and np.isscalar(dg_min):
        dg_min = [dg_min]*n_modes
    if dg_max is not None and np.isscalar(dg_max):
        dg_max = [dg_max]*n_modes

    def model(times, strains, ls, fps, fcs,
              predictive : bool = predictive, 
              store_h_det : bool = store_h_det, 
              store_h_det_mode : bool = store_h_det_mode):
        """The ringdown model.

        Arguments
        ---------
        times : array_like
            The times at which the ringdown waveform should be evaluated; list
            of 1D arrays for each IFO, or a 2D array with shape (n_det,
            n_times).
        strains : array_like
            The strain data; list of 1D arrays for each IFO, or a 2D array with
            shape (n_det, n_times).
        ls : array_like
            The noise covariance matrices; list of 2D arrays for each IFO, or a
            3D array with shape (n_det, n_times, n_times).
        fps : array_like
            The "plus" polarization coefficients for each IFO; length `n_det`.
        fcs : array_like
            The "cross" polarization coefficients for each IFO; length `n_det`.
        """
        times, strains, ls, fps, fcs = map(jnp.array, (times, strains, ls, fps, fcs))

        n_det = times.shape[0]

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
            if mode_ordering == 'f':
                # Sure would be nice if numpyro had `OrderedBoundedVector` or
                # something similar.  But because it doesn't, we have to do it
                # with transformations.  This takes:
                #
                # Unconstrained reals -> ordered reals -> (0,1) ordered reals ->
                # (f_min, f_max) ordered reals
                #
                # Since we want a flat prior on f, but the sampler sees
                # f_latent, we need a Jacobian factor:
                #
                # log_jac = log(d(f)/d(f_latent))
                #
                # which, happily, is provided by the composed transformation
                f_latent = numpyro.sample('f_latent',
                    dist.ImproperUniform(dist.constraints.real, (), (n_modes,)))
                f_transform = dist.transforms.ComposeTransform([
                    dist.transforms.OrderedTransform(),
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(f_min, f_max - f_min)
                ])
                f = numpyro.deterministic('f', f_transform(f_latent))
                numpyro.factor('f_transform', f_transform.log_abs_det_jacobian(f_latent, f))

                g = numpyro.sample('g', dist.Uniform(g_min, g_max),
                                    sample_shape=(modes,))
            elif mode_ordering == 'g':
                f = numpyro.sample('f', dist.Uniform(f_min, f_max), 
                                   sample_shape=(modes,))
                
                g_latent = numpyro.sample('g_latent', dist.ImproperUniform(
                    dist.constraints.real, (), (n_modes,)))
                g_transform = dist.transforms.ComposeTransform([
                    dist.transforms.OrderedTransform(),
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(g_min, g_max - g_min)
                ])
                g = numpyro.deterministic('g', g_transform(g_latent))
                numpyro.factor('g_transform', g_transform.log_abs_det_jacobian(g_latent, g))
            else:
                f = numpyro.sample('f', dist.Uniform(f_min, f_max))
                g = numpyro.sample('g', dist.Uniform(g_min, g_max))
        elif isinstance(modes, list):
            fcoeffs = []
            gcoeffs = []
            for mode in modes:
                c = qnms.KerrMode(mode).coefficients
                fcoeffs.append(c[0])
                gcoeffs.append(c[1])
            fcoeffs = jnp.array(fcoeffs)
            gcoeffs = jnp.array(gcoeffs)

            m = numpyro.sample('m', dist.Uniform(m_min, m_max))
            chi = numpyro.sample('chi', dist.Uniform(chi_min, chi_max))

            f0 = 1 / (m*qnms.T_MSUN)
            f_gr = f0*chi_factors(chi, fcoeffs)
            g_gr = f0*chi_factors(chi, gcoeffs)

            if df_min is None or df_max is None:
                f = numpyro.deterministic('f', f_gr)
            else:
                df_unit = numpyro.sample('df_unit', dist.Uniform(0, 1), sample_shape=(n_modes,1))
                # Don't want to shadow df_min and df_max
                df_low = jnp.array([0.0 if x is None else x for x in df_min])
                df_high = jnp.array([0.0 if x is None else x for x in df_max])
                
                df = numpyro.deterministic('df', df_unit*(df_high - df_low) + df_low)
                f = numpyro.deterministic('f', f_gr * jnp.exp(df))

            if dg_min is None or dg_max is None:
                g = numpyro.deterministic('g', g_gr)
            else:
                dg_unit = numpyro.sample('dg_unit', dist.Uniform(0, 1), sample_shape=(n_modes,))
                dg_low = jnp.array([0.0 if x is None else x for x in dg_min])
                dg_high = jnp.array([0.0 if x is None else x for x in dg_max])
                
                dg = numpyro.deterministic('dg', dg_unit*(dg_high - dg_low) + dg_low)
                g = numpyro.deterministic('g', g_gr * jnp.exp(dg))
        # At this point the frequencies `f` and damping rates `g` of the various
        # modes should be established, and we can proceed with the rest of the
        # model.

        tau = numpyro.deterministic('tau', 1/g)
        omega = numpyro.deterministic('omega', 2*np.pi*f)
        quality = numpyro.deterministic('quality', np.pi*f*tau)

        if marginalized:
            a_scale = numpyro.sample('a_scale', dist.Uniform(0, a_scale_max), 
                                     sample_shape=(n_modes,))
            design_matrices = rd_design_matrix(times, f, g, fps, fcs, a_scale)

            mu = jnp.zeros(4*n_modes)
            lambda_inv = jnp.eye(4*n_modes)
            lambda_inv_chol = jnp.eye(4*n_modes)
            if not prior:
                for i in range(n_det):
                    mm = design_matrices[i,:,:].T # (ndet, 4*nmode, ntime) => (i, ntime, 4*nmode)
                    l = ls[i,:,:]
                    s = strains[i,:]

                    a_inv = lambda_inv + jnp.dot(mm.T, jsp.linalg.cho_solve((l, True), mm))
                    a_inv_chol = jsp.linalg.cholesky(a_inv, lower=True)

                    a = jsp.linalg.cho_solve((a_inv_chol, True), jnp.dot(lambda_inv, mu) + jnp.dot(mm.T, jsp.linalg.cho_solve((l, True), s)))

                    b = jnp.dot(mm, mu)

                    blogsqrtdet = jnp.sum(jnp.log(jnp.diag(l))) - jnp.sum(jnp.log(jnp.diag(lambda_inv_chol))) + jnp.sum(jnp.log(jnp.diag(a_inv_chol)))

                    r = s - b
                    cinv_r = jsp.linalg.cho_solve((l, True), r)
                    mamtcinv_r = jnp.dot(mm, jsp.linalg.cho_solve((a_inv_chol, True), jnp.dot(mm.T, cinv_r)))
                    cinvmamtcinv_r = jsp.linalg.cho_solve((l, True), mamtcinv_r)
                    logl = -0.5*jnp.dot(r, cinv_r - cinvmamtcinv_r) - blogsqrtdet

                    numpyro.factor(f'logl_{i}', logl)

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
                apx_unit = numpyro.sample('apx_unit', dist.Normal(0, 1), sample_shape=(n_modes,))
                apy_unit = numpyro.sample('apy_unit', dist.Normal(0, 1), sample_shape=(n_modes,))
                acx_unit = numpyro.sample('acx_unit', dist.Normal(0, 1), sample_shape=(n_modes,))
                acy_unit = numpyro.sample('acy_unit', dist.Normal(0, 1), sample_shape=(n_modes,))

                quads = mu + jsp.linalg.solve(lambda_inv_chol.T, jnp.concatenate((apx_unit, apy_unit, acx_unit, acy_unit)))

                get_quad_derived_quantities(design_matrices, quads, a_scale, store_h_det, store_h_det_mode)
        else:
            a_scales = a_scale_max*jnp.ones(n_modes)
            design_matrices = rd_design_matrix(times, f, g, fps, fcs, a_scales)
            apx_unit = numpyro.sample('apx_unit', dist.Normal(0, 1), sample_shape=(n_modes,))
            apy_unit = numpyro.sample('apy_unit', dist.Normal(0, 1), sample_shape=(n_modes,))
            acx_unit = numpyro.sample('acx_unit', dist.Normal(0, 1), sample_shape=(n_modes,))
            acy_unit = numpyro.sample('acy_unit', dist.Normal(0, 1), sample_shape=(n_modes,))

            quads = jnp.concatenate((apx_unit, apy_unit, acx_unit, acy_unit))
            a, h_det = get_quad_derived_quantities(design_matrices, quads, a_scale_max, 
                                                   store_h_det, store_h_det_mode, compute_h_det=(not prior))
            
            if flat_amplitude_prior:
                # We need a Jacobian that is A^-3
                numpyro.factor('flat_a_prior', -3*jnp.sum(jnp.log(a)) + \
                               0.5*jnp.sum((jnp.square(apx_unit) + jnp.square(apy_unit) + \
                               jnp.square(acx_unit) + jnp.square(acy_unit))))
                
                if prior:
                    raise ValueError('you did not want to impose a flat amplitude prior without a likelihood')

            if not prior:
                for i, strain in enumerate(strains):
                    numpyro.sample(f'logl_{i}', dist.MultivariateNormal(h_det[i,:], scale_tril=ls[i, :, :]), obs=strain)

    return model

MODEL_VARIABLES_BY_MODE = ['a_scale', 'a', 'acx', 'acy', 'apx', 'apy', 
                           'acx_unit', 'acy_unit', 'apx_unit', 'apy_unit',
                           'ellip', 'f', 'g', 'omega', 'phi', 'phi_l',
                           'phi_r', 'quality', 'tau', 'theta']
MODEL_DIMENSIONS = {k: ['mode'] for k in MODEL_VARIABLES_BY_MODE}
MODEL_DIMENSIONS['h_det'] = ['ifo', 'time_index']
MODEL_DIMENSIONS['h_det_mode'] = ['ifo', 'mode', 'time_index']

def get_arviz(sampler,
              modes : list | None = None,
              ifos : list | None = None,
              injections : list | None = None,
              epoch : list | None = None,
              attrs : dict | None = None, 
              store_data : bool = True):
    """Convert a numpyro sampler to an arviz dataset.
    
    Arguments
    ---------
    sampler : numpyro.MCMC
        The sampler to convert after running.
    modes : None or array_like
        Coordinates of modes to include in the dataset; if `None`, then all
        modes are included and indexed by integers.
    ifos : None or array_like
        The ifos to include in the dataset.  If `None`, then all ifos are
        included.
    injections : None or array_like
        The injections to include in the dataset.  If `None`, then no injections
        are included.
    epoch : None or array_like
        The epoch of each ifo.  If `None`, then all epochs are set to zero.
    attrs : None or dict
        Attributes to include in the arviz dataset.
    record_data : bool
        Whether to record the observed data and auxiliary quantities in the
        arviz dataset.

    Returns
    -------
    dataset : arviz.InferenceData
        The arviz dataset.
    """
    samples = sampler.get_samples()
    params_in_model = samples.keys()
    # get dimensions
    dims = {k: v for k,v in MODEL_DIMENSIONS.items() if k in params_in_model}
    for x in params_in_model:
        if x not in MODEL_DIMENSIONS and len(samples[x].shape) > 1:
            warnings.warn(f'{x} not in model dimensions; please report issue')
    
    # get coordinates
    # assume that all fits will have an 'f' parameter
    n_mode = samples['f'].shape[1]
    modes = ModeIndexList(modes or n_mode).get_coordinates()
    if len(modes) != n_mode:
        raise ValueError(f'expected {n_mode} modes, got {len(modes)}')
    # get ifo from shape of Fc, assuming it's last argument provided to model
    n_ifo = len(sampler._args[-1])
    if ifos is None:
        ifos = np.arange(n_ifo, dtype=int)
    elif len(ifos) != n_ifo:
        raise ValueError(f'expected {n_ifo} ifos, got {len(ifos)}')
    # get epochs
    if epoch is None:
        epoch = np.zeros(n_ifo)
    elif len(epoch) != n_ifo:
        raise ValueError(f'expected {n_ifo} epochs, got {len(epoch)}')
    # get times from model arguments
    n_analyze = len(sampler._args[0][0])
    time_index = np.arange(n_analyze, dtype=int)
    coords = {'ifo': ifos, 'mode': modes, 'time_index': time_index,
              'time_index_1': time_index}
    if store_data:
        # get constant_data
        in_dims = {
            'time': ['ifo', 'time_index'],
            'strain': ['ifo', 'time_index'],
            'cholesky_factor': ['ifo', 'time_index', 'time_index_1'],
            'fp': ['ifo'],
            'fc': ['ifo'],
            'epoch': ['ifo']
        }
        in_data = {k: np.array(v) for k,v in zip(in_dims.keys(),sampler._args)}
        in_data['epoch'] = np.array(epoch)
        # get injections, if provided
        if injections is not None:
            in_data['injection'] = np.array(injections)
            in_dims['injection'] = ['ifo', 'time_index']
        dims.update(in_dims)
        obs_data = {'strain': in_data.pop('strain')}
    else:
        in_data = None

    result = az.from_numpyro(sampler, dims=dims, coords=coords, 
                             constant_data=in_data)
    result.attrs.update(attrs or {})
    
    if store_data:
        # add observed data
        if hasattr(result, 'observed_data'):
            logging.info("added strain to observed data")
            result.observed_data['strain'] = (in_dims['strain'], 
                                              obs_data['strain'])
        else:
            logging.info("creating observed data in arviz dataset")
            # We assume that observed_data isn't created yet.
            result.add_groups(dict(
                observed_data=dict_to_dataset(
                    obs_data,
                    coords=result.posterior.coords,
                    dims={'strain': in_dims['strain']},
            )))
    return Result(result)

def get_neff_from_numpyro(sampler):
    """Get the effective sample size from a numpyro sampler.
    
    Arguments
    ---------
    sampler : numpyro.MCMC
        The sampler to compute the effective sample size from.
    
    Returns
    -------
    neff : dict
        The effective sample size for each parameter.
    """
    import io
    import contextlib
    import pandas as pd

    # Create a string buffer to capture the output
    buffer = io.StringIO()
    # Redirect standard output to the buffer
    with contextlib.redirect_stdout(buffer):
        sampler.print_summary()
    # Get the content from the buffer
    output = buffer.getvalue()
    
    neff = pd.read_csv(io.StringIO(output), sep='\s+')['n_eff'].drop('Number')
    
    return neff
