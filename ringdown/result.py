"""Module defining the core :class:`Result` class.
"""

__all__ = ['Result']

import numpy as np
import arviz as az
import scipy.linalg as sl
from arviz.data.base import dict_to_dataset
import logging

_WHITENED_LOGLIKE_KEY = 'whitened_pointwise_loglike'

class Result(az.InferenceData):

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], az.InferenceData):
            # modeled after from_netcdf
            # https://python.arviz.org/en/stable/_modules/arviz/data/inference_data.html#
            # az.InferenceData(fit.result.attrs, **{k: getattr(fit.result, k) for k in fit.result._groups})
            super().__init__(args[0].attrs, **{k: getattr(args[0], k) for k in args[0]._groups})
        else:
            super().__init__(*args, **kwargs)
            
    @classmethod
    def from_netcdf(cls, *args, **kwargs):
        # Load the data using the base class method
        data = super().from_netcdf(*args, **kwargs)
        # Create an instance of the subclass with the loaded data
        return cls(data)
    
    @classmethod
    def from_zarr(cls, *args, **kwargs):
        # Load the data using the base class method
        data = super().from_zarr(*args, **kwargs)
        # Create an instance of the subclass with the loaded data
        return cls(data)

    def draw_sample(self, map=False, prior=False, rng=None, seed=None):
        """Draw a sample from the posterior.

        Arguments
        ---------
        map : bool
           return maximum-probability sample; otherwise, returns random draw
           (def., `False`) 
        prior : bool
            draw from prior instead of posterior samples
        rng : numpy.random._generator.Generator
            random number generator (optional)
        seed : int
            seed to initialize new random number generator (optional)
        
        Returns
        -------
        i : int
            location of draw in stacked samples (i.e., samples obtained by
            calling ``posterior.stack(sample=('chain', 'draw'))``)
        pars : xarray.core.dataset.DataVariables
            object containing drawn parameters (can be treated as dict)
        """
        samples = self.posterior.stack(sample=('chain', 'draw'))
        if map:
            # select maximum probability sample
            logp = self.sample_stats.lp.stack(sample=('chain', 'draw'))
            i = np.argmax(logp.values)
        else:
            # pick random sample
            rng = rng or np.random.default_rng(seed)
            i = rng.integers(len(samples['sample']))
        sample = samples.isel(sample=i)
        pars = sample.data_vars
        return i, pars 
        
    @property
    def ifos(self):
        return self.posterior.ifo
    
    @property
    def cholesky_factors(self):
        if 'L' in self.constant_data:
            return self.constant_data.L
        else:
            return self.constant_data.cholesky_factor
        
    def whiten(self, datas):
        chols = self.cholesky_factors
        if isinstance(datas, dict):
            wds = {}    
            for i, data in datas.items():
                L = chols.sel(ifo=i)
                wds[i] = sl.solve_triangular(L, data, lower=True)
        else:
            # whiten the reconstructions using the Cholesky factors, L, with shape
            # (ifo, time, time). the resulting object will have shape (ifo, time, sample)
            wds = np.array([sl.solve_triangular(L, d, lower=True)
                            for L, d in zip(chols, datas)])
        return wds

    @property
    def whitened_templates(self) -> np.ndarray:
        """Whitened templates corresponding to each posterior sample, as
        were seen by the sampler.

        Dimensions will be ``(ifo, time, sample)``.

        Corresponding unwhitened templates can be obtained from posterior by
        doing::

          result.posterior.h_det.stack(sample=('chain', 'draw'))
        """
        # get reconstructions from posterior, shaped as (chain, draw, ifo, time)
        # and stack into (ifo, time, sample)
        hs = self.posterior.h_det.stack(samples=('chain', 'draw'))
        return  self.whiten(hs)

    def compute_posterior_snrs(self, optimal=True, network=True) -> np.ndarray:
        """Efficiently computes signal-to-noise ratios from posterior samples,
        reproducing the computation internally carried out by the sampler.

        Depending on the ``optimal`` argument, returns either the optimal SNR::

          snr_opt = sqrt(dot(template, template))

        or the matched filter SNR::

          snr_mf = dot(data, template) / snr_opt

        Arguments
        ---------
        optimal : bool
            return optimal SNR, instead of matched filter SNR (def., ``True``)
        network : bool
            return network SNR, instead of individual-detector SNRs (def.,
            ``True``)

        Returns
        -------
        snrs : array
            stacked array of SNRs, with shape ``(samples,)`` if ``network =
            True``, or ``(ifo, samples)`` otherwise; the number of samples
            equals the number of chains times the number of draws.
        """
        # get whitened reconstructions from posterior (ifo, time, sample)
        whs = self.whitened_templates
        # take the norm across time to get optimal snrs for each (ifo, sample)
        opt_ifo_snrs = np.linalg.norm(whs, axis=1)
        if optimal:
            snrs = opt_ifo_snrs
        else:
            # get analysis data, shaped as (ifo, time)
            if "strain" in self.observed_data:
                # strain values stored in "old" PyStan structure
                ds = self.observed_data.strain
            elif "strain" in self.constant_data:
                ds = self.constant_data.strain
            else:
                # strain values stored in "new" PyMC structure
                ds = np.array([d.values for d in self.observed_data.values()])
            # whiten it with the Cholesky factors, so shape will remain (ifo, time)
            wds = self.whiten(ds)
            # take inner product between whitened template and data, and normalize
            snrs = np.einsum('ijk,ij->ik', whs, wds)/opt_ifo_snrs
        if network:
            # take norm across detectors
            return np.linalg.norm(snrs, axis=0)
        else:
            return snrs

    @property
    def waic(self):
        """Returns the 'widely applicable information criterion' predictive
        accuracy metric for the fit.
        
        See https://arxiv.org/abs/1507.04544 for definitions and discussion.  A
        larger WAIC indicates that the model has better predictive accuarcy on
        the fitted data set."""
        if not _WHITENED_LOGLIKE_KEY in self.get('log_likelihood', {}):
            self._generate_whitened_residuals()
        return az.waic(self, var_name=_WHITENED_LOGLIKE_KEY)
    
    @property
    def loo(self):
        """Returns a leave-one-out estimate of the predictive accuracy of the
        model.
        
        See https://arxiv.org/abs/1507.04544 for definitions and discussion,
        including discussion of the 'Pareto stabilization' algorithm for
        reducing the variance of the leave-one-out estimate.  The LOO is an
        estimate of the expected log predictive density (log of the likelihood
        evaluated on hypothetical data from a replication of the observation
        averaged over the posterior) of the model; larger LOO values indicate
        higher predictive accuracy (i.e. explanatory power) for the model."""
        if not _WHITENED_LOGLIKE_KEY in self.get('log_likelihood', {}):
            self._generate_whitened_residuals()
        return az.loo(self, var_name=_WHITENED_LOGLIKE_KEY)
    
    def _generate_whitened_residuals(self):
        """Adduct the whitened residuals to the result.
        """
        residuals = {}
        residuals_stacked = {}
        for ifo in self.ifos.values.astype(str):
            r = self.constant_data.strain.sel(ifo=ifo) -\
                self.posterior.h_det.sel(ifo=ifo)
            residuals[ifo] = r.transpose('chain', 'draw', 'time_index')
            residuals_stacked[ifo] = residuals[ifo].stack(sample=['chain',
                                                                  'draw'])
        residuals_whitened = self.whiten(residuals_stacked)
        d = self.posterior.sizes
        residuals_whitened = {
            i: v.reshape((d['time_index'], d['chain'], d['draw']))
            for i,v in residuals_whitened.items()
        }
        resid = np.stack([residuals_whitened[i] 
                          for i in self.ifos.values.astype(str)], axis=-1)
        keys = ('time_index', 'chain', 'draw', 'ifo')
        self.posterior['whitened_residual'] = (keys, resid)
        keys = ('chain', 'draw', 'ifo', 'time_index')
        self.posterior['whitened_residual'] = \
            self.posterior.whitened_residual.transpose(*keys)
        lnlike = -self.posterior.whitened_residual**2/2
        try:
            self.log_likelihood[_WHITENED_LOGLIKE_KEY] = lnlike    
        except AttributeError:
            # We assume that log-likelihood isn't created yet.
            self.add_groups(dict(
                log_likelihood=dict_to_dataset(
                    {_WHITENED_LOGLIKE_KEY: lnlike},
                    coords=self.posterior.coords,
                    dims={_WHITENED_LOGLIKE_KEY: list(keys)}
                    )))
            
    @property
    def ess(self):
        """Minimum effective sample size for all parameters in the result.
        """
        # check effective number of samples and rerun if necessary
        ess = az.ess(self)
        mess = ess.min()
        mess_arr = np.array([mess[k].values[()] for k in mess.keys()])
        return np.min(mess_arr)
