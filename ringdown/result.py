"""Module defining the core :class:`Result` class.
"""

__all__ = ['Result']

import numpy as np
import arviz as az
import scipy.linalg as sl
from arviz.data.base import dict_to_dataset
import logging
from . import qnms
from . import data
from .target import construct_target
import pandas as pd
import json
import configparser

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
        self._whitened_templates = None
        self._config_dict = None
        self._target = None
        
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
        
    @property
    def config(self):
        if self._config_dict is None:
            if 'config' in self.attrs:
                config_string = self.attrs['config']
                self._config_dict = json.loads(config_string)
            else:
                self._config_dict = {}
        return self._config_dict
    
    @property
    def _config_object(self):
        config = configparser.ConfigParser()
        # Populate the ConfigParser with data from the dictionary
        for section, settings in self.config.items():
            config.add_section(section)
            for key, value in settings.items():
                config.set(section, key, value)
        return config
    
    @property
    def target(self):
        if self._target is None:
            if 'target' in self.config:
                self._target = construct_target(**self.config['target'])
        return self._target
    
    @property
    def epoch(self):
        if 'epoch' in self.constant_data:
            return self.constant_data.epoch
        elif 'target' in self.config:
            ifos = self.posterior.ifo.values.astype(str)
            epochs = list(self.target.get_detector_times_dict(ifos).values())
            shape = self.constant_data.fp.shape
            return np.array(epochs).reshape(shape)
        else:
            return np.zeros_like(self.constant_data.fp)
        
    @property
    def sample_times(self):
        return self.constant_data.time + self.epoch
    
    def get_fit(self, **kwargs):
        if self.config:
            from .fit import Fit
            return Fit.from_config(self._config_object, result=self, **kwargs)

    def draw_sample(self, map : bool = False, rng : np.random.Generator = None,
                    seed : int = None) -> tuple[int, dict]:
        """Draw a sample from the posterior.

        Arguments
        ---------
        map : bool
           return maximum-probability sample; otherwise, returns random draw
           (def., `False`) 
        rng : numpy.random.Generator
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
        samples = self.stacked_samples
        if map:
            # select maximum probability sample
            if 'lp' in self.sample_stats:
                logp = self.sample_stats.lp
            else:
                logp = sum([v for k,v in self.log_likelihood.items()
                            if k.startswith('logl_')])
            i = np.argmax(logp.stack(sample=('chain', 'draw')).values)
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
    def cholesky_factors(self) -> np.ndarray :
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
        if self._whitened_templates is None:

            # get reconstructions from posterior, shaped as (chain, draw, ifo, time)
            # and stack into (ifo, time, sample)
            hs = self.posterior.h_det.stack(samples=('chain', 'draw'))
            self._whitened_templates = self.whiten(hs)
        return self._whitened_templates

    def compute_posterior_snrs(self, optimal : bool = True, 
                               network : bool = True) -> np.ndarray:
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
    
    def _generate_whitened_residuals(self) -> None:
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
    def ess(self) -> float:
        """Minimum effective sample size for all parameters in the result.
        """
        # check effective number of samples and rerun if necessary
        ess = az.ess(self)
        mess = ess.min()
        mess_arr = np.array([mess[k].values[()] for k in mess.keys()])
        return np.min(mess_arr)
    
    @property
    def stacked_samples(self):
        """Stacked samples for all parameters in the result.
        """
        return self.posterior.stack(sample=('chain', 'draw'))
    
    _PARAMETER_KEY_MAP = {
        'm': '$M / M_\\odot$',
        'chi': '$\\chi$',
        'f': '$f_{{{}}} / \\mathrm{{Hz}}$',
        'g': '$\\gamma_{{{}}} / \\mathrm{{Hz}}$',
        'a': '$A_{{{}}}$',
        'phi': '$\\phi_{{{}}}$',
        'theta': '$\\theta_{{{}}}$',
        'ellip': '$\\epsilon_{{{}}}$',
    }
    
    _STRAIN_KEY_MAP = {
        'h_det': '$h(t) [\\mathrm{{{ifo}}}]$',
        'h_det_mode': '$h_{{{mode}}}(t) [\\mathrm{{{ifo}}}]$',
    }
    
    def get_parameter_key_map(self, modes : bool = True, **kws) -> dict:
        key_map = {}
        for key, key_latex in self._PARAMETER_KEY_MAP.items():
            if key in self.posterior:
                x = self.posterior[key]
                if modes and 'mode' in x.dims:
                    for mode in x.mode.values:
                        label = qnms.get_mode_label(mode, **kws)
                        mode_key = f'{key}_{label}'
                        key_map[mode_key] = key_latex.format(label)
                elif 'mode' in x.dims:
                    key_map[key] = key_latex.replace('_{{{}}}', '')
                else:
                    key_map[key] = key_latex
        return key_map
    
    def get_strain_key_map(self, ifos : bool = True, modes : bool = True,
                            **kws) -> dict :
        strain_map = {}
        for k, v in self._STRAIN_KEY_MAP.items():
            if k in self.posterior:
                x = self.posterior[k]
                if ifos:
                    for ifo in x.ifo.values:
                        k_ifo = f'{k}_{ifo}'
                        if modes and 'mode' in x.dims:
                            for m in x.mode.values:
                                label = qnms.get_mode_label(m, **kws)
                                mode_k = f'{k_ifo}_{label}'
                                strain_map[mode_k] = v.format(mode=label, ifo=ifo)
                        elif 'mode' in x.dims:
                            strain_map[k_ifo] = v.replace('_{{{mode}}}','').format(ifo=ifo)
                        else:
                            strain_map[k_ifo] = v.format(ifo=ifo)
                elif modes and 'mode' in x.dims:
                    for m in x.mode.values:
                        label = qnms.get_mode_label(m, **kws)
                        mode_k = f'{k}_{label}'
                        strain_map[mode_k] = v.replace(' [\\mathrm{{{ifo}}}]', '').format(mode=label)
                else:
                    strain_map[k] = v.replace('_{{{mode}}}', '').replace(' [\\mathrm{{{ifo}}}]', '')
        return strain_map
    
    def get_full_key_map(self, **kws):
        key_map = self.get_parameter_key_map(**kws)
        strain_map = self.get_strain_key_map(**kws)
        key_map.update(strain_map)
        return key_map
    
    def get_parameter_dataframe(self, latex_keys : bool = False,
                                 **kws) -> pd.DataFrame :
        samples = self.stacked_samples
        df = pd.DataFrame()
        for key, key_latex in self._PARAMETER_KEY_MAP.items():
            if key in samples:
                x = samples[key]
                if 'mode' in x.dims:
                    for mode in x.mode.values:
                        label = qnms.get_mode_label(mode, **kws)
                        if latex_keys:
                            key_df = key_latex.format(label)
                        else:
                            key_df = f'{key}_{label}'
                        df[key_df] = x.sel(mode=mode).values
                else:
                    key_df = key_latex if latex_keys else key
                    df[key_df] = x.values
        return df
    
    def get_mode_parameter_dataframe(self, latex_keys : bool = False,
                                     ignore_index : bool = True,
                                     **kws) -> pd.DataFrame :
        samples = self.stacked_samples
        dfs = []
        for mode in self.posterior.mode.values:
            label = qnms.get_mode_label(mode, **kws)
            df = pd.DataFrame()
            for key, key_latex in self.get_parameter_key_map(modes=False).items():
                if key in samples:
                    x = samples[key]
                    if 'mode' in x.dims:
                        key_df = key_latex.format(label) if latex_keys else key
                        df[key_df] = x.sel(mode=mode).values
            df['mode'] = label
            dfs.append(df)
        return pd.concat(dfs, ignore_index=ignore_index)
        
    def get_strain_quantile(self, q : float, ifo : str = None,
                            mode : str | tuple | qnms.ModeIndex | bytes = None)\
                             -> dict[data.Data] | data.Data :
        """Get the quantile of the strain reconstruction.
        
        Arguments
        ---------
        q : float
            quantile to compute
        ifo : str
            detector to extract (optional)
        mode : str, tuple, ModeIndex, or bytes
            mode to extract (optional)

        Returns
        -------
        h : dict[data.Data] | data.Data
            dictionary of strain reconstructions, with keys corresponding to
            detector names, or a single data object if ``ifo`` is not provided
        """
        if mode is None:
            key = 'h_det'
        else:
            mode = qnms.get_mode_coordinate(mode)
            key = 'h_det_mode'
        sel = {k: v for k, v in dict(ifo=ifo, mode=mode).items()
               if v is not None}
        x = self.posterior[key].sel(**sel)
        hq = x.quantile(q, dim=('chain', 'draw'))
        hdict = {}
        for i in hq.ifo.values.astype(str):
            time = self.sample_times.sel(ifo=i).values
            hdict[i] = data.Data(hq.sel(ifo=i).values, index=time)
        h = hdict if ifo is None else hdict[ifo]
        return h
    
    def get_strain_sample(self, 
                          idx : int | None= None,
                          map : bool = False,
                          ifo : str | None = None,
                          mode : str | tuple | qnms.ModeIndex | bytes | None = None,
                          rng : int | np.random.Generator | None = None,
                          seed : int | None = None) \
                          -> dict[data.Data] | data.Data:
        """Get a sample of the strain reconstruction.
        
        Arguments
        ---------
        ifo : str
            detector to extract
        mode : str, tuple, ModeIndex, or bytes
            mode to extract (optional)

        Returns
        -------
        h : dict[data.Data] | data.Data
            dictionary of strain reconstructions, with keys corresponding to
            detector names, or a single data object if ``ifo`` is not provided
        """
        if mode is None:
            key = 'h_det'
        else:
            mode = qnms.get_mode_coordinate(mode)
            key = 'h_det_mode'
        if idx == None:
            idx, x = self.draw_sample(map=map, rng=rng, seed=seed)
        elif isinstance(idx, int):
            x = self.stacked_samples.isel(sample=idx)
        elif isinstance(idx, dict):
            x = self.posterior.isel(**idx)
        elif len(idx) == 2:
            x = self.posterior.isel(chain=idx[0], draw=idx[1])
        else:
            raise ValueError(f"Invalid index: {idx}")
        sel = {k: v for k, v in dict(ifo=ifo, mode=mode).items()
               if v is not None}
        h = x[key].sel(**sel)
        hdict = {}
        for i in h.ifo.values.astype(str):
            time = self.sample_times.sel(ifo=i).values
            hdict[i] = data.Data(h.sel(ifo=i).values, index=time)
        hdata = hdict if ifo is None else hdict[ifo]
        return hdata