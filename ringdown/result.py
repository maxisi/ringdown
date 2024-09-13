"""Module defining the core :class:`Result` class.
"""

__all__ = ['Result', 'ResultCollection']

import os
import numpy as np
import arviz as az
import scipy.linalg as sl
from arviz.data.base import dict_to_dataset
from . import qnms
from . import indexing
from . import data
from .target import Target, TargetCollection
from . import utils
import pandas as pd
import json
import configparser
from glob import glob
from parse import parse
import logging

_WHITENED_LOGLIKE_KEY = 'whitened_pointwise_loglike'

_DATAFRAME_PARAMETERS = ['m', 'chi', 'f', 'g',
                         'a', 'phi', 'theta', 'ellip', 'df', 'dg']


class Result(az.InferenceData):
    """Result from a ringdown fit."""

    def __init__(self, *args, config=None, produce_h_det=True, **kwargs):
        """Initialize a result from a ringdown fit.

        Arguments
        ---------
        *args : list
            arguments to pass to the `az.InferenceData` constructor.
        config : str
            path to configuration file (optional).
        produce_h_det : bool
            produce h_det from modes if not already present (def., `True`).
        **kwargs : dict
            additional keyword arguments to pass to the `az.InferenceData`
            constructor.
        """
        # get config file input if provided
        if len(args) == 1 and isinstance(args[0], az.InferenceData):
            # modeled after from_netcdf
            # https://python.arviz.org/en/stable/_modules/arviz/data/inference_data.html#
            # az.InferenceData(fit.result.attrs, **{k: getattr(fit.result, k)
            # for k in fit.result._groups})
            super().__init__(args[0].attrs, **{k: getattr(args[0], k)
                                               for k in args[0]._groups})
        else:
            super().__init__(*args, **kwargs)
        self._whitened_templates = None
        self._target = None
        self._modes = None
        # settings for formatting DataFrames
        self._default_label_format = {}
        # try to load config
        if config is not None:
            self._config_dict = utils.load_config_dict(config)
        else:
            self._config_dict = None
        # produce h_det (i.e., sum of all modes) if not already present
        if produce_h_det:
            self.h_det

    @property
    def _df_parameters(self) -> dict[str, qnms.ParameterLabel]:
        """Default parameters for DataFrames."""
        df_parameters = {}
        for m in _DATAFRAME_PARAMETERS:
            if m in getattr(self, 'posterior', {}):
                df_parameters[m] = qnms.ParameterLabel(m)
            elif m.upper() in getattr(self, 'posterior', {}):
                df_parameters[m.upper()] = qnms.ParameterLabel(m)
        return df_parameters

    @property
    def strain_scale(self) -> float:
        """Scale factor for strain data.
        """
        s = self.get('constant_data', {}).get('scale', 1.0)
        return float(s)

    @property
    def h_det(self):
        """Alias for `posterior.h_det`, in case this is not already present,
        it gets computed from individual modes."""
        if 'h_det' in self.posterior:
            return self.posterior.h_det
        elif 'h_det_mode' in self.posterior:
            self.posterior['h_det'] = self.posterior.h_det_mode.sum('mode')
            return self.posterior.h_det
        else:
            return None

    @property
    def h_det_mode(self):
        """Alias for `posterior.h_det_mode`, in case this is not already
        present,  it gets computed from the sum of all modes."""
        if 'h_det_mode' in self.posterior:
            return self.posterior.h_det_mode
        else:
            return None

    def rescale_strain(self, scale=None) -> None:
        """Autoscale the strain data in the result.
        """
        scale = scale or self.strain_scale
        logging.info(f"rescaling strain by {scale}")
        if 'h_det' in self.posterior:
            self.posterior['h_det'] = scale * self.posterior['h_det']
        if 'h_det_mode' in self.posterior:
            self.posterior['h_det_mode'] = \
                scale * self.posterior['h_det_mode']
        if 'a' in self.posterior:
            self.posterior['a'] = scale * self.posterior['a']
        if 'a_scale' in self.posterior:
            self.posterior['a_scale'] = scale * self.posterior['a_scale']
        if 'observed_data' in self and 'strain' in self.observed_data:
            self.observed_data['strain'] = \
                scale * self.observed_data['strain']
        if 'constant_data' in self:
            if 'scale' in self.constant_data:
                self.constant_data['scale'] = \
                    self.constant_data['scale'] / scale
            if 'injection' in self.constant_data:
                self.constant_data['injection'] = \
                    scale * self.constant_data['injection']
            if 'cholesky_factor' in self.constant_data:
                self.constant_data['cholesky_factor'] = \
                    scale * self.constant_data['cholesky_factor']

    @property
    def default_label_format(self) -> dict:
        """Default formatting options for DataFrames.
        """
        kws = dict(
            label_prograde=any([not m.is_prograde for m in self.modes]),
        )
        kws.update(self._default_label_format)
        return kws

    def update_default_label_format(self, **kws) -> None:
        """Update the default formatting options for DataFrames.
        """
        self._default_label_format.update(kws)

    @classmethod
    def from_netcdf(cls, *args, config=None, **kwargs) -> 'Result':
        data = super().from_netcdf(*args, **kwargs)
        return cls(data, config=config)
    from_netcdf.__doc__ = az.InferenceData.from_netcdf.__doc__

    @classmethod
    def from_zarr(cls, *args, **kwargs):
        data = super().from_zarr(*args, **kwargs)
        return cls(data)
    from_zarr.__doc__ = az.InferenceData.from_zarr.__doc__

    @property
    def config(self) -> dict[str, dict[str, str]]:
        """Configuration dictionary for the result. Entries represent sections
        of the configuration file.
        """
        if self._config_dict is None:
            if 'config' in self.attrs:
                config_string = self.attrs['config']
                raw_config_dict = json.loads(config_string)
                self._config_dict = {kk: {k: utils.try_parse(v)
                                          for k, v in vv.items()}
                                     for kk, vv in raw_config_dict.items()}
            else:
                self._config_dict = {}
        return self._config_dict

    @property
    def _config_object(self) -> configparser.ConfigParser:
        """Configuration file stored as ConfigParser object."""
        config = configparser.ConfigParser()
        # Populate the ConfigParser with data from the dictionary
        for section, settings in self.config.items():
            config.add_section(section)
            for key, value in settings.items():
                config.set(section, key, utils.form_opt(value))
        return config

    @property
    def target(self) -> Target | None:
        """Target used in the analysis."""
        if self._target is None:
            if 'target' in self.config:
                self._target = Target.construct(**self.config['target'])
        return self._target

    @property
    def t0(self) -> float | None:
        """Reference time for the analysis."""
        return getattr(self.target, 't0', None)

    @property
    def epoch(self):
        """Epoch for detector times; corresponds to `fit.start_times`
        """
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
        """Sample times for the analysis; corresponds to
        `fit.analysis_data[i].time`."""
        shape = (self.posterior.sizes['ifo'], 1)
        return self.constant_data.time + np.array(self.epoch).reshape(shape)

    def get_fit(self, **kwargs):
        """Get a Fit object from the result."""
        if self.config:
            from .fit import Fit
            return Fit.from_config(self._config_object, result=self, **kwargs)

    def draw_sample(self,
                    idx: int | tuple[int, int] | dict = None,
                    map: bool = False,
                    rng: np.random.Generator = None,
                    seed: int = None) -> tuple[int, dict]:
        """Draw a sample from the posterior.

        Arguments
        ---------
        idx : int, dict, or tuple
            index of sample to draw; if an integer, it is the index of the
            sample in the stacked samples; if a tuple, it is the (chain, draw)
            index of the sample in the posterior; if a dictionary, it is the
            index of the sample with keys corresponding to the dimensions of
            the posterior (chains, draws); (def., `None`)
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
        if idx is not None:
            if isinstance(idx, int):
                i = idx
            elif isinstance(idx, dict):
                i = tuple(list(idx.values()))
                sample = samples.isel(**idx)
            elif len(idx) == 2:
                i = tuple(idx)
                sample = self.posterior.isel(chain=idx[0], draw=idx[1])
        elif map:
            # select maximum probability sample
            if 'lp' in self.sample_stats:
                logp = self.sample_stats.lp
            else:
                logp = sum([v for k, v in self.log_likelihood.items()
                            if k.startswith('logl_')])
            i = np.argmax(logp.stack(sample=('chain', 'draw')).values)
            sample = samples.isel(sample=i)
        else:
            # pick random sample
            rng = rng or np.random.default_rng(seed)
            i = rng.integers(len(samples['sample']))
            sample = samples.isel(sample=i)
        pars = sample.data_vars
        return i, pars

    @property
    def ifos(self) -> list:
        """Detectors used in analysis."""
        return self.posterior.ifo

    @property
    def modes(self) -> list | None:
        """Modes used in analysis."""
        if self._modes is None:
            self._modes = indexing.ModeIndexList(self.posterior.mode.values
                                                 if 'mode' in self.posterior
                                                 else [])
        return self._modes

    @property
    def cholesky_factors(self) -> np.ndarray:
        """Cholesky L factors used in analysis."""
        if 'L' in self.constant_data:
            return self.constant_data.L
        else:
            return self.constant_data.cholesky_factor

    def whiten(self, datas: dict | np.ndarray) -> dict | np.ndarray:
        """Whiten data using the Cholesky factors from the analysis.

        Arguments
        ---------
        datas : dict | np.ndarray
            data to whiten; if a dictionary, the keys should correspond to the
            detectors used in the analysis.

        Returns
        -------
        wds : dict | np.ndarray
            whitened data; if a dictionary, the keys correspond to the
            detectors used in the analysis.
        """
        chols = self.cholesky_factors
        if isinstance(datas, dict):
            wds = {}
            for i, d in datas.items():
                L = chols.sel(ifo=i)
                wd = sl.solve_triangular(L, d, lower=True)
                if isinstance(d, data.Data):
                    wd = data.Data(wd, index=d.index, ifo=i, info=d.info)
                wds[i] = wd
        else:
            # whiten the reconstructions using the Cholesky factors, L, with
            # shape (ifo, time, time). the resulting object will have shape
            # (ifo, time, sample)
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

          result.h_det.stack(sample=('chain', 'draw'))
        """
        if self._whitened_templates is None:
            # get reconstructions from posterior, shaped as
            # (chain, draw, ifo, time)
            # and stack into (ifo, time, sample)
            hs = self.h_det.stack(samples=('chain', 'draw'))
            self._whitened_templates = self.whiten(hs)
        return self._whitened_templates

    def compute_posterior_snrs(self, optimal: bool = True,
                               network: bool = True) -> np.ndarray:
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
            ds = self.observed_strain
            # whiten it with the Cholesky factors,
            # so shape will remain (ifo, time)
            wds = self.whiten(ds)
            # take inner product between whitened template and data,
            # and normalize
            snrs = np.einsum('ijk,ij->ik', whs, wds)/opt_ifo_snrs
        if network:
            # take norm across detectors
            return np.linalg.norm(snrs, axis=0)
        else:
            return snrs

    def compute_posterior_snr_timeseries(self, optimal: bool = True,
                                         network: bool = True) -> np.ndarray:
        """Efficiently computes cumulative signal-to-noise ratio from
        posterior samples as a function of time.

        Depending on the ``optimal`` argument, returns either the optimal SNR::

          snr_opt = sqrt(dot(template, template))

        or the matched filter SNR::

          snr_mf = dot(data, template) / snr_opt

        NOTE: the last time sample of the returned SNR timeseries corresponds
              to the total accumulated SNR, which is the same as returned
              by :meth:`compute_posterior_snrs`; however, that function is
              10x faster, so use it if you only need the total SNR.

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
            stacked array of cumulative SNRs, with shape ``(time, samples,)``
            if ``network = True``, or ``(ifo, time, samples)`` otherwise;
            the number of samples equals the number of chains times the number
            of draws.

        See Also
        --------
        compute_posterior_snrs : Computes the overall signal-to-noise ratio.
        """
        # get whitened reconstructions from posterior (ifo, time, sample)
        whs = self.whitened_templates
        # get series of cumulative optimal SNRs for each (ifo, time, sample)
        opt_ifo_snrs = np.sqrt(np.cumsum(whs * whs, axis=1))
        if optimal:
            snrs = opt_ifo_snrs
        else:
            # get analysis data, shaped as (ifo, time)
            ds = self.observed_strain
            # whiten it with the Cholesky factors,
            # so shape will remain (ifo, time)
            wds = self.whiten(ds)
            # take inner product between whitened template and data,
            # and normalize
            snrs = np.cumsum(wds[:, :, None]*whs, axis=1) / opt_ifo_snrs
        if network:
            # take norm across detectors
            return np.linalg.norm(snrs, axis=0)
        else:
            return snrs

    @property
    def observed_strain(self):
        if "strain" in self.observed_data:
            return self.observed_data.strain
        elif "strain" in self.constant_data:
            return self.constant_data.strain
        else:
            raise KeyError("No observed strain found in result.")

    @property
    def waic(self) -> az.ELPDData:
        """Returns the 'widely applicable information criterion' predictive
        accuracy metric for the fit.

        See https://arxiv.org/abs/1507.04544 for definitions and discussion.  A
        larger WAIC indicates that the model has better predictive accuarcy on
        the fitted data set."""
        if _WHITENED_LOGLIKE_KEY not in self.get('log_likelihood', {}):
            self._generate_whitened_residuals()
        return az.waic(self, var_name=_WHITENED_LOGLIKE_KEY)

    @property
    def loo(self) -> az.ELPDData:
        """Returns a leave-one-out estimate of the predictive accuracy of the
        model.

        See https://arxiv.org/abs/1507.04544 for definitions and discussion,
        including discussion of the 'Pareto stabilization' algorithm for
        reducing the variance of the leave-one-out estimate.  The LOO is an
        estimate of the expected log predictive density (log of the likelihood
        evaluated on hypothetical data from a replication of the observation
        averaged over the posterior) of the model; larger LOO values indicate
        higher predictive accuracy (i.e. explanatory power) for the model."""
        if _WHITENED_LOGLIKE_KEY not in self.get('log_likelihood', {}):
            self._generate_whitened_residuals()
        return az.loo(self, var_name=_WHITENED_LOGLIKE_KEY)

    def _generate_whitened_residuals(self) -> None:
        """Adduct the whitened residuals to the result.
        """
        residuals = {}
        residuals_stacked = {}
        for ifo in self.ifos.values.astype(str):
            r = self.observed_strain[list(self.ifos.values.astype(str)).index(ifo)] -\
                self.h_det.sel(ifo=ifo)
            residuals[ifo] = r.transpose('chain', 'draw', 'time_index')
            residuals_stacked[ifo] = residuals[ifo].stack(sample=['chain',
                                                                  'draw'])
        residuals_whitened = self.whiten(residuals_stacked)
        d = self.posterior.sizes
        residuals_whitened = {
            i: v.reshape((d['time_index'], d['chain'], d['draw']))
            for i, v in residuals_whitened.items()
        }
        resid = np.stack([residuals_whitened[i]
                          for i in self.ifos.values.astype(str)], axis=-1)
        keys = ('time_index', 'chain', 'draw', 'ifo')
        self.posterior['whitened_residual'] = (keys, resid)
        keys = ('chain', 'draw', 'ifo', 'time_index')
        self.posterior['whitened_residual'] = \
            self.posterior.whitened_residual.transpose(*keys)
        lnlike = -self.posterior.whitened_residual**2/2
        if hasattr(self, 'log_likelihood'):
            self.log_likelihood[_WHITENED_LOGLIKE_KEY] = lnlike
        else:
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

    def set_dataframe_parameters(self, parameters: list[str]) -> None:
        """Set the parameters to be included in DataFrames derived
        from this result."""
        pars = []
        for par in parameters:
            p = par.lower()
            if p in self.posterior:
                pars.append(p)
            else:
                raise ValueError(f"Parameter {par} not found in posterior.")
        self._df_parameters.update({p: qnms.ParameterLabel(p) for p in pars})

    def get_parameter_key_map(self, modes: bool = True, **kws) -> dict:
        """Get a dictionary of parameter labels for the result."""
        kws['latex'] = True
        if modes:
            x = {}
            for m in self.modes:
                x.update({p.get_label(mode=m, latex=False): p.get_label(mode=m,
                                                                        **kws)
                          for p in self._df_parameters.values()})
        else:
            x = {k: p.get_label(**kws) for k, p in self._df_parameters.items()}
        return x

    def get_parameter_dataframe(self, nsamp: int | None = None,
                                rng: int | np.random.Generator = None,
                                ignore_index=False,
                                **kws) -> pd.DataFrame:
        """Get a DataFrame of parameter samples drawn from the posterior.

        The columns correspond to parameters and the index to the sample,
        which can be used to locate this row in the `Result.stacked_samples`.
        If `ignore_index`, the index will be reset rather than showing the
        location in the original set of samples.

        The parameters are labeled using the `qnms.ParameterLabel` class.

        Arguments
        ---------
        nsamp : int
            number of samples to draw from the posterior (optional).
        rng : numpy.random.Generator | int
            random number generator or seed (optional).
        ignore_index : bool
            reset index rather than showing location in original samples
            (def., `False`).
        **kws : dict
            additional keyword arguments to pass to the `get_label` method of
            :class:`qnms.ParameterLabel`.
        """
        # set labeling options (e.g., whether to show p index)
        fmt = self.default_label_format.copy()
        fmt.update(kws)
        # get samples
        samples = self.stacked_samples
        if nsamp is not None:
            rng = rng or np.random.default_rng(rng)
            idxs = rng.choice(samples.sizes['sample'], nsamp, replace=False)
            samples = samples.isel(sample=idxs)
        else:
            idxs = None
        df = pd.DataFrame(index=idxs if not ignore_index else None)
        for p, par in self._df_parameters.items():
            x = samples[p]
            if 'mode' in x.dims:
                for mode in x.mode.values:
                    key_df = par.get_label(mode=mode, **fmt)
                    df[key_df] = x.sel(mode=mode).values
            else:
                key_df = par.get_label(**fmt)
                df[key_df] = x.values
        return df

    def get_mode_parameter_dataframe(self, nsamp: int | None = None,
                                     ignore_index: bool = False,
                                     rng: int | np.random.Generator |
                                     None = None,
                                     **kws) -> pd.DataFrame:
        """Get a DataFrame of parameter samples drawn from the posterior, with
        columns for different modes.

        This is similar to :meth:`get_parameter_dataframe`, but splits the
        parameters for each mode into unique columns, rather than stacking
        them and labeling the modes through a `mode` column.

        Arguments
        ---------
        nsamp : int
            number of samples to draw from the posterior (optional).
        ignore_index : bool
            reset index rather than showing location in original samples
            (def., `False`).
        rng : numpy.random.Generator | int
            random number generator or seed (optional).
        **kws : dict
            additional keyword arguments to pass to the `get_label` method of
            :class:`qnms.ParameterLabel`.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of parameter samples.
        """
        # set labeling options (e.g., whether to show p index)
        fmt = self.default_label_format.copy()
        fmt.update(kws)
        # get samples
        samples = self.stacked_samples
        if nsamp is not None:
            rng = rng or np.random.default_rng(rng)
            idxs = rng.choice(samples.sizes['sample'], nsamp, replace=False)
            samples = samples.isel(sample=idxs)
        else:
            idxs = None
        dfs = []
        for mode, m in zip(self.modes, self.posterior.mode.values):
            df = pd.DataFrame(index=idxs)
            for p, par in self._df_parameters.items():
                if p in samples and 'mode' in samples[p].dims:
                    x = samples[p]
                    key_df = par.get_label(mode=None, **fmt)
                    df[key_df] = x.sel(mode=m).values
            df['mode'] = mode.get_label(**fmt)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=ignore_index)

    def get_single_mode_dataframe(self,
                                  mode: str | tuple | indexing.ModeIndex |
                                  bytes,
                                  *args,
                                  **kws) -> pd.DataFrame:
        """Get a DataFrame of parameter samples drawn from the posterior for a
        specific mode.

        Arguments
        ---------
        mode : str, tuple, ModeIndex, or bytes
            mode to extract.
        *args : list
            additional arguments to pass to
            :meth:`get_mode_parameter_dataframe`.
        **kws : dict
            additional keyword arguments to pass to the `get_mode_label` method
            of :class:`qnms.ParameterLabel`.
        """
        df = self.get_mode_parameter_dataframe(*args, **kws)
        return df[df['mode'] == indexing.get_mode_label(mode, **kws)]

    def get_strain_quantile(self, q: float, ifo: str = None,
                            mode: str | tuple | indexing.ModeIndex |
                            bytes = None) -> dict[data.Data] | data.Data:
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
            # make sure h_det exists
            self.h_det
        else:
            mode = indexing.get_mode_coordinate(mode)
            if mode not in self.posterior.mode:
                raise ValueError("Mode requested not in result")
            key = 'h_det_mode'
        sel = {k: v for k, v in dict(mode=mode).items()
               if v is not None}
        x = self.posterior[key].sel(**sel)
        hq = x.quantile(q, dim=('chain', 'draw'))
        hdict = {}
        for i in hq.ifo.values.astype(str):
            time = self.sample_times.sel(ifo=i).values
            hdict[i] = data.Data(hq.sel(ifo=i).values, index=time, ifo=i)
        h = hdict if ifo is None else hdict[ifo]
        return h

    def draw_strain_sample(self,
                           idx: int | None = None,
                           map: bool = False,
                           ifo: str | None = None,
                           mode: str | tuple | indexing.ModeIndex | bytes |
                           None = None,
                           rng: int | np.random.Generator | None = None,
                           seed: int | None = None) \
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
            # make sure h_det exists
            self.h_det
        else:
            mode = indexing.get_mode_coordinate(mode)
            if mode not in self.posterior.mode:
                raise ValueError("Mode requested not in result")
            key = 'h_det_mode'
        idx, x = self.draw_sample(idx=idx, map=map, rng=rng, seed=seed)
        sel = {k: v for k, v in dict(mode=mode).items()
               if v is not None}
        h = x[key].sel(**sel)
        info = {k: v.values for k, v in x.items()}
        info['idx'] = idx
        hdict = {}
        for i in h.ifo.values.astype(str):
            time = self.sample_times.sel(ifo=i).values
            hdict[i] = data.Data(h.sel(ifo=i).values, ifo=i, index=time,
                                 attrs=info)
        hdata = hdict if ifo is None else hdict[ifo]
        return hdata

    @property
    def injected_strain(self) -> dict[data.Data] | None:
        """Injections used in the analysis."""
        if 'injection' in self.constant_data:
            h = self.constant_data.injection
            hdict = {}
            for i in h.ifo.values.astype(str):
                time = self.sample_times.sel(ifo=i).values
                hdict[i] = data.Data(h.sel(ifo=i).values, ifo=i, index=time)
            return hdict
        else:
            return None


class ResultCollection(utils.MultiIndexCollection):
    """Collection of results from ringdown fits."""

    def __init__(self, results: list | None = None,
                 index: list | None = None,
                 reference_mass: float | None = None,
                 reference_time: float | None = None) -> None:
        _results = []
        for r in results:
            if isinstance(r, Result):
                _results.append(r)
            else:
                _results.append(Result(r))
        super().__init__(_results, index, reference_mass, reference_time)
        self._targets = None

    def __repr__(self):
        return f"ResultCollection({self.index})"

    @property
    def results(self) -> list[Result]:
        """List of results in the collection.
        """
        return self.data

    @property
    def targets(self) -> TargetCollection:
        """Targets associated with the results in the collection.
        """
        if self._targets is None:
            self._targets = TargetCollection([r.target for r in self.results])
        elif len(self._targets) != len(self.results):
            logging.warning("Number of targets does not match results."
                            "Recomputing targets")
            self._targets = TargetCollection([r.target for r in self.results])
        return self._targets

    @property
    def reference_mass(self) -> float | None:
        """Reference mass in solar masses used for time-step labeling.
        """
        if self.targets.reference_mass is None:
            if self._reference_mass is None:
                logging.info("No reference mass specified; trying to infer "
                             "from result configurations.")
                m0 = None
                for r in self.results:
                    m0 = r.config.get('pipe', {}).get(self.targets._mref_key)
                    break
                self._reference_mass = m0
            self.targets.set_reference_mass(self._reference_mass)
        return self.targets.reference_mass

    @property
    def reference_time(self) -> float | None:
        """Reference time used for time-step labeling.
        """
        if self.targets.reference_time is None:
            if self._reference_time is None:
                logging.info("No reference time specified; trying to infer "
                             "from result configurations.")
                t0 = None
                for r in self.results:
                    t0 = r.config.get('pipe', {}).get(self.targets._tref_key)
                    break
                self._reference_time = t0
            self.targets.set_reference_time(self._reference_time)
        return self.targets.reference_time

    def set_reference_mass(self, reference_mass: float | None) -> None:
        """Set the reference mass for the collection."""
        self._reference_mass = reference_mass

    def set_reference_time(self, reference_time: float | None) -> None:
        """Set the reference time for the collection."""
        self._reference_time = reference_time

    def update_default_label_format(self, **kws) -> None:
        """Update the default formatting options for DataFrames."""
        for result in self.results:
            result.update_default_label_format(**kws)

    def get_t0s(self,
                reference_mass: float | bool | None = None) -> np.ndarray:
        """Get analysis start times for the collection.
        """
        if reference_mass:
            if reference_mass is None:
                # reference mass is true, but no value specified so assume
                # default
                reference_mass = self._reference_mass
            targets = self.targets
            targets.set_reference_mass(reference_mass)
            t0s = targets.t0m
        else:
            t0s = [result.t0 for result in self.results]
        return np.array(t0s)

    def reindex_by_t0(self,
                      reference_mass: bool | float | None = None) -> None:
        """Reindex the collection by the analysis start time.
        """
        t0s = self.get_t0s(reference_mass)
        if np.any(t0s is None):
            raise ValueError("Cannot reindex by t0 if any t0 values are None.")
        self.reindex(t0s)

    @classmethod
    def from_netcdf(cls, path_input: str | list, index: list = None,
                    config: str | list | None = None,
                    progress: bool = False, **kws):
        """Load a collection of results from NetCDF files.

        Arguments
        ---------
        path_input : str or list
            template path to NetCDF file or list of paths; if a string,
            expected to be a template like `path/to/many/files/*.nc` or
            `path/to/many/files/{}.nc` where `{}` or `*` is replaced by the
            index, or used to glob for files.
        index : list
            list of indices for the results; if not provided, will be inferred
            from the paths of found files.
        config : str or list
            template path to configuration file or list of paths; if a string,
            expected to be a template like `path/to/many/files/*.ini` or
            `path/to/many/files/{}.ini` where `{}` or `*` is replaced by the
            index, or used to glob for files.
        progress : bool
            show progress bar (def., `False`)
        **kws : dict
            additional keyword arguments to pass to the constructor, like
            reference_mass or reference_time
        """
        index = index or []
        cpaths = []
        if isinstance(path_input, str):
            paths = sorted(glob(path_input))
            logging.info(f"loading {len(paths)} results from {path_input}")
            for path in paths:
                pattern = parse(path_input.replace('*', '{}'), path).fixed
                idx = tuple([utils.try_parse(k) for k in pattern])
                index.append(idx)
                if isinstance(config, str):
                    cpath = config.replace('*', '{}').format(*pattern)
                    if os.path.exists(cpath):
                        cpaths.append(cpath)
        else:
            paths = path_input
        if config is not None:
            if len(cpaths) != len(paths):
                raise ValueError("Number of configuration files does not "
                                 "match number of result files.")
        else:
            cpaths = [None]*len(paths)
        results = []
        custom_tqdm = utils.get_tqdm(progress)
        for path, cpath in custom_tqdm(zip(paths, cpaths), total=len(paths)):
            results.append(Result.from_netcdf(path, config=cpath))
        info = kws.get('info', {})
        info['provenance'] = paths
        return cls(results, index, **kws)

    def get_parameter_dataframe(self, ndraw: int | None = None,
                                index_label: str = 'run',
                                split_index: bool = False,
                                t0: bool = False,
                                reference_mass: bool | float | None = None,
                                draw_kws: dict | None = None,
                                progress: bool = False,
                                **kws) -> pd.DataFrame:
        """Get a combined DataFrame of parameter samples for all results in
        the collection, with a new index column identifying provenance of each
        sample.

        Arguments
        ---------
        ndraw : int
            number of samples to draw from each result (optional)
        index_label : str
            name of provenance column (def., 'run')
        split_index : bool
            split tuple index into multiple columns (def., `False`)
        t0 : bool
            include reference time in DataFrame (def., `False`)
        reference_mass : bool or float
            mass in solar masses used to label time steps if including t0; if
            provided 't0m' will be used instead of 't0' (optional) so that the
            time is unit of mass rather than seconds.
        draw_kws : dict
            keyword arguments to pass to the `sample` method when drawing
            random samples from each result (optional)
        progress : bool
            show progress bar (def., `False`)
        **kws : dict
            additional keyword arguments to pass to the
            `get_parameter_dataframe`  method of each result
        """
        dfs = []
        key_size = self._key_size
        # get t0 values if requested
        if t0:
            t0s = self.get_t0s(reference_mass)
        # iterate over results and get DataFrames for each
        # figure out wheter to print a progress bar
        custom_tqdm = utils.get_tqdm(progress)
        n = len(self)
        for i, (key, result) in custom_tqdm(enumerate(self.items()), total=n):
            df = result.get_parameter_dataframe(**kws)
            if key_size == 1:
                df[index_label] = key[0]
            elif split_index:
                for j, k in enumerate(key):
                    df[f'{index_label}_{j}'] = k
            else:
                df[index_label] = [key] * len(df)
            if t0:
                df['t0m' if reference_mass else 't0'] = t0s[i]
            if ndraw is not None:
                dfs.append(df.sample(ndraw, **(draw_kws or {})))
            else:
                dfs.append(df)
        # return combined DataFrame
        return pd.concat(dfs, ignore_index=True)

    def get_mode_parameter_dataframe(self, ndraw: int | None = None,
                                     index_label: str = 'run',
                                     split_index: bool = False,
                                     t0: bool = False,
                                     reference_mass: bool | float |
                                     None = None,
                                     draw_kws: dict | None = None,
                                     **kws) -> pd.DataFrame:
        """Get a combined parameter DataFrame of mode parameter samples for all
        results in the collection, with a new index column identifying
        provenance of each sample. One column per parameter, with a
        mode-indexing column.

        Arguments
        ---------
        ndraw : int
            number of samples to draw from each result (optional)
        index_label : str
            name of provenance column (def., 'run')
        split_index : bool
            split tuple index into multiple columns (def., `False`)
        t0 : bool
            include reference time in DataFrame (def., `False`)
        reference_mass : bool or float
            mass in solar masses used to label time steps if including t0; if
            provided 't0m' will be used instead of 't0' (optional) so that the
            time is unit of mass rather than seconds.
        draw_kws : dict
            keyword arguments to pass to the `sample` method when drawing
            random samples from each result (optional)
        **kws : dict
            additional keyword arguments to pass to the
            `get_mode_parameter_dataframe`
            method of each result
        """
        dfs = []
        key_size = self._key_size
        if t0:
            t0s = self.get_t0s(reference_mass)
        for i, (key, result) in enumerate(self.items()):
            df = result.get_mode_parameter_dataframe(**kws)
            if key_size == 1:
                df[index_label] = key[0]
            elif split_index:
                for j, k in enumerate(key):
                    df[f'{index_label}_{j}'] = k
            else:
                df[index_label] = [key] * len(df)
            if t0:
                df['t0m' if reference_mass else 't0'] = t0s[i]
            if ndraw is not None:
                dfs.append(df.sample(ndraw, **(draw_kws or {})))
            else:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
