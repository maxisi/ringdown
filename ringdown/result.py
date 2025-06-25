"""Module defining the core :class:`Result` class."""

__all__ = ["Result", "ResultCollection", "PPResult"]

import os
import numpy as np
import arviz as az
import scipy.linalg as sl
from arviz.data.base import dict_to_dataset
from . import indexing
from . import data
from .imr import IMRResult
from .target import Target, TargetCollection
from . import utils
from .utils import stats
from .config import WHITENED_LOGLIKE_KEY
import pandas as pd
import json
import configparser
from glob import glob
from parse import parse
import logging
from scipy.stats import gaussian_kde
import xarray as xr
from .labeling import ParameterLabel, get_latex_from_key
import h5py

logger = logging.getLogger(__name__)

_DATAFRAME_PARAMETERS = [
    "m",
    "chi",
    "f",
    "g",
    "a",
    "phi",
    "theta",
    "ellip",
    "df",
    "dg",
]

DEFAULT_COLLECTION_KEY = "run"


class Result(az.InferenceData):
    """Result from a ringdown fit."""

    def __init__(self, *args, config=None, produce_h_det=False, **kwargs):
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
            super().__init__(
                args[0].attrs,
                **{k: getattr(args[0], k) for k in args[0]._groups},
            )
        else:
            super().__init__(*args, **kwargs)
        self._whitened_templates = None
        self._target = None
        self._modes = None
        self._imr_result = None
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
        self._fit = None

    @property
    def has_imr_result(self) -> bool:
        """Check if an IMR result is loaded."""
        return self.imr_result is not None and not self.imr_result.empty

    @property
    def imr_result(self) -> IMRResult:
        """Reference IMR result."""
        if self._imr_result is None:
            logger.info("Looking for IMR result in config.")
            if "imr" in self.config:
                logger.info("IMR section found in config")
                self._imr_result = IMRResult.from_config(self.config)
                logger.info("IMR result loaded.")
            else:
                logger.info("No IMR section found in config.")
                return IMRResult()
            self._imr_result.set_ringdown_reference(self)
        return self._imr_result

    def set_imr_result(self, imr_result: IMRResult) -> None:
        if isinstance(imr_result, IMRResult):
            self._imr_result = imr_result
        else:
            self._imr_result = IMRResult(imr_result)
        self._imr_result.set_ringdown_reference(self)

    @property
    def _df_parameters(self) -> dict[str, ParameterLabel]:
        """Default parameters for DataFrames."""
        df_parameters = {}
        for m in _DATAFRAME_PARAMETERS:
            if m in getattr(self, "posterior", {}):
                df_parameters[m] = ParameterLabel(m)
            elif m.upper() in getattr(self, "posterior", {}):
                df_parameters[m.upper()] = ParameterLabel(m)
        return df_parameters

    @property
    def strain_scale(self) -> float:
        """Scale factor for strain data."""
        s = self.get("constant_data", {}).get("scale", 1.0)
        return float(s)

    @property
    def h_det(self):
        """Alias for `posterior.h_det`, in case this is not already present,
        it gets computed from individual modes."""
        if "h_det" in self.posterior:
            return self.posterior.h_det
        elif "h_det_mode" in self.posterior:
            self.posterior["h_det"] = self.posterior.h_det_mode.sum("mode")
            return self.posterior.h_det
        else:
            return None

    @property
    def h_det_mode(self):
        """Alias for `posterior.h_det_mode`, in case this is not already
        present,  it gets computed from the sum of all modes."""
        if "h_det_mode" in self.posterior:
            return self.posterior.h_det_mode
        else:
            return None

    def rescale_strain(self, scale=None) -> None:
        """Autoscale the strain data in the result."""
        scale = scale or self.strain_scale
        logger.info(f"rescaling strain by {scale}")
        if "h_det" in self.posterior:
            self.posterior["h_det"] = scale * self.posterior["h_det"]
        if "h_det_mode" in self.posterior:
            self.posterior["h_det_mode"] = scale * self.posterior["h_det_mode"]
        if "a" in self.posterior:
            self.posterior["a"] = scale * self.posterior["a"]
        if "a_scale" in self.posterior:
            self.posterior["a_scale"] = scale * self.posterior["a_scale"]
        if "observed_data" in self and "strain" in self.observed_data:
            self.observed_data["strain"] = scale * self.observed_data["strain"]
        if "constant_data" in self:
            if "scale" in self.constant_data:
                self.constant_data["scale"] = (
                    self.constant_data["scale"] / scale
                )
            if "injection" in self.constant_data:
                self.constant_data["injection"] = (
                    scale * self.constant_data["injection"]
                )
            if "cholesky_factor" in self.constant_data:
                self.constant_data["cholesky_factor"] = (
                    scale * self.constant_data["cholesky_factor"]
                )

    @property
    def default_label_format(self) -> dict:
        """Default formatting options for DataFrames."""
        kws = dict(
            label_prograde=any([not m.is_prograde for m in self.modes]),
        )
        kws.update(self._default_label_format)
        return kws

    def update_default_label_format(self, **kws) -> None:
        """Update the default formatting options for DataFrames."""
        self._default_label_format.update(kws)

    @classmethod
    def from_netcdf(cls, *args, load_h_det_mode=True, produce_h_det=False,
                    config=None, **kwargs) -> "Result":
        group_kwargs = kwargs.pop("group_kwargs", {})
        if not load_h_det_mode:
            group_kwargs["posterior"] = {"drop_variables": ["h_det_mode"]}
        data = super().from_netcdf(*args, group_kwargs=group_kwargs, **kwargs)
        # if h_det_mode is not loaded, we cannot produce h_det
        produce_h_det = produce_h_det and load_h_det_mode
        return cls(data, produce_h_det=produce_h_det, config=config)

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
            if "config" in self.attrs:
                config_string = self.attrs["config"]
                raw_config_dict = json.loads(config_string)
                self._config_dict = {
                    kk: {k: utils.try_parse(v) for k, v in vv.items()}
                    for kk, vv in raw_config_dict.items()
                }
            else:
                self._config_dict = {}
        return self._config_dict

    @property
    def info(self) -> dict[str, dict[str, str]]:
        """Alias for `config`."""
        return self.config

    @property
    def _config_object(self) -> configparser.ConfigParser:
        """Configuration file stored as ConfigParser object."""
        config = configparser.ConfigParser()
        # Populate the ConfigParser with data from the dictionary
        for section, settings in self.config.items():
            config.add_section(section)
            for key, value in settings.items():
                config.set(section, key, utils.form_opt(value, key=key))
        return config

    @property
    def target(self) -> Target | None:
        """Target used in the analysis."""
        if self._target is None:
            if "target" in self.config:
                self._target = Target.construct(**self.config["target"])
        return self._target

    @property
    def t0(self) -> float | None:
        """Reference time for the analysis."""
        return getattr(self.target, "t0", None)

    @property
    def epoch(self):
        """Epoch for detector times; corresponds to `fit.start_times`"""
        if "epoch" in self.constant_data:
            return self.constant_data.epoch
        elif "target" in self.config:
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
        shape = (self.posterior.sizes["ifo"], 1)
        return self.constant_data.time + np.array(self.epoch).reshape(shape)

    @property
    def analysis_data(self):
        """Same as observed_strain but in dict format as in Fit"""
        return dict(zip(self.ifos.values, self.observed_strain.values))

    @property
    def start_times(self) -> dict:
        """Same as epoch but in dict format as in Fit"""
        return dict(zip(self.ifos.values, self.epoch.values))

    @property
    def n_analyze(self) -> int:
        """Number of samples in the analysis."""
        return self.constant_data.sizes["time_index"]

    @property
    def a_scale_max(self) -> float:
        """Maximum amplitude scale assumed in the analysis."""
        amax = self.config.get("model", {}).get("a_scale_max")
        if amax is None:
            logger.warning("No maximum amplitude scale found in config")
            amax = self.posterior.a_scale.max().values
        return float(amax)

    def get_fit(self, **kwargs):
        """Get a Fit object from the result."""
        if self._fit is None and self.config:
            from .fit import Fit

            self._fit = Fit.from_config(
                self._config_object, result=self, **kwargs
            )
        return self._fit

    def draw_sample(
        self,
        idx: int | tuple[int, int] | dict = None,
        map: bool = False,
        prng: np.random.Generator = None,
        seed: int = None,
    ) -> tuple[int, dict]:
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
        prng : numpy.random.Generator
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
            if "lp" in self.sample_stats:
                logp = self.sample_stats.lp
            else:
                logp = sum(
                    [
                        v
                        for k, v in self.log_likelihood.items()
                        if k.startswith("logl_")
                    ]
                )
            i = np.argmax(logp.stack(sample=("chain", "draw")).values)
            sample = samples.isel(sample=i)
        else:
            # pick random sample
            prng = prng or np.random.default_rng(seed)
            i = prng.integers(len(samples["sample"]))
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
            self._modes = indexing.ModeIndexList(
                self.posterior.mode.values if "mode" in self.posterior else []
            )
        return self._modes

    @property
    def cholesky_factors(self) -> np.ndarray:
        """Cholesky L factors used in analysis."""
        if "L" in self.constant_data:
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
            wds = np.array(
                [
                    sl.solve_triangular(L, d, lower=True)
                    for L, d in zip(chols, datas)
                ]
            )
        return wds

    @property
    def whitened_data(self) -> np.ndarray:
        """Whitened data used in the analysis."""
        return self.whiten(self.observed_strain)

    @property
    def templates(self) -> data.StrainStack:
        """Templates corresponding to each posterior sample, as were seen by
        the sampler.

        Dimensions will be ``(ifo, time, sample)``.

        Corresponding whitened templates can be obtained from posterior by
        doing::

          result.h_det.stack(sample=('chain', 'draw'))
        """
        return data.StrainStack(self.h_det)

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
            hs = self.h_det.stack(samples=("chain", "draw"))
            self._whitened_templates = self.whiten(hs)
        return self._whitened_templates

    def compute_posterior_snrs(
        self,
        optimal: bool = True,
        network: bool = False,
        cumulative: bool = False,
    ) -> np.ndarray:
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
            ``False``)
        cumulative : bool
            return cumulative SNR, instead of instantaneous SNR (def.,
            ``False``)

        Returns
        -------
        snrs : array
            stacked array of SNRs; if ``cumulative = False``, the shape is
            ``(samples,)``  if ``network = True``, or ``(ifo, samples)``
            otherwise; if ``cumulative = True``, the shape is
            ``(time, samples)`` if ``network = True``, or
            ``(ifo, time, samples)`` otherwise; the number of samples
            equals the number of chains times the number of draws.
        """
        # get whitened reconstructions from posterior (ifo, time, sample)
        whs = self.whitened_templates
        # take the norm across time to get optimal snrs for each (ifo, sample)
        if cumulative:
            opt_ifo_snrs = np.sqrt(np.cumsum(whs * whs, axis=1))
        else:
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
            if cumulative:
                snrs = np.cumsum(wds[..., None] * whs, axis=1) / opt_ifo_snrs
            else:
                snrs = np.sum(wds[..., None] * whs, axis=1) / opt_ifo_snrs
        if network:
            # take norm across detectors
            return np.linalg.norm(snrs, axis=0)
        else:
            return snrs

    @property
    def log_likelihood_timeseries(self):
        """Compute the likelihood timeseries for the posterior samples.

        Returns
        -------
        likelihoods : array
            array of likelihoods, with shape ``(time, samples,)``; the number
            of samples equals the number of chains times the number of draws.
        """
        # get whitened residuals from posterior: (chain, draw, ifo, time)
        whr = self.posterior.whitened_residual
        # compute likelihood timeseries
        return -0.5 * np.sum(np.cumsum(whr * whr, axis=3), axis=2)

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
        if WHITENED_LOGLIKE_KEY not in self.get("log_likelihood", {}):
            self._generate_whitened_residuals()
        return az.waic(self, var_name=WHITENED_LOGLIKE_KEY)

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
        if WHITENED_LOGLIKE_KEY not in self.get("log_likelihood", {}):
            self._generate_whitened_residuals()
        return az.loo(self, var_name=WHITENED_LOGLIKE_KEY)

    def _generate_whitened_residuals(self) -> None:
        """Adduct the whitened residuals to the result."""
        residuals = {}
        residuals_stacked = {}
        ifo_list = list(self.ifos.values.astype(str))
        for ifo in ifo_list:
            r = self.observed_strain[ifo_list.index(ifo)] - self.h_det.sel(
                ifo=ifo
            )
            residuals[ifo] = r.transpose("chain", "draw", "time_index")
            residuals_stacked[ifo] = residuals[ifo].stack(
                sample=["chain", "draw"]
            )
        residuals_whitened = self.whiten(residuals_stacked)
        d = self.posterior.sizes
        residuals_whitened = {
            i: v.reshape((d["time_index"], d["chain"], d["draw"]))
            for i, v in residuals_whitened.items()
        }
        resid = np.stack([residuals_whitened[i] for i in ifo_list], axis=-1)
        keys = ("time_index", "chain", "draw", "ifo")
        self.posterior["whitened_residual"] = (keys, resid)
        keys = ("chain", "draw", "ifo", "time_index")
        self.posterior["whitened_residual"] = (
            self.posterior.whitened_residual.transpose(*keys)
        )
        lnlike = -(self.posterior.whitened_residual**2) / 2
        if hasattr(self, "log_likelihood"):
            self.log_likelihood[WHITENED_LOGLIKE_KEY] = lnlike
        else:
            # We assume that log-likelihood isn't created yet.
            self.add_groups(
                dict(
                    log_likelihood=dict_to_dataset(
                        {WHITENED_LOGLIKE_KEY: lnlike},
                        coords=self.posterior.coords,
                        dims={WHITENED_LOGLIKE_KEY: list(keys)},
                    )
                )
            )

    @property
    def whitened_residuals(self) -> np.ndarray:
        """Whitened residuals from the analysis."""
        if "whitened_residual" not in self.posterior:
            self._generate_whitened_residuals()
        return self.posterior.whitened_residual

    @property
    def ess(self) -> float:
        """Minimum effective sample size for all parameters in the result."""
        # check effective number of samples and rerun if necessary
        ess = az.ess(self)
        mess = ess.min()
        mess_arr = np.array([mess[k].values[()] for k in mess.keys()])
        return np.min(mess_arr)

    @property
    def stacked_samples(self):
        """Stacked samples for all parameters in the result."""
        if "chain" in self.posterior.dims and "draw" in self.posterior.dims:
            return self.posterior.stack(sample=("chain", "draw"))
        else:
            logger.info("No chain or draw dimensions found in posterior.")
            return self.posterior

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
        self._df_parameters.update({p: ParameterLabel(p) for p in pars})

    def get_parameter_key_map(self, modes: bool = True, **kws) -> dict:
        """Get a dictionary of parameter labels for the result."""
        kws["latex"] = True
        if modes:
            x = {}
            for m in self.modes:
                x.update(
                    {
                        p.get_label(mode=m, latex=False): p.get_label(
                            mode=m, **kws
                        )
                        for p in self._df_parameters.values()
                    }
                )
        else:
            x = {k: p.get_label(**kws) for k, p in self._df_parameters.items()}
        return x

    def get_parameter_dataframe(
        self,
        nsamp: int | None = None,
        prng: int | np.random.Generator = None,
        ignore_index=False,
        **kws,
    ) -> pd.DataFrame:
        """Get a DataFrame of parameter samples drawn from the posterior.

        The columns correspond to parameters and the index to the sample,
        which can be used to locate this row in the `Result.stacked_samples`.
        If `ignore_index`, the index will be reset rather than showing the
        location in the original set of samples.

        The parameters are labeled using the `ParameterLabel` class.

        Arguments
        ---------
        nsamp : int
            number of samples to draw from the posterior (optional).
        prng : numpy.random.Generator | int
            random number generator or seed (optional).
        ignore_index : bool
            reset index rather than showing location in original samples
            (def., `False`).
        **kws : dict
            additional keyword arguments to pass to the `get_label` method of
            :class:`ParameterLabel`.
        """
        # set labeling options (e.g., whether to show p index)
        fmt = self.default_label_format.copy()
        fmt.update(kws)
        # get samples
        samples = self.stacked_samples
        if nsamp is not None:
            prng = prng or np.random.default_rng(prng)
            idxs = prng.choice(samples.sizes["sample"], nsamp, replace=False)
            samples = samples.isel(sample=idxs)
        else:
            idxs = None
        df = pd.DataFrame(index=idxs if not ignore_index else None)
        for p, par in self._df_parameters.items():
            x = samples[p]
            if "mode" in x.dims:
                for mode in x.mode.values:
                    key_df = par.get_label(mode=mode, **fmt)
                    df[key_df] = x.sel(mode=mode).values
            else:
                key_df = par.get_label(**fmt)
                df[key_df] = x.values
        return df

    def get_mode_parameter_dataframe(
        self,
        nsamp: int | None = None,
        ignore_index: bool = False,
        prng: int | np.random.Generator | None = None,
        **kws,
    ) -> pd.DataFrame:
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
        prng : numpy.random.Generator | int
            random number generator or seed (optional).
        **kws : dict
            additional keyword arguments to pass to the `get_label` method of
            :class:`ParameterLabel`.

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
            prng = prng or np.random.default_rng(prng)
            idxs = prng.choice(samples.sizes["sample"], nsamp, replace=False)
            samples = samples.isel(sample=idxs)
        else:
            idxs = None
        dfs = []
        for mode, m in zip(self.modes, self.posterior.mode.values):
            df = pd.DataFrame(index=idxs)
            for p, par in self._df_parameters.items():
                if p in samples and "mode" in samples[p].dims:
                    x = samples[p]
                    key_df = par.get_label(mode=None, **fmt)
                    df[key_df] = x.sel(mode=m).values
            df["mode"] = mode.get_label(**fmt)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=ignore_index)

    def get_single_mode_dataframe(
        self, mode: str | tuple | indexing.ModeIndex | bytes, *args, **kws
    ) -> pd.DataFrame:
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
            of :class:`ParameterLabel`.
        """
        df = self.get_mode_parameter_dataframe(*args, **kws)
        return df[df["mode"] == indexing.get_mode_label(mode, **kws)]

    def get_strain_quantile(
        self,
        q: float,
        ifo: str = None,
        mode: str | tuple | indexing.ModeIndex | bytes = None,
    ) -> dict[data.Data] | data.Data:
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
            key = "h_det"
            # make sure h_det exists
            self.h_det
        else:
            mode = indexing.get_mode_coordinate(mode)
            if mode not in self.posterior.mode:
                raise ValueError("Mode requested not in result")
            key = "h_det_mode"
        sel = {k: v for k, v in dict(mode=mode).items() if v is not None}
        x = self.posterior[key].sel(**sel)
        hq = x.quantile(q, dim=("chain", "draw"))
        hdict = {}
        for i in hq.ifo.values.astype(str):
            time = self.sample_times.sel(ifo=i).values
            hdict[i] = data.Data(hq.sel(ifo=i).values, index=time, ifo=i)
        h = hdict if ifo is None else hdict[ifo]
        return h

    def draw_strain_sample(
        self,
        idx: int | None = None,
        map: bool = False,
        ifo: str | None = None,
        mode: str | tuple | indexing.ModeIndex | bytes | None = None,
        prng: int | np.random.Generator | None = None,
        seed: int | None = None,
    ) -> dict[data.Data] | data.Data:
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
            key = "h_det"
            # make sure h_det exists
            self.h_det
        else:
            mode = indexing.get_mode_coordinate(mode)
            if mode not in self.posterior.mode:
                raise ValueError("Mode requested not in result")
            key = "h_det_mode"
        idx, x = self.draw_sample(idx=idx, map=map, prng=prng, seed=seed)
        sel = {k: v for k, v in dict(mode=mode).items() if v is not None}
        h = x[key].sel(**sel)
        info = {k: v.values for k, v in x.items()}
        info["idx"] = idx
        hdict = {}
        for i in h.ifo.values.astype(str):
            time = self.sample_times.sel(ifo=i).values
            hdict[i] = data.Data(
                h.sel(ifo=i).values, ifo=i, index=time, attrs=info
            )
        hdata = hdict if ifo is None else hdict[ifo]
        return hdata

    @property
    def injected_strain(self) -> dict[data.Data] | None:
        """Injections used in the analysis."""
        if "injection" in self.constant_data:
            h = self.constant_data.injection
            hdict = {}
            for i in h.ifo.values.astype(str):
                time = self.sample_times.sel(ifo=i).values
                hdict[i] = data.Data(h.sel(ifo=i).values, ifo=i, index=time)
            return hdict
        else:
            return None

    def resample_to_uniform_amplitude(
        self,
        nsamp: int | None = None,
        prng: int | np.random.Generator | None = None,
    ) -> "Result":
        """Reweight the posterior to a uniform amplitude prior.

        The "primal" posterior has a density :math:`p(a, a_{\\rm scale}) =
        N(a; 0, a_{\\rm scale}) 1/a_{\\rm scale max}`. The target posterior
        we want has a density :math:`p(a, a_{\\rm scale}) \\propto
        \\frac{1}{|a|^{n-1}} \\frac{1}{a_{\\rm scale max}}` for :math:`n`
        quadratures.

        Therefore the importance weighs are

        .. math::
                w = \\frac{1}{|a|^{n-1} N(a; 0, a_{\\rm scale})}

        WARNING: this method can be unstable unless you have an extremely
        large number of samples; we do not recommend it when the model has
        more than 2 quadratures (as in the z-parity symmetric models).

        Arguments
        ---------
        nsamp : int
            number of samples to draw from the posterior (optional, defaults to
            all samples).
        prng : numpy.random.Generator | int
            random number generator or seed (optional).

        Returns
        -------
        new_result : Result
            result with samples reweighted to a uniform amplitude prior;
            posterior will have stacked chains and draws.
        """
        samples = self.stacked_samples
        # get amplitudes and scales
        a = samples.a
        a_scale = samples.a_scale
        n = len([k for k in samples.keys() if k.endswith("_unit")])
        # compute weights
        w = (
            1
            / (
                a ** (n - 1) * np.exp(-0.5 * (a / a_scale) ** 2) / a_scale**n
            ).values
        )
        w = np.prod(w, axis=0)
        # zero out weights for samples with amplitudes above the maximum
        a_max = self.a_scale_max / self.strain_scale
        w[np.any(a > a_max, axis=0)] = 0
        w /= np.sum(w)
        # draw samples
        if nsamp is None:
            nsamp = samples.sizes["sample"]
        if isinstance(prng, int):
            prng = np.random.default_rng(prng)
        elif prng is None:
            prng = np.random.default_rng()
        idxs = prng.choice(samples.sizes["sample"], nsamp, p=w, replace=False)
        # create updated result
        new_result = self.copy()
        new_result.posterior = samples.isel(sample=idxs)
        return new_result

    def imr_consistency(
        self,
        coords: str = "mchi",
        ndraw_rd: int | None = None,
        ndraw_imr: int | None = 1000,
        prng: int | np.random.Generator | None = None,
        kde_kws: dict | None = None,
    ) -> np.ndarray:
        """Computes credible levels (CLs) at which each IMR sample is found
        relative to the ringdown posterior, returning a distribution of CLs.

        The comparison is done in coordinates specified by the ``coords``
        argument (NOTE: only 'mchi' currently supported).

        Arguments
        ---------
        coords : str
            coordinates to use for comparison (def., 'mchi'), currently only
            'mchi' is supported.
        ndraw_rd : int
            number of RD samples to draw (def., all samples)
        ndraw_imr : int
            number of IMR samples to draw (def., 1000)
        prng : numpy.random.Generator
            random number generator or seed (optional)
        kde_kws : dict
            additional keyword arguments to pass to `gaussian_kde`

        Returns
        -------
        qs : array
            distribution of CLs at which each IMR sample is found relative to
            the ringdown posterior.
        """
        if coords.lower() != "mchi":
            raise NotImplementedError("Only mchi coordinates are supported.")
        if not self.has_imr_result:
            raise ValueError("No IMR result loaded.")
        if "m" not in self.posterior or "chi" not in self.posterior:
            raise ValueError("No mass or chi parameters found in posterior.")

        # get random subset of M-chi samples from ringdown analysis
        samples = self.stacked_samples
        prng = prng or np.random.default_rng(prng)
        n = min(ndraw_rd or len(samples["sample"]), len(samples["sample"]))
        idxs = prng.choice(samples.sizes["sample"], n, replace=False)
        samples = samples.isel(sample=idxs)
        xy_rd = samples[["m", "chi"]].to_array().values

        # get random subset of M-chi samples from IMR analysis
        n = min(ndraw_imr or len(self.imr_result), len(self.imr_result))
        idxs = prng.choice(len(self.imr_result), n, replace=False)
        xy_imr = [
            self.imr_result.final_mass[idxs],
            self.imr_result.final_spin[idxs],
        ]

        # compute support of RD distribution for each RD and IMR sample
        kde = gaussian_kde(xy_rd, **(kde_kws or {}))
        p_rd = kde(xy_rd)
        p_imr = kde(xy_imr)

        # compute CL of RD distribution at each IMR sample
        qs = []
        for p in p_imr:
            q = np.sum(p_rd > p) / len(p_rd)
            qs.append(q)

        # return distribution of CLs
        return np.array(qs)

    def imr_consistency_summary(
        self,
        coords: str = "mchi",
        imr_weight: str = "rd",
        ndraw_rd: int | None = None,
        ndraw_imr: int | None = 1000,
        imr_cl: float = 0.9,
        prng: int | np.random.Generator | None = None,
        kde_kws: dict | None = None,
    ) -> float:
        """Compute ringdown credible level that fully encompasses certain IMR
        credible level specified, or that encompasses a certain fraction of
        IMR samples.

        If `imr_weight` is 'imr', the output is the smallest credible level of
        the RD posterior that encompasses the entirety of the IMR credible
        level specified by `imr_cl`.

        If `imr_weight` is 'rd', the output is the ringdown credible level
        that encompasses `imr_cl` of the IMR samples.

        Arguments
        ---------
        coords : str
            coordinates to use for comparison (def., 'mchi')
        imr_weight : str
            distribution to use to define the IMR credible level, 'imr'
            or 'rd' (def., 'rd')
        ndraw_rd : int
            number of RD samples to draw (def., all samples)
        ndraw_imr : int
            number of IMR samples to draw (def., 1000)
        imr_cl : float
            IMR credible level (def., 0.9)
        prng : numpy.random.Generator
            random number generator or seed (optional)
        kde_kws : dict
            additional keyword arguments to pass to `gaussian_kde`

        Returns
        -------
        q : float
            credible of the RD posterior that fully encompasses the IMR
            credible level specified by `imr_cl`.
        """
        if coords.lower() != "mchi":
            raise NotImplementedError("Only mchi coordinates are supported.")
        if not self.has_imr_result:
            raise ValueError("No IMR result loaded.")
        if "m" not in self.posterior or "chi" not in self.posterior:
            raise ValueError("No mass or chi parameters found in posterior.")

        if imr_weight not in ("imr", "rd"):
            raise ValueError("kind must be 'imr' or 'rd'.")

        prng = prng or np.random.default_rng(prng)

        # select a random subset of IMR samples
        n = min(ndraw_imr or len(self.imr_result), len(self.imr_result))
        idxs = prng.choice(len(self.imr_result), n, replace=False)
        imr_samples = np.array(
            [self.imr_result.final_mass[idxs], self.imr_result.final_spin[idxs]]
        )

        if imr_weight == "imr":
            # further subselect IMR samples to thosw within the IMR credible
            # level specified by `imr_cl`
            kde_imr = utils.Bounded_2d_kde(imr_samples.T, **(kde_kws or {}))
            imr_idxs = np.argsort(kde_imr(imr_samples.T))[::-1][
                : int(imr_cl * n)
            ]
            imr_samples = imr_samples[:, imr_idxs]

        # select a random subset of RD samples and KDE them
        samples = self.stacked_samples
        n = min(ndraw_rd or len(samples["sample"]), len(samples["sample"]))
        idxs = prng.choice(samples.sizes["sample"], n, replace=False)
        samples = samples.isel(sample=idxs)
        xy_rd = samples[["m", "chi"]].to_array().values
        kde_rd = utils.Bounded_2d_kde(xy_rd.T, **(kde_kws or {}))

        # evaluate RD KDE on IMR samples
        p_imr = kde_rd(imr_samples.T)

        # set the thrshold CL based on the RD posterior
        if imr_weight == "imr":
            # find the IMR sample inside the `imr_cl` IMR region that
            # has the lowest amount of RD posterior probability and record
            # that value as a threshold
            rd_kde_thresh = np.min(p_imr)
        else:
            # rank all IMR samples based on the RD posterior and take
            # the `1-imr_cl`-th quantile, thus identifying the value of the
            # RD posterior that encompasses `imr_cl` of the IMR samples
            rd_kde_thresh = np.quantile(p_imr, 1 - imr_cl)

        # compute the fraction of RD samples above the IMR threshold
        p_rd = kde_rd(xy_rd.T)
        q = np.sum(p_rd > rd_kde_thresh) / n
        return q

    def amplitude_significance(self, kind="quantile") -> pd.Series:
        """Compute a measure of support for non-vanishing mode amplitudes.

        If `kind` is 'quantile', the output is the quantile of the amplitude
        distribution at zero, computed using the HPD.

        If `kind` is 'zscore', translates the quantile into a z-score using
        the standard normal distribution.

        Arguments
        ---------
        kind : str
            method to compute significance, 'quantile' or 'zscore' (def.,
            'quantile')

        Returns
        -------
        p : pd.Series
            p-value of the amplitude of the signal.
        """
        amps = self.stacked_samples["a"]
        qs = {}
        for a in amps:
            label = indexing.get_mode_label(a.mode.values)
            q = stats.quantile_at_value(a)
            if kind == "quantile":
                qs[label] = q
            elif kind == "zscore":
                qs[label] = stats.z_score(q)
            else:
                raise ValueError("kind must be 'quantile' or 'zscore'.")
        return pd.Series(qs)

    def get_marginal_quantiles(
        self, reference_values: dict | None = None
    ) -> xr.Dataset:
        """Compute the marginal quantiles of the injection parameters.

        Arguments
        ---------
        truth : list[str] | None
            list of parameters to compute quantiles for (def., all parameters).

        Returns
        -------
        quantiles : xr.Dataset
            Dataset of marginal quantiles.
        """
        samples = self.posterior
        d = ("chain", "draw")
        qs = {}
        for k, v in reference_values.items():
            if v is None or k not in samples:
                continue
            if np.isscalar(v):
                # compute quantile for scalar parameter
                # counting number of samples below the reference value
                qs[k] = (samples[k] <= v).mean(dim=d)
            else:
                # vector parameter
                qs[k] = (samples[k] <= np.array(v)[None, None, :]).mean(dim=d)
        return xr.Dataset(data_vars=qs)

    @property
    def injection_marginal_quantiles(self) -> xr.Dataset:
        """Compute the marginal quantiles of the injection parameters."""
        if not self.config.get("injection", None):
            return xr.Dataset()
        return self.get_marginal_quantiles(self.config["injection"])

    def get_injection_marginal_quantiles_series(self, **kws) -> pd.Series:
        """Compute the marginal quantiles of the injection parameters."""
        # set labeling options (e.g., whether to show p index)
        fmt = self.default_label_format.copy()
        fmt.update(kws)
        # get quantile Dataset
        qs = self.injection_marginal_quantiles
        # generate labeled dictionary of quantiles
        qdict = {}
        for k, q in qs.items():
            if k in self._df_parameters:
                par = self._df_parameters[k]
                if "mode" in q.dims:
                    for mode in q.mode.values:
                        key_df = par.get_label(mode=mode, **fmt)
                        qdict[key_df] = q.sel(mode=mode).values
                else:
                    key_df = par.get_label(**fmt)
                    qdict[key_df] = q.values
        return pd.Series(qdict, dtype=float)

    def get_injection_parameters(
        self,
        include_opt_snr: bool = False,
        include_mf_snr: bool = False,
        latex: bool = False,
        **kws,
    ) -> pd.Series:
        """Get injection parameters as a pandas Series.

        Arguments
        ---------
        include_opt_snr : bool
            include optimal SNR (def., `False`).
        include_mf_snr : bool
            include matched-filter SNR (def., `False`).
        latex : bool
            use LaTeX formatting for the labels (def., `False`).

        Returns
        -------
        params : pd.Series
            injection parameters as a pandas Series.
        """
        # set labeling options (e.g., whether to format as LaTeX)
        fmt = self.default_label_format.copy()
        fmt.update(kws)
        fmt["latex"] = latex
        # get injection parameters
        inj = self.config.get("injection", {})
        qdict = {}
        for k, v in inj.items():
            if k in self._df_parameters:
                par = self._df_parameters[k]
                if np.isscalar(v):
                    key = par.get_label(**fmt)
                    qdict[key] = v
                else:
                    for mode, val in zip(self.modes, v):
                        key = par.get_label(mode=mode, **fmt)
                        qdict[key] = val
        # add SNRs if requested
        if include_opt_snr:
            qdict["snr_opt"] = self.compute_injected_snrs(optimal=True)
        if include_mf_snr:
            qdict["snr_mf"] = self.compute_injected_snrs(optimal=False)
        return pd.Series(qdict, dtype=float)

    @property
    def injection(self) -> data.StrainStack:
        """Injection waveforms as StrainStack."""
        if "injection" in self.constant_data:
            return data.StrainStack(self.constant_data.injection)
        else:
            return None

    @property
    def whitened_injection(self) -> data.StrainStack:
        """Whiten the injection waveforms."""
        if self.injection is None:
            return None
        return self.injection.whiten(self.cholesky_factors.values)

    def compute_injected_snrs(
        self, optimal: bool = True, network: bool = True
    ) -> np.ndarray | float:
        """Compute the injected SNRs for the result.

        Arguments
        ---------
        optimal : bool
            compute optimal SNR (def., `True`).
        network : bool
            compute network SNR (def., `True`).

        Returns
        -------
        snr : np.ndarray | float
            injected SNRs; if `network` is `True`, returns the quadrature sum
            of the SNRs and the output will be a float; if `network` is `False`,
            returns an array of SNRs for each detector.
        """
        if self.injection is None:
            return None
        data = None if optimal else self.observed_strain
        snr = self.injection.compute_snr(self.cholesky_factors, data)
        if network:
            return float(np.linalg.norm(snr, axis=0))
        else:
            return snr

    # ------------------------------------------------------------------------
    # PLOTS

    def plot_trace(
        self,
        var_names: list[str] = ["a"],
        compact: bool = True,
        *args,
        **kwargs,
    ):
        """Alias for :func:`arviz.plot_trace`."""
        return az.plot_trace(
            self, compact=compact, var_names=var_names, *args, **kwargs
        )

    def plot_mass_spin(
        self,
        ndraw: int = 500,
        imr: bool = True,
        joint_kws: dict | None = None,
        marginal_kws: dict | None = None,
        imr_kws: dict | None = None,
        df_kws: dict | None = None,
        prng: int | np.random.Generator | None = None,
        palette=None,
        dropna: bool = False,
        height: float = 6,
        ratio: float = 5,
        space: float = 0.2,
        xlim: tuple | None = None,
        ylim: tuple | None = (0, 1),
        marginal_ticks: bool = False,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = 0,
        y_max: float | None = 1,
        engine: str = "auto",
        **kws,
    ) -> None:
        """Plot the mass-spin distribution for the collection.
        Based on seaborn's jointplot but with the ability to use a truncated
        KDE (1D and 2D), controlled by the `x_min`, `x_max`, `y_min`, and
        `y_max` arguments.

        Arguments
        ---------
        ndraw : int
            number of samples to draw from the posterior (optional).
        imr : bool
            plot IMR samples (def., `True`).
        joint_kws : dict
            keyword arguments to pass to the `kdeplot` method for the joint
            distribution (optional).
        marginal_kws : dict
            keyword arguments to pass to the `kdeplot` method for the marginal
            distributions (optional).
        imr_kws : dict
            keyword arguments plot IMR result, accepts: `color`, `linewidth`
            and `linestyle`.
        df_kws : dict
            keyword arguments to pass to the `get_parameter_dataframe` method
            (optional).
        prng : numpy.random.Generator | int
            random number generator or seed (optional).
        palette : str
            color palette for hue variable (optional).
        dropna : bool
            drop NaN values from DataFrame (def., `False`).
        height : float
            height of the plot (def., 6).
        ratio : float
            aspect ratio of the plot (def., 5).
        space : float
            space between axes (def., 0.2).
        xlim : tuple
            x-axis limits (optional).
        ylim : tuple
            y-axis limits (def., (0, 1)).
        marginal_ticks : bool
            show ticks on marginal plots (def., `False`).
        x_min : float
            minimum mass value for KDE truncation (optional).
        x_max : float
            maximum mass value for KDE truncation (optional).
        y_min : float
            minimum spin value for KDE truncation (optional).
        y_max : float
            maximum spin value for KDE truncation (optional).
        **kws : dict
            additional keyword arguments to pass to the joint plot.

        Returns
        -------
        grid : seaborn.JointGrid
            joint plot object.
        df_rd : pandas.DataFrame
            DataFrame of parameter samples drawn from the posterior
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        imr_kws = {} if imr_kws is None else imr_kws.copy()
        df_kws = {} if df_kws is None else df_kws.copy()

        # get data
        df_rd = self.get_parameter_dataframe(ndraw=ndraw, prng=prng, **df_kws)

        if "m" not in df_rd.columns or "chi" not in df_rd.columns:
            raise ValueError("Mass and spin columns not found in results.")

        if engine == "auto":
            if any(k is not None for k in [x_min, x_max, y_min, y_max]):
                engine = "bounded"
            else:
                engine = "seaborn"
        elif engine not in ["seaborn", "bounded"]:
            raise ValueError(
                "Invalid engine, choose from: 'auto', 'seaborn', 'bounded'."
            )

        if engine == "seaborn":
            # simply call the seaborn jointplot method
            if "levels" in kws:
                # NOTE: our bounded kdeplot and sns.kdeplot have different
                # definitions of levels!
                kws["levels"] = [1 - c for c in kws["levels"]]
            grid = sns.jointplot(
                data=df_rd,
                x="m",
                y="chi",
                palette=palette,
                dropna=dropna,
                height=height,
                ratio=ratio,
                space=space,
                xlim=xlim,
                ylim=ylim,
                marginal_ticks=marginal_ticks,
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                **kws,
            )
        else:
            color = "C0"

            # parse arguments
            kws.update(
                {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
            )
            joint_kws = {} if joint_kws is None else joint_kws.copy()
            joint_kws.update(kws)
            marginal_kws = {} if marginal_kws is None else marginal_kws.copy()
            for k in ["x_min", "x_max", "y_min", "y_max"]:
                if k in kws and k not in marginal_kws:
                    marginal_kws[k] = kws[k]

            # Initialize the JointGrid object (based on sns.jointplot)
            grid = sns.JointGrid(
                data=df_rd,
                x="m",
                y="chi",
                palette=palette,
                dropna=dropna,
                height=height,
                ratio=ratio,
                space=space,
                xlim=xlim,
                ylim=ylim,
                marginal_ticks=marginal_ticks,
            )

            # if grid.hue is not None:
            #     marginal_kws.setdefault("legend", False)

            joint_kws.setdefault("color", color)
            grid.plot_joint(utils.kdeplot, **joint_kws)

            marginal_kws.setdefault("color", color)
            if "fill" in joint_kws:
                marginal_kws.setdefault("fill", joint_kws["fill"])

            grid.plot_marginals(utils.kdeplot, **marginal_kws)

        # plot IMR result
        if imr and self.has_imr_result:
            n_imr = len(self.imr_result)
            if ndraw > n_imr:
                logger.warning(
                    f"Using fewer IMR samples ({n_imr}) than "
                    f"requested ({ndraw})."
                )
                ndraw = n_imr
            df_imr = pd.DataFrame(
                {
                    "m": self.imr_result.final_mass,
                    "chi": self.imr_result.final_spin,
                }
            ).sample(ndraw, replace=False, random_state=prng)

            levels = imr_kws.pop("levels", None)
            if levels is None:
                if "levels" in kws:
                    # NOTE: our bounded kdeplot and sns.kdeplot have different
                    # definitions of levels!
                    levels = [1 - c for c in kws["levels"]]
                else:
                    # default to 90% CL
                    levels = [
                        0.1,
                    ]
            imr_kwargs = dict(fill=False, color="k", linestyle="--")
            imr_kwargs.update(imr_kws)
            sns.kdeplot(
                data=df_imr,
                x="m",
                y="chi",
                levels=levels,
                ax=grid.ax_joint,
                **imr_kwargs,
            )

            imr_q = imr_kws.pop("quantile", 0.90)
            if imr_q is not None and imr_q > 0:
                hi, lo = (1 - imr_q) / 2, 1 - (1 - imr_q) / 2
                cis = df_imr.quantile([hi, 0.5, lo])
            else:
                hi, lo = None, None

            # plot IMR median
            lkws = dict(color="k", linestyle=":")
            for k in ["color", "linestyle", "linewidth"]:
                if k in imr_kws:
                    lkws[k] = imr_kws[k]
            grid.ax_joint.axvline(cis["m"][0.5], **lkws)
            grid.ax_joint.axhline(cis["chi"][0.5], **lkws)

            grid.ax_marg_x.axvline(cis["m"][0.5], **lkws)
            grid.ax_marg_y.axhline(cis["chi"][0.5], **lkws)

            # plor IMR CLs
            if hi is not None:
                bkws = dict(alpha=0.1, color=lkws.get("color", "k"))
                grid.ax_marg_x.axvspan(cis["m"][lo], cis["m"][hi], **bkws)
                grid.ax_marg_y.axhspan(cis["chi"][lo], cis["chi"][hi], **bkws)

        # Make the main axes active in the matplotlib state machine
        plt.sca(grid.ax_joint)
        return grid, df_rd


class ResultCollection(utils.MultiIndexCollection):
    """Collection of results from ringdown fits."""

    def __init__(
        self,
        results: list | None = None,
        index: list | None = None,
        reference_mass: float | None = None,
        reference_time: float | None = None,
        info: dict | None = None,
    ) -> None:
        _results = []
        for r in results:
            if isinstance(r, Result):
                _results.append(r)
            else:
                _results.append(Result(r))
        super().__init__(_results, index, reference_mass, reference_time, info)
        self._targets = None
        self._imr_result = None
        self.collection_key = DEFAULT_COLLECTION_KEY

    @property
    def has_imr_result(self) -> bool:
        """Check if the collection has an IMR result."""
        return self.imr_result is not None and not self.imr_result.empty

    @property
    def imr_result(self) -> IMRResult:
        """Reference IMR result"""
        if self._imr_result is None:
            logger.info("Looking for IMR result in first collection item")
            return self.results[0].imr_result
        return self._imr_result

    def set_imr_result(self, imr_result: IMRResult,
                       inherit: bool = True) -> None:
        """Set the reference IMR result for the collection.

        Arguments
        ---------
        imr_result : IMRResult
            IMR result to set as reference.
        """
        old_imr_result = self.imr_result
        if old_imr_result is not None and not old_imr_result.empty:
            logger.warning("Overwriting existing IMR result.")
        self._imr_result = imr_result
        if inherit:
            for r in self.results:
                r.set_imr_result(imr_result)

    @property
    def results(self) -> list[Result]:
        """List of results in the collection."""
        return self.data

    @property
    def targets(self) -> TargetCollection:
        """Targets associated with the results in the collection."""
        if self._targets is None:
            self._targets = TargetCollection([r.target for r in self.results])
        elif len(self._targets) != len(self.results):
            logger.warning(
                "Number of targets does not match results.Recomputing targets"
            )
            self._targets = TargetCollection([r.target for r in self.results])
        return self._targets

    @property
    def reference_mass(self) -> float | None:
        """Reference mass in solar masses used for time-step labeling."""
        if self.targets.reference_mass is None:
            if self._reference_mass is None:
                logger.info(
                    "No reference mass specified; trying to infer "
                    "from result configurations."
                )
                m0 = None
                for r in self.results:
                    m0 = r.config.get("pipe", {}).get(self.targets._mref_key)
                    break
                self._reference_mass = m0
            self.targets.set_reference_mass(self._reference_mass)
        return self.targets.reference_mass

    @property
    def reference_time(self) -> float | None:
        """Reference time used for time-step labeling."""
        if self.targets.reference_time is None:
            if self._reference_time is None:
                logger.info(
                    "No reference time specified; trying to infer "
                    "from result configurations."
                )
                t0 = None
                for r in self.results:
                    t0 = r.config.get("pipe", {}).get(self.targets._tref_key)
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

    def get_t0s(
        self,
        reference_mass: float | bool | None = None,
        reference_time: float | None = None,
        decimals: int | None = None,
    ) -> np.ndarray:
        """Get analysis start times for the collection.

        Arguments
        ---------
        reference_mass : float or bool
            reference mass to use for time labeling; if `True`, use the
            reference mass of the targets; if `False`, do not use a reference
            mass and return time as recorded (def., `None`)
        reference_time : float
            reference time to use for time labeling (def., `0`)
        decimals : int
            number of decimal places to round the times to (optional)

        Returns
        -------
        t0s : np.ndarray
            array of analysis start times.
        """
        if reference_mass:
            targets = self.targets
            if not isinstance(reference_mass, bool):
                targets.set_reference_mass(reference_mass)
                # reference mass is true, but no value specified so assume
                # default
            elif not targets.reference_mass:
                targets.set_reference_mass(self.reference_mass)
            if not targets.reference_time:
                targets.set_reference_time(self.reference_time)
            t0s = targets.t0m
        else:
            if reference_time:
                if isinstance(reference_time, bool):
                    reference_time = self.reference_time
            else:
                reference_time = 0
            t0s = [result.t0 - reference_time for result in self.results]
        if decimals is not None:
            t0s = np.round(t0s, decimals)
        return np.array(t0s)

    def reindex_by_t0(
        self,
        reference_mass: bool | float | None = None,
        reference_time: float | None = None,
        decimals: int | None = None,
    ) -> None:
        """Reindex the collection by the analysis start time."""
        t0s = self.get_t0s(reference_mass, reference_time, decimals)
        if np.any(t0s is None):
            raise ValueError("Cannot reindex by t0 if any t0 values are None.")
        self.reindex(t0s)

    @classmethod
    def from_netcdf(
        cls,
        path_input: str | list,
        index: list = None,
        config: str | list | None = None,
        load_h_det_mode: bool = True,
        produce_h_det: bool = False,
        progress: bool = True,
        **kws,
    ):
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
            show progress bar (def., `True`)
        **kws : dict
            additional keyword arguments to pass to the constructor, like
            reference_mass or reference_time
        """
        paths = []
        indxs = []
        cpaths = []
        if index is not None:
            index = [tuple(np.atleast_1d(idx)) for idx in index]
        if isinstance(path_input, str):
            all_paths = sorted(glob(path_input))
            logger.info(f"loading {len(all_paths)} results from {path_input}")
            for path in all_paths:
                pattern = parse(path_input.replace("*", "{}"), path).fixed
                idx = tuple([utils.try_parse(k) for k in pattern])
                if index is None or idx in index:
                    indxs.append(idx)
                    paths.append(path)
                else:
                    logger.debug(f"skipping {idx} because it is not in index")
                if isinstance(config, str):
                    cpath = config.replace("*", "{}").format(*pattern)
                    if os.path.exists(cpath):
                        cpaths.append(cpath)
        else:
            paths = path_input
        if len(paths) == 0:
            raise ValueError("No results found.")
        if index is not None and len(index) != len(paths):
            for idx in index:
                if idx not in indxs:
                    logger.warning(f"index {idx} not found")
        if config is not None:
            if len(cpaths) != len(paths):
                raise ValueError(
                    "Number of configuration files does not "
                    "match number of result files."
                )
        else:
            cpaths = [None] * len(paths)
        results = []
        tqdm = utils.get_tqdm(progress)
        for path, cpath in tqdm(
            zip(paths, cpaths), total=len(paths), desc="results"
        ):
            results.append(Result.from_netcdf(path, config=cpath,
                                              load_h_det_mode=load_h_det_mode,
                                              produce_h_det=produce_h_det))
        info = kws.get("info", {})
        info["provenance"] = paths
        return cls(results, indxs, info=info, **kws)

    def to_netcdf(self, paths: str | None = None, **kws) -> None:
        """Save the collection of results to NetCDF files.

        Arguments
        ---------
        paths : str
            template path to NetCDF file or list of paths; if a string,
            expected to be a template like `path/to/many/files/*.nc` or
            `path/to/many/files/{}.nc` where `{}` or `*` is replaced by the
            index value.
        **kws : dict
            additional keyword arguments to pass to the `to_netcdf` method of
            each result
        """
        index = self.index
        if isinstance(paths, str):
            paths = [paths.replace("*", "{}").format(idx) for idx in index]
        elif len(paths) != len(index):
            raise ValueError("Number of paths does not match results.")
        for path, result in zip(paths, self.results):
            dirname = os.path.abspath(os.path.dirname(path))
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            result.to_netcdf(path, **kws)

    def get_parameter_dataframe(
        self,
        ndraw: int | None = None,
        index_label: str = None,
        split_index: bool = False,
        t0: bool = False,
        reference_mass: bool | float | None = None,
        draw_kws: dict | None = None,
        progress: bool = False,
        **kws,
    ) -> pd.DataFrame:
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
        index_label = index_label or self.collection_key
        # get t0 values if requested
        if t0:
            t0s = self.get_t0s(reference_mass)
        # iterate over results and get DataFrames for each
        # figure out wheter to print a progress bar
        tqdm = utils.get_tqdm(progress)
        n = len(self)
        dkws = {"random_state": kws.get("prng", None)}
        dkws.update(draw_kws or {})
        for i, (key, result) in tqdm(
            enumerate(self.items()), total=n, desc="results"
        ):
            df = result.get_parameter_dataframe(**kws)
            if key_size == 1:
                df[index_label] = key[0]
            elif split_index:
                for j, k in enumerate(key):
                    df[f"{index_label}_{j}"] = k
            else:
                df[index_label] = [key] * len(df)
            if t0:
                df["t0m" if reference_mass else "t0"] = t0s[i]
            if ndraw is not None:
                dfs.append(df.sample(ndraw, **dkws))
            else:
                dfs.append(df)
        # return combined DataFrame
        return pd.concat(dfs, ignore_index=True)

    def get_mode_parameter_dataframe(
        self,
        ndraw: int | None = None,
        index_label: str = None,
        split_index: bool = False,
        t0: bool = False,
        reference_mass: bool | float | None = None,
        draw_kws: dict | None = None,
        **kws,
    ) -> pd.DataFrame:
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
        index_label = index_label or self.collection_key
        dkws = {"random_state": kws.get("prng", None)}
        dkws.update(draw_kws or {})
        if t0:
            t0s = self.get_t0s(reference_mass)
        for i, (key, result) in enumerate(self.items()):
            df = result.get_mode_parameter_dataframe(**kws)
            if key_size == 1:
                df[index_label] = key[0]
            elif split_index:
                for j, k in enumerate(key):
                    df[f"{index_label}_{j}"] = k
            else:
                df[index_label] = [key] * len(df)
            if t0:
                df["t0m" if reference_mass else "t0"] = t0s[i]
            if ndraw is not None:
                dfs.append(df.sample(ndraw, **dkws))
            else:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def get_injection_parameters_dataframe(self, **kws) -> pd.DataFrame:
        """Get injection parameters as a pandas DataFrame."""
        qs = {i: r.get_injection_parameters(**kws) for i, r in self.items()}
        return pd.DataFrame(qs).T

    def get_injection_marginal_quantiles_dataframe(self, **kws) -> pd.DataFrame:
        """Compute the marginal quantiles of the injection parameters."""
        qs = {
            i: r.get_injection_marginal_quantiles_series(**kws)
            for i, r in self.items()
        }
        return pd.DataFrame(qs).T

    def compute_injected_snrs(
        self,
        optimal: bool = True,
        network: bool = True,
        progress: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """Get the injected SNRs for the result."""
        tqdm = utils.get_tqdm(progress)
        snrs = {
            i: r.compute_injected_snrs(optimal=optimal, network=network)
            for i, r in tqdm(self.items(), desc="results", total=len(self))
        }
        if network:
            return pd.Series(snrs)
        else:
            return pd.DataFrame(snrs).T

    def imr_consistency(self, *args, progress=False, **kwargs) -> pd.DataFrame:
        """Compute the IMR consistency for the collection.
        See :meth:`Result.imr_consistency` for details.
        """
        tqdm = utils.get_tqdm(progress)
        q = {
            k: r.imr_consistency(*args, **kwargs)
            for k, r in tqdm(self, desc="results")
        }
        return pd.DataFrame(q)

    def imr_consistency_summary(
        self, *args, progress: bool = False, simplify_index: bool = True, **kws
    ) -> pd.Series:
        """Compute the IMR consistency summary for each element in the
        collection.

        See :meth:`Result.imr_consistency_summary` for details.

        Arguments
        ---------
        progress : bool
            show progress bar (def., `False`)
        simplify_index : bool
            simplify the index to a single column (def., `True`)
        **kws : dict
            additional keyword arguments to pass to the `imr
            consistency_summary` method of each result

        Returns
        -------
        q : pd.Series
            summary of IMR consistency for each result in the collection
        """
        tqdm = utils.get_tqdm(progress)
        q = [
            r.imr_consistency_summary(*args, **kws)
            for r in tqdm(self.results, desc="results")
        ]
        if simplify_index:
            index = self.simplified_index
        else:
            index = self.index
        return pd.Series(q, index=index)

    def amplitude_significance(
        self, simplified_index: bool = True, **kws
    ) -> pd.DataFrame:
        """Compute the significance for non-vanishing mode amplitudes for each
        result in the collection.

        See :meth:`Result.amplitude_significance` for details.

        Arguments
        ---------
        **kws : dict
            additional keyword arguments to pass to the
            `amplitude_significance` method of each result

        Returns
        -------
        p : pd.DataFrame
            DataFrame of amplitude significance for each result.
        """
        if simplified_index:
            index = self.simplified_index
        else:
            index = self.index
        return pd.DataFrame(
            {
                k: r.amplitude_significance(**kws)
                for k, r in zip(index, self.data)
            }
        ).T

    @property
    def simplified_index(self) -> list:
        """Simplified index for the collection, with unit-lenght tuples
        converted to standalone items."""
        if len(self) > 0 and len(self.index[0]) == 1:
            index = [k[0] for k in self.index]
        else:
            index = self.index
        return index

    def compute_posterior_snrs(self, **kws) -> np.ndarray:
        """Compute the posterior SNRs for each result in the collection.
        See :meth:`Result.compute_posterior_snrs` for details.

        Returns an array of shape (n_collection, n_ifo, n_samples) if
        network is False, and (n_collection, n_samples) if network is True.
        """
        return np.stack([r.compute_posterior_snrs(**kws) for r in self.data])

    def compute_imr_snrs(self, **kws) -> np.ndarray:
        """Compute the IMR SNRs for each result in the collection.
        See :meth:`IMRResult.compute_ringdown_snrs` for details.

        Returns an array of shape (n_collection, n_ifo, n_samples) if
        network is False, and (n_collection, n_samples) if network is True.
        """
        return np.stack(
            [r.imr_result.compute_ringdown_snrs(**kws) for r in self.data]
        )

    def compute_imr_snrs_by_t0(
        self,
        optimal: bool = True,
        network: bool = False,
        cumulative: bool = False,
        approximate: bool = True,
        progress: bool = False,
        **kws,
    ) -> np.ndarray:
        """ """
        snrs = []
        t0s = self.get_t0s()
        tqdm = utils.get_tqdm(progress)
        if approximate:
            # get earliest fit as reference
            reference_result = self.results[np.argmin(self.get_t0s())]
            wfs = reference_result.imr_result.get_waveforms(
                ringdown_slice=True, progress=False, **kws
            )
            times = reference_result.sample_times
            data = reference_result.observed_strain
            for _, r in tqdm(sorted(zip(t0s, self.results)), desc="results"):
                # get indices for start_times in times
                delta_times = (times - r.epoch).values
                i0s = np.argmin(np.abs(delta_times), axis=1)
                # slice waveforms
                n = min(times.shape[-1] - i0s)
                wfs_sliced = wfs.slice(i0s, n)
                # compute SNRs based on sliced waveforms
                chol = r.cholesky_factors[:, :n, :n]
                if optimal:
                    d = None
                else:
                    d = [dd[i0: i0 + n] for i0, dd in zip(i0s, data)]
                snr = wfs_sliced.compute_snr(
                    chol, data=d, network=network, cumulative=cumulative
                )
                snrs.append(snr)
        else:
            # directly compute SNRs by recreating a fit for each result
            # (this takes longer than the above, but is more representative
            # of the actual rigdown analysis)
            for _, r in tqdm(sorted(zip(t0s, self.results)), desc="results"):
                snr = r.imr_result.compute_ringdown_snrs(
                    progress=False,
                    network=network,
                    cumulative=cumulative,
                    optimal=optimal,
                    **kws,
                )
                snrs.append(snr)
        return np.stack(snrs)

    # -----------------------------------------------------------------------
    # PLOTS

    def to_pp_result(self, prior: Result | None = None) -> "PPResult":
        """Convert the ResultCollection to a PPResult.

        Arguments
        ---------
        prior : Result | None
            prior result to use for the PP plot. If None, the prior will be
            inferred from the data.

        Returns
        -------
        pp_result : PPResult
            PPResult object.
        """
        return PPResult.from_results_collection(self, prior)

    def plot_mass_spin(
        self,
        ndraw: int = 500,
        imr: bool = True,
        joint_kws: dict | None = None,
        marginal_kws: dict | None = None,
        imr_kws: dict | None = None,
        df_kws: dict | None = None,
        prng: int | np.random.Generator | None = None,
        index_label: str = None,
        hue: str = None,
        hue_order: list | None = None,
        palette=None,
        hue_norm=None,
        dropna: bool = False,
        height: float = 6,
        ratio: float = 5,
        space: float = 0.2,
        xlim: tuple | None = None,
        ylim: tuple | None = (0, 1),
        marginal_ticks: bool = False,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = 0,
        y_max: float | None = 1,
        **kws,
    ) -> None:
        """Plot the mass-spin distribution for the collection.
        Based on seaborn's jointplot but with the ability to use a truncated
        KDE (1D and 2D), controlled by the `x_min`, `x_max`, `y_min`, and
        `y_max` arguments.

        Arguments
        ---------
        ndraw : int
            number of samples to draw from the posterior (optional).
        imr : bool
            plot IMR samples (def., `True`).
        joint_kws : dict
            keyword arguments to pass to the `kdeplot` method for the joint
            distribution (optional).
        marginal_kws : dict
            keyword arguments to pass to the `kdeplot` method for the marginal
            distributions (optional).
        imr_kws : dict
            keyword arguments plot IMR result, accepts: `color`, `linewidth`
            and `linestyle`.
        df_kws : dict
            keyword arguments to pass to the `get_parameter_dataframe` method
            (optional).
        prng : numpy.random.Generator | int
            random number generator or seed (optional).
        index_label : str
            label for the index column in the DataFrame (optional).
        hue : str
            alias for index_label (optional).
        hue_order : list
            order of hue variable (optional).
        palette : str
            color palette for hue variable (optional).
        hue_norm : tuple
            normalization tuple for hue variable (optional).
        dropna : bool
            drop NaN values from DataFrame (def., `False`).
        height : float
            height of the plot (def., 6).
        ratio : float
            aspect ratio of the plot (def., 5).
        space : float
            space between axes (def., 0.2).
        xlim : tuple
            x-axis limits (optional).
        ylim : tuple
            y-axis limits (def., (0, 1)).
        marginal_ticks : bool
            show ticks on marginal plots (def., `False`).
        x_min : float
            minimum mass value for KDE truncation (optional).
        x_max : float
            maximum mass value for KDE truncation (optional).
        y_min : float
            minimum spin value for KDE truncation (optional).
        y_max : float
            maximum spin value for KDE truncation (optional).
        **kws : dict
            additional keyword arguments to pass to the joint plot.

        Returns
        -------
        grid : seaborn.JointGrid
            joint plot object.
        df_rd : pandas.DataFrame
            DataFrame of parameter samples drawn from the posterior
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # parse arguments
        kws.update(
            {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        )
        joint_kws = {} if joint_kws is None else joint_kws.copy()
        joint_kws.setdefault("palette", palette)
        joint_kws.update(kws)
        marginal_kws = {} if marginal_kws is None else marginal_kws.copy()
        for k in ["x_min", "x_max", "y_min", "y_max"]:
            if k in kws and k not in marginal_kws:
                marginal_kws[k] = kws[k]
        imr_kws = {} if imr_kws is None else imr_kws.copy()
        df_kws = {} if df_kws is None else df_kws.copy()

        color = "C0"
        index_label = index_label or hue or self.collection_key

        # get data
        df_rd = self.get_parameter_dataframe(
            ndraw=ndraw, prng=prng, index_label=index_label, **df_kws
        )

        if "m" not in df_rd.columns or "chi" not in df_rd.columns:
            raise ValueError("Mass and spin columns not found in results.")

        # Initialize the JointGrid object (based on sns.jointplot)
        grid = sns.JointGrid(
            data=df_rd,
            x="m",
            y="chi",
            hue=index_label,
            palette=palette,
            hue_norm=hue_norm,
            hue_order=hue_order,
            dropna=dropna,
            height=height,
            ratio=ratio,
            space=space,
            xlim=xlim,
            ylim=ylim,
            marginal_ticks=marginal_ticks,
        )

        if grid.hue is not None:
            marginal_kws.setdefault("legend", False)

        joint_kws.setdefault("color", color)
        grid.plot_joint(utils.kdeplot, **joint_kws)

        marginal_kws.setdefault("color", color)
        marginal_kws.setdefault("palette", joint_kws["palette"])
        if "fill" in joint_kws:
            marginal_kws.setdefault("fill", joint_kws["fill"])

        grid.plot_marginals(utils.kdeplot, **marginal_kws)

        # plot IMR result
        if imr and self.has_imr_result:
            n_imr = len(self.imr_result)
            if ndraw > n_imr:
                logger.warning(
                    f"Using fewer IMR samples ({n_imr}) than "
                    f"requested ({ndraw})."
                )
                ndraw = n_imr
            df_imr = pd.DataFrame(
                {
                    "m": self.imr_result.final_mass,
                    "chi": self.imr_result.final_spin,
                }
            ).sample(ndraw, replace=False, random_state=prng)

            levels = imr_kws.pop("levels", None)
            if levels is None:
                if "levels" in kws:
                    # NOTE: our bounded kdeplot and sns.kdeplot have different
                    # definitions of levels!
                    levels = [1 - c for c in kws["levels"]]
                else:
                    # default to 90% CL
                    levels = [
                        0.1,
                    ]
            imr_kwargs = dict(fill=False, color="k", linestyle="--")
            imr_kwargs.update(imr_kws)
            sns.kdeplot(
                data=df_imr,
                x="m",
                y="chi",
                levels=levels,
                ax=grid.ax_joint,
                **imr_kwargs,
            )

            imr_q = imr_kws.pop("quantile", 0.90)
            if imr_q is not None and imr_q > 0:
                hi, lo = (1 - imr_q) / 2, 1 - (1 - imr_q) / 2
                cis = df_imr.quantile([hi, 0.5, lo])
            else:
                hi, lo = None, None

            # plot IMR median
            lkws = dict(color="k", linestyle=":")
            for k in ["color", "linestyle", "linewidth"]:
                if k in imr_kws:
                    lkws[k] = imr_kws[k]
            grid.ax_joint.axvline(cis["m"][0.5], **lkws)
            grid.ax_joint.axhline(cis["chi"][0.5], **lkws)

            grid.ax_marg_x.axvline(cis["m"][0.5], **lkws)
            grid.ax_marg_y.axhline(cis["chi"][0.5], **lkws)

            # plor IMR CLs
            if hi is not None:
                bkws = dict(alpha=0.1, color=lkws.get("color", "k"))
                grid.ax_marg_x.axvspan(cis["m"][lo], cis["m"][hi], **bkws)
                grid.ax_marg_y.axhspan(cis["chi"][lo], cis["chi"][hi], **bkws)

        # Make the main axes active in the matplotlib state machine
        plt.sca(grid.ax_joint)
        return grid, df_rd


class PPResult(object):
    """PP results container with P–P plotting utilities."""

    _truth_group = "truths"
    _quantile_group = "quantiles"

    def __init__(
        self,
        quantiles: pd.DataFrame,
        truth: pd.DataFrame,
        prior: Result | None = None,
        rundir: str | None = None,
        info: dict | None = None,
    ):
        self.quantiles = quantiles
        self.truths = truth
        self.prior = prior
        self.rundir = rundir or ""
        self._info = info
        self._config = None
        self._null_cum_hists = {}
        self._null_bands = {}

    def __len__(self):
        return len(self.quantiles)

    def __str__(self):
        return f"PPResult('{self.rundir}', N={len(self)})"

    def __repr__(self):
        return str(self)

    @property
    def config(self) -> configparser.ConfigParser | None:
        if self._config is None:
            # attempt to read in config from rundir
            config_path = os.path.join(self.rundir, "config.ini")
            if os.path.exists(config_path):
                self._config = utils.load_config(config_path)
        return self._config

    @property
    def info(self) -> dict:
        if not self._info and self.config is not None:
            # Convert ConfigParser to dict
            self._info = {
                s: dict(self.config.items(s)) for s in self.config.sections()
            }
        return self._info

    @classmethod
    def from_results_collection(
        cls, results: ResultCollection, prior: Result | None = None
    ):
        """Construct a PPResult from a ResultCollection."""
        quantiles = results.get_injection_marginal_quantiles_dataframe()
        truth = results.get_injection_parameters_dataframe(
            include_mf_snr=True, include_opt_snr=True
        )
        if "provenance" in results.info:
            if isinstance(results.info["provenance"], str):
                path = results.info["provenance"]
            else:
                # assume list of paths
                path = results.info["provenance"][0]
            if "engine" in path:
                rundir = path.split("engine")[0]
            else:
                rundir = path
        else:
            rundir = ""
        return cls(quantiles, truth, prior=prior, rundir=rundir)

    def to_hdf5(self, path: str | None = None) -> None:
        """Save the PPResult to an HDF5 file.

        The file is saved to path under different groups: "quantiles" and
        "truths". The run directory is saved as an attribute of the file.

        Arguments
        ---------
        path : str
            path to the HDF5 file
        """
        if path is None:
            path = os.path.join(self.rundir, "pp_result.h5")
        self.quantiles.to_hdf(path, key=self._quantile_group, mode="w")
        self.truths.to_hdf(path, key=self._truth_group, mode="a")
        with h5py.File(path, "a") as f:
            f.attrs["rundir"] = self.rundir
            # JSON-encode the dict so it can be stored as a HDF5 attribute
            f.attrs["config"] = json.dumps(self.info)
        logger.info(f"Saved PP results: {path}")

    @classmethod
    def from_hdf5(cls, path: str) -> "PPResult":
        """Read a PPResult from an HDF5 file."""
        # Read DataFrames directly from the HDF5 file path
        quantiles = pd.read_hdf(path, key=cls._quantile_group)
        truth = pd.read_hdf(path, key=cls._truth_group)
        # Read rundir and config attributes
        with h5py.File(path, "r") as f:
            rundir = f.attrs.get("rundir", "")
            config_attr = f.attrs.get("config", "{}")
        # JSON-decode the config attribute
        if isinstance(config_attr, (bytes, bytearray)):
            config_str = config_attr.decode("utf-8")
        else:
            config_str = config_attr
        info = json.loads(config_str)
        return cls(quantiles, truth, rundir=rundir, info=info)

    @staticmethod
    def _null_var(x, n):
        """Returns the theoretical variance for the ECDF of n samples drawn
        from a uniform distribution between 0 and 1.

        The true CDF for a U[0, 1] distribution is F(x) = x, which means that
        if ECDF_n(x) is the ECDF for n samples, then n * ECDF_n(x) is a
        drawn from a Binomial distribution with n trials and probability
        p = x. The variance of a Binomial distribution is
        var(X) = n * p * (1 - p), so the variance of the ECDF is
        var(F(x)) = x * (1 - x) / n.

        Arguments
        ---------
        x : float
            the value at which to evaluate the variance, 0 <= x <= 1
        n : int
            the number of samples drawn from the uniform distribution

        Returns
        -------
        var : float
            the variance of the difference of the ECDFs
        """
        return x * (1 - x) / n

    def plot(
        self,
        keys: list[str] | None = None,
        nmax: int | None = None,
        nbins: int = 50,
        nsamp: int = 200,
        nhist: int = 10000,
        ax: None = None,
        bands: tuple[float, ...] = (3, 2, 1),
        difference: bool = True,
        latex: bool = False,
        palette: str | list[str] | None = None,
        legend: bool = True,
        legend_kws: dict | None = None,
        line_kws: dict | None = None,
    ):
        """P–P plot of injection marginal quantiles.

        Arguments
        ---------
        keys : list
            list of parameters to plot (optional).
        nmax : int
            maximum number of results to plot (optional).
        nbins : int
            number of bins in the histogram (def., `50`).
        nsamp : int
            number of samples to draw from the null distribution
            (def., `200`).
        nhist : int
            number of histograms to draw from the null distribution
            (def., `10000`).
        ax : matplotlib.axes.Axes
            matplotlib axes object (optional).
        bands : tuple
            tuple of sigmas to plot for the null distribution
            (def., `(3, 2, 1)`).
        difference : bool
            whether to plot the difference between the empirical and null
            distributions (def., `True`).
        legend : bool
            whether to plot the legend (def., `True`).
        legend_kws : dict
            keyword arguments for the legend (optional).
        line_kws : dict
            keyword arguments for the PP lines (optional).

        Returns
        -------
        ax : matplotlib.axes.Axes
            matplotlib axes object.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.quantiles is None or self.quantiles.empty:
            raise ValueError("no results loaded!")
        # get quantile DataFrame for selected parameters, dropping NaNs
        qdf = self.quantiles[keys] if keys is not None else self.quantiles
        qdf = qdf.dropna()
        if len(qdf) < len(self):
            logger.warning(f"Dropped {len(self) - len(qdf)} rows with NaNs")
        if latex:
            qdf = qdf.copy()
            qdf.rename(columns=get_latex_from_key, inplace=True)
        N = len(qdf) if nmax is None else min(nmax, len(qdf))
        # initialize figure
        if ax is None:
            _, ax = plt.subplots()
        # plot null distribution
        ks = np.linspace(0, 1, nbins + 1)
        m = np.zeros_like(ks[1:]) if difference else ks[1:]
        for sigma in bands:
            half_band = sigma * np.sqrt(self._null_var(ks[1:], N))
            ax.fill_between(ks[:-1], m + half_band, m - half_band, step="post",
                            color="gray", alpha=0.15)
        ax.step(ks[:-1], m, c="k", where="post")
        # plot results
        colors = sns.color_palette(palette, n_colors=len(qdf.columns))
        m = ks[1:] if difference else 0
        for k, c in zip(qdf.columns, colors):
            y, _ = np.histogram(qdf[k].iloc[:N], bins=ks)
            ax.step(ks[:-1], np.cumsum(y) / N - m, label=k, c=c, where="post",
                    **(line_kws or {}))
        ncol = 2 if len(qdf.columns) > 16 else 1
        if legend:
            lkws = dict(bbox_to_anchor=(1.05, 1), loc="upper left",
                        frameon=False, ncol=ncol)
            lkws.update(legend_kws or {})
            ax.legend(**lkws)
        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$p-p$" if difference else r"$p$")
        ax.set_title(f"$N = {N}$")
        return ax
