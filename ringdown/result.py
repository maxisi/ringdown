"""Module defining the core :class:`Result` class."""

__all__ = ["Result", "ResultCollection"]

import os
import numpy as np
import arviz as az
import scipy.linalg as sl
from arviz.data.base import dict_to_dataset
from . import qnms
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
    def _df_parameters(self) -> dict[str, qnms.ParameterLabel]:
        """Default parameters for DataFrames."""
        df_parameters = {}
        for m in _DATAFRAME_PARAMETERS:
            if m in getattr(self, "posterior", {}):
                df_parameters[m] = qnms.ParameterLabel(m)
            elif m.upper() in getattr(self, "posterior", {}):
                df_parameters[m.upper()] = qnms.ParameterLabel(m)
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
    def from_netcdf(cls, *args, config=None, **kwargs) -> "Result":
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
        if self.config:
            from .fit import Fit

            return Fit.from_config(self._config_object, result=self, **kwargs)

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
        self._df_parameters.update({p: qnms.ParameterLabel(p) for p in pars})

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

        The parameters are labeled using the `qnms.ParameterLabel` class.

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
            :class:`qnms.ParameterLabel`.
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
            of :class:`qnms.ParameterLabel`.
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

        The “primal” posterior has a density :math:`p(a, a_{\\rm scale}) =
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
    
    def get_generic_amplitude(self):

        if 'cosi' not in self.posterior.keys(): 
            raise KeyError('Must be result of fit using aligned model')
        else:
            A_j = [] # collect the computed generic amplitudes per mode, to be stacked at the end
                
            for mode in self.modes:
                ### C is the "amplitude" returned by the aligned model, with angular factors still factored out
                C = self.posterior.a.sel(mode=bytes('1,-2,{},{},{}'.format(mode.l, mode.m, mode.n), 'utf-8')).values
                cosi = self.posterior.cosi.values
                swsh = utils.swsh.construct_sYlm(-2, mode.l, mode.m)
                ylm_p = swsh(cosi)
                ylm_m = swsh(-cosi)
                A = C * (np.abs(ylm_p) + np.abs(ylm_m))
                A_j.append(A)
            
            return np.stack(A_j, axis=-1)

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
    ) -> None:
        _results = []
        for r in results:
            if isinstance(r, Result):
                _results.append(r)
            else:
                _results.append(Result(r))
        super().__init__(_results, index, reference_mass, reference_time)
        self._targets = None
        self._imr_result = None
        self.collection_key = DEFAULT_COLLECTION_KEY

    def __repr__(self):
        return f"ResultCollection({self.index})"

    def thin(self, n: int, start_loc: int = 0) -> "ResultCollection":
        """Thin the collection by taking every `n`th result.

        Arguments
        ---------
        n : int
            number of results to skip between each result.
        start_loc : int
            starting location in the collection to thin from (def., 0).

        Returns
        -------
        new_collection : ResultCollection
            thinned collection.
        """
        results = self.results[start_loc::n]
        index = self.index[start_loc::n]
        return ResultCollection(
            results=results,
            index=index,
            reference_mass=self.reference_mass,
            reference_time=self.reference_time,
        )

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

    def set_imr_result(self, imr_result: IMRResult) -> None:
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
        index = index or []
        cpaths = []
        if isinstance(path_input, str):
            paths = sorted(glob(path_input))
            logger.info(f"loading {len(paths)} results from {path_input}")
            for path in paths:
                pattern = parse(path_input.replace("*", "{}"), path).fixed
                idx = tuple([utils.try_parse(k) for k in pattern])
                index.append(idx)
                if isinstance(config, str):
                    cpath = config.replace("*", "{}").format(*pattern)
                    if os.path.exists(cpath):
                        cpaths.append(cpath)
        else:
            paths = path_input
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
            results.append(Result.from_netcdf(path, config=cpath))
        info = kws.get("info", {})
        info["provenance"] = paths
        return cls(results, index, **kws)

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
                    d = [dd[i0 : i0 + n] for i0, dd in zip(i0s, data)]
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
