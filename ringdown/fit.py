"""Module defining the core :class:`Fit` class.
"""

__all__ = ['Fit']

import numpy as np
import arviz as az
import json
from ast import literal_eval
import configparser
import copy as cp
import os
import warnings
import inspect
import jax
import numpyro.infer
import jaxlib.xla_extension
import xarray as xr
import lal
import logging
from .data import *
from . import utils
from .target import Target
from .result import Result
from .model import make_model, get_arviz, MODEL_DIMENSIONS
from . import indexing
from . import waveforms

# TODO: support different samplers?
KERNEL = numpyro.infer.NUTS
SAMPLER = numpyro.infer.MCMC

KERNEL_ARGS = inspect.signature(KERNEL).parameters.keys()
SAMPLER_ARGS = inspect.signature(SAMPLER).parameters.keys()
# for safety check that there are no overlapping keys
for k in SAMPLER_ARGS:
    if k in KERNEL_ARGS:
        logging.warning(f"overlapping keys in {KERNEL} and {SAMPLER}: {k}")

MODEL_ARGS = inspect.signature(make_model).parameters.keys()
RUNTIME_MODEL_ARGS = ['modes', 'prior', 'predictive', 'store_h_det', 'store_h_det_mode']

DEF_RUN_KWS = dict(dense_mass=True, num_warmup=1000, num_samples=1000,
                    num_chains=4)

class Fit(object):
    """ A ringdown fit. Contains all the information required to setup and run
    a ringdown inference analysis, as well as to manipulate the result.

    Example usage::

        import ringdown as rd
        fit = rd.Fit(modes=[(1,-2,2,2,0), (1,-2,2,2,1)])
        fit.load_data('{i}-{i}1_GWOSC_16KHZ_R1-1126259447-32.hdf5', ifos=['H1', 'L1'], kind='gwosc')
        fit.set_target(1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82, duration=0.05)
        fit.condition_data(ds=8)
        fit.update_model(A_scale=1e-21, M_min=50, M_max=150)
        fit.run()

    Attributes
    ----------
    data : dict
        dictionary containing data, indexed by detector name.
    acfs : dict
        dictionary containing autocovariance functions corresponding to data,
        if already computed.
    start_times : dict
        target truncation time for each detector.
    antenna_patterns : dict
        dictionary of tuples (Fp, Fc) with plus and cross antenna patterns
        for each detector (only applicable depending on model).
    target : Target
        information about truncation time at geocenter and, if applicable,
        source right ascension, declination and polarization angle.
    result : Result, arviz.data.inference_data.InferenceData
        if model has been run, arviz object containing fit result
    prior : Result, arviz.data.inference_data.InferenceData
        if model prior has been run, arviz object containing prior
    modes : list
        if applicable, list of (p, s, l, m, n) tuples identifying modes to be
        fit (else, None).
    n_modes : int
        number of modes to be fit.
    ifos : list
        list of detector names.
    t0 : float
        target geocenter start time.
    sky : tuple
        tuple with source right ascension, declination and polarization angle.
    analysis_data : dict
        dictionary of truncated analysis data that will be fed to Stan model.
    model_data : dict
        arguments passed to Stan model internally.
    info : dict
        information that can be used to reproduce a fit (e.g., data provenance,
        or conditioning options), stored as dictionary of dictionaries whose
        outer (inner) keys will be interpreted as sections (options) when
        creating a configuration file through :meth:`Fit.to_config`.
    """

    def __init__(self, modes=None, **kws):
        self.info = {}
        self.data = {}
        self.injections = {}
        self.acfs = {}
        self.target = None
        self.result = None
        self.prior = None
        self._n_analyze = None
        self._raw_data = None
        # set modes dynamically
        self.modes = None
        self.set_modes(modes)
        # assume rest of kwargs are to be passed to make_model
        self._model_settings = {}
        self.update_model(**kws)
        
    def __repr__(self):
        return f"Fit({self.modes}, ifos={self.ifos})"
    
    def copy(self):
        """Produce a deep copy of this `Fit` object.

        Returns
        -------
        fit_copy : Fit
            deep copy of `Fit`.
        """
        return cp.deepcopy(self)
    
    @property
    def has_target(self) -> bool:
        """Whether an analysis target has been set with
        :meth:`Fit.set_target`.
        """
        return self.target is not None
    
    @property
    def has_injections(self) -> bool:
        """Whether injections have been added with :meth:`Fit.inject`.
        """
        return bool(self.injections)
    
    @property
    def start_times(self):
        if self.has_target:
            start_times = self.target.get_detector_times_dict(self.ifos)
        else:
            start_times = {}
        return start_times
    
    @property
    def antenna_patterns(self):
        if self.has_target:
            aps = self.target.get_antenna_patterns_dict(self.ifos)
        else:
            aps = {}
        return aps

    @property
    def n_modes(self) -> int:
        """ Number of damped sinusoids to be included in template.
        """
        return len(self.modes)

    @property
    def ifos(self) -> list:
        """ Instruments to be analyzed.
        """
        return list(self.data.keys())

    @property
    def t0(self) -> float:
        """ Target truncation time (defined at geocenter if model accepts
        multiple detectors).
        """
        return self.target.t0 if self.has_target else None

    @property
    def analysis_data(self) -> dict:
        """Slice of data to be analyzed for each detector. Extracted from
        :attr:`Fit.data` based on information in analysis target
        :attr:`Fit.target`.
        """
        data = {}
        i0s = self.start_indices
        for i, d in self.data.items():
            data[i] = d.iloc[i0s[i]:i0s[i] + self.n_analyze]
        return data
    
    @property
    def analysis_injections(self) -> dict:
        """Slice of injection to be analyzed for each detector. Extracted from
        :attr:`Fit.conditioned_injections` based on information in analysis
        target :attr:`Fit.target`.
        """
        data = {}
        i0s = self.start_indices
        for i, d in self.conditioned_injections.items():
            data[i] = d.iloc[i0s[i]:i0s[i] + self.n_analyze]
        return data
    
    @property
    def analysis_times(self) -> dict:
        """Time arrays for analysis data.
        """
        return {i: d.time.values for i,d in self.analysis_data.items()}

    @property
    def raw_data(self) -> dict:
        if self._raw_data is None:
            return self.data
        else:
            return self._raw_data
        
    @property
    def model_settings(self) -> dict:
        """Prior options as currently set.
        """
        return self._model_settings

    @property
    def valid_model_settings(self) -> list:
        """Valid prior parameters for the selected model. These can be set
        through :meth:`Fit.update_model`.
        """
        return [k for k in list(MODEL_ARGS) if k not in RUNTIME_MODEL_ARGS]

    def update_model(self, **kws):
        """Set or modify prior options or other model settings.  For example,
        ``fit.update_model(A_scale=1e-21)`` sets the `A_scale` parameter to
        `1e-21`.

        Valid arguments for the selected model can be found in
        :attr:`Fit.valid_model_settings`.
        """
        if self.result is not None:
            logging.warning("updating prior of Fit with preexisting results!")
        if self.prior is not None:
            logging.warning("updating prior of Fit with preexisting prior results!")

        # check whether the option is valid, regardless of case
        for k, v in kws.items():
            if k in self.valid_model_settings:
                self._model_settings[k] = v
            elif k.lower() in self.valid_model_settings:
                self._model_settings[k] = v
            elif k.upper() in self.valid_model_settings:
                self._model_settings[k] = v
            else:
                logging.warning(f"unknown model argument: {k}")
    
    def update_prior(self, *args, **kwargs):
        warnings.warn("update_prior is deprecated, use update_model instead")
        self.update_model(*args, **kwargs)

    @property
    def run_input(self) -> dict:
        """Arguments to be passed to model function at runtime:
        [times, strains, ls, fp, fc].
        """
        if not self.acfs:
            logging.warning("computing ACFs with default settings")
            self.compute_acfs()

        data_dict = self.analysis_data

        fpfc = [self.antenna_patterns[i] for i in self.ifos]
        fp = [x[0] for x in fpfc]
        fc = [x[1] for x in fpfc]

        times = [np.array(d.time) - self.start_times[i] 
                 for i,d in data_dict.items()]

        # arguments to be passed to function returned by model_function
        # make sure this agrees with that function call!
        # [times, strains, ls, fp, fc]
        input = [
            times,
            [s.values for s in data_dict.values()],
            [a.iloc[:self.n_analyze].cholesky for a in self.acfs.values()],
            fp,
            fc
        ]
        return input

    @classmethod
    def from_config(cls, config_input : str | configparser.ConfigParser,
                    no_cond: bool = False, result : str | None = None):
        """Creates a :class:`Fit` instance from a configuration file.
        
        Has the ability to load and condition data, as well as to inject a
        simulated signal and to compute or load ACFs. Does not run the fit
        automatically.

        Arguments
        ---------
        config_input : str, configparser.ConfigParser
            path to config file on disk, or preloaded
            :class:`configparser.ConfigParser`
        no_cond : bool
            option to ignore conditioning

        Returns
        -------
        fit : Fit
            Ringdown :class:`Fit` object.
        """
        config = utils.load_config(config_input)
                    
        # parse model options
        model_opts = {k: utils.try_parse(v) for k,v in config['model'].items()}
        if 'name' in model_opts:
            warnings.warn("model name is deprecated, use explicit mode options instead")
            logging.info("trying to guess mode configuration based on model name")
            name = model_opts.pop('name')
            if 'aligned' in name:
                raise NotImplementedError("aligned model not yet supported")
            model_opts['marginalized'] = 'marginal' in name

        if config.has_section('prior'):
            prior = {k: utils.try_parse(v) for k,v in config['prior'].items()}
            model_opts.update(prior)

        # look for some legacy options and replace them with current arguments
        # accepted by make_model
        legacy = {
            'a_scale_max': ['A_scale', 'a_scale', 'A_SCALE'],
            'flat_amplitude_prior': ['flat_a', 'flat_A', 'FLAT_A'],
            'g_min': ['gamma_min'],
            'g_max': ['gamma_max']
        }
        for new, old in legacy.items():
            for k in old:
                if k in model_opts:
                    warnings.warn(f"replacing deprecated option {k} with {new}")
                    model_opts[new] = model_opts.pop(k)
        
        if 'perturb_f' in model_opts:
            warnings.warn("perturb_f is deprecated, just use df_min/max instead")
            perturb_f = model_opts.pop('perturb_f')
            for k in ['df_min', 'df_max']:
                if k in model_opts:
                    model_opts[k] *= perturb_f
        
        if 'perturb_tau' in model_opts:
            warnings.warn("perturb_tau is deprecated, just use dg_min/max instead")
            perturb_tau = model_opts.pop('perturb_tau')
            if 'dtau_max' in model_opts:
                model_opts['dg_min'] = - model_opts.pop('dtau_max') * perturb_tau
            if 'dtau_min' in model_opts:
                model_opts['dg_max'] = - model_opts.pop('dtau_min') * perturb_tau
                
        if 'order_fs' in model_opts:
            warnings.warn("order_fs is deprecated, use `mode_ordering = 'f'` instead")
            if bool(model_opts.pop('order_fs')):
                model_opts['mode_ordering'] = 'f'
            
        if 'order_gammas' in model_opts:
            warnings.warn("order_gammas is deprecated, use `mode_ordering = 'g'` instead")
            if bool(model_opts.pop('order_gammas')):
                model_opts['mode_ordering'] = 'g'
            
        # create fit object
        fit = cls(**model_opts)
        
        if 'data' not in config:
            # the rest of the options require loading data, so if no pointer to
            # data was provided, just exit
            return fit
        
        # load data
        ifo_input = config.get('data', 'ifos', fallback='')
        try:
            ifos = literal_eval(ifo_input)
        except (ValueError,SyntaxError):
            ifos = [i.strip() for i in ifo_input.split(',')]
        path_input = config['data']['path']
        
        # TODO: add ability to generate synthetic data here
        # NOTE: not popping in order to preserve original ConfigParser
        kws = {k: utils.try_parse(v) for k,v in config['data'].items()
                  if k not in ['ifos', 'path']}
        fit.load_data(path_input, ifos, **kws)

        # add target
        fit.set_target(**{k: utils.try_parse(v) for k,v in config['target'].items()})
        
        # inject signal if requested
        if config.has_section('injection'):
            inj_kws = {k: utils.try_parse(v) for k,v in config['injection'].items()}
            if 'path' in inj_kws:
                # attempt to read injection parameters from JSON file
                injpath = os.path.abspath(inj_kws.pop('path'))
                try:
                    with open(injpath, 'r') as f:
                        json_kws = json.load(f)
                    # check if there's an overlap between JSON and INI
                    overlap = set(json_kws.keys()).intersection(inj_kws.keys())
                    if overlap:
                        logging.warn("overwriting injection file options "
                                     f"with config: {overlap}")
                    # merge injection settings from JSON and INI
                    # NOTE: config file overwrites JSON!
                    json_kws.update(inj_kws)
                    inj_kws = json_kws
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise IOError(f"unable to read JSON file: {injpath}")
            no_noise = inj_kws.get('no_noise', False)
            post_cond = inj_kws.get('post_cond', False)
            if no_noise:
                # create injection but do not add it to data quite yet, in case
                # we need to estimate ACFs from data first
                fit.injections = fit.get_templates(**inj_kws)
                fit.update_info('injection', **inj_kws)
            elif not post_cond:
                # unless we have to wait after conditioning (post_cond) inject
                # signal into data now
                fit.inject(**inj_kws)
        else:
            # no injection requested, so set some dummy defaults
            no_noise = False
            post_cond = False
        
        # condition data if requested
        if config.has_section('condition') and not no_cond:
            cond_kws = {k: utils.try_parse(v) for k,v in config['condition'].items()}
            fit.condition_data(**cond_kws)
        
        # load or produce ACFs
        if config.get('acf', 'path', fallback=False):
            kws = {k: utils.try_parse(v) for k,v in config['acf'].items()
                   if k not in ['path']}
            fit.load_acfs(config['acf']['path'], **kws)
        else:
            acf_kws = {} if 'acf' not in config else config['acf']
            fit.compute_acfs(**{k: utils.try_parse(v) for k,v in acf_kws.items()})
        
        if no_noise:
            # no-noise injection, so replace data by simulated signal
            if post_cond:
                # post_cond means the injection must not be conditioned, but it
                # should be evaluated on the decimated time array (if
                # applicable); as a hack, just zero out the data and call
                # fit.inject(), thus adding the injection to a bunch of zeros
                # while guaranteeing that the injection gets produced on the
                # right time array
                fit.data = {i: 0*v for i,v in fit.data.items()}
                fit.inject(**inj_kws)
            else:
                # the injection must be conditioned, so replace the data with
                # the injection and condition it explicitly
                fit.data = fit.injections
                if config.has_section('condition') and not post_cond:
                    fit.condition_data(preserve_acfs=True, **cond_kws)
        elif post_cond:
            # now that we are done conditioning, inject the requested signal 
            fit.inject(**inj_kws)
        
        if result:
            if isinstance(result, az.InferenceData):
                fit.result = Result(result)
            elif isinstance(result, str) and os.path.exists(result):
                logging.warning("loading result from disk with "
                                "no guarantee of fit correspondence!")
                try:
                    if result.endswith('.nc'):
                        fit.result = Result(az.from_netcdf(result))
                    elif result.endswith('.json'):
                        fit.result = Result(az.from_json(result))
                    else:
                        logging.error(f"unknown result format: {result}")
                except Exception as e:
                    logging.error(f"unable to read result from {result}: {e}")
            else:
                logging.error(f"result file {result} not found")
        return fit

    def to_config(self, path=None):
        """Create configuration file to reproduce this fit by calling
        :meth:`Fit.from_config`.

        .. note::
            This will only result in a working configuration file if all 
            data provenance information is available in :attr:`Fit.info`.
            This field is automatically populated if the :meth:`Fit.load_data`
            method is used to add data to fit.

        Arguments
        ---------
        path : str
            optional destination path for configuration file.

        Returns
        -------
        config : configparser.ConfigParser
            configuration file object.
        """
        config = configparser.ConfigParser()
        # model options
        config['model'] = {}
        config['model']['modes'] = str(self.modes)
        # prior options
        config['model'].update({k: utils.form_opt(v) for k,v 
                                in self.model_settings.items()})
        # rest of options require data, so exit of none were added
        if not self.ifos:
            return config
        # data, injection, conditioning and acf options
        for sec, opts in self.info.items():
            config[sec] = {k: utils.form_opt(v) for k,v in opts.items()}
        config['target'] = {k: str(v) for k,v in self.info['target'].items()}
        # write file to disk if requested
        if path is not None:
            with open(path, 'w') as f:
                config.write(f)
        return config

    def update_info(self, section: str, **kws) -> None:
        """Update fit information stored in :attr:`Fit.info`, e.g., data
        provenance or injection properties. If creating a config file through
        :meth:`Fit.to_config`, `section` will operate as a name for a section
        with options determined by the keyword arguments passed here.

        Keyword arguments are stored as "options" for the given section,
        e.g.,::

            fit.update_info('data', path='path/to/{ifo}-data.h5')

        adds an entry to ``fit.info`` like::
        
            {'data': {'path': 'path/to/{ifo}-data.h5')}}

        Arguments
        ---------
        section : str
            name of information category, e.g., `data`, `injection`,
            `condition`. 
        """
        self.info[section] = self.info.get(section, {})
        self.info[section].update(**kws)

    def condition_data(self, preserve_acfs : bool = False, **kwargs):
        """Condition data for all detectors by calling
        :meth:`ringdown.data.Data.condition`. Docstring for that function
        below.

        The `preserve_acfs` argument determines whether to preserve original
        ACFs in fit after conditioning (default False).

        """
        # record all arguments
        settings = {k: v for k,v in locals().items() if k != 'self'}
        for k, v in settings.pop('kwargs').items():
            settings[k] = v
            
        new_data = {}
        for k, d in self.data.items():
            t0 = self.start_times[k]
            new_data[k] = d.condition(t0=t0, **kwargs)
        self._raw_data = self.data
        self.data = new_data
        if not preserve_acfs:
            self.acfs = {} # Just to be sure that these stay consistent
        elif self.acfs:
            logging.warning("preserving existing ACFs after conditioning")
        # record conditioning settings
        self.update_info('condition', **settings)
    condition_data.__doc__ += Data.condition.__doc__

    def get_templates(self, signal_buffer='auto', **kws):
        """Produce templates at each detector for a given set of parameters.
        Can be used to generate waveforms from model samples, or a full
        coalescence.

        This is a wrapper around
        :func:`ringdown.waveforms.get_detector_signals`.

        Arguments
        ---------
        signal_buffer : float, str
            span of time around target for which to evaluate polarizations, to
            avoid doing so over a very long time array (for speed). By default,
            ``signal_buffer='auto'`` sets this to a multiple of the analysis
            duration; otherwise this should be a float, or `inf` for no
            signal_buffer.  (see docs for
            :meth:`ringdown.waveforms.Ringdown.from_parameters`; this option
            has no effect for coalescence signals)
        \*\*kws :
            arguments passed to :func:`ringdown.waveforms.get_detector_signals`.
        
        Returns
        -------
        waveforms : dict
            dictionary of :class:`Data` waveforms for each detector.
        """
        if signal_buffer == 'auto':
            if self.duration is not None:
                kws['signal_buffer'] = 10*self.duration
        else:
            kws['signal_buffer'] = signal_buffer

        # if no sky location given, use provided APs or default to target
        sky_keys = ['ra', 'dec', 'psi']
        if not all([k in kws for k in sky_keys]):
            kws['antenna_patterns'] = kws.pop('antenna_patterns', None) or \
                                      self.antenna_patterns
        for k in sky_keys:
            kws[k] = kws.get(k, getattr(self.target, k, None))

        kws['times'] = {ifo: d.time.values for ifo,d in self.data.items()}
        kws['t0_default'] = self.t0
        return waveforms.get_detector_signals(**kws)

    def inject(self, no_noise=False, **kws):
        """Add simulated signal to data, and records it in
        :attr:`Fit.injections`.

        .. warning::
          This method overwrites data stored in in :attr:`Fit.data`.

        Arguments are passed to :meth:`Fit.get_templates`.

        Arguments
        ---------
        no_noise : bool
            if true, replaces data with injection, instead of adding the two.
            (def. False)

        """
        # record all arguments
        settings = {k: v for k,v in locals().items() if k != 'self'}
        for k, v in settings.pop('kws').items():
            settings[k] = v
            
        self.injections = self.get_templates(**kws)
        for i, h in self.injections.items():
            if no_noise:
                self.data[i] = h
            else:
                self.data[i] = self.data[i] + h
        self.update_info('injection', **settings)
    
    @property
    def injection_parameters(self) -> dict:
        return self.info.get('injection', {})
        
    @property
    def conditioned_injections(self) -> dict:
        """Conditioned injections, if available.
        """
        if self.injections:
            if 'condition' in self.info:
                # inspect h.condition to get valid arguments
                x = inspect.signature(Data.condition).parameters.keys()
                c = {k: v for k,v in self.info['condition'].items() if k in x}
                hdict = {i: h.condition(t0=self.start_times[i], **c) 
                        for i, h in self.injections.items()}
            else:
                hdict = self.injections
        else:
            hdict = {}
        return hdict
    
    def run(self,
            prior : bool = False,
            predictive : bool = True,
            store_h_det : bool = False,
            store_h_det_mode : bool = True,
            store_residuals : bool = False,
            suppress_warnings : bool = True, 
            min_ess : int | None = None,
            prng : jaxlib.xla_extension.ArrayImpl | int | None = None,
            **kwargs):
        """Fit model.

        Additional keyword arguments not listed below are passed to the sampler
        with the following defaults when sampling:

        {}

        See docs for :func:`numpyro.infer.NUTS` and :func:`numpyro.infer.MCMC`
        to see all available options.

        Arguments
        ---------
        prior : bool
            whether to sample the prior (def. `False`).

        suppress_warnings : bool
            suppress some sampler warnings (def. `True`).

        store_residuals : bool
            compute whitened residuals point-wise and store in ``Fit.result``.

        min_ess: number
            if given, keep re-running the sampling with longer chains until the
            minimum effective sample size exceeds `min_ess` (def. `None`).

        \*\*kwargs :
            arguments passed to sampler.
        """
        # record all arguments
        settings = {k: v for k,v in locals().items() if k != 'self'}
        for k, v in settings.pop('kwargs').items():
            settings[k] = v
        self.update_info('run', **settings)
        
        ess_run = -1.0 # ess after sampling finishes, to be set by loop below
        min_ess = 0.0 if min_ess is None else min_ess

        if not self.acfs:
            logging.warning("computing ACFs with default settings")
            self.compute_acfs()

        # ensure delta_t of ACFs is equal to delta_t of data
        for ifo in self.ifos:
            if not np.isclose(self.acfs[ifo].delta_t, self.data[ifo].delta_t):
                e = "{} ACF delta_t ({:.1e}) does not match data ({:.1e})."
                raise AssertionError(e.format(ifo, self.acfs[ifo].delta_t,
                                          self.data[ifo].delta_t))

        # parse keyword arguments
        filter = 'ignore' if suppress_warnings else 'default'

        # create model
        model = make_model(self.modes.value, prior=prior, predictive=False,
                           store_h_det=False, store_h_det_mode=False,
                           **self.model_settings)
        
        logging.info('running {} mode fit'.format(self.modes))
        logging.info('prior run: {}'.format(prior))
        logging.info('model settings: {}'.format(self._model_settings))

        # parse keyword arguments to be passed to KERNEL and SAMPLER, with
        # defaults based on DEF_RUN_KWS
        kws = cp.deepcopy(DEF_RUN_KWS)
        kws.update(kwargs)

        # tease out kernel options dynamically
        kernel_kws = kws.pop('kernel', {})
        kernel_kws.update({k: v for k,v in kws.items() if k in KERNEL_ARGS})
        logging.info('kernel settings: {}'.format(kernel_kws))

        # tease out sampler options dynamically
        sampler_kws = kws.pop('sampler', {})
        sampler_kws.update({k: v for k,v in kws.items() if k in SAMPLER_ARGS})
        logging.info('sampler settings: {}'.format(sampler_kws))
        
        # assume leftover arguments will be passed to run method
        run_kws = {k: v for k,v in kws.items() if k not in SAMPLER_ARGS and 
                   k not in KERNEL_ARGS}
        logging.info('run settings: {}'.format(run_kws))
        
        # create random number generator
        if isinstance(prng, int):
            prng = jax.random.PRNGKey(prng)
        elif prng is None:
            prng = jax.random.PRNGKey(np.random.randint(1<<31))

        # log some runtime information
        jax_device_count = jax.device_count()
        platform = jax.lib.xla_bridge.get_backend().platform.upper()
        omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
        logging.info(f"running on {jax_device_count} {platform} using "
                     f"{omp_num_threads} OMP threads")

        # get run input and run
        run_input = self.run_input
        start_times  = self.start_times
        epoch = [start_times[i] for i in self.ifos]
        if self.has_injections:
            inj = [self.analysis_injections[i] for i in self.ifos]
        else:
            inj = None

        run_count = 1
        with warnings.catch_warnings():
            warnings.simplefilter(filter)
            while ess_run < min_ess:
                if not np.isscalar(min_ess):
                    raise ValueError("min_ess is not a number")
         
                # make kernel, sampler and run
                kernel = KERNEL(model, **kernel_kws)
                sampler = SAMPLER(kernel, **sampler_kws)
                sampler.run(prng, *run_input, **run_kws)

                # turn sampler into Result object and store
                # (recall that Result is a wrapper for arviz.InferenceData)
                result = get_arviz(sampler, ifos=self.ifos, modes=self.modes,
                                   injections=inj, epoch=epoch, 
                                   attrs=self.attrs)

                # check effective number of samples and rerun if necessary
                ess_run = result.ess
                logging.info(f"ess = {int(ess_run)} after {run_count} runs")

                if ess_run < min_ess:
                    run_count += 1
                    
                    # if we need to run again, double the number of tuning steps
                    # and samples
                    new_kws = dict(
                        num_warmup = 2*sampler_kws.get('num_warmup', 1000),
                        num_samples = 2*sampler_kws.get('num_samples', 1000)
                    )
                    sampler_kws.update(new_kws)
                        
                    logging.warning(
                        f"""ess = {ess_run:.1f} below threshold {min_ess}; 
                        fitting again with "{new_kws['num_warmup']} tuning
                        steps and {new_kws['num_samples']} samples"""
                    )

                    kwargs.update(kws)
                
        if predictive or store_h_det or store_h_det_mode:
            logging.info("obtaining predictive distribution")
            predictive = numpyro.infer.Predictive(model, sampler.get_samples())
            pred = predictive(prng, *run_input, predictive=predictive, 
                              store_h_det=store_h_det,
                              store_h_det_mode=store_h_det_mode)
            
            # adduct posterior predictive to result
            chain_draw = ['chain', 'draw']
            shape = [result.posterior.sizes[k] for k in chain_draw]
            coord = dict(ifo=self.ifos,
                         mode=self.modes.get_coordinates(),
                         time_index=np.arange(self.n_analyze, dtype=int))
            for k, v in pred.items():
                if k not in result.posterior and k not in result.observed_data:
                    # get dimension names
                    d = tuple(chain_draw + list(MODEL_DIMENSIONS.get(k, ())))
                    # get coordinates
                    c = {c: coord[c] for c in d if c not in chain_draw}
                    # get data array replacing first dimension (samples) with
                    # chain and draw
                    v = np.reshape(v, tuple(shape + list(v.shape[1:])))
                    result.posterior[k] = xr.DataArray(v, coords=c, dims=d)
                    logging.info(f"added {k} to posterior")

        if prior:
            self.prior = result
        else:
            if store_residuals:
                result._generate_whitened_residuals()
            self.result = result
        self._numpyro_sampler = sampler
    run.__doc__ = run.__doc__.format(DEF_RUN_KWS)

    @property
    def settings(self):
        config = self.to_config()
        return {section: dict(config[section]) for section in config.sections()}

    def to_json(self, indent=4, **kws):
        return json.dumps(self.settings, indent=indent, **kws)
    
    @property
    def attrs(self):
        from . import __version__
        return dict(config=self.to_json(), ringdown_version=__version__)

    def add_data(self, data, time=None, ifo=None, acf=None):
        """Add data to fit.

        Arguments
        ---------
        data : array,Data
            time series to be added.
        time : array
            array of time stamps (only required if `data` is not
            :class:`ringdown.data.Data`).
        ifo : str
            interferometer key (optional).
        acf : array,AutoCovariance
            autocovariance series corresponding to these data (optional).
        """
        if not isinstance(data, Data):
            data = Data(data, index=getattr(data, 'time', time), ifo=ifo)
        self.data[data.ifo] = data
        if acf is not None:
            self.acfs[data.ifo] = acf
    
    def load_data(self, path=None, ifos=None, channel=None,
                  frametype=None, **kws):
        """Load data from disk.

        Additional arguments are passed to :meth:`ringdown.data.Data.read`.

        Arguments
        ---------
        path : dict, str
            dictionary of data paths indexed by interferometer keys, or path
            string replacement pattern, e.g.,
            ``'path/to/{i}-{ifo}_GWOSC_16KHZ_R1-1126259447-32.hdf5'`` where `i`
            and `ifo` will be respectively replaced by the first letter and key
            for each detector listed in `ifos` (e.g., `H` and `H1` for LIGO
            Hanford).

        ifos : list
            list of detector keys (e.g., ``['H1', 'L1']``), not required if
            `path_input` is a dictionary.
        
        channel : dict, str
            dictionary of channel names indexed by interferometer keys, or
            channel name string replacement pattern, e.g.,
            ``'{ifo}:GWOSC-16KHZ_R1_STRAIN'`` where `i` and `ifo` will be
            respectively replaced by the first letter and key for each detector
            listed in `ifos` (e.g., `H` and `H1` for LIGO Hanford).  Only used
            when `kind = 'frame'`.
            
        frametype : dict, str
            dictionary of frame types indexed by interferometer keys, or frame
            type string replacement pattern, e.g., `'H1_HOFT_C00'`, with 
            same replacement rules as for `channel` and `path`. Only used when
            `kind = 'discover'`.
        """
        # record all arguments
        settings = {k: v for k,v in locals().items() if k != 'self'}
        for k, v in settings.pop('kws').items():
            settings[k] = v
        
        if ifos is None:
            if hasattr(path, 'keys'):
                ifos = list(path.keys())
            else:
                raise ValueError("no ifos provided")
        
        if path is None:
            path_dict = {k: None for k in ifos}
        else:
            path_dict = utils.get_dict_from_pattern(path, ifos)
        
        if channel is not None:
            channel_dict = {k: None for k in path_dict.keys()}
        else:
            channel_dict = utils.get_dict_from_pattern(channel, ifos)
        
        if frametype is not None:
            frametype_dict = {k: None for k in path_dict.keys()}
        else:
            frametype_dict = utils.get_dict_from_pattern(frametype, ifos)
            
        tslide = kws.pop('slide', {}) or {}
        for ifo, path in path_dict.items():
            self.add_data(Data.read(path, ifo=ifo, channel=channel_dict[ifo],
                                    frametype=frametype_dict[ifo], **kws))
        # apply time slide if requested
        for i, dt in tslide.items():
            d = self.data[i]
            new_d = Data(np.roll(d, int(dt / d.delta_t)), ifo=i, index=d.time)
            self.add_data(new_d)
        # record data provenance
        settings['path'] = path_dict
        self.update_info('data', **settings)
    
    def compute_acfs(self, shared=False, ifos=None, **kws):
        """Compute ACFs for all data sets in `Fit.data`.

        Arguments
        ---------
        shared : bool
            specifies if all IFOs are to share a single ACF, in which case the
            ACF is only computed once from the data of the first IFO (useful
            for simulated data) (default False)

        ifos : list
            specific set of IFOs for which to compute ACF, otherwise computes
            it for all

        extra kwargs are passed to ACF constructor
        """
        # record all arguments
        settings = {k: v for k,v in locals().items() if k != 'self'}
        for k, v in settings.pop('kws').items():
            settings[k] = v
        
        ifos = self.ifos if ifos is None else ifos
        if len(ifos) == 0:
            raise ValueError("first add data")

        # Try to set a safe `nperseg` if we are using `fd` estimation
        if self.n_analyze is not None:
            nperseg_safe = utils.np2(16*self.n_analyze)
            if kws.get('method', 'fd') == 'fd':
                if not ('nperseg' in kws):
                    kws['nperseg'] = nperseg_safe
        
        # if shared, compute a single ACF
        acf = self.data[ifos[0]].get_acf(**kws) if shared else None
        for ifo in ifos:
            self.acfs[ifo] = acf if shared else self.data[ifo].get_acf(**kws)
        # record ACF computation options
        self.update_info('acf', **settings)

    def load_acfs(self, path, ifos=None, from_psd=False, **kws):
        """Load autocovariances from disk. Can read in a PSD, instead of an
        ACF, if using the `from_psd` argument.

        Additional arguments are passed to
        :meth:`ringdown.data.AutoCovariance.read` (or,
        :meth:`ringdown.data.PowerSpectrum.read` if ``from_psd``).

        Arguments
        ---------
        path : dict, str
            dictionary of ACF paths indexed by interferometer keys, or path
            string replacement pattern, e.g.,
            ``'path/to/acf_{i}_{ifo}.dat'`` where `i` and `ifo` will be
            respectively replaced by the first letter and key for each detector
            listed in `ifos` (e.g., `H` and `H1` for LIGO Hanford).

        ifos : list
            list of detector keys (e.g., ``['H1', 'L1']``), not required 
            if `path_input` is a dictionary.

        from_psd : bool
            read in a PSD and convert to ACF.
        """
        # record all arguments
        settings = {k: v for k,v in locals().items() if k != 'self'}
        for k, v in settings.pop('kws').items():
            settings[k] = v
        
        if isinstance(path, str) and ifos is None:
            ifos = self.ifos
        path_dict = utils.get_dict_from_pattern(path, ifos)
        for ifo, p in path_dict.items():
            if from_psd:
                self.acfs[ifo] = PowerSpectrum.read(p, **kws).to_acf()
            else:
                self.acfs[ifo] = AutoCovariance.read(p, **kws)
        # record ACF computation options
        settings['path'] = path_dict
        self.update_info('acf', **settings)

    def set_tone_sequence(self, nmode, p=1, s=-2, l=2, m=2):
        """Set template modes to be a sequence of overtones with a given
        angular structure.

        To set an arbitrary set of modes, use :meth:`Fit.set_modes`

        Arguments
        ---------
        nmode : int
          number of tones (`nmode=1` includes only fundamental mode).
        p : int
          prograde (`p=1`) vs retrograde (`p=-1`) flag.
        s : int
          spin-weight.
        l : int
          azimuthal quantum number.
        m : int
          magnetic quantum number.
        """
        indexes = [(p, s, l, m, n) for n in range(nmode)]
        self.set_modes(indexes)

    def set_modes(self, modes : int | list[tuple[int, int, int, int, int]]):
        """Establish list of modes to include in analysis template.

        Modes can be an integer, in which case `n` arbitrary damped sinusoids
        will be fit.

        Modes identified by their `(p, s, l, m, n)` indices, where:
          - `p` is `1` for prograde modes, and `-1` for retrograde modes;
          - `s` is the spin-weight (`-2` for gravitational waves);
          - `l` is the azimuthal quantum number;
          - `m` is the magnetic quantum number;
          - `n` is the overtone number.

        See :class:`ringdown.indexing.ModeIndexList`.

        Arguments
        ---------
        modes : list
            list of tuples with quasinormal mode `(p, s, l, m, n)` numbers.
        """
        self.modes = indexing.ModeIndexList(modes)

    def set_target(self, t0 : float | dict, ra : float | None = None,
                   dec : float | None = None, psi : float | None = None,
                   duration : float | None = None,
                   reference_ifo : str | None = None,
                   antenna_patterns: dict | None = None,
                   n_analyze : int | None = None):
        """ Establish truncation target, stored to `self.target`.

        Provide a targetted analysis start time `t0` to serve as beginning of
        truncated analysis segment; this will be compared against timestamps
        in `fit.data` objects so that the closest sample to `t0` is preserved
        after conditioning and taken as the first sample of the analysis 
        segment.

        .. important::
          If the model accepts multiple detectors, `t0` is assumed to be
          defined at geocenter; truncation time at individual detectors will
          be determined based on specified sky location.

        The source sky location and orientation can be specified by the `ra`,
        `dec`, and `psi` arguments. These are use to both determine the
        truncation time at different detectors, as well as to compute the
        corresponding antenna patterns. Specifying a sky location is only
        required if the model can handle data from multiple detectors.

        The argument `duration` specifies the length of the analysis segment in
        the unit of time used to index the data (e.g., s). Based on the sampling
        rate, this argument is used to compute the number of samples to be
        included in the segment, beginning from the first sample identified from
        `t0`.

        Alternatively, the `n_analyze` argument can be specified directly. If
        neither `duration` nor `n_analyze` are provided, the duration will be
        set based on the shortest available data series in the `Fit` object.

        .. warning::
          Failing to explicitly specify `duration` or `n_analyze` risks
          inadvertently extremely long analysis segments, with correspondingly
          long run times.

        Arguments
        ---------
        t0 : float, dict
            target time (at geocenter for a detector network, if no
            `reference_ifo` is specified), or a dictionary of start
            times.
        ra : float
            source right ascension (rad).
        dec : float
            source declination (rad).
        psi : float
            source polarization angle (rad).
        duration : float
            analysis segment length in seconds, or time unit indexing data
            (overrides `n_analyze`).
        antenna_patterns : dict
            dictionary with tuples for plus and cross antenna patterns for
            each detector `{ifo: (Fp, Fc)}` (optional)
        reference_ifo : str
            if specified, use this detector as reference for delays and
            antenna patterns, otherwise assume t0 defined at geocenter.
        """
        # turn float into LIGOTimeGPS object to ensure we get the right 
        # number of digits when converting to string
        t0 = lal.LIGOTimeGPS(t0) if isinstance(t0, float) else t0
        
        # record all arguments for provenance
        settings = {k: v for k,v in locals().items() if k != 'self'}
        
        if self.result is not None:
            raise ValueError("cannot set target with preexisting results")
        
        if n_analyze:
            if duration:
                logging.warning("ignoring duration in favor of n_analyze")
                duration = None
            self._n_analyze = int(n_analyze)
        elif not duration:
            logging.warning("no duration or n_analyze specified")
        else:
            self._duration = float(duration)
        
        self.target = Target.construct(t0, ra, dec, psi, reference_ifo,
                                       antenna_patterns)
                
        # make sure that start times are encompassed by data (if data exist)
        for i, data in self.data.items():
            t0_i = self.start_times[i] 
            if t0_i < data.time[0] or t0_i > data.time[-1]:
                raise ValueError("{} start time not in data".format(i))
        # record state
        self.update_info('target', **settings)

    @property
    def duration(self) -> float:
        """Analysis duration in the units of time presumed by the
        :attr:`Fit.data` and :attr:`Fit.acfs` objects
        (usually seconds). Defined as :math:`T = N\\times\Delta t`, where
        :math:`N` is the number of analysis samples
        (:attr:`n_analyze`) and :math:`\Delta t` is the time
        sample spacing.
        """
        if self._n_analyze and not self._duration:
            if self.data:
                return self._n_analyze*self.data[self.ifos[0]].delta_t
            else:
                logging.warning("add data to compute duration "
                                "(n_analyze = {})".format(self._n_analyze))
                return None
        else:
            return self._duration

    @property
    def start_indices(self) -> dict:
        """Locations of first samples in :attr:`Fit.data`
        to be included in the ringdown analysis for each detector.
        """
        i0_dict = {}
        if self.has_target:
            # make sure that start times are encompassed by data
            for i, t0_i in self.start_times.items():
                if t0_i < self.data[i].time[0] or t0_i > self.data[i].time[-1]:
                    raise ValueError("{} start time not in data".format(i))
            # find sample closest to (but no later than) requested start time
            for ifo, d in self.data.items():
                t0 = self.start_times[ifo]
                i0_dict[ifo] = np.argmin(abs(d.time - t0))
        return i0_dict

    @property
    def n_analyze(self) -> int:
        """Number of data points included in analysis for each detector.
        """
        if self.duration and not self._n_analyze:
            # set n_analyze based on specified duration in seconds
            if self.data:
                dt = self.data[self.ifos[0]].delta_t
                return int(round(self.duration/dt))
            else:
                logging.warning("add data to compute n_analyze "
                                "(duration = {})".format(self.duration))
                return None
        elif self.data and self.has_target:
            # set n_analyze to fit shortest data set
            i0s = self.start_indices
            return min([len(d.iloc[i0s[i]:]) for i, d in self.data.items()])
        else:
            return self._n_analyze

    def whiten(self, datas: dict) -> dict:
        """Return whiten data for all detectors using ACFs stored in
        :attr:`Fit.acfs`.

        See also :meth:`ringdown.data.AutoCovariance.whiten`.

        Arguments
        ---------
        datas : dict
            dictionary of data to be whitened for each detector.

        Returns
        -------
        whitened_datas : dict
            dictionary of :class:`ringdown.data.Data` with whitened data for
            each detector.
        """
        return {i: self.acfs[i].whiten(d) for i, d in datas.items()}