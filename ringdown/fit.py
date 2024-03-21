"""Module defining the core :class:`Fit` class.
"""

__all__ = ['Target', 'Fit']

import numpy as np
import arviz as az
import json
from arviz.data.base import dict_to_dataset
from ast import literal_eval
from collections import namedtuple
import configparser
import copy as cp
from .data import *
import lal
import logging
from .model import make_model, get_arviz
import os
from . import qnms
import warnings
from . import waveforms
import inspect
import jax
import numpyro.infer

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

DEF_RUN_KWS = dict(dense_mass=True, num_warmup=1000, num_samples=1000,
                       num_chains=4)

def form_opt(x):
    """Utility to format options in config."""
    return np.array2string(np.array(x), separator=', ')

def try_parse(x):
    """Attempt to parse a string as a number, dict, or list."""
    try:
        return float(x)
    except (TypeError,ValueError):
        try:
            return literal_eval(x)
        except (TypeError,ValueError,SyntaxError):
            if x == "inf":
                return np.inf
            else:
                return x

def np2(x):
    """Returns the next power of two as big as or larger than x."""
    p = 1
    while p < x:
        p = p << 1
    return p

Target = namedtuple('Target', ['t0', 'ra', 'dec', 'psi'])

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
    result : arviz.data.inference_data.InferenceData
        if model has been run, arviz object containing fit result
    prior : arviz.data.inference_data.InferenceData
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
        self.start_times = {}
        self.antenna_patterns = {}
        self.target = Target(None, None, None, None)
        self.result = None
        self.prior = None
        self._duration = None
        self._n_analyze = None
        # set modes dynamically
        self.modes = None
        self.set_modes(modes)
        # assume rest of kwargs are to be passed to make_model
        self._model_settings = {}
        self.update_model(**kws)

    @property
    def n_modes(self) -> int:
        """ Number of damped sinusoids to be included in template.
        """
        if isinstance(self.modes, int):
            return self.modes
        else:
            return len(self.modes)
    
    @property
    def injection_parameters(self) -> dict:
        return self.info.get('injection', {})

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
        return self.target.t0

    @property
    def sky(self) -> tuple:
        """ Tuple of source right ascension, declination and polarization angle
        (all in radians). This can be set using
        :meth:`Fit.set_target`.
        """
        return (self.target.ra, self.target.dec, self.target.psi)

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
    def model_settings(self) -> dict:
        """Prior options as currently set.
        """
        return self._model_settings

    @property
    def valid_model_settings(self) -> list:
        """Valid prior parameters for the selected model. These can be set
        through :meth:`Fit.update_model`.
        """
        return list(MODEL_ARGS)

    def update_model(self, **kws):
        """Set or modify prior options or other model settings.  For example,
        ``fit.update_model(A_scale=1e-21)`` sets the `A_scale` parameter to
        `1e-21`.

        Valid arguments for the selected model can be found in
        :attr:`Fit.valid_model_options`.
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

        fpfc = self.antenna_patterns.values()
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
                    no_cond: bool = False):
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
        # determine whether to read config from disk
        if isinstance(config_input, configparser.ConfigParser):
            config = config_input
        else:
            if not os.path.exists(config_input):
                raise FileNotFoundError(config_input)
            config = configparser.ConfigParser()
            config.read(config_input)
                    
        # parse model options
        model_opts = {k: try_parse(v) for k,v in config['model'].items()}
        if 'name' in model_opts:
            warnings.warn("model name is deprecated, use explicit mode options instead")
            logging.info("trying to guess mode configuration based on model name")
            name = model_opts.pop('name')
            if 'aligned' in name:
                raise NotImplementedError("aligned model not yet supported")
            model_opts['marginalized'] = 'marginal' in name

        if config.has_section('prior'):
            prior = {k: try_parse(v) for k,v in config['prior'].items()}
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
            if 'dtau_min' in model_opts:
                model_opts['dg_min'] = - model_opts.pop('dtau_max') * perturb_tau
            if 'dtau_max' in model_opts:
                model_opts['dg_max'] = - model_opts.pop('dtau_min') * perturb_tau
            
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
        
        # NOTE: not popping in order to preserve original ConfigParser
        kws = {k: try_parse(v) for k,v in config['data'].items()
                  if k not in ['ifos', 'path']}
        fit.load_data(path_input, ifos, **kws)

        # add target
        fit.set_target(**{k: try_parse(v) for k,v in config['target'].items()})
        
        # inject signal if requested
        if config.has_section('injection'):
            inj_kws = {k: try_parse(v) for k,v in config['injection'].items()}
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
                except (UnicodeDecodeError,json.JSONDecodeError):
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
            cond_kws = {k: try_parse(v) for k,v in config['condition'].items()}
            fit.condition_data(**cond_kws)
        
        # load or produce ACFs
        if config.get('acf', 'path', fallback=False):
            kws = {k: try_parse(v) for k,v in config['acf'].items()
                   if k not in ['path']}
            fit.load_acfs(config['acf']['path'], **kws)
        else:
            acf_kws = {} if 'acf' not in config else config['acf']
            fit.compute_acfs(**{k: try_parse(v) for k,v in acf_kws.items()})
        
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
        if isinstance(self.modes, int):
            config['model']['modes'] = str(self.n_modes)
        else:
            config['model']['modes'] = str([tuple(m) for m in self.modes])
        # prior options
        config['model'].update({k: form_opt(v) for k,v 
                                in self.model_settings.items()})
        # rest of options require data, so exit of none were added
        if not self.ifos:
            return config
        # data, injection, conditioning and acf options
        for sec, opts in self.info.items():
            config[sec] = {k: form_opt(v) for k,v in opts.items()}
        # target options
        config['target'] = {k: str(v) for k,v in self.target._asdict().items()}
        config['target']['duration'] = str(self.duration)
        # write file to disk if requested
        if path is not None:
            with open(path, 'w') as f:
                config.write(f)
        return config

    def update_info(self, section, **kws):
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

    def copy(self):
        """Produce a deep copy of this `Fit` object.

        Returns
        -------
        fit_copy : Fit
            deep copy of `Fit`.
        """
        # we can't deepcopy the PyMC model (aesara will throw an error)
        # so remove the cached model before copying (it can always be
        # recompiled if needed)
        self._pymc_model = None
        return cp.deepcopy(self)

    def condition_data(self, preserve_acfs=False, **kwargs):
        """Condition data for all detectors by calling
        :meth:`ringdown.data.Data.condition`. Docstring for that function
        below.

        The `preserve_acfs` argument determines whether to preserve original
        ACFs in fit after conditioning.

        """
        new_data = {}
        for k, d in self.data.items():
            t0 = self.start_times[k]
            new_data[k] = d.condition(t0=t0, **kwargs)
        self.data = new_data
        if not preserve_acfs:
            self.acfs = {} # Just to be sure that these stay consistent
        elif self.acfs:
            logging.warning("preserving existing ACFs after conditioning")
        # record conditioning settings
        self.update_info('condition', **kwargs)
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
        if not all([k in kws for k in ['ra', 'dec']]):
            kws['antenna_patterns'] = kws.pop('antenna_patterns', None) or \
                                      self.antenna_patterns
        for k in ['ra', 'dec', 'psi']:
            kws[k] = kws.get(k, self.target._asdict()[k])

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
        self.injections = self.get_templates(**kws)
        for i, h in self.injections.items():
            if no_noise:
                self.data[i] = h
            else:
                self.data[i] = self.data[i] + h
        self.update_info('injection', no_noise=no_noise, **kws)
    
    def run(self, prior=False, suppress_warnings=True, store_residuals=True,
            min_ess=None, prng=None, **kwargs):
        """Fit model.

        Additional keyword arguments not listed below are passed to the sampler,
        with the following defaults when sampling:

        {}

        See docs for :func:`pymc.sample` to see all available options.

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
        ess_run = -1.0 # ess after sampling finishes, to be set by loop below
        if min_ess is None:
            min_ess = 0.0

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
        model = make_model(self.modes, prior=prior, **self.model_settings)

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
        logging.info(f"running on {jax_device_count} {platform}")

        omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
        logging.info(f"using {omp_num_threads} OMP threads")

        # get run input and run
        run_input = self.run_input

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

                # turn sampler into arviz object and store
                result = get_arviz(sampler, ifos=self.ifos, modes=self.modes)
                if prior:
                    self.prior = result
                else:
                    self.result = result

                # check effective number of samples and rerun if necessary
                ess = az.ess(result)
                mess = ess.min()
                mess_arr = np.array([mess[k].values[()] for k in mess.keys()])
                ess_run = np.min(mess_arr)
                
                logging.info(f"min ess = {int(ess_run)} after {run_count} runs")
                run_count += 1

                if ess_run < min_ess:
                    # if we need to run again, double the number of tuning steps
                    # and samples
                    new_kws = dict(
                        num_warmup=2*sampler_kws.get('num_warmup', 1000),
                        num_samples=2*sampler_kws.get('num_samples', 1000)
                    )
                    sampler_kws.update(new_kws)
                        
                    logging.warning(f"min ess = {ess_run:.1f} below threshold {min_ess}")
                    logging.warning(f"fitting again with {new_kws['num_warmup']}"
                                    f"tuning steps and {new_kws['num_samples']} samples")

                    kwargs.update(kws)
        self.update_info('run', **kwargs)

        if not prior and store_residuals:
            self._generate_whitened_residuals()

    run.__doc__ = run.__doc__.format(DEF_RUN_KWS)

    def _generate_whitened_residuals(self):
        # Adduct the whitened residuals to the result.
        residuals = {}
        residuals_stacked = {}
        for ifo in self.ifos:
            if ifo in self.result.posterior.ifo:
                ifo_key = ifo
            elif bytes(str(ifo), 'utf-8') in self.result.posterior.ifo:
                # IFO coordinates are bytestrings
                ifo_key = bytes(str(ifo), 'utf-8') # IFO coordinates are bytestrings
            else:
                raise KeyError(f"IFO {ifo} is not a valid indexing ifo for the result posterior")
            r = self.result.constant_data['strain'].sel(ifo=ifo_key) -\
                self.result.posterior.h_det.sel(ifo=ifo_key)
            residuals[ifo] = r.transpose('chain', 'draw', 'time_index')
            residuals_stacked[ifo] = residuals[ifo].stack(sample=['chain',
                                                                  'draw'])
        residuals_whitened = self.whiten(residuals_stacked)
        d = self.result.posterior.sizes
        residuals_whitened = {
            i: v.reshape((d['time_index'], d['chain'], d['draw']))
            for i,v in residuals_whitened.items()
        }
        resid = np.stack([residuals_whitened[i] for i in self.ifos], axis=-1)
        keys = ('time_index', 'chain', 'draw', 'ifo')
        self.result.posterior['whitened_residual'] = (keys, resid)
        keys = ('chain', 'draw', 'ifo', 'time_index')
        self.result.posterior['whitened_residual'] = \
            self.result.posterior.whitened_residual.transpose(*keys)
        lnlike = -self.result.posterior.whitened_residual**2/2
        try:
            self.result.log_likelihood['whitened_pointwise_loglike'] = lnlike    
        except AttributeError:
            # We assume that log-likelihood isn't created yet.
            self.result.add_groups(dict(
                log_likelihood=dict_to_dataset(
                    {'whitened_pointwise_loglike': lnlike},
                    coords=self.result.posterior.coords,
                    dims={'whitened_pointwise_loglike': list(keys)}
                    )))

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

    def load_data(self, path_input, ifos=None, channel=None, **kws):
        """Load data from disk.

        Additional arguments are passed to :meth:`ringdown.data.Data.read`.

        Arguments
        ---------
        path_input : dict, str
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
            ``'H1:GWOSC-16KHZ_R1_STRAIN'`` where `i` and `ifo` will be
            respectively replaced by the first letter and key for each detector
            listed in `ifos` (e.g., `H` and `H1` for LIGO Hanford).  Only used
            when `kind = 'frame'`.
        """
        # TODO: add ability to generate synthetic data here?
        if isinstance(path_input, str):
            try:
                path_dict = literal_eval(path_input)
            except (ValueError,SyntaxError):
                if ifos is None:
                    raise ValueError("must provide IFO list.")
                path_dict = {}
                for ifo in ifos:
                    i = '' if not ifo else ifo[0]
                    path_dict[ifo] = path_input.format(i=i, ifo=ifo)
        else:
            path_dict = path_input
        path_dict = {k: os.path.abspath(v) for k,v in path_dict.items()}
        if channel is not None:
            if isinstance(channel, str):
                try:
                    channel_dict = literal_eval(channel)
                except (ValueError,SyntaxError):
                    if ifos is None:
                        raise ValueError("must provide IFO list.")
                    channel_dict = {}
                    for ifo in ifos:
                        i = '' if not ifo else ifo[0]
                        channel_dict[ifo] = channel.format(i=i, ifo=ifo)
            else:
                channel_dict = channel
        else:
            channel_dict = {k: None for k in path_dict.keys()}
        tslide = kws.pop('slide', {}) or {}
        for ifo, path in path_dict.items():
            self.add_data(Data.read(path, ifo=ifo, channel=channel_dict[ifo], **kws))
        # apply time slide if requested
        for i, dt in tslide.items():
            d = self.data[i]
            new_d = Data(np.roll(d, int(dt / d.delta_t)), ifo=i, index=d.time)
            self.add_data(new_d)
        # record data provenance
        self.update_info('data', path=path_dict, **kws)
    
    def compute_acfs(self, shared=False, ifos=None, **kws):
        """Compute ACFs for all data sets in `Fit.data`.

        Arguments
        ---------
        shared : bool
            specifices if all IFOs are to share a single ACF, in which case the
            ACF is only computed once from the data of the first IFO (useful
            for simulated data) (default False)

        ifos : list
            specific set of IFOs for which to compute ACF, otherwise computes
            it for all

        extra kwargs are passed to ACF constructor
        """
        ifos = self.ifos if ifos is None else ifos
        if len(ifos) == 0:
            raise ValueError("first add data")

        # Try to set a safe `nperseg` if we are using `fd` estimation
        if self.n_analyze is not None:
            nperseg_safe = np2(16*self.n_analyze)
            if kws.get('method', 'fd') == 'fd':
                if not ('nperseg' in kws):
                    kws['nperseg'] = nperseg_safe
        
        # if shared, compute a single ACF
        acf = self.data[ifos[0]].get_acf(**kws) if shared else None
        for ifo in ifos:
            self.acfs[ifo] = acf if shared else self.data[ifo].get_acf(**kws)
        # record ACF computation options
        self.update_info('acf', shared=shared, **kws)

    def load_acfs(self, path_input, ifos=None, from_psd=False, **kws):
        """Load autocovariances from disk. Can read in a PSD, instead of an
        ACF, if using the `from_psd` argument.

        Additional arguments are passed to
        :meth:`ringdown.data.AutoCovariance.read` (or,
        :meth:`ringdown.data.PowerSpectrum.read` if ``from_psd``).

        Arguments
        ---------
        path_input : dict, str
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
        if isinstance(path_input, str):
            try:
                path_dict = literal_eval(path_input)
            except (ValueError,SyntaxError):
                if ifos is None:
                    ifos = self.ifos 
                path_dict = {}
                for ifo in ifos:
                    i = '' if not ifo else ifo[0]
                    path_dict[ifo] = path_input.format(i=i, ifo=ifo)
        else:
            path_dict = path_input
        path_dict = {k: os.path.abspath(v) for k,v in path_dict.items()}
        for ifo, p in path_dict.items():
            if from_psd:
                self.acfs[ifo] = PowerSpectrum.read(p, **kws).to_acf()
            else:
                self.acfs[ifo] = AutoCovariance.read(p, **kws)
        # record ACF computation options
        self.update_info('acf', path=path_dict, from_psd=from_psd, **kws)

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

        See :meth:`ringdown.qnms.construct_mode_list`.

        Arguments
        ---------
        modes : list
            list of tuples with quasinormal mode `(p, s, l, m, n)` numbers.
        """
        try:
            # if modes is integer, interpret as number of modes
            self.modes = int(modes)
        except Exception:
            # otherwise, assume it is a mode index list
            self.modes = qnms.construct_mode_list(modes)

    def set_target(self, t0, ra=None, dec=None, psi=None, delays=None,
                   antenna_patterns=None, duration=None, n_analyze=None):
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

        Alternatively, antenna patterns and geocenter-delays can be specified
        directly through the `antenna_patterns` and `delays` arguments.

        For all models, the argument `duration` specifies the length of the 
        analysis segment in the unit of time used to index the data (e.g., s).
        Based on the sampling rate, this argument is used to compute the number
        of samples to be included in the segment, beginning from the first
        sample identified from `t0`.

        Alternatively, the `n_analyze` argument can be specified directly. If
        neither `duration` nor `n_analyze` are provided, the duration will be
        set based on the shortest available data series in the `Fit` object.

        .. warning::
          Failing to explicitly specify `duration` or `n_analyze` risks
          inadvertedly extremely long analysis segments, with correspondingly
          long runtimes.

        Arguments
        ---------
        t0 : float
            target time (at geocenter for a detector network).
        ra : float
            source right ascension (rad).
        dec : float
            source declination (rad).
        psi : float
            source polarization angle (rad).
        duration : float
            analysis segment length in seconds, or time unit indexing data
            (overrides `n_analyze`).
        n_analyze : int
            number of data points to include in analysis segment.
        delays : dict
            dictionary with delays from geocenter for each detector, as would
            be computed by `lal.TimeDelayFromEarthCenter` (optional).
        antenna_patterns : dict
            dictionary with tuples for plus and cross antenna patterns for
            each detector `{ifo: (Fp, Fc)}` (optional)
        """
        if not self.data:
            raise ValueError("must add data before setting target.")
        tgps = lal.LIGOTimeGPS(t0)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        delays = delays or {}
        antenna_patterns = antenna_patterns or {}
        for ifo, data in self.data.items():
            # TODO: should we have an elliptical+ftau model?
            if ifo is None:
                dt_ifo = 0
                self.antenna_patterns[ifo] = (1, 1)
            else:
                det = data.detector
                dt_ifo = delays.get(ifo,
                    lal.TimeDelayFromEarthCenter(det.location, ra, dec, tgps))
                self.antenna_patterns[ifo] = antenna_patterns.get(ifo,
                    lal.ComputeDetAMResponse(det.response, ra, dec, psi, gmst))
            self.start_times[ifo] = t0 + dt_ifo
        self.target = Target(t0, ra, dec, psi)
        # also specify analysis duration if requested
        if duration:
            self._duration = duration
        elif n_analyze:
            self._n_analyze = int(n_analyze)
        # make sure that start times are encompassed by data
        for i, t0_i in self.start_times.items():
            if t0_i < self.data[i].time[0] or t0_i > self.data[i].time[-1]:
                raise ValueError("{} start time not in data".format(i))

    def update_target(self, **kws):
        """Modify analysis target. See also :meth:`Fit.set_target`.
        """
        if self.result is not None:
            # TODO: fail?
            logging.warn("updating target of Fit with preexisting results!")
        target = self.target._asdict()
        target.update({k: getattr(self,k) for k in
                       ['duration', 'n_analyze', 'antenna_patterns']})
        target.update(kws)
        self.set_target(**target)

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
    def has_target(self) -> bool:
        """Whether an analysis target has been set with
        :meth:`Fit.set_target`.
        """
        return self.target.t0 is not None

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
        if self._duration and not self._n_analyze:
            # set n_analyze based on specified duration in seconds
            if self.data:
                dt = self.data[self.ifos[0]].delta_t
                return int(round(self._duration/dt))
            else:
                logging.warning("add data to compute n_analyze "
                                "(duration = {})".format(self._duration))
                return None
        elif self.data and self.has_target:
            # set n_analyze to fit shortest data set
            i0s = self.start_indices
            return min([len(d.iloc[i0s[i]:]) for i, d in self.data.items()])
        else:
            return self._n_analyze

    def whiten(self, datas) -> dict:
        """Return whiten data for all detectors.

        See also :meth:`ringdown.data.AutoCovariance.whiten`.

        Arguments
        ---------
        datas : dict
            dictionary of data to be whitened for each detector.

        Returns
        -------
        wdatas : dict
            dictionary of :class:`ringdown.data.Data` with whitned data for
            each detector.
        """
        return {i: self.acfs[i].whiten(d) for i,d in datas.items()}

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
        if prior and not self.prior:
            raise ValueError("no prior samples available")
        elif not prior and not self.result:
            raise ValueError("no posterior samples available")
        # stack samples (prior or result)
        result = self.prior if prior else self.result
        samples = result.posterior.stack(sample=('chain', 'draw'))
        if map:
            # select maximum probability sample
            logp = result.sample_stats.lp.stack(sample=('chain', 'draw'))
            i = np.argmax(logp.values)
        else:
            # pick random sample
            rng = rng or np.random.default_rng(seed)
            i = rng.integers(len(samples['sample']))
        sample = samples.isel(sample=i)
        pars = sample.data_vars
        return i, pars

    @property
    def whitened_templates(self) -> np.ndarray:
        """Whitened templates corresponding to each posterior sample, as
        were seen by the sampler.

        Dimensions will be ``(ifo, time, sample)``.

        Corresponding unwhitened templates can be obtained from posterior by
        doing::

          fit.result.posterior.h_det.stack(sample=('chain', 'draw'))
        """
        if self.result is None:
            return None
        # get reconstructions from posterior, shaped as (chain, draw, ifo, time)
        # and stack into (ifo, time, sample)
        hs = self.result.posterior.h_det.stack(samples=('chain', 'draw'))
        # whiten the reconstructions using the Cholesky factors, L, with shape
        # (ifo, time, time). the resulting object will have shape (ifo, time, sample)
        return np.linalg.solve(self.result.constant_data.L, hs)

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
        if self.result is None:
            raise RuntimeError("no results available")
        # get whitened reconstructions from posterior (ifo, time, sample)
        whs = self.whitened_templates
        # take the norm across time to get optimal snrs for each (ifo, sample)
        opt_ifo_snrs = np.linalg.norm(whs, axis=1)
        if optimal:
            snrs = opt_ifo_snrs
        else:
            # get analysis data, shaped as (ifo, time)
            if "strain" in self.result.observed_data:
                # strain values stored in "old" PyStan structure
                ds = self.result.observed_data.strain
            else:
                # strain values stored in "new" PyMC structure
                ds = np.array([d.values for d in self.result.observed_data.values()])
            # whiten it with the Cholesky factors, so shape will remain (ifo, time)
            wds = np.linalg.solve(self.result.constant_data.L, ds)
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
        accuarcy metric for the fit.
        
        See https://arxiv.org/abs/1507.04544 for definitions and discussion.  A
        larger WAIC indicates that the model has better predictive accuarcy on
        the fitted data set."""
        if self.result is None:
            raise RuntimeError("no results available")
        return az.waic(self.result, var_name='whitened_pointwise_loglike')
    
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
        if self.result is None:
            raise RuntimeError("no results available")
        return az.loo(self.result, var_name='whitened_pointwise_loglike')
