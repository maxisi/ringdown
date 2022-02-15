__all__ = ['Target', 'Fit', 'MODELS']

import copy as cp
from pylab import *
from .data import *
from . import qnms
import lal
from collections import namedtuple
import pkg_resources
import arviz as az

# def get_raw_time_ifo(tgps, raw_time, duration=None, ds=None):
#     ds = ds or 1
#     duration = inf if duration is None else duration
#     m = abs(raw_time - tgps) < 0.5*duration
#     i = argmin(abs(raw_time - tgps))
#     return roll(raw_time, -(i % ds))[m]

Target = namedtuple('Target', ['t0', 'ra', 'dec', 'psi'])

MODELS = ('ftau', 'mchi', 'mchi_aligned')

class Fit(object):
    """ A ringdown fit.

    Attributes
    ----------
    model : str
        name of Stan model to be fit.
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
    spectral_coefficients : tuple
        tuple of arrays containing dimensionless frequency and damping rate
        fit coefficients to be passed internally to Stan model.
    model_data : dict
        arguments passed to Stan model internally.
    """


    _compiled_models = {}

    def __init__(self, model='mchi', modes=None, **kws):
        self.data = {}
        self.acfs = {}
        self.start_times = {}
        self.antenna_patterns = {}
        self.target = Target(None, None, None, None)
        if model.lower() in MODELS:
            self.model = model.lower()
        else:
            raise ValueError('invalid model {:s}; options are {}'.format(model,
                                                                         MODELS))
        self.result = None
        self.prior = None
        try:
            # if modes is integer, interpret as number of modes
            self._n_modes = int(modes)
            self.modes = None
        except TypeError:
            # otherwise, assume it's mode index list
            self.set_modes(modes)
            self._n_modes = None
        self._duration = None
        self._n_analyze = None
        # assume rest of kwargs are to be passed to stan_data (e.g. prior)
        self._prior_settings = kws

    @property
    def n_modes(self) -> int:
        """ Number of damped sinusoids to be included in template.
        """
        return self._n_modes or len(self.modes)

    @property
    def _model(self):
        if self.model is None:
            raise ValueError('you must specify a model')
        elif self.model not in self._compiled_models:
            if self.model in MODELS:
                self.compile()
            else:
                raise ValueError('unrecognized model %r' % self.model)
        return self._compiled_models[self.model]

    def compile(self, verbose=False, force=False):
        """ Compile `Stan` model.

        Arguments
        ---------
        verbose : bool
            print out all messages from compiler.
        force : bool
            force recompile.
        """
        if force or self.model not in self._compiled_models:
            # compile model and cache in class variable
            code = pkg_resources.resource_string(__name__,
                'stan/ringdown_{}.stan'.format(self.model)
            )
            import pystan
            kws = dict(model_code=code.decode("utf-8"))
            if not verbose:
                kws['extra_compile_args'] = ["-w"]
            self._compiled_models[self.model] = pystan.StanModel(**kws)

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
    def sky(self) -> tuple[float]:
        """ Tuple of source right ascension, declination and polarization
        angle (all in radians).
        """
        return (self.target.ra, self.target.dec, self.target.psi)

    # this can be generalized for charged bhs based on model name
    @property
    def spectral_coefficients(self):
        f_coeffs = []
        g_coeffs = []
        for mode in self.modes:
            coeffs = qnms.KerrMode(mode).coefficients
            f_coeffs.append(coeffs[0])
            g_coeffs.append(coeffs[1])
        return array(f_coeffs), array(g_coeffs)

    @property
    def analysis_data(self) -> dict[Data]:
        data = {}
        i0s = self.start_indices
        for i, d in self.data.items():
            data[i] = d.iloc[i0s[i]:i0s[i] + self.n_analyze]
        return data

    @property
    def _default_prior(self):
        default = {'A_scale': None,
                   'drift_scale': 0.1}
        if self.model == 'ftau':
            # TODO: set default priors based on sampling rate and duration
            default.update(dict(
                f_max=None,
                f_min=None,
                gamma_max=None,
                gamma_min=None
            ))
        elif self.model == 'mchi':
            default.update(dict(
                perturb_f=zeros(self.n_modes or 1),
                perturb_tau=zeros(self.n_modes or 1),
                df_max=0.5,
                dtau_max=0.5,
                M_min=None,
                M_max=None,
                chi_min=0,
                chi_max=0.99,
                flat_A_ellip=0
            ))
        elif self.model == 'mchi_aligned':
            default.update(dict(
                perturb_f=zeros(self.n_modes or 1),
                perturb_tau=zeros(self.n_modes or 1),
                df_max=0.5,
                dtau_max=0.5,
                M_min=None,
                M_max=None,
                chi_min=0,
                chi_max=0.99,
                cosi_min=-1,
                cosi_max=1,
                flat_A=0
            ))
        return default

    @property
    def prior_settings(self):
        prior = self._default_prior
        prior.update(self._prior_settings)
        return prior

    @property
    def valid_model_options(self):
        return list(self._default_prior.keys())

    # TODO: warn or fail if self.results is not None?
    def update_prior(self, **kws):
        valid_keys = self.valid_model_options
        for k, v in kws.items():
            if k in valid_keys:
                self._prior_settings[k] = v
            else:
                raise ValueError('{} is not a valid model argument.'
                                 'Valid options are: {}'.format(k, valid_keys))

    @property
    def model_input(self):
        if not self.acfs:
            print('WARNING: computing ACFs with default settings.')
            self.compute_acfs()

        data_dict = self.analysis_data

        stan_data = dict(
            # data related quantities
            nsamp=self.n_analyze,
            nmode=self.n_modes,
            nobs=len(data_dict),
            t0=list(self.start_times.values()),
            times=[d.time for d in data_dict.values()],
            strain=list(data_dict.values()),
            L=[acf.iloc[:self.n_analyze].cholesky for acf in self.acfs.values()],
            FpFc = list(self.antenna_patterns.values()),
            # default priors
            dt_min=-1E-6,
            dt_max=1E-6
        )

        if 'mchi' in self.model:
            f_coeff, g_coeff = self.spectral_coefficients
            stan_data.update(dict(
                f_coeffs=f_coeff,
                g_coeffs=g_coeff,
        ))

        stan_data.update(self.prior_settings)

        for k, v in stan_data.items():
            if v is None:
                raise ValueError('please specify {}'.format(k))
        return stan_data

    def copy(self):
        return cp.deepcopy(self)

    def condition_data(self, **kwargs):
        """ Condition data for all detectors.
        """
        new_data = {}
        for k, d in self.data.items():
            t0 = self.start_times[k]
            new_data[k] = d.condition(t0=t0, **kwargs)

        self.data = new_data
        self.acfs = {} # Just to be sure that these stay consistent

    def run(self, prior=False, **kws):
        """ Fit model.

        Arguments
        ---------
        prior : bool
            whether to sample the prior (def. False).

        additional kwargs are passed to pystan.model.sampling
        """
        # get model input
        stan_data = self.model_input
        stan_data['only_prior'] = int(prior)
        # get sampler settings
        n = kws.pop('thin', 1)
        chains = kws.pop('chains', 4)
        n_jobs = kws.pop('n_jobs', chains)
        n_iter = kws.pop('iter', 2000*n)
        metric = kws.pop('metric', 'dense_e')
        stan_kws = {
            'iter': n_iter,
            'thin': n,
            'init': (kws.pop('init_dict', {}),)*chains,
            'n_jobs': n_jobs,
            'chains': chains,
            'control': {'metric': metric}
        }
        stan_kws.update(kws)
        # run model and store
        print('Running {}'.format(self.model))
        result = self._model.sampling(data=stan_data, **stan_kws)
        if prior:
            self.prior = az.convert_to_inference_data(result)
        else:
            self.result = az.convert_to_inference_data(result)

    def add_data(self, data, time=None, ifo=None, acf=None):
        if not isinstance(data, Data):
            data = Data(data, index=getattr(data, 'time', time), ifo=ifo)
        self.data[data.ifo] = data
        if acf is not None:
            self.acfs[data.ifo] = acf

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
        # if shared, compute a single ACF
        acf = self.data[ifos[0]].get_acf(**kws) if shared else None
        for ifo in ifos:
            self.acfs[ifo] = acf if shared else self.data[ifo].get_acf(**kws)

    def set_tone_sequence(self, nmode, p=1, s=-2, l=2, m=2):
        """ Set template modes to be a sequence of overtones with a given
        angular structure.

        To set an arbitrary set of modes, use :meth:`ringdown.fit.Fit.set_modes`

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

    def set_modes(self, modes):
        """ Establish list of modes to include in analysis template.

        Modes identified by their `(p, s, l, m, n)` indices, where:
          - `p` is `1` for prograde modes, and `-1` for retrograde modes;
          - `s` is the spin-weight (`-2` for gravitational waves);
          - `l` is the azimuthal quantum number;
          - `m` is the magnetic quantum number;
          - `n` is the overtone number.

        Arguments
        ---------
        modes : list
            list of tuples with quasinormal mode `(p, s, l, m, n)` numbers.
        """
        self.modes = qnms.construct_mode_list(modes)
        if self.model == 'mchi_aligned':
            ls_valid = [mode.l == 2 for mode in self.modes]
            ms_valid = [abs(mode.m) == 2 for mode in self.modes]
            if not (all(ls_valid) and all(ms_valid)):
                raise ValueError("mchi_aligned model only accepts l=m=2 modes")

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
            number of datapoints to include in analysis segment.
        delays : dict
            dictionary with delayes from geocenter for each detector, as would
            be computed by `lal.TimeDelayFromEarthCenter` (optional).
        antenna_patterns : dict
            dictionary with tuples for plus and cross antenna patterns for
            each detector `{ifo: (Fp, Fc)}` (optional)
        """
        tgps = lal.LIGOTimeGPS(t0)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        delays = delays or {}
        antenna_patterns = antenna_patterns or {}
        for ifo, data in self.data.items():
            # TODO: should we have an elliptical+ftau model?
            if ifo is None or self.model=='ftau':
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

    # TODO: warn or fail if self.results is not None?
    def update_target(self, **kws):
        target = dict(**self.target)
        target.update({k: getattr(self,k) for k in
                       ['duration', 'n_analyze', 'antenna_patterns']})
        target.update(kws)
        self.set_target(**target)

    @property
    def duration(self):
        if self._n_analyze and not self._duration:
            if self.data:
                return self._n_analyze*self.data[self.ifos[0]].delta_t
            else:
                print("Add data to compute duration (n_analyze = {})".format(
                      self._n_analyze))
                return None
        else:
            return self._duration

    @property
    def has_target(self):
        return self.target.t0 is not None

    @property
    def start_indices(self):
        i0_dict = {}
        if self.has_target:
            for ifo, d in self.data.items():
                t0 = self.start_times[ifo]
                i0_dict[ifo] = argmin(abs(d.time - t0))
        return i0_dict

    @property
    def n_analyze(self):
        if self._duration and not self._n_analyze:
            # set n_analyze based on specified duration in seconds
            if self.data:
                dt = self.data[self.ifos[0]].delta_t
                return int(round(self._duration/dt))
            else:
                print("Add data to compute n_analyze (duration = {})".format(
                      self._duration))
                return None
        elif self.data and self.has_target:
            # set n_analyze to fit shortest data set
            i0s = self.start_indices
            return min([len(d.iloc[i0s[i]:]) for i, d in self.data.items()])
        else:
            return self._n_analyze

    def whiten(self, datas, drifts=None):
        if drifts is None:
            drifts = {i : 1 for i in datas.keys()}
        return {i: Data(self.acfs[i].whiten(d, drift=drifts[i]), ifo=i) for i,d in datas.items()}
        
