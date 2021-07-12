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

MODELS = ('ftau', 'mchi')

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
    sky: tuple
        tuple with source right ascension, declination and polarization angle.
    analysis_data : dict
        dictionary of truncated analysis data that will be fed to Stan model.
    spectral_coefficients: tuple
        tuple of arrays containing dimensionless frequency and damping rate
        fit coefficients to be passed internally to Stan model.
    model_data: dict
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
            self.modes = qnms.construct_mode_list(modes)
            self._n_modes = None
        self._duration = None
        self._n_analyze = None
        # assume rest of kwargs are to be passed to stan_data (e.g. prior)
        self._prior_settings = kws

    @property
    def n_modes(self):
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

    def compile(self, force=False):
        if force or self.model not in self._compiled_models:
            # compile model and cache in class variable
            code = pkg_resources.resource_string(__name__,
                'stan/ringdown_{}.stan'.format(self.model)
            )
            import pystan
            model = pystan.StanModel(model_code=code.decode("utf-8"))
            self._compiled_models[self.model] = model

    @property
    def ifos(self):
        return list(self.data.keys())

    @property
    def t0(self):
        return self.target.t0

    @property
    def sky(self):
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
    def analysis_data(self):
        data = {}
        i0s = self.start_indices
        for i, d in self.data.items():
            data[i] = d.iloc[i0s[i]:i0s[i] + self.n_analyze]
        return data

    @property
    def _default_prior(self):
        default = {'A_scale': None}
        if self.model == 'ftau':
            # TODO: set default priors based on sampling rate and duration
            default.update(dict(
                f_max=None,
                f_min=None,
                gamma_max=None,
                gamma_min=None,
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

        if self.model == 'mchi':
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
        return cp.copy(self)

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
        """Compute ACFs for all data sets in Fit.

        Arguments
        ---------
        shared: bool
            specifices if all IFOs are to share a single ACF, in which case the
            ACF is only computed once from the data of the first IFO (useful
            for simulated data) (default False)

        ifos: list
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
        """ Set fit modes to be a sequence of overtones.
        """
        indexes = [(p, s, l, m, n) for n in range(nmode)]
        self.set_modes(indexes)

    def set_modes(self, modes):
        self.modes = qnms.construct_mode_list(modes)

    def set_target(self, t0, ra=None, dec=None, psi=None, delays=None,
                   antenna_patterns=None, duration=None, n_analyze=None):
        """ Establish truncation target, stored to `self.target`.

        Arguments
        ---------
        t0: float
            target time (at geocenter for a detector network)
        ra: float
            source right ascension
        dec: float
            source declination
        delays: dict
            dictionary with delayes from geocenter for each detector, as would
            be computed by `lal.TimeDelayFromEarthCenter` (optional)
        antenna_patterns: dict
            dictionary with tuples for plus and cross antenna patterns for
            each detector {ifo: (Fp, Fc)} (optional)
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
            self._nanalyze = int(n_analyze)

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

    def whiten(self, tseries_dict):
        wtseries = {}

        for ifo, ts in tseries_dict.items():
            L = self.acfs[ifo].iloc[:self.n_analyze].cholesky

            wtseries[ifo] = np.linalg.solve(L, ts)

        return wtseries

