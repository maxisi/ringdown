from pylab import *
from .data import *
from . import qnms
import lal
from collections import namedtuple
import pkg_resources

# def get_raw_time_ifo(tgps, raw_time, duration=None, ds=None):
#     ds = ds or 1
#     duration = inf if duration is None else duration
#     m = abs(raw_time - tgps) < 0.5*duration
#     i = argmin(abs(raw_time - tgps))
#     return roll(raw_time, -(i % ds))[m]

Target = namedtuple('Target', ['t0', 'ra', 'dec', 'psi'])

MODELS = ['ftau', 'kerr']

# TODO: might be better to subclass this
class Fit(object):

    _compiled_models = {}

    def __init__(self, model=None, modes=None, **kws):
        self.data = {}
        self.acfs = {}
        self.start_times = {}
        self.antenna_patterns = {}
        self.target = Target(None, None, None, None)
        self.model = model.lower() if model is not None else model
        try:
            # if modes is integer, interpret as number of modes
            self._nmodes = int(modes)
            self.modes = None
        except TypeError:
            # otherwise, assume it's mode index list
            self.modes = qnms.construct_mode_list(modes)
            self._nmodes = None
        # assume rest of kwargs are to be passed to stan_data (e.g. prior)
        self._model_input = kws
        
    @property
    def nmodes(self):
        return self._nmodes or len(self.modes)

    @property
    def _model(self):
        if self.model is None:
            raise ValueError('you must specify a model')
        elif self.model in self._compiled_models:
            # look for model in cache
            model = self._compiled_models[self.model]
        elif self.model in MODELS:
            # compile model and cache in class variable
            code = pkg_resources.resource_string(__name__,
                'stan/gwtones_{}.stan'.format(self.model)
            )
            import pystan
            model = pystan.StanModel(model_code=code.decode("utf-8"))
            self._compiled_models[self.model] = model
        else:
            raise ValueError('unrecognized model %r' % self.model)
        return model

    @property
    def ifos(self):
        return list(self.data.keys())

    @property
    def t0(self):
        return self.target.t0
        
    @property
    def sky(self):
        return (self.target.ra, self.target.dec, self.target.psi)

    # # this can be generalized for charged bhs based on model name
    # def _get_mode(self, i):
    #     index = self.modes[i]
    #     if index not in self._modes:
    #         self._modes[index] = qnms.KerrMode(index)
    #     return self._modes[index]

    @property
    def spectral_coefficients(self):
        f_coeffs = []
        g_coeffs = []
        for i in range(len(self.modes)):
            coeffs = qnms.KerrMode(i).coefficients
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
    def model_input(self):
        if not self.acfs:
            print('WARNING: computing ACFs with default settings.')
            self.compute_acfs()

        data_dict = self.analysis_data

        stan_data = dict(
            # data related quantities
            nsamp=self.n_analyze,
            nmode=self.nmodes,
            nobs=len(self.data),
            t0=list(self.start_times.values()),
            times=[d.time for d in data_dict.values()],
            strain=list(self.data_dict.values()),
            L=[acf[:self.n_analyze].cholesky for acf in self.acfs.values()], 
            # default priors
            dt_min=1E-6,
            dt_max=1E-6,
            only_prior=0,
        )

        if self.model == 'ftau':
            # TODO: set default priors based on sampling rate and duration
            pass
        elif self.model == 'kerr':
            stan_data.update(dict(
                perturb_f=zeros(self.nmodes),
                perturb_tau=zeros(self.nmodes),
                df_max=0.9,
                dtau_max=0.9,
                chi_min=0,
                chi_max=0.99,
            ))

        stan_data.update(self._model_input)
        return stan_data

    def add_data(self, data, time=None, ifo=None, acf=None):
        if not isinstance(data, Data):
            data = Data(data, index=getattr(data, 'time', time), ifo=ifo)
        self.data[data.ifo] = data
        if acf is not None:
            self.acfs[ifo] = acf

    def compute_acfs(self, shared=False, ifos=None, **kws):
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
        self.modes = qnms.construct_mode_list(indexes)
        
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
            if ifo is None:
                dt_ifo = 0
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

    @property
    def duration(self):
        if self._nanalyze and not self._duration:
            if self.data:
                return self._nanalyze*self.data[self.ifos[0]].delta_t
            else:
                print("Add data to compute duration (n_analyze = {})".format(
                      self._nanalyze)
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
            for i, d in self.data.items():
                t0 = self.start_times[ifo]
                i0_dict[i] = argmin(abs(d.time - t0))
        return i0_dict

    @property
    def n_analyze(self):
        if self._duration and not self._analyze:
            # set n_analyze based on specified duration in seconds
            if self.data:
                dt = self.data[self.ifos[0]].delta_t
                return int(round(self._duration/dt))
            else:
                print("Add data to compute n_analyze (duration = {})".format(
                      self._duration)
                return None
        elif self.data and self.has_target:
            # set n_analyze to fit shortest data set
            i0s = self.start_indices
            return min([len(d.iloc[i0s[i]:]) for i, d in self.data.items()])
        else:
            return self._n_analyze
