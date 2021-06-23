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
    def model_input(self):
        if not self.acfs:
            print('WARNING: computing ACFs with default settings.')
            self.compute_acfs()

        stan_data = dict(
            nmode=self.nmodes,
            nobs=len(self.data),
            times=[d.time for d in self.data.values()],
            strain=list(self.data.values()),
            L=[acf.cholesky for acf in self.acfs.values()],  # must check if acfs populated
        )

    @property
    def ifos(self):
        return list(self.data.keys())

    @property
    def t0(self):
        return self.target.t0
        
    @property
    def sky(self):
        return (self.target.ra, self.target.dec, self.target.psi)
        
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
        self.modes = qnms.construct_mode_list([(p, s, l, m, n) for n in range(nmode)])
        
    def set_target(self, t0, ra=None, dec=None, psi=None, delays=None,
                   antenna_patterns=None):
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
    
