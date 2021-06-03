from pylab import *
from .data import *
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

class Fit(object):

    _compiled_models = {}

    def __init__(self, model=None):
        self.data = {}
        self.acfs = {}
        self.start_times = {}
        self.antenna_patterns = {}
        self.target = Target(None, None, None, None)
        self.model = model.lower() if model is not None else model
        
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
        
    def add_data(self, data, time=None, ifo=None, acf=None):
        if not isinstance(data, Data):
            data = Data(data, index=getattr(data, 'time', time),
                        ifo=ifo)
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
            acf[ifo] = self.data[ifo].get_acf(**kws) if acf is None else acf

    def set_target(self, t0, ra=None, dec=None, psi=None):
        tgps = lal.LIGOTimeGPS(t0)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        for ifo, data in self.data.items():
            if ifo is None:
                dt_ifo = 0
            else:
                dt_ifo = lal.TimeDelayFromEarthCenter(data.detector.location,
                                                      ra, dec, tgps)
                self.antenna_patterns[ifo] = lal.ComputeDetAMResponse(
                    data.detector.response, ra, dec, psi, gmst
                )
            self.start_times[ifo] = t0 + dt_ifo
        self.target = Target(t0, ra, dec, psi)
    
