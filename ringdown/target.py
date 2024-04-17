"""Module defining the core :class:`Target` class.
"""

__all__ = ['construct_target']

import numpy as np
import lal
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from .utils import load_config, try_parse
from .qnms import T_MSUN

class Target(ABC):
    def as_dict(self) -> dict:
        return asdict(self)
    
    @abstractmethod
    def get_detector_time(self, ifo) -> float | None:
        pass
    
    @abstractmethod
    def get_antenna_patterns(self, ifo) -> tuple[float, float] | None:
        pass
    
    @property
    @abstractmethod
    def t0(self) -> float | None:
        pass
    
    @property
    @abstractmethod
    def sky(self) -> tuple[float | None]:
        pass
    
    @property
    def has_sky(self) -> bool:
        return self.sky[0] is not None
    
    @property
    def is_set(self) -> bool:
        return any([x is not None for x in self.as_dict().values()])
    
    def get_detector_times_dict(self, ifos):
        return {ifo: self.get_detector_time(ifo) for ifo in ifos}
    
    def get_antenna_patterns_dict(self, ifos):
        return {ifo: self.get_antenna_patterns(ifo) for ifo in ifos}
    
@dataclass
class SkyTarget(Target):
    geocenter_time : float | None = None
    ra : float | None = None
    dec : float | None = None
    psi : float | None = None
    
    def __post_init__(self):
        # validate input: floats or None
        for k, v in self.as_dict().items():
            if v is not None and not isinstance(v, lal.LIGOTimeGPS):
                setattr(self, k, float(v))
        # make sure options are not contradictory                   
        if self.is_set:
            for k,v in self.as_dict().items():
                if v is None:
                    raise ValueError(f"missing {k}")
                    
    @property
    def t0(self):
        """Alias for time_geocenter."""
        if self.geocenter_time is None:
            return None
        else:
            return float(self.geocenter_time)
    
    @property
    def sky(self):
        return (self.ra, self.dec, self.psi)
    
    def get_detector_time(self, ifo):
        """Compute detector times based on sky location.
        
        Arguments
        ---------
        ifos : list
            list of detector names.
        
        Returns
        -------
        times : dict
            dictionary of detector times.
        """
        det = lal.cached_detector_by_prefix[ifo]
        tgps = lal.LIGOTimeGPS(self.geocenter_time)
        dt = lal.TimeDelayFromEarthCenter(det.location, self.ra, 
                                            self.dec, tgps)
        t0 = self.geocenter_time + dt
        return float(t0)
    
    def get_antenna_patterns(self, ifo):
        """Compute antenna patterns based on sky location.
        
        Arguments
        ---------
        ifos : list
            list of detector names.
        
        Returns
        -------
        antenna_patterns : dict
            dictionary of antenna patterns.
        """
        det = lal.cached_detector_by_prefix[ifo]
        tgps = lal.LIGOTimeGPS(self.geocenter_time)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        fpfc = lal.ComputeDetAMResponse(det.response, self.ra, self.dec,
                                        self.psi, gmst)
        return fpfc
    
    @classmethod
    def construct(cls, t0 : float, ra : float, dec : float, psi : float,
                 reference_ifo : str | None = None):
        """Create a sky location from a reference time, either a specific
        detector or geocenter.
        
        Arguments
        ---------
        t0 : float
            detector time.
        ra : float
            source right ascension.
        dec : float
            source declination.
        psi : float
            source polarization angle.
        reference_ifo : str, None
            detector name, or `None` for geocenter (default `None`)
        
        Returns
        -------
        sky : Target
            a target object.
        """
        if reference_ifo is None:
            tgeo = t0
        else:
            det = lal.cached_detector_by_prefix[reference_ifo]
            tgps = lal.LIGOTimeGPS(t0)
            dt = lal.TimeDelayFromEarthCenter(det.location, ra, dec, tgps)
            tgeo = t0 - dt
        return cls(lal.LIGOTimeGPS(tgeo), ra, dec, psi)
    
@dataclass
class DetectorTarget(Target):
    detector_times : dict | None = None
    antenna_patterns : dict | None = None
    
    def __post_init__(self):
        # validate input: floats or None
        for k, v in self.as_dict().items():
            if v is not None:
                if k == 'detector_times':
                    self.detector_times = {i: float(v) for i,v in v.items()}
                elif k == 'antenna_patterns':
                    aps = {}
                    for i, fpfc in v.items():
                        if len(i) != 2:
                            raise ValueError("antenna patterns must be (Fp, Fc)")
                        aps[k] = (float(fpfc[0]), float(fpfc[1]))
                    self.antenna_patterns = aps
        # make sure options are not contradictory                   
        if self.is_set:
            for k, v in self.as_dict().items():
                if v is None:
                    raise ValueError(f"missing {k}")
    
    @property
    def t0(self):  
        return None
    
    @property
    def sky(self):
        return (None, None, None)
    
    def get_antenna_patterns(self, ifo):
        return self.antenna_patterns[ifo]
    
    def get_detector_time(self, ifo):
        return self.detector_times[ifo]
    
    @classmethod
    def construct(cls, detector_times, antenna_patterns):
        # TODO: merge this with __post_init__
        if not hasattr(antenna_patterns, 'keys'):
            # assume antenna_patterns is (Fp, Fc); will check below
            antenna_patterns = {None: antenna_patterns}
        else:
            pass
        # antenna patterns have been explicitly provided, validate their
        # structure and store for later
        _antenna_patterns = {}
        for i in antenna_patterns.keys():
            ap = antenna_patterns[i]
            if len(ap) != 2:
                raise ValueError("antenna patterns must be (Fp, Fc)")
            _antenna_patterns[i] = (float(ap[0]), float(ap[1]))

        if not hasattr(detector_times, 'keys'):
            # assume t0 is a single time, will check below
            logging.warning("setting same start time for all detectors")
            detector_times = {i: float(detector_times) 
                          for i in _antenna_patterns.keys()}
        else:
            pass
        # construct start-time dictionary based on antenna patterns    
        _detector_times = {}
        for i in _antenna_patterns.keys():
            if i in detector_times:
                _detector_times[i] = float(detector_times[i])
            else:
                raise ValueError(f"missing start time for {i}")
        # check that there were no extra start times
        extra_times = set(_detector_times.keys()) -\
                        set(_antenna_patterns.keys())
        if extra_times:
            raise ValueError("detectors without antenna patterns: "
                             f"{extra_times}")
        else:
            pass
        return cls(_detector_times, _antenna_patterns)
    
def construct_target(t0 : float | dict, ra : float | None = None,
                    dec : float | None = None, psi : float | None = None,
                    reference_ifo : str | None = None,
                    antenna_patterns: dict | None = None, **kws):
    if antenna_patterns is None:
        return SkyTarget.construct(t0, ra, dec, psi, reference_ifo)
    else:
        return DetectorTarget.construct(t0, antenna_patterns)
    
class TargetCollection(object):
    def __init__(self, targets : list, index=None, info=None):
        for target in targets:
            if not isinstance(target, Target):
                raise ValueError("targets must be instances of Target")
        self.targets = targets
        self.info = info or {}
        self._reference_time = None
        self._reference_mass = None
        self._index = None
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.targets})"
    
    def __getitem__(self, i):
        return self.targets[i]
    
    def __len__(self):
        return len(self.targets)
    
    def get(self, key):
        if key.lower() == 'delta-t0':
            return np.array(self.get('t0')) - self.reference_time
        elif key.lower() == 'delta-m':
            return np.array(self.get('delta-t0')) / self.step_time
        return [getattr(t, key) for t in self.targets]
    
    def get_detector_times(self, ifo):
        return [t.get_detector_time(ifo=ifo) for t in self.targets]
    
    def get_antenna_patterns(self, ifo):
        return [t.get_antenna_patterns(ifo=ifo) for t in self.targets]
    
    def update_info(self, section: str, **kws) -> None:
        """Update fit information stored in :attr:`TargetCollection.info`
        """
        self.info[section] = self.info.get(section, {})
        self.info[section].update(**kws)
    
    @property
    def index(self):
        if self._index is None:
            self._index = [t.t0 for t in self.targets]
            if any([t is None for t in self._index]):
                self._index = None
        return self._index
    
    @property
    def reference_time(self):
        if self._reference_time is None:
            self._reference_time = self.info.get('t0-ref', self.info.get('pipe', {}).get('t0-ref', 0.))
        return self._reference_time
    
    @property
    def reference_mass(self):
        if self._reference_mass is None:
            self._reference_mass = self.info.get('m-ref', self.info.get('pipe', {}).get('m-ref', None))
        return self._reference_mass
    
    @property
    def _step(self):
        return self.info.get('t0-step', self.info.get('pipe', {}).get('t0-step', None))
    
    @property
    def step_time(self):
        mref = self.reference_mass
        if mref:
            tstep = self.step_mass * T_MSUN
        else:
            tstep= self._step
        return tstep
    
    @property
    def step_mass(self):
        mref = self.reference_mass
        if mref:
            mstep = self._step *  mref
        else:
            mstep = None
        return mstep
    
    @classmethod
    def from_config(cls, config_input, t0_sect='pipe', sky_sect='target'):
        """Identify target analysis times. There will be three possibilities:
            1- listing the times explicitly
            2- listing time differences with respect to a reference time
            3- providing start, stop, step instructions to construct start times
               (potentially relative to a reference time)
        Time steps/differences can be specified in seconds or M, if a reference mass
        is provided (in solar masses).
        """
        config = load_config(config_input)

        # Define valid options to specify the start times
        T0_KEYS = {
            'ref': 't0-ref',
            'list': 't0-list',
            'delta': 't0-delta-list',
            'step': 't0-step',
            'start': 't0-start',
            'stop': 't0-stop'
        }
        start_stop_step = [T0_KEYS[k] for k in ['start', 'stop', 'step']]

        # First make sure that only compatible t0 options were provided
        incompatible_sets = [['ref', 'list'], ['delta', 'list']]
        incompatible_sets += [[k, 'delta'] for k in ['start', 'stop', 'step']]
        for bad_set in incompatible_sets:
            opt_names = [T0_KEYS[k] for k in bad_set]
            if all([k in config[t0_sect] for k in opt_names]):
                raise ValueError("incompatible T0 options: {}".format(opt_names))

        # Look for a reference mass, to be used when stepping in time
        m_ref = config.getfloat(t0_sect, 'M-ref', fallback=None)
        if m_ref:
            # reference time translating from solar masses
            tm_ref = m_ref * T_MSUN
            logging.info("Reference mass: {} Msun ({} s)".format(m_ref, tm_ref))
        else:
            # no reference mass provided, so will default to seconds
            tm_ref = 1

        # Look for reference time to be used to construct start times
        t0ref = config.getfloat(t0_sect, T0_KEYS['ref'], fallback=0)

        # Now we can safely interpret the options assuming one of three cases
        if T0_KEYS['list'] in config[t0_sect]:
            t0s = np.array(try_parse(config.get(t0_sect, T0_KEYS['list'])))
        elif T0_KEYS['delta'] in config[t0_sect]:
            dt0s = np.array(try_parse(config.get(t0_sect, T0_KEYS['delta'])))
            t0s = dt0s*tm_ref + t0ref
        elif any([k in config[t0_sect] for k in start_stop_step]):
            if not all([k in config[t0_sect] for k in start_stop_step]):
                missing = [k for k in start_stop_step if k not in config[t0_sect]]
                raise ValueError("missing start/stop/step options: {}".format(missing))
            # add a safety check here, in case the user mistakenly requests stepping
            # based on a GPS time and provides a reference GPS time
            start, stop, step  = [config.getfloat(t0_sect, k) for k in start_stop_step]
            if start > 500 and t0ref > 1E8:
                logging.warning("high reference time and stepping start---did you "
                                "accidentally provide GPS times twice?")
            t0s = np.arange(start, stop, step)*tm_ref + t0ref
        else:
            raise ValueError("no timing instructions in [{}] section; valid options "
                            "are: {}".format(t0_sect, list(T0_KEYS.values())))
            
        # get sky location arguments if provided
        sky_dict = {k.lower(): try_parse(v) for k,v in config[sky_sect].items()}
        
        targets = [construct_target(t0, **sky_dict) for t0 in t0s]
        info = {
            t0_sect : {k.lower(): try_parse(v) for k,v in config[t0_sect].items()},
            sky_sect : sky_dict
        }
        
        return cls(targets, info=info)
        