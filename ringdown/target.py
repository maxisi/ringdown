"""Module defining the core :class:`Target` class.
"""

__all__ = ['construct_target']

from ast import literal_eval
from .data import *
import lal
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

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