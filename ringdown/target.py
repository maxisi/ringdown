"""Module defining the core :class:`Target` class.
"""

__all__ = ['Target', 'SkyTarget', 'DetectorTarget', 'TargetCollection']

import numpy as np
import lal
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from .utils import utils
from .utils.utils import try_parse
from .config import T_MSUN, IMR_CONFIG_SECTION

# Define valid options to specify the start times
T0_KEYS = {
    'ref': 't0-ref',
    'list': 't0-list',
    'delta': 't0-delta-list',
    'step': 't0-step',
    'start': 't0-start',
    'stop': 't0-stop'
}
T0_KWARGS = {k: v.replace('-', '_') for k, v in T0_KEYS.items()}
START_STOP_STEP = [T0_KEYS[k] for k in ['start', 'stop', 'step']]
T0_INCOMPATIBLE_OPTS = [['ref', 'list'], ['delta', 'list']]
T0_INCOMPATIBLE_OPTS += [[k, 'delta'] for k in ['start', 'stop', 'step']]
MREF_KEY = 'm-ref'
TREF_KEY = T0_KEYS['ref']
PIPE_SEC = 'pipe'


class Target(ABC):
    """Abstract class to define a target for a ringdown analysis. A target can
    be defined either by a sky location or by detector times and antenna
    patterns. Use the `construct` method to create a target object from a
    dictionary or keyword arguments.
    """

    duration: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __eq__(self, other) -> bool:
        if isinstance(other, Target):
            return self.as_dict() == other.as_dict()
        else:
            return False

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
    def settings(self) -> dict:
        return self.as_dict()

    @property
    def is_set(self) -> bool:
        return any([x is not None for x in self.as_dict().values()])

    def get_detector_times_dict(self, ifos) -> dict:
        return {ifo: self.get_detector_time(ifo) for ifo in ifos}

    def get_antenna_patterns_dict(self, ifos) -> dict:
        return {ifo: self.get_antenna_patterns(ifo) for ifo in ifos}

    @staticmethod
    def construct(t0: float | dict, ra: float | None = None,
                  dec: float | None = None, psi: float | None = None,
                  reference_ifo: str | None = None,
                  antenna_patterns: dict | None = None, **kws):
        """Create a target object from a dictionary or keyword arguments.
        The source sky location and orientation can be specified by the `ra`,
        `dec`, and `psi` arguments. These are use to both determine the
        truncation time at different detectors, as well as to compute the
        corresponding antenna patterns.

        Alternatively, analysis times and antenna patterns can be specified
        directly for each detector. In this case, the `t0` argument should be a
        dictionary with detector names as keys and start times as values.

        Cannot simultaneously provide both sky location and detector times.

        Arguments
        ---------
        t0 : float, dict
            start time for the analysis, either a single time or a dictionary
            of times for each detector.
        ra : float, None
            source right ascension.
        dec : float, None
            source declination.
        psi : float, None
            source polarization angle.
        reference_ifo : str, None
            detector name for time reference, or `None` for geocenter (default
            `None`)
        antenna_patterns : dict, None
            dictionary of antenna patterns for each detector, or `None` to
            compute from sky location.
        """
        if ra is None:
            return DetectorTarget.construct(t0, antenna_patterns, **kws)
        else:
            return SkyTarget.construct(t0, ra, dec, psi, reference_ifo, **kws)


@dataclass
class SkyTarget(Target):
    """Sky location target for a ringdown analysis.
    """
    geocenter_time: float | lal.LIGOTimeGPS | None = None
    ra: float | None = None
    dec: float | None = None
    psi: float | None = None
    duration: float = 0

    def __post_init__(self):
        # validate input: floats or None
        for k, v in self.as_dict().items():
            if v is not None and not isinstance(v, lal.LIGOTimeGPS):
                setattr(self, k, float(v))
        # make sure options are not contradictory
        if self.is_set:
            for k, v in self.as_dict().items():
                if v is None:
                    raise ValueError(f"missing {k}")

    @property
    def t0(self) -> float | None:
        """Alias for time_geocenter."""
        if self.geocenter_time is None:
            return None
        else:
            return float(self.geocenter_time)

    @property
    def sky(self) -> tuple[float | None]:
        """Return sky location as a tuple (ra, dec, psi)."""
        return (self.ra, self.dec, self.psi)

    def get_detector_time(self, ifo) -> float:
        """Compute detector times based on sky location.

        Arguments
        ---------
        ifo : list
            detector name.

        Returns
        -------
        times : float
            time for specified detector.
        """
        if ifo in lal.cached_detector_by_prefix:
            det = lal.cached_detector_by_prefix[ifo]
        else:
            raise ValueError(f"unrecognized detector {ifo}")
        tgps = lal.LIGOTimeGPS(self.geocenter_time)
        dt = lal.TimeDelayFromEarthCenter(det.location, self.ra,
                                          self.dec, tgps)
        t0 = self.geocenter_time + dt
        return float(t0)

    def get_antenna_patterns(self, ifo) -> tuple[float, float]:
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
        if ifo in lal.cached_detector_by_prefix:
            det = lal.cached_detector_by_prefix[ifo]
        else:
            raise ValueError(f"unrecognized detector {ifo}")
        tgps = lal.LIGOTimeGPS(self.geocenter_time)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        fpfc = lal.ComputeDetAMResponse(det.response, self.ra, self.dec,
                                        self.psi, gmst)
        return tuple(fpfc)

    @classmethod
    def construct(cls, t0: float, ra: float, dec: float, psi: float,
                  reference_ifo: str | None = None, duration: float = 0.,
                  **kws):
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
        duration : float
            analysis duration (default 0.)

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
        if kws:
            logging.info(f"unused keyword arguments: {kws}")
        return cls(lal.LIGOTimeGPS(tgeo), ra, dec, psi, duration)

    @property
    def settings(self) -> dict:
        """Return a dictionary of settings for the target."""
        s = {k: v for k, v in self.as_dict().items() if 'time' not in k}
        s['t0'] = self.t0
        return s


@dataclass
class DetectorTarget(Target):
    """Target for a ringdown analysis defined from explicit detector times and
    antenna patterns, rather than a sky location and a geocenter time.
    """
    detector_times: dict | None = None
    antenna_patterns: dict | None = None
    duration: float = 0

    def __post_init__(self):

        # validate antenna patterns
        if not hasattr(self.antenna_patterns, 'keys'):
            # assume antenna_patterns is (Fp, Fc); will check below
            _antenna_patterns = {None: self.antenna_patterns}
        else:
            # make sure antenna patterns is a dictionary
            _antenna_patterns = {k: self.antenna_patterns[k]
                                 for k in self.antenna_patterns.keys()}
        for i, fpfc in _antenna_patterns.items():
            if fpfc is None:
                if len(_antenna_patterns) > 1 or i is not None:
                    logging.warning(
                        f"defaulting to (1, 0) antenna patterns for {i} ifo")
                else:
                    logging.info("defaulting to (1, 0) antenna patterns")
                fpfc = (1, 0)
            if len(fpfc) != 2:
                raise ValueError("antenna patterns must be (Fp, Fc)")
            _antenna_patterns[i] = (float(fpfc[0]), float(fpfc[1]))
        self.antenna_patterns = _antenna_patterns

        # validate times
        if not hasattr(self.detector_times, 'keys'):
            # assume t0 is a single time, will check below
            if len(_antenna_patterns) > 1:
                logging.warning("setting same start time for all detectors")
            self.detector_times = {i: float(self.detector_times)
                                   for i in _antenna_patterns.keys()}
        else:
            pass

        # construct start-time dictionary based on antenna patterns
        _detector_times = {}
        for i in _antenna_patterns.keys():
            if i in self.detector_times:
                _detector_times[i] = float(self.detector_times[i])
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
        self.detector_times = _detector_times

    @property
    def t0(self) -> None:
        """Geocenter reference time is `None`"""
        return None

    @property
    def sky(self) -> tuple[None]:
        """Sky location is `(None, None, None)`"""
        return (None, None, None)

    def get_antenna_patterns(self, ifo) -> tuple[float, float]:
        if ifo in self.antenna_patterns:
            return self.antenna_patterns[ifo]
        else:
            raise ValueError(f"antenna patterns unavailable for {ifo}; "
                             "you might need to reset the target.")

    def get_detector_time(self, ifo) -> float:
        if ifo in self.detector_times:
            return self.detector_times[ifo]
        else:
            raise ValueError(f"start time unavailable for {ifo}; "
                             "you might need to reset the target.")

    @classmethod
    def construct(cls, detector_times: dict | float | list,
                  antenna_patterns: dict | list[tuple[float]] |
                  tuple[float, float], ifos: list | str = None,
                  duration: float = 0.):
        """Create a target object from a set of detector times and antenna
        patterns.

        The preferred input is two dictionaries of detector times and antenna
        pattern (Fp, Fc) tuples indexed by interferometer name. Alternatively,
        lists of values can be provided for both `detector_times` and
        `antenna_patterns` to be indexed by a list of interferometer names
        `ifo`.

        If a single time is provided, it will be used for all detectors. If a
        single tuple is provided for antenna patterns, it will be used if only
        a single detector was specified through `ifos`.

        If no detector names are specified as either keys of `antenna_patterns`
        or the `ifos` argument, then will assume a single detector named
        `None`.
        """
        if isinstance(ifos, str):
            ifos = [ifos]
        if ifos and antenna_patterns is None:
            # ifos have been provided but no antenna patterns
            antenna_patterns = {ifo: None for ifo in ifos}
            if len(ifos) == 1 and not ifos[0] is None:
                # the IFO has a distinct name, warn about this
                logging.info(
                    "assuming the unlabeled antenna patterns were meant "
                    f"for {ifos[0]}")
        elif ifos and not hasattr(antenna_patterns, 'keys'):
            # ifos have been provided and antenna_patterns is not dict-like
            if antenna_patterns is not None:
                # check whether antenna patterns is a tuple or a list of tuples
                if len(antenna_patterns) == 2 and len(ifos) == 1:
                    # this is ambiguous: we can have (Fp, Fc), or [(Fp, Fc),
                    # (Fp, Fc)] check that antenna patterns is tuple (Fp, Fc)
                    # by trying to convert elements to float
                    try:
                        antenna_patterns = \
                            {ifos[0]: [float(a) for a in antenna_patterns]}
                        if ifos[0] is not None:
                            # the IFO has a distinct name, warn about this
                            logging.info(
                                "assuming the unlabeled antenna patterns "
                                f"were meant for {ifos[0]}")
                    except TypeError:
                        # the antenna patterns are not a tuple of floats,
                        # so it may be an list of tuples with length ifos
                        pass
        # check again and construct AP dictionary if necessary
        if ifos and antenna_patterns and not hasattr(antenna_patterns, 'keys'):
            if len(ifos) == len(antenna_patterns):
                antenna_patterns = dict(zip(ifos, antenna_patterns))
                if not hasattr(detector_times, 'keys'):
                    # check if detector_times is also a list of values
                    try:
                        if len(detector_times) == len(antenna_patterns):
                            detector_times = dict(zip(ifos, detector_times))
                        else:
                            raise ValueError("detector_times and "
                                             "antenna_patterns "
                                             "must have the same length")
                    except TypeError:
                        pass
            elif antenna_patterns is not None and ifos is not None:
                raise ValueError("antenna patterns must be a dictionary, "
                                 "a list of tuples with the same length as "
                                 "ifos, or a single tuple for a single ifo")
        elif ifos:
            logging.info("ignoring ifos argument")
        return cls(detector_times, antenna_patterns, duration)


class TargetCollection(utils.MultiIndexCollection):
    """Collection of targets for a ringdown analysis. The collection can be
    initialized with a list of `Target` objects, or by specifying the start
    times and sky locations for each target. The collection can also store a
    reference time to label start times, as well as a reference mass (in solar
    masses) to label time steps in units of mass. The collection can be indexed
    by start times, and can provide time differences with respect to the
    reference time, or time steps in units of mass."""

    def __init__(self, targets: list | None = None, index=None,
                 reference_mass=None, reference_time=None, info=None):
        if targets is None:
            targets = []
        if all([t is None for t in targets]):
            targets = []
        for target in targets:
            if not isinstance(target, Target):
                raise ValueError("targets must be instances of Target")
        super().__init__(targets, index=index, reference_mass=reference_mass,
                         reference_time=reference_time, info=info)
        self._mref_key = MREF_KEY
        self._tref_key = TREF_KEY

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.targets})"

    @property
    def targets(self) -> list:
        """List of targets in the collection."""
        return self.data

    def get(self, key) -> list | np.ndarray:
        """Get attribute `key` for each target in the collection.
        Special keys include:
        - `delta-t0`: time differences with respect to the reference time
        - `delta-m`: time differences in units of reference mass

        Arguments
        ---------
        key : str
            attribute name.

        Returns
        -------
        values : list
            list of attribute values.
        """
        if key.lower() == 'delta-t0':
            t0 = 0 if self.reference_time is None else self.reference_time
            return np.array(self.get('t0')) - t0
        elif key.lower() == 'delta-m':
            if self.step_time:
                return self.get('delta-t0') / self.step_time
            elif self.reference_mass and self.reference_time:
                return self.get('delta-t0') / self.reference_mass_seconds
            else:
                return [None] * len(self)
        return [getattr(t, key) for t in self.targets]

    def get_detector_times(self, ifo) -> list[float]:
        """Get detector times for each target in the collection."""
        return [t.get_detector_time(ifo=ifo) for t in self.targets]

    def get_antenna_patterns(self, ifo) -> list[tuple[float, float]]:
        """Get antenna patterns for each target in the collection."""
        return [t.get_antenna_patterns(ifo=ifo) for t in self.targets]

    def update_info(self, section: str, **kws) -> None:
        """Update fit information stored in :attr:`TargetCollection.info`
        """
        self.info[section] = self.info.get(section, {})
        self.info[section].update(**kws)

    @property
    def t0(self) -> np.ndarray:
        """Start times for each target in the collection."""
        return np.array(self.get('t0'))

    @property
    def t0m(self) -> np.ndarray:
        """Start times relative to the reference time in units of mass."""
        return self.get('delta-m')

    @property
    def index(self) -> list:
        """Index for the collection; defaults to start times if none
        provided."""
        if self._index is None:
            self._index = [t.t0 for t in self.targets]
            if any([t is None for t in self._index]):
                self._index = None
        return self._index

    @property
    def _step(self) -> float | None:
        tdef = self.info.get(PIPE_SEC, {}).get('t0-step', None)
        return self.info.get('t0-step', tdef)

    @property
    def step_time(self) -> float | None:
        """Time step for the collection, in seconds."""
        mref = self.reference_mass
        if mref and self.step_mass:
            tstep = self.step_mass * T_MSUN
        else:
            tstep = self._step
        return tstep

    @property
    def step_mass(self) -> float | None:
        """Time step for the collection, in units of mass."""
        mref = self.reference_mass
        if mref and self._step:
            mstep = self._step * mref
        else:
            mstep = None
        return mstep

    @classmethod
    def construct(cls, *args, reference_mass: float | None = None,
                  info: dict | None = None, **kws) -> 'TargetCollection':
        """Construct a collection of targets from a set of keyword arguments.
        There are three ways to specify the start times:
            1- listing the times explicitly
            2- listing time differences with respect to a reference time
            3- providing start, stop, step instructions to construct start
            times (potentially relative to a reference time)
        Time steps/differences can be specified in seconds or M, if a
        reference mass is provided (in solar masses).

        Accepted arguments to set target time are:
        - `t0_list`: list of start times
        - `t0_delta_list`: list of time differences with respect to a reference
        - `t0_ref`: reference time to be used for time differences
        - `t0_start`, `t0_stop`, `t0_step`: start, stop, step instructions

        Additional arguments can be provided to specify the sky location and
        duration of the analysis.

        Arguments
        ---------
        reference_mass : float, None
            reference mass in solar masses to use for time steps.
        info : dict, None
            dictionary of additional information to store with the collection.
        kws : dict
            keyword arguments to specify target times and sky locations.

        Returns
        -------
        targets : TargetCollection
            collection of target objects.
        """
        if len(args) == 1:
            if isinstance(args[0], TargetCollection):
                return args[0]
            else:
                return cls(targets=args[0], reference_mass=reference_mass,
                           info=info)
        elif len(args) > 1:
            raise ValueError("too many arguments")

        t0kws = {k: kws.pop(k) for k in T0_KWARGS.values() if k in kws}
        for bad_set in T0_INCOMPATIBLE_OPTS:
            opt_names = [T0_KEYS[k] for k in bad_set]
            if all([k in t0kws for k in opt_names]):
                raise ValueError(
                    "incompatible T0 options: {}".format(opt_names))

        # look for reference mass to be used when stepping in time
        if reference_mass:
            # obtain stepping time in seconds from reference mass
            tm_ref = reference_mass * T_MSUN
            logging.info(f"Reference mass: {reference_mass} Msun ({tm_ref} s)")
        else:
            # no reference mass provided, so will default to seconds
            tm_ref = 1

        # look for reference time to be used to construct start times
        t0ref = t0kws.get(T0_KWARGS['ref'], 0)

        # Now we can safely interpret the options assuming one of three cases
        start_stop_step = [k.replace('-', '_') for k in START_STOP_STEP]
        if T0_KWARGS['list'] in t0kws:
            t0s = np.array(t0kws[T0_KWARGS['list']])
            t0ref = None
        elif T0_KWARGS['delta'] in t0kws:
            dt0s = np.array(t0kws[T0_KWARGS['delta']])
            t0s = dt0s*tm_ref + t0ref
        elif any([k in t0kws for k in start_stop_step]):
            if not all([k in t0kws for k in start_stop_step]):
                missing = [k for k in start_stop_step if k not in t0kws]
                raise ValueError(f"missing start/stop/step options: {missing}")
            # add a safety check here, in case the user mistakenly requests
            # stepping based on a GPS time and provides a reference GPS time
            start, stop, step = [t0kws[k] for k in start_stop_step]
            if start > 500 and t0ref > 1E8:
                logging.warning("high reference time and stepping start---did "
                                "you accidentally provide GPS times twice?")
            t0s = np.arange(start, stop, step)*tm_ref + t0ref

        targets = [Target.construct(t0, **kws) for t0 in t0s]
        return cls(targets, reference_mass=reference_mass,
                   reference_time=t0ref, info=info)

    @classmethod
    def from_config(cls, config_input, t0_sect=PIPE_SEC, sky_sect='target',
                    imr_result=None):
        """Create a collection of targets from a configuration file.

        All arguments in the `t0_sect` section of the configuration file are
        passed to the `construct` method to create the target collection 
        (see docs for `construct` for details), with additional special 
        handling if an IMR result is referenced.

        If any of the arguments to `construct` are set to 'imr', then attempts
        to derive them from IMR result: if `imr_result` is not provided, then
        the IMR section of the configuration file is used to load the IMR
        result.

        Arguments
        ---------
        config_input : str, ConfigParser
            configuration file or dictionary.
        t0_sect : str
            section of the configuration file to use for target times.
        sky_sect : str
            section of the configuration file to use for sky location.
        imr_result : IMRResult, None
            IMR result to use for reference values. If `None`, then the IMR
            section of the configuration file is used to load the IMR result
            if needed.

        Returns
        -------
        targets : TargetCollection
            collection of target objects.
        """
        config = utils.load_config(config_input)

        # Check if we are to get reference values from IMR result
        reference_imr = any([
            config.get(t0_sect, k, fallback='').lower() == 'imr'
            for k in [MREF_KEY, TREF_KEY]])
        if reference_imr:
            if imr_result is None:
                if not config.has_section(IMR_CONFIG_SECTION):
                    raise ValueError("IMR reference requested but no result "
                                     "provided or IMR section in config file")
                # get the IMR result
                from .imr import IMRResult
                pkws = {k: try_parse(v) 
                        for k, v in config[IMR_CONFIG_SECTION].items()
                        if k not in ['initialize_fit', 'psds']}
                path = pkws.pop('path')
                logging.info(f"loading IMR result from from {path} and {pkws}")
                imr_result = IMRResult.construct(path, **pkws)
            # get IMR target options
            pkws = dict(cache=True)
            pkws.update(try_parse(config.get(IMR_CONFIG_SECTION, 
                                             'peak_kws', fallback={})))
            imr_target = imr_result.get_best_peak_target(**pkws).settings
            # set the reference mass and time
            if MREF_KEY in config[t0_sect]:
                if config[t0_sect][MREF_KEY].lower() == 'imr':
                    logging.info("using IMR result for reference mass")
                    config[t0_sect][MREF_KEY] = \
                        str(imr_result.remnant_mass_scale_reference)
            else:
                logging.warning("no reference mass requested")

            if TREF_KEY in config[t0_sect]:
                if config[t0_sect][TREF_KEY].lower() == 'imr':
                    logging.info("using IMR result for reference time")
                    config[t0_sect][TREF_KEY] = str(imr_target.pop('t0'))
            else:
                logging.warning("no reference time requested")
            if sky_sect not in config:
                config[sky_sect] = {k: str(v) for k, v in imr_target.items()
                                    if k != 't0'}

        # Look for a reference mass, to be used when stepping in time
        m_ref = config.getfloat(t0_sect, MREF_KEY, fallback=None)

        t0kws = {}
        for k, conf_key in T0_KEYS.items():
            if config.has_option(t0_sect, conf_key):
                t0kws[T0_KWARGS[k]] = try_parse(config.get(t0_sect, conf_key))

        # get sky location and duration arguments if provided
        sky_dict = {k.lower(): try_parse(v)
                    for k, v in config[sky_sect].items()}

        info = {
            t0_sect: {k.lower(): try_parse(v)
                      for k, v in config[t0_sect].items()},
            sky_sect: sky_dict
        }
        return cls.construct(reference_mass=m_ref, info=info, **t0kws,
                             **sky_dict)
