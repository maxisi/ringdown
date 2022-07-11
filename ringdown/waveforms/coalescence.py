__all__ = ['Coalescence', 'Parameters']

from pylab import *
import lal
from .core import *
from scipy.signal import tukey
import lal
import lalsimulation as ls
from dataclasses import dataclass, asdict, fields
import inspect
import h5py

def docstring_parameter(*args, **kwargs):
    def dec(obj):
        if not obj.__doc__ is None:
            obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj
    return dec

def m1m2_from_mtotq(mtot, q):
    m1 = mtot / (1 + q)
    m2 = m1 * q
    return m1, m2

def m1m2_from_mcq(mc, q):
    m1 = mc*(1 + q)**(1/5)/q**(3/5)
    m2 = m1 * q
    return m1, m2

@dataclass
class Parameters:
    """Container for CBC parameters.

    Spin information is encoded in Cartesian components (`LALSimulation`
    convention).
    """
#
#    Attributes
#    ----------
#    mass_1 : float
#        heaviest component mass in solar masses
#    mass_2 : float
#        lightest component mass in solar masses
#    spin_1x : float
#        x-component of first spin
#    spin_1y : float
#        y-component of first spin
#    spin_1z : float
#        z-component of first spin
#    spin_2x : float
#        x-component of second spin
#    spin_2y : float
#        y-component of second spin
#    spin_2z : float
#        z-component of second spin
#    luminosity_distance : float
#        source luminosity distance in Mpc
#    iota : float
#        source inclination in rad
#    phase : float
#        refrence phase
#    long_asc_nodes : float
#        longitude of ascending nodes (def., 0, following `LALSuite` convention)
#    eccentricity : float
#        system eccentricity (def. 0)
#    mean_per_ano : float
#        mean annomaly of periastron (def., 0)
#    f_low : float
#        waveform starting frequency
#    f_ref : float
#        reference frequency
#    psi : float
#        source polarization angle in rad
#    ra : float
#        source right ascension in rad
#    dec : float
#        source declination in rad
#    trigger_time : float
#        trigger time at geocenter

    # masses
    mass_1: float
    mass_2: float
    # spins
    spin_1x: float = 0
    spin_1y: float = 0
    spin_1z: float = 0
    spin_2x: float = 0
    spin_2y: float = 0
    spin_2z: float = 0
    luminosity_distance: float = None
    iota: float = 0
    phase: float = 0
    # additional
    long_asc_nodes: float = 0
    eccentricity: float = 0
    mean_per_ano: float = 0
    # frequency
    f_low: float = 20
    f_ref: float = 20
    # location
    psi: float = 0
    ra: float = None
    dec: float = None
    trigger_time: float = 0

    def __post_init__(self):
        # make sure all values are floats, or LALSim will complain
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                setattr(self, f.name, float(value))

    def __getitem__(self, *args, **kwargs):
        return getattr(self, *args, **kwargs)
    
    def __setitem__(self, *args, **kwargs):
        return setattr(self, *args, **kwargs)
    
    def to_dict(self) -> dict:
            return asdict(self)

    def items(self, *args, **kwargs):
        return self.to_dict().items(*args, **kwargs)
    
    def keys(self, *args, **kwargs):
        return self.to_dict().keys(*args, **kwargs)
    
    def values(self, *args, **kwargs):
        return self.to_dict().values(*args, **kwargs)

    _EXTRINSIC_KEYS = ['ra', 'dec', 'trigger_time']

    _SPIN_KEYS_LALINF = ['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12',
                         'a_1', 'a_2']
    
    _SPIN_COMP_KEYS = ['spin_{}{}'.format(i,x) for i in [1,2] for x in 'xyz']
    _SPIN_KEYS_LALSIM = ['iota'] + _SPIN_COMP_KEYS
    
    _ALIASES = {
        'trigger_time': Signal._T0_ALIASES,
        'mass_1': ['m1'],
        'mass_2': ['m2'],
        'total_mass': ['mtot', 'm', 'mtotal', 'mtot_msun', 'mtotal_msun'],
        'chirp_mass': ['mc', 'mchirp'],
        'mass_ratio': ['q'],
        'luminosity_distance': ['dist', 'dl', 'distance', 'dist_mpc'],
        'iota': ['inclination', 'inc'],
        'f_ref': ['fref', 'reference_frequency'],
        'f_low': ['flow', 'fmin', 'f_min'],
    }
    _ALIASES_STR = ''
    for _k,_v in _ALIASES.items():
        _ALIASES_STR += f"* `{_k}`: ``{_v}\n``"

    for _k in _SPIN_KEYS_LALINF:
        _ALIASES[_k] = [_k.replace('_', '')]
    del _k, _v
    
    @property
    def intrinsic(self):
        """Intrinsic parameters."""
        return {k: v for k,v in self.items() if k not in self._EXTRINSIC}

    @property
    def extrinsic(self):
        """Extrinsic parameters."""
        return {k: v for k,v in self.items() if k in self._EXTRINSIC}

    @classmethod
    def construct(cls, **kws):                                            
        """Construct :class:`Parameters` instance from keyword arguments.
        
        Arguments will be treated as ``param_name = value``, and will attempt
        to match ``param_name`` to one of the known CBC parameters (or their
        aliases). As part of this process, the method will recognize spin
        parameters provided in the `LALInference` convention (magnitude and
        angles) and automatically convert them to the `LALSimulation`
        convention (Cartesian components) for storage.

        A number of aliases for common parameters are recognized (e.g., ``mc``
        is an alias of ``chirp_mass``); all valid aliases are listed below.
        Unrecognized arguments are ignored.  Missing parameters will be set to
        default (see :class:`Parameters` docs).

        Valid aliases:
        {}

        Arguments
        ---------
        \*\*kws
            parameter names and values

        Returns
        -------
        pars : Parameters
            coalescence parameters container object
        """.format(cls._ALIASES_STR)
        kws['f_ref'] = kws.get('f_ref', kws.get('f_low'))
        for par, aliases in cls._ALIASES.items():
            for k in aliases:
                if k in kws:
                    kws[par] = kws.pop(k)
        # compose component masses
        if 'mass_1' not in kws or 'mass_2' not in kws:                                                               
            if 'total_mass' in kws and 'mass_ratio' in kws:                   
                kws['mass_1'], kws['mass_2'] = m1m2_from_mtotq(kws['total_mass'],
                                                               kws['mass_ratio'])
            elif 'chirp_mass' in kws and 'mass_ratio' in kws:
                kws['mass_1'], kws['mass_2'] = m1m2_from_mcq(kws['chirp_mass'],
                                                             kws['mass_ratio'])
        # compose spins
        lsim_given = [k in kws for k in cls._SPIN_KEYS_LALSIM if k!='iota']
        linf_given = [k in kws for k in cls._SPIN_KEYS_LALINF if k!='theta_jn']
        if not all(lsim_given) and any(linf_given):
            try:
                a = [kws[k] for k in cls._SPIN_KEYS_LALINF] + \
                       [kws['mass_1']*lal.MSUN_SI, kws['mass_2']*lal.MSUN_SI,
                        kws['f_ref'], kws['phase']]
            except KeyError as e:
                raise ValueError(f"unable to parse spins, missing: {e}")
            b = ls.SimInspiralTransformPrecessingNewInitialConditions(*a)
            kws.update(dict(zip(cls._SPIN_KEYS_LALSIM, b)))
            
        return cls(**{k: v for k,v in kws.items() 
                      if k in inspect.signature(cls).parameters})

    @property
    def total_mass(self):
        """Total mass in solar masses, :math:`M = m_1 + m_2`.
        """
        return self.mass_1 + self.mass_2

    @property
    def mass_ratio(self):
        """Mass ratio :math:`q = m_2/m_1` for :math:`m_1 \geq m_2`.
        """
        return self.mass_2 / self.mass_1

    @property
    def chirp_mass(self):
        """Chirp mass :math:`\\mathcal{M}_c = \\left(m_1 m_2\\right)^{3/5}/\\left(m_1 + m_2\\right)^{1/5}`.
        """
        return (self.mass_1 * self.mass_2)**(3/5) / \
               (self.mass_1 + self.mass_2)**(1/5)

    @property
    def spin_1(self):
        """3D dimensionless spin for first component.
        """
        return array([self.spin_1x, self.spin_1y, self.spin_1z])

    @property
    def spin_2(self):
        """3D dimensionless spin for second component.
        """
        return array([self.spin_2x, self.spin_2y, self.spin_2z])

    @property
    def spin_1_mag(self):
        """Dimensionless spin magnitude for first component.
        """
        return linalg.norm(self.spin_1)

    @property
    def spin_2_mag(self):
        """Dimensionless spin magnitude for second component.
        """
        return linalg.norm(self.spin_2)

    @property
    def cos_iota(self):
        """Cosine of the Newtonian inclination at reference frequency.
        """
        return cos(self.iota)

    @property
    def luminosity_distance_si(self):
        """Luminosity distance in meters.
        """
        return self.luminosity_distance * 1e6 * lal.PC_SI

    @property
    def mass_1_si(self):
        """First component mass in kg.
        """
        return self.mass_1 * lal.MSUN_SI
    
    @property
    def mass_2_si(self):
        """Second component mass in kg.
        """
        return self.mass_2 * lal.MSUN_SI

    def get_choosetdwaveform_args(self, delta_t):
        """Construct input for :func:`ls.SimInspiralChooseTDWaveform`.
        
        Arguments
        ---------
        delta_t: float
            time spacing for waveform array

        Returns
        -------
        args : list
            list of arguments ready for :func:`ls.SimInspiralChooseTDWaveform`
        """
        args = [self.mass_1_si, self.mass_2_si, *self.spin_1, *self.spin_2,
                self.luminosity_distance_si, self.iota, self.phase,
                self.long_asc_nodes, self.eccentricity, self.mean_per_ano,
                float(delta_t), self.f_low, self.f_ref]
        return args 

    def get_choosetdmodes_args(self, delta_t):
        """Construct input for :func:`ls.SimInspiralChooseTDModes`.
        
        Arguments
        ---------
        delta_t: float
            time spacing for waveform array

        Returns
        -------
        args : list
            list of arguments ready for :func:`ls.SimInspiralChooseTDModes`
        """
        # Arguments taken by ChooseTDModes (from docstring):
        # UNUSED REAL8 phiRef,                        /**< reference orbital phase (rad). This variable is not used and only kept here for backwards compatibility */
        # REAL8 deltaT,                               /**< sampling interval (s) */
        # REAL8 m1,                                   /**< mass of companion 1 (kg) */
        # REAL8 m2,                                   /**< mass of companion 2 (kg) */
        # REAL8 S1x,                                  /**< x-component of the dimensionless spin of object 1 */
        # REAL8 S1y,                                  /**< y-component of the dimensionless spin of object 1 */
        # REAL8 S1z,                                  /**< z-component of the dimensionless spin of object 1 */
        # REAL8 S2x,                                  /**< x-component of the dimensionless spin of object 2 */
        # REAL8 S2y,                                  /**< y-component of the dimensionless spin of object 2 */
        # REAL8 S2z,                                  /**< z-component of the dimensionless spin of object 2 */
        # REAL8 f_min,                                /**< starting GW frequency (Hz) */
        # REAL8 f_ref,                                /**< reference GW frequency (Hz) */
        # REAL8 r,                                    /**< distance of source (m) */
        # LALDict *LALpars,                           /**< LAL dictionary containing accessory parameters */
        # int lmax,                                   /**< generate all modes with l <= lmax */
        # Approximant approximant                     /**< post-Newtonian approximant to use for waveform production */
        args = [0.0, float(delta_t), self.mass_1_si, self.mass_2_si,
                *self.spin_1, *self.spin_2, self.f_low, self.f_ref,
                self.luminosity_distance_si]
        return args 

class Coalescence(Signal):
    """An inspiral-merger-ringdown signal from a compact binary coalescence.
    """
    _DEF_TUKEY_ALPHA = 0.125

    # register names of all available LALSimulation approximants
    _MODELS = [
        ls.GetStringFromApproximant(a) for a in range(ls.NumApproximants) if
        ls.SimInspiralImplementedFDApproximants(a) or
        ls.SimInspiralImplementedTDApproximants(a)
    ]

    def __init__(self, *args, modes=None, **kwargs):
        super(Coalescence, self).__init__(*args, **kwargs)
        self._invariant_peak = None

    @property
    def _constructor(self):
        return Coalescence

    @classmethod
    @docstring_parameter(_DEF_TUKEY_ALPHA)
    def from_parameters(cls, time, model=None, approximant=None, ell_max=None,
                        single_mode=None, window=_DEF_TUKEY_ALPHA,
                        manual_epoch=False, subsample_placement=False, **kws):
        """Construct coalescence waveform from compact-binary parameters.

        Additional keyword arguments are passed to :class:`Parameters` to
        construct the input for :func:`ls.SimInspiralChooseTDWaveform`. This
        should include masses, spins, inclination, refrence phase, reference
        frequency, minimum frequency, and trigger time.

        In order to reproduce waveform placement for time-domain waveforms as
        carried out by :mod:`LALInference`, use `manual_epoch = False`.
        Otherwise, will place waveform such that the empirical peak of the
        waveform envelope falls on the specified trigger time. If
        `subsample_placement`, the peak is identified via quadratic sub-sample
        interpolation and the waveform is shifted in the Fourier domain so that
        the interpolated peak falls on a sample (this option necessarily
        reduces performance).

        By default, a one-sided Tukey window with a ramp up of :math:`\\alpha =
        {0}` is applied to the left-hand side of the signal. This means that
        frequencies below :math:`\\approx 1/\\left(\\alpha \\times T\\right)`
        will be corrupted, where :math:`T` is the intrinsic duration of the
        waveform as determined by `f_low`. This behavior is turned off if
        `window` is 0 or False, but at risk of having a sharp turn on of
        time-domain waveforms in band (this should be safe for frequency-domain
        waveforms).

        (Note that an estimate on :math:`T` can be obtained through
        ``XLALSimInspiralChirpTimeBound(f_min, m1, m2, S1z, S2z)``.)

        Arguments
        ---------
        time : array
            time array over which to evaluate waveform.
        model : str
            name of waveform approximant (e.g., "NRSur7dq4").
        approximant : str
            alias for `model`.
        ell_max : int
            highest angular harmonic ``ell`` mode to include, defined in the
            coprecessing frame per the convention inherited from LALSimulation
            Defaults to None, which includes all available modes.
        single_mode : tuple
            single ``(ell, m)`` spherical mode to include, e.g., ``(2, 2)``;
            includes both left and right handed contributions, i.e., ``(l, m)``
            and ``(l, -m)``. Defaults to None, which includes all available
            modes.
        window : float, bool
            window signal from the left to avoid sharp turn ons in time-domain
            approximants; numerical value is interpreted as fraction of time
            for :func:`tukey` window ramp up. Defaults to {0}.
        manual_epoch : bool
            align waveform based on empirically found waveform peak (i.e., peak
            of :math:`h_+^2 + h_\\times^2`), not the epoch reported by
            approximant. Defaults to False.
        subsample_placement : bool
            if ``manual_epoch``, find waveform peak using quadratic subsample
            interpolation; else, find maximum of waveform envelope in samples.
            Defaults to False.
        kws
            additional keyword arguments passed to :class:`Parameters`.

        Returns
        -------
        h : Coalescence
            compact binary coalescence signal.
        """
        approximant = model or approximant

        all_kws = {k: v for k,v in locals().items() if k not in ['cls','time']}
        all_kws.update(all_kws.pop('kws'))

        dt = time[1] - time[0]
        approx = ls.SimInspiralGetApproximantFromString(approximant)
        pars = Parameters.construct(**kws)

        param_dict = lal.CreateDict()
        # NR handling based on https://arxiv.org/abs/1703.01076
        if approximant == 'NR_hdf5':
            nr_path = kws['nr_path']
            # get masses
            mtot_msun = pars['total_mass']
            with h5py.File(nr_path, 'r') as f:
                m1 = f.attrs['mass1']
                m2 = f.attrs['mass2']
                pars['mass_1'] = m1 * mtot_msun/(m1 + m2)
                pars['mass_2'] = m2 * mtot_msun/(m1 + m2)
            # compute spin components in lalsim frame
            s = ls.SimInspiralNRWaveformGetSpinsFromHDF5File(pars['f_ref'],
                                                             mtot_msun,
                                                             nr_path)
            for k, v in zip(pars._SPIN_COMP_KEYS, s):
                pars[k] = v
            # add pointer to NR file to dictionary
            ls.SimInspiralWaveformParamsInsertNumRelData(param_dict, nr_path)
            # TODO: is this right??
            pars['long_asc_nodes'] = np.pi / 2

        if single_mode is not None and ell_max is not None:
            raise ValueError("Specify only one of single_mode or ell_max")

        if ell_max is not None:
            # if ell_max, load all modes with ell <= ell_max
            ma = ls.SimInspiralCreateModeArray()
            for ell in range(2, ell_max+1):
                ls.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
            ls.SimInspiralWaveformParamsInsertModeArray(param_dict, ma)
        elif single_mode is not None:
            l, m = single_mode
            # if a single_mode is given, load only that mode (l,m) and (l,-m)
            ma = ls.SimInspiralCreateModeArray()
            # add (l,m) and (l,-m) modes
            ls.SimInspiralModeArrayActivateMode(ma, l, m)
            ls.SimInspiralModeArrayActivateMode(ma, l, -m)    
            # then insert the ModeArray into the LALDict params
            ls.SimInspiralWaveformParamsInsertModeArray(param_dict, ma)

        args = pars.get_choosetdwaveform_args(dt)
        hp, hc = ls.SimInspiralChooseTDWaveform(*args, param_dict, approx)

        # align waveform to trigger time, following LALInferenceTemplate
        # https://git.ligo.org/lscsoft/lalsuite/blob/master/lalinference/lib/LALInferenceTemplate.c#L1124
        # (keeping variable names the same to facilitate comparison)
        bufLength = len(time)
        tStart = time[0]
        tEnd = tStart + dt * bufLength

        # /* The nearest sample in model buffer to the desired tc. */
        tcSample = round((pars['trigger_time'] - tStart)/dt)

        # /* The actual coalescence time that corresponds to the buffer
        #    sample on which the waveform's tC lands. */
        # i.e. the nearest time in the buffer
        injTc = tStart + tcSample*dt

        hp_d = hp.data.data
        hc_d = hc.data.data
        n = hp.data.length

        # /* The sample at which the waveform reaches tc. */
        if manual_epoch:
            # manually find peak of the waveform envelope
            if subsample_placement:
                # time-shift waveform in frequency domain so that the
                # quadratically-interpolated peak falls on a sample
                ipeak = n - _ishift(hp_d, hc_d)
                ishift = ipeak - round(ipeak)
                w = tukey(n, window)
                shift = exp(1j*2*pi*np.fft.rfftfreq(n, dt)*ishift*dt)
                hp_d = np.fft.irfft(np.fft.rfft(w*hp_d)*shift, n=n)
                hc_d = np.fft.irfft(np.fft.rfft(w*hc_d)*shift, n=n)
            waveTcSample = argmax(hp_d**2 + hc_d**2)
        else:
            # this is what is done in LALInference
            hp_epoch = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds*1E-9
            waveTcSample = round(-hp_epoch/dt)

        # /* 1 + (number of samples post-tc in waveform) */
        wavePostTc = n - waveTcSample

        if tcSample >= waveTcSample:
            bufStartIndex = int(tcSample - waveTcSample)
        else:
            bufStartIndex = 0

        if tcSample + wavePostTc <= bufLength:
            bufEndIndex= int(tcSample + wavePostTc)
        else:
            bufEndIndex= bufLength

        bufWaveLength = bufEndIndex - bufStartIndex

        if tcSample >= waveTcSample:
            waveStartIndex = 0
        else:
            waveStartIndex = waveTcSample - tcSample

        if window and tcSample >= waveTcSample and not subsample_placement:
            # smoothly turn on waveform
            w = tukey(bufWaveLength, window)
            w[int(0.5*bufWaveLength):] = 1.
        else:
            # no window
            w = 1

        h = np.zeros(bufLength, dtype=complex)
        h[bufStartIndex:bufEndIndex] = w*(hp_d[waveStartIndex:waveStartIndex+bufWaveLength] -\
                                       1j*hc_d[waveStartIndex:waveStartIndex+bufWaveLength])
        all_kws.update(pars.to_dict())
        return cls(h, index=time, parameters=all_kws)

    def get_invariant_peak_time(self, ell_max=None, force=False):
        """Compute time of the peak of the invariant strain :math:`H^2`, which
        is defined as

        .. math::
            H^2(t) \equiv \sum_{\ell m} H_{\ell m}^2(t)

        where :math:`H_{\ell m}(t)` are the coefficients associated with the
        spherical harmonic factors :math:`{}_{-2} Y_{\ell m}` in the strain,
        namely

        .. math::
            h(t) = \sum_{\ell m} H_{\ell m}(t) {}_{-2} Y_{\ell m}

        .. note::
            This currently only works with some time domain approximants (e.g.,
            `NRSur7dq4`), as it relies on :func:`ls.SimInspiralChooseTDModes`.

        Arguments
        ---------
        ell_max : int
            maximum :math:`\ell` to include in the computation of the peak
        force : bool
            redo the computation ignoring cached results

        Returns
        -------
        t_peak : float
            peak time of the invariant strain
        """
        # NOTE: there are functions in LALSimulation to obtain a list of
        # individual modes for a given source but these do not work with all
        # approximants; here I take the simpler, fail safe approach, which is
        # to call ChooseTDWaveform several times with different mode content.

        #     LALDict *LALpars,                           /**< LAL dictionary containing accessory parameters */
        #     int lmax,                                   /**< generate all modes with l <= lmax */
        #     Approximant approximant                     /**< post-Newtonian approximant to use for waveform production */

        if self._invariant_peak is None or ell_max or force:
            kws = self.parameters.copy()
            approximant = kws.get("model", kws.get("approximant"))
            approx = ls.SimInspiralGetApproximantFromString(approximant)
            # the ell_max argument is not actually used by
            # `SimInspiralChooseTDModes`, which just returns all modes, so we
            # need to handle this manually ourselves
            dict_params = lal.CreateDict()
            if ell_max is not None:
                ma = ls.SimInspiralCreateModeArray()
                for ell in range(2, ell_max+1):
                    ls.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
                    ls.SimInspiralWaveformParamsInsertModeArray(dict_params, ma)

            pars = Parameters.construct(**kws)
            args = pars.get_choosetdmodes_args(self.delta_t)
            hlms = ls.SimInspiralChooseTDModes(*args, dict_params, 5, approx)
            # get the time array corresponding to this waveform, such that the
            # reference time (wrt which epoch is defined) is placed at the
            # requested trigger time
            t = np.arange(len(hlms.mode.data.data))*self.delta_t + \
                (float(hlms.mode.epoch) + pars['trigger_time'])
            # iterate over modes and compute magnitude squared
            sum_h_squared = Coalescence(zeros_like(t), index=t,
                                        parameters=self.parameters)
            while hlms is not None:
                sum_h_squared += real(hlms.mode.data.data)**2 + imag(hlms.mode.data.data)**2
                hlms = hlms.next
            t_peak = sum_h_squared.peak_time
            if ell_max is not None:
                # return value without caching
                return t_peak
            self._invariant_peak = t_peak
        return self._invariant_peak
Signal._register_model(Coalescence)


class IMR(Signal):
    _metadata = ['parameters','posterior_sample', 't_dict']

    def __init__(self, *args, posterior_sample=None, t_dict=None, **kwargs):
        warnings.warn("IMR is deprecated; use Coalescence", warnings.DeprecationWarning)
        if isinstance(posterior_sample,pd.DataFrame):
            posterior_sample = posterior_sample.squeeze().to_dict()

        super(IMR, self).__init__(*args, **kwargs)
        self.posterior_sample = posterior_sample
        self.t_dict = t_dict

    @classmethod
    def from_posterior(cls, posterior_sample, wf=None, dt=(1/4096),
                            interpolation_times=None, f_low=20.0, f_ref=20.0):
        """
        Constructs the IMR signal from a posterior sample

        Arguments
        ---------
        posterior_sample : dataframe or dictionary
            contains one particular posterior sample, with sky location
            and geocent_time
        wf : int
            One can provide a particular waveform code they would like to run
            with
        dt: float
            duration of the new interpolated signal
        interpolation_times: dictionary of time arrays for each ifo
            if there is a dictionary of times for each detector, this will
            interpolate at each ifo
        """
        if isinstance(posterior_sample,pd.DataFrame) or isinstance(posterior_sample,pd.Series):
            posterior_sample = posterior_sample.squeeze().to_dict()

        if not isinstance(posterior_sample, dict):
            raise ValueError("Expected posterior_sample to be a dict or a one-row pandas DataFrame or Series")

        if wf is None:
            wf = int(posterior_sample['waveform_code'])

        waveform_dt = dt

        t_peak, t_dict, hp,hc = complex_strain_peak_time_td(posterior_sample, wf=wf,
                                                                 dt=dt, f_low=f_low, f_ref=f_ref)
        signal_dict = {}
        # At Geocent
        geocent_time = posterior_sample['geocent_time']
        ts_geocent = (np.arange(len(hp.data.data)))*waveform_dt + float(hp.epoch) + geocent_time
        hpdf = Data(hp.data.data,index=ts_geocent)
        hcdf = Data(hc.data.data,index=ts_geocent)
        signal_dict['geocent'] = {'hp': hpdf,'hc': hcdf}

        # Get Times
        tgps = lal.LIGOTimeGPS(geocent_time)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
            
        params = posterior_sample.copy()
        params.update({(k+'_peak'):v for k,v in t_dict.items()})
        params['t0'] = t_peak

        main = hpdf - 1j*hcdf
        result = cls(main, index=ts_geocent, posterior_sample=posterior_sample, 
                            parameters=params, t_dict=t_dict)
            
        return result

    def time_delay(self, t0_geocent, ifo):
        tgps = lal.LIGOTimeGPS(t0_geocent)
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        det = lal.cached_detector_by_prefix[ifo]

        if self.posterior_sample is not {}:
            ra = self.posterior_sample['ra']
            dec = self.posterior_sample['dec']
            psi = self.posterior_sample['psi']
        else:
            raise KeyError("Posterior Samples haven't been filled into this object")

        # Get patterns and timedelay
        timedelay = lal.TimeDelayFromEarthCenter(det.location,  ra, dec, tgps)
        return timedelay

    @property
    def _constructor(self):
        return IMR

    @property
    def projections(self):
        """
        A dictionary of projections onto each detector.

        Returns
        --------
        dict with the key value pairing {ifo : Data ...}

        Note
        ---------
        This method projects with a timeshift, which means that
        the time index of the projection will be shifted with respect
        to the Signal time index so that there is an element by element
        correspondence between the Signal values and the projected Data
        values.
        """
        return {ifo: getattr(self,ifo) for ifo in ['H1','L1','V1']}

    def project_with_timeshift(self, ifo):
        """
        Given a detector name, it projects the signal onto that
        interferometer so that:

            h = Fp*hp + Fc*hc
            t = t + detectortimedelay

        This makes sure that if we have a signal that starts at
        the geocent signal peak, then the projected signal start
        time corresponds to the geocent signal peak time delayed 
        to H1 (a.k.a H1_peak)


        Arguments
        ---------
        ifo : str, one of 'H1','L1' or 'V1'
            name of the interferometer you would like to project to

        Note
        ---------
        This method projects with a timeshift, which means that
        the time index of the projection will be shifted with respect
        to the Signal time index so that there is an element by element
        correspondence between the Signal values and the projected Data
        values.
        """
        tgps = lal.LIGOTimeGPS(self.posterior_sample['geocent_time'])
        gmst = lal.GreenwichMeanSiderealTime(tgps)
        det = lal.cached_detector_by_prefix[ifo]

        if self.posterior_sample is not {}:
            ra = self.posterior_sample['ra']
            dec = self.posterior_sample['dec']
            psi = self.posterior_sample['psi']
        else:
            raise KeyError("Posterior Samples haven't been filled into this object")

        # Get patterns and timedelay
        Fp, Fc = lal.ComputeDetAMResponse(det.response, ra, dec, psi, gmst)
        timedelay = lal.TimeDelayFromEarthCenter(det.location,  ra, dec, tgps)

        # Get new time detector time index
        ts_detector = self.time + timedelay 

        # Construct the data objects and use the antenna_patterns to project
        hpdf = Data(self._hp, index=ts_detector,ifo=ifo)
        hcdf = Data(self._hc, index=ts_detector, ifo=ifo)
        signal = Fp*hpdf + Fc*hcdf
        return signal

    @property
    def H1(self):
        """
        Finds the signal at H1
        """ 
        return self.project_with_timeshift('H1')

    @property
    def L1(self):
        """
        Finds the signal at L1
        """
        return self.project_with_timeshift('L1')

    @property
    def V1(self):
        """
        Finds the signal at V1
        """
        return self.project_with_timeshift('V1')

    def whitened_projections(self, acfs, interpolate_data=False):
        """
        Whitens the data after projecting onto detectors
        based on the passed acf dictionary

        Returns: A dictionary of Data objects labelled by detectors
        where the data is whitened by the corresponding acf provided

        Arguments
        ---------
        acfs : dict 
            dict must have key value pairs: {ifo: AutocorrelationFunction}

        interpolate_data: bool (default=True)
            Interpolates the data to the sample rate of the PSD so that
            whitening can take place

        """
        whitened = {}

        for ifo, acf in acfs.items():
            data = getattr(self,ifo)
            if interpolate_data:
                # Interpolate to the new sample rate 
                # (rest of the things remain the same)
                duration = data.time.max() - data.time.min()
                data = data.interpolate(times=acf.time[acf.time <= duration].time, t0=data.time[0])
            else:
                if acf.delta_t != data.delta_t:
                    raise ValueError("""The data delta_t doesnot match the delta_t of the acf.
                                        If you would like to interpolate the signal, then
                                        set interpolate_data=True as a kwarg of this function""")
            whitened[ifo] = acf.whiten(data)

        return whitened

    def whiten_with_data(self, data, acfs, t0=None, duration=None, flow=None):
        """
        Given a dictionary of data and a dictionary of acfs, this returns
        the whitened data and the whitened IMR signal for each detector. One can set the
        start time (with respect to geocent) and the duration of the required
        returned data

        Arguments:
        -----------
        data: dict
            A dictionary pairing ifo strings with Data objects. This will be
            the strain data for each detector
        acfs: dict
            A dictionary pairing ifo strings with Autocovariance objects. This will
            be the acf for the noise of each detector.
        t0: float
            All returned timestamps will be more than this t0 (when shifted to geocent)
        duration: float
            All returned timestamps will be less than t0+duration (when shifted to geocent)
        flow: float
            The low frequency cutoff needed to condition the provided data. If not provided, it
            is assumed the data is already conditioned
        """
        # Initialize the dictionaries
        whitened_data = {}
        whitened_signal = {}
        signal_projections = self.projections

        # Loop over the detectors
        for ifo in data.keys():
            # If t0 and duration not provided, set them to good default values (i.e. donot change the 
            # length or start time of the data array)
            start_ifo = data[ifo].time.min() if t0 is None else (t0 + self.time_delay(t0, ifo))
            end_ifo = data[ifo].time.max() if duration is None else (start_ifo + duration)

            # Get this detector's data, condition and chop it if needed
            ds = int(round(data[ifo].fsamp/acfs[ifo].fsamp))
            data_ifo = data[ifo].condition(flow=flow, ds=ds)[start_ifo:end_ifo]

            # Interpolate this IMR signal's detector data to match the provided timestamps
            signal_ifo = signal_projections[ifo].interpolate(times=data_ifo.time)

            # Check if the provided ACF has the right sample frequency
            if not (np.isclose(acfs[ifo].delta_t, data_ifo.delta_t) and np.isclose(acfs[ifo].delta_t, signal_ifo.delta_t)):
                raise ValueError("The sample rates of the ACF, data and signal donot match")

            # Whiten both data and the signal and append it to the dictionaries
            whitened_data[ifo] = acfs[ifo].whiten(data_ifo)
            whitened_signal[ifo] = acfs[ifo].whiten(signal_ifo)

        return whitened_signal, whitened_data


    def calculate_snr(self, data, acfs, t0=None, duration=None, flow=None):
        """
        Calculates the time-bounded matched filter SNR based on the 
        provided data and the covariance matrix extracted from the
        acfs provided.
        
        Arguments:
        -----------
        data: dict
            A dictionary pairing ifo strings with Data objects. This will be
            the strain data for each detector
        acfs: dict
            A dictionary pairing ifo strings with Autocovariance objects. This will
            be the acf for the noise of each detector.
        t0: float
            All returned timestamps will be more than this t0 (when shifted to geocent)
        duration: float
            All returned timestamps will be less than t0+duration (when shifted to geocent)
        flow: float
            The low frequency cutoff needed to condition the provided data. If not provided, it
            is assumed the data is already conditioned
        """
        whitened_signal, whitened_data = self.whiten_with_data(data=data,acfs=acfs,t0=t0,duration=duration,flow=flow)
        SNR_squared = 0.0
        for ifo in whitened_data.keys():
            signal_optimal_SNR_squared = np.dot(whitened_signal[ifo].values, whitened_signal[ifo].values)
            SNR_squared += (np.dot(whitened_signal[ifo].values, whitened_data[ifo].values)**2)/signal_optimal_SNR_squared
        SNR = np.sqrt(SNR_squared)
        return SNR

    def plot_whitened_with_data(self, data, acfs, t0=None, duration=None, flow=None):
        """
        Plots the comparison of the whitened data and the IMR signal. 
        The t0 and duration of the plot can be set

        Arguments:
        -----------
        data: dict
            A dictionary pairing ifo strings with Data objects. This will be
            the strain data for each detector
        acfs: dict
            A dictionary pairing ifo strings with Autocovariance objects. This will
            be the acf for the noise of each detector.
        t0: float
            All returned timestamps will be more than this t0 (when shifted to geocent)
        duration: float
            All returned timestamps will be less than t0+duration (when shifted to geocent)
        """
        whitened_signal, whitened_data = self.whiten_with_data(data=data,acfs=acfs,t0=t0,duration=duration, flow=flow)
        fig,axes = subplots(len(whitened_data.keys()))
        for i,ifo in enumerate(whitened_data.keys()):
            data = whitened_data[ifo]
            signal_ifo = whitened_signal[ifo]
            axes[i].errorbar(data.time, data.values, yerr=ones_like(data.values), fmt='.', 
             alpha=0.5, label='Data')
            axes[i].plot(signal_ifo.time, signal_ifo.values, color="black", label='IMR')
            axes[i].set_xlabel(r'$t / \mathrm{s}$')
            axes[i].set_ylabel(r'$h_%s(t)$ (whitened)' % ifo[0])

            # Add t_peak if needed
            if (data.time.min() < self.t_dict[ifo]) and (self.t_dict[ifo] < data.time.max()):
                axes[i].axvline(self.t_dict[ifo], linestyle='dashed', c='r', label='Peak Time', alpha=0.5)

            axes[i].legend(loc='best')
        show()
        return fig, axes
