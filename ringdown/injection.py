__all__ = ['Signal', 'Ringdown', 'simulated_template', 'IMR']

from pylab import *
import lal
from .data import *
from .peak import *
from . import qnms
import pandas as pd
from scipy.interpolate import interp1d
from inspect import getfullargspec

class Signal(TimeSeries):
    _metadata = ['parameters']

    def __init__(self, *args, parameters=None, **kwargs):
        super(Signal, self).__init__(*args, **kwargs)
        self.parameters = parameters or {}

    @property
    def _constructor(self):
        return Signal

    @property
    def t0(self):
        return self.get_parameter('t0', None)

    def get_parameter(self, k, *args):
        return self.parameters.get(k.lower(), *args) 

    @property
    def _hp(self):
        return np.real(self) 

    @property
    def hp(self):
        return Signal(self._hp, index=self.index, parameters=self.parameters)

    @property
    def _hc(self):
        return -np.imag(self) 

    @property
    def hc(self):
        return Signal(self._hc, index=self.index, parameters=self.parameters)

    def project(self, ifo=None, t0=None, antenna_patterns=None, delay=None,
                ra=None, dec=None, psi=None, fd_shift=False, interpolate=False):
        if antenna_patterns is None and ifo:
            tgps = lal.LIGOTimeGPS(t0 or self.t0)
            gmst = lal.GreenwichMeanSiderealTime(tgps)
            det = lal.cached_detector_by_prefix[ifo]
            antenna_patterns = lal.ComputeDetAMResponse(det.response, ra, dec,
                                                        psi, gmst)
        Fp, Fc = antenna_patterns
        h = Fp*self._hp + Fc*self._hc
        if isinstance(delay, str):
            if delay.lower() == 'from_geo':
                tgps = lal.LIGOTimeGPS(t0 or self.t0)
                det = lal.cached_detector_by_prefix[ifo]
                delay = lal.TimeDelayFromEarthCenter(det.location,ra,dec,tgps)
            else:
                raise ValueError("invalid delay reference: {}".format(delay))
        else:
            delay = delay or 0
        if fd_shift:
            h_fd = np.fft.rfft(h)
            frequencies = np.fft.rfftfreq(len(h), d=self.delta_t)
            timeshift_vector = np.exp(-2.*1j*np.pi*delay*frequencies)
            h = np.fft.irfft(h_fd * timeshift_vector)
        else:
            idt = int(round(delay * self.fsamp))
            if interpolate:
                hint = interp1d(self.time, h, kind='cubic', fill_value=0,
                                bounds_error=False)
                dt = (idt - delay*self.fsamp)*self.delta_t
                h = hint(self.time + dt)
            h = np.roll(h, idt)
        return Data(h, ifo=ifo, index=self.time)

    def interpolate(self, times=None, t0=None, duration=None, fsamp=None):
        """
        Arguments
        ---------
        times : list or numpy array or pd.Series
            array of GPS times (seconds) to label the times
        t0 : float
            instead of an array of times, one can provide the start time t0
            at geocenter, the duration or/and the sample rate. If both this and
            times is provided then this takes the duration and fsamp from the times
            array and sets the t0 to be the initial one

        duration: float
            duration of the new interpolated signal
        fsamp: float
            sample rate of the new interpolated signal
        """
        if times is None:
            # Set default values if needed
            t0 = t0 or self.time.min()
            duration = duration or (self.time.max() - t0)
            fsamp = fsamp or self.fsamp

            # Find Number of points
            # +1 because int(round(duration*fsamp)) counts the number of intervals
            N = int(round(duration*fsamp)) + 1

            # Create the timing array
            times = np.arange(N)/fsamp + t0

            # Make sure we don't include points outside of the index
            if times.max() > self.time.max():
                times = times[times <= self.time.max()]
        elif t0 is not None:
            # Use the times array for the delta_t and duration, but set 
            # the t0 provided
            times = times - times[0] + t0

        # Interpolate to the new times
        hp_interp_func = interp1d(self.time, self.hp.values, kind='cubic', fill_value=0, bounds_error=False);
        hc_interp_func = interp1d(self.time, self.hc.values, kind='cubic', fill_value=0, bounds_error=False);
        hp_interp = hp_interp_func(times)
        hc_interp = hc_interp_func(times)
        signal_val = hp_interp - 1j*hc_interp

        # Set up the parameters for a new copy of this object
        kwargs = {'index': times}
        for keyword in self._metadata:
            kwargs[keyword] = getattr(self, keyword, None)

        return self._constructor(signal_val, **kwargs)

    def plot(self):
        """
        Plot the function's hp and hc components.
        Remember that the value of this timeseries is h = hp -1j*hc
        """
        fig,ax = subplots(1)
        ax.plot(self.time, self.hp,label="hp")
        ax.plot(self.time, self.hc, label="hc")
        legend(loc='best')
        show()

    def find_peak(self):
        ipeak = len(self) - _ishift(self._hp, self._hc)
        tpeak = self.delta_t*ipeak + float(self.time[0])
        return tpeak


class IMR(Signal):
    _metadata = ['parameters','posterior_sample', 't_dict']

    def __init__(self, *args, posterior_sample=None, t_dict=None, **kwargs):
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
            print(ifo)
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




class Ringdown(Signal):
    _metadata = ['modes']

    def __init__(self, *args, modes=None, **kwargs):
        super(Ringdown, self).__init__(*args, **kwargs)
        self.modes = qnms.construct_mode_list(modes)

    @property
    def _constructor(self):
        return Ringdown

    @property
    def n_modes(self):
        return len(self.get_parameter('A', []))

    @staticmethod
    def _theta_phi_from_phip_phim(phip, phim):
        return -0.5*(phip + phim), 0.5*(phip - phim)

    @staticmethod
    def _phip_phim_from_theta_phi(theta, phi):
        return phi - theta, -(phi + theta)

    @staticmethod
    def _construct_parameters(ndmin=0, **kws):
        kws = {k.lower(): v for k,v in kws.items()}
        # define parameters that will take precedence for storage
        keys = ['a', 'ellip', 'theta', 'phi', 'omega', 'gamma']
        pars = {k: array(kws.pop(k), ndmin=ndmin) for k in list(kws.keys())
                if k in keys}
        # possibly interpret parameters as coming from a specific stan model
        if 'model' in kws:
            if kws['model'] == 'mchi_aligned':
                # this assumes a template for the oscillatory part like
                #     hp = (1 + cosi**2) * A * cos(wt - phi)
                #     hc = 2*cosi * A * sin(wt - phi)
                # in the parameterization adopted here, this corresponds to 
                #     A = A*(1 + cosi**2), e = 2*cosi / (1 + cosi**2)
                #     phi = phi, theta = 0
                # the ellipticity is already computed within stan, but we need
                # to readjust the definition of the amplitude and add theta
                pars['cosi'] = kws.pop('cosi')*ones_like(pars['a'])
                pars['a'] = pars['a']*(1 + pars['cosi']**2)
                kws['theta'] = zeros_like(pars['a'])
                if 'ellip' not in kws:
                    kws['ellip'] = 2*pars['cosi'] / (1 + pars['cosi']**2)
            elif kws['model'] == 'ftau':
                # this assumes a template for the oscillatory part like
                #     hp = Ax*cos(wt) + Ay*sin(wt) = A*cos(wt - phi)
                #     hc = 0
                # with phi = atan2(Ay, Ax) and A = sqrt(Ax**2 + Ay**2)
                # in the parameterization adopted here, this corresponds to
                #     A = A, ellip = 0, theta = 0, phi = phi
                kws['theta'] = zeros_like(pars['a'])
                kws['ellip'] = zeros_like(pars['a'])
        # obtain frequencies from remnant parameters if necessary
        freq_keys = ['omega', 'gamma', 'f', 'tau']
        if 'modes' in kws and not any([k in kws for k in freq_keys]):
            if 'M' in kws:
                kws['m'] = kws.pop('M')
            kws['approx'] = kws.get('approx', False)
            kws['f'], kws['tau'] = [], []
            for m in kws['modes']:
                f, tau = qnms.KerrMode(m).ftau(kws['chi'], kws['m'],
                                               kws['approx'])
                kws['f'].append(f)
                kws['tau'].append(tau)
        # frequency parameters
        if 'f' in kws and 'omega' not in kws:
            pars['omega'] = 2*pi*array(kws.pop('f'), ndmin=ndmin)
        if 'tau' in kws and 'gamma' not in kws:
            pars['gamma'] = 1/array(kws.pop('tau'), ndmin=ndmin)
        # phase parameters
        if 'phip' in kws and 'phim' in kws:
            theta, phi = Ringdown._theta_phi_from_phip_phim(kws['phip'],
                                                            kws['phim'])
            pars['theta'] = array(theta, ndmin=ndmin)
            pars['phi'] = array(phi, ndmin=ndmin)
        pars.update(kws)
        return pars

    @staticmethod
    def complex_mode(time, omega, gamma, A, ellip, theta, phi):
        """Eq. (8) in Isi & Farr (2021).
        """
        phi_p = phi - theta
        phi_m = - (phi + theta)
        iwt = 1j*omega*time
        vt = gamma*time
        # h = 0.5*A*exp(-time*gamma)*((1 + ellip)*exp(-1j*(wt - phi_p)) +
        #                             (1 - ellip)*exp(1j*(wt + phi_m)))
        h = ((0.5*A)*(1 + ellip))*exp(1j*phi_p - iwt - vt) + \
            ((0.5*A)*(1 - ellip))*exp(1j*phi_m + iwt - vt)
        return h

    #_MODE_PARS = [k.lower() for k in getfullargspec(Ringdown.complex_mode)[0][1:]]
    _MODE_PARS = ['omega', 'gamma', 'a', 'ellip', 'theta', 'phi']

    @classmethod
    def from_parameters(cls, time, t0=0, window=inf, two_sided=True, df_pre=0,
                        dtau_pre=0, mode_isel=None, **kws):
        """Create injection: a sinusoid up to t0, then a damped sinusoiud. The
        (A_pre, df_pre, dtau_pre) parameters can turn the initial sinusoid into
        a sinegaussian, to produce a ring-up.  Can incorporate several modes,
        if (A, phi0, f, tau) are 1D.

        Arguments
        ---------
        mode_isel : list, int
            index or indices of modes to include in template; if `None` 
            includes all modes (default).
        """
        # parse arguments
        all_kws = {k: v for k,v in locals().items() if k not in ['cls','time']}
        all_kws.update(all_kws.pop('kws'))
        modes = all_kws.get('modes', None)

        # reshape arrays (to handle multiple modes)
        t = reshape(time, (len(time), 1))
        signal = zeros(len(time), dtype=complex)

        pars = cls._construct_parameters(ndmin=1, **all_kws)

        # define some masks (pre and post t0) to avoid evaluating the waveform
        # over extremely long time arrays [optional]
        mpost = (time >= t0) & (time < t0 + 0.5*window) 

        # each mode will be a sinusoid up to t0, then a damped sinusoid
        if mode_isel == None:
            mode_isel = slice(None)
        margs = {k: array(pars[k][mode_isel], ndmin=2) for k in cls._MODE_PARS}
        if modes:
            if len(modes) > len(mode_args[0][0]):
                raise ValueError("insufficient parameters provided")
        signal[mpost] = sum(cls.complex_mode(t[mpost]-t0, *margs.values()),
                            axis=1)

        # add a damping to the amplitude near t0 for t < t0
        if two_sided:
            margs['omega'] = margs['omega']*exp(df_pre)
            margs['gamma'] = -margs['gamma']*exp(-dtau_pre)
            mpre = (time < t0) & (time > t0 - 0.5*window)
            signal[mpre] = sum(cls.complex_mode(t[mpre]-t0, *margs.values()),
                               axis=1)
        return cls(signal, index=time, parameters=pars, modes=modes)
    
    def get_parameter(self, k, *args, **kwargs):
        k = k.lower()
        if k == 'f':
            return self.get_parameter('omega', *args, **kwargs)/ (2*pi)
        elif k == 'tau':
            return 1/self.get_parameter('gamma', *args, **kwargs)
        elif k == 'phip' or k == 'phim':
            th = self.get_parameter('theta', *args, **kwargs)
            ph = self.get_parameter('phi', *args, **kwargs)
            d = dict(zip(['phip', 'phim'],
                         self._phip_phim_from_theta_phi(th, ph)))
            return d[k]
        elif k == 'quality':
            # Q = pi*f*tau
            w = self.get_parameter('omega', *args, **kwargs)
            v = self.get_parameter('gamma', *args, **kwargs)
            return 0.5 * w / v
        else:
            return super().get_parameter(k, *args, **kwargs)

    def get_mode_parameters(self, mode):
        try:
            n = int(mode)
        except (TypeError, ValueError):
            n = self.modes.index(qnms.ModeIndex(*mode))
        pars = {k: self.parameters[k][n] for k in self._MODE_PARS}
        return pars

        


def simulated_template(freq, tau, smprate, duration, theta, phi, amplitude, ra,
                       dec, ifos, tgps, ellip, psi=0):
    """Function to make a simulated signal, as in Isi & Farr (2021)
    [`arXiv:2107.05609 <https://arxiv.org/abs/2107.05609>`_] Eqns. (11)-(13),
    here using a "ring-up" followed by a ring-down as modeled by a time decay
    of :math:`\exp(-|t-t_\mathrm{gps}-\Delta t_\mathrm{ifo}|\gamma)`.
            
    Note that in general this function works best for durations much longer
    than the time delay. No noise is overlaid on this template.

    Arguments
    ---------
    freq : list
        list of frequencies of each tone in Hz.
    tau : list
        damping time of each tone in seconds.
    theta : list
        angle of ellipse in :math:`h_+,h_\\times` space, in radians.
    phi : list
        phase offset of sinusoids, in radians.
    ellip : list
        ellipticity of each tone, :math:`-1 \leq \epsilon \leq 1`.
    smprate : float
        sample rate of signal in Hz.
    duration : float
        length of time of the full signal in seconds.
    ra : float
        source right ascension in radians.
    dec : float
        source declination in radians.
    psi : float
        polarization angles for which to evaluate antenna patterns (def. 0).
    ifos : list
        list of ifos as strings, e.g., `["H1", "L1"]`.
    tgps : float
        GPS time for peak of signal.
    amplitude : list
        amplitude of each tone.
        
    Returns
    -------
    sig_dict : dict
        Dict of signal :class:`ringdown.data.TimeSeries` for each IFO.
    modes_dict : dict
        Dict of mode :class:`ringdown.data.TimeSeries` for each ifo.
    time_delay_dict : dict
        Dict of `lal.TimeDelayFromEarthCenter` for each ifo.
    """
    N = int((duration*smprate))
    t = arange(N)/smprate+tgps-duration/2.0 #list of times that will be used as fake data input
    s = TimeSeries(zeros_like(t), index=t) #array for template signal
    hplus = TimeSeries(zeros_like(t), index=t) #array for h_plus polarization, h_plus = hcos*cos(theta) - epsilon*hsin*sin(theta)
    hcross = TimeSeries(zeros_like(t), index=t) #array for h_cross polarization, h_cross = hcos*sin(theta) + epsilon*hsin*cos(theta)
    hsin = TimeSeries(zeros_like(t), index=t) #sine quadrature
    hsin = TimeSeries(zeros_like(t), index=t) #cosine quadrature
    sig_dict = {} #dicts used for template output
    lal_det = {} #ifo information
    modes_dict = {} #individual mode information
    antenna_patterns = {} #antenna patterns projected onto modes
    time_delay_dict = {} #time delays from Earth center
    omega = 2*pi*array(freq) #frequencies
    gamma = 1./array(tau) #damping
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    for ifo in ifos:
        s = s-s #hacky way to zero out the arrays for fresh signal in each ifo
        lal_det[ifo] = lal.cached_detector_by_prefix[ifo]
        antenna_patterns[ifo] = lal.ComputeDetAMResponse(lal_det[ifo].response, ra, dec, psi, gmst)
        time_delay_dict[ifo] = lal.TimeDelayFromEarthCenter(lal_det[ifo].location, ra, dec, tgps)
        modes_dict[ifo]=[]
        for (w,v,A,E,n,th,ph) in zip(omega,gamma,amplitude,ellip,arange(len(omega)),theta,phi):
            hsin = TimeSeries(A*exp(-abs(t-tgps-time_delay_dict[ifo])*v)*sin(w*(t-tgps-time_delay_dict[ifo])-ph), index=t)
            hcos = TimeSeries(A*exp(-abs(t-tgps-time_delay_dict[ifo])*v)*cos(w*(t-tgps-time_delay_dict[ifo])-ph), index=t)
            hplus= hcos*cos(th)-E*hsin*sin(th)
            hcross= hcos*sin(th)+E*hsin*cos(th)
            m = antenna_patterns[ifo][0]*hplus+antenna_patterns[ifo][1]*hcross
            s += m
            modes_dict[ifo].append(m)
        sig_dict[ifo]=s    
    return sig_dict, modes_dict, time_delay_dict
