__all__ = ['Ringdown']

import numpy as np
from .core import Signal
from .. import qnms
from .. import indexing


class Ringdown(Signal):
    _metadata = ['modes']
    _MODELS = ['ringdown']

    def __init__(self, *args, modes=None, **kwargs):
        super(Ringdown, self).__init__(*args, **kwargs)
        self.modes = indexing.ModeIndexList(modes)

    @property
    def _constructor(self):
        return Ringdown

    @property
    def n_modes(self):
        return len(self.get_parameter('f', []))

    @staticmethod
    def _theta_phi_from_phip_phim(phip, phim):
        return -0.5*(phip + phim), 0.5*(phip - phim)

    @staticmethod
    def _phip_phim_from_theta_phi(theta, phi):
        return phi - theta, -(phi + theta)

    @staticmethod
    def complex_mode(time: np.ndarray,
                     omega: float | np.ndarray,
                     gamma: float | np.ndarray,
                     a: float | np.ndarray,
                     ellip: float | np.ndarray,
                     theta: float | np.ndarray,
                     phi: float | np.ndarray) -> np.ndarray:
        """ Compute complex-valued ringdown mode waveform, as given by
        Eq. (8) in Isi & Farr (2021), namely

        .. math::
            h = \\frac{1}{2} A e^{-t \\gamma}*\\left((1 + \\epsilon)
            e^{-i(\\omega t - \\phi_p)} + (1 - \\epsilon)
            e^{i (\\omega*t + \\phi_m)}\\right)

        where :math:`\\phi_p = \\phi - \\theta` and
        :math:`\\phi_m = -(\\phi + \\theta)`.

        Arguments
        ---------
        time : array
            time array [t].
        omega : float
            angular frequency of the mode [1/t].
        gamma : float
            damping rate of the mode [1/t].
        a : float
            amplitude of the mode.
        ellip : float
            ellipticity of the mode.
        theta : float
            angle of ellipse in :math:`h_+,h_\\times` space, in radians.
        phi : float
            phase angle in radians.

        Returns
        -------
        h : array
            complex-valued waveform.
        """
        phi_p = phi - theta
        phi_m = - (phi + theta)
        iwt = 1j*omega*time
        vt = gamma*time
        # h = 0.5*A*exp(-time*gamma)*((1 + ellip)*exp(-1j*(wt - phi_p)) +
        #                             (1 - ellip)*exp(1j*(wt + phi_m)))
        h = ((0.5*a)*(1 + ellip))*np.exp(1j*phi_p - iwt - vt) + \
            ((0.5*a)*(1 - ellip))*np.exp(1j*phi_m + iwt - vt)
        return h

    # _MODE_PARS = [k.lower()
    # for k in getfullargspec(Ringdown.complex_mode)[0][1:]]
    _MODE_PARS = ['omega', 'gamma', 'a', 'ellip', 'theta', 'phi']

    @staticmethod
    def _construct_parameters(ndmin=0, **kws):
        kws = {k.lower(): v for k, v in kws.items()}
        # define parameters that will take precedence for storage
        pars = {k: np.array(kws.pop(k), ndmin=ndmin) for k in list(kws.keys())
                if k in Ringdown._MODE_PARS}
        # obtain frequencies from remnant parameters if necessary
        freq_keys = ['omega', 'gamma', 'f', 'tau']
        if 'modes' in kws and not any([k in kws for k in freq_keys]):
            kws['approx'] = kws.get('approx', False)
            kws['f'], kws['tau'] = [], []
            for m in kws['modes']:
                f, tau = qnms.KerrMode(m).ftau(kws['chi'], kws['m'],
                                               kws['approx'])
                kws['f'].append(f)
                kws['tau'].append(tau)
        # frequency parameters
        if 'f' in kws and 'omega' not in kws:
            pars['omega'] = 2*np.pi*np.array(kws.pop('f'), ndmin=ndmin)
        if 'tau' in kws and 'gamma' not in kws:
            pars['gamma'] = 1/np.array(kws.pop('tau'), ndmin=ndmin)
        pars.update(kws)
        return pars

    @classmethod
    def from_parameters(cls,
                        time: np.array,
                        t0: float = 0,
                        signal_buffer: float = np.inf,
                        two_sided: bool = True,
                        df_pre: float = 0,
                        dtau_pre: float = 0,
                        mode_isel: float = None,
                        **kws):
        """Create injection: a sinusoid up to t0, then a damped sinusoid. The
        (A_pre, df_pre, dtau_pre) parameters can extend the damped sinusoid
        backwards into  a sinusoid or a ring-up, to produce a ring-up.  Can
        incorporate several modes, if (A, phi0, f, tau) are 1D.

        Arguments
        ---------
        time : array
            time array [t].
        t0 : float
            ringdown model start time.
        signal_buffer : float
            window of time around t0 over which to evaluate model (default
            np.inf, the full time segment).
        two_sided : bool
            whether to include a ring-up in times preceding t0 (default True).
        df_pre : float
            frequency shift for ring-up (default 0).
        dtau_pre : float
            damping shift for ring-up (default 0).
        mode_isel : list, int
            index or indices of modes to include in template;
            if `None` includes all modes (default).
        """
        # parse arguments
        all_kws = {k: v for k, v in locals().items() if k not in [
            'cls', 'time']}
        all_kws.update(all_kws.pop('kws'))
        modes = all_kws.get('modes')

        # reshape arrays (to handle multiple modes)
        t = np.reshape(time, (len(time), 1))
        signal = np.zeros(len(time), dtype=complex)

        pars = cls._construct_parameters(ndmin=1, **all_kws)

        # define some masks (pre and post t0) to avoid evaluating the waveform
        # over extremely long time arrays [optional]
        mpost = (time >= t0) & (time < t0 + 0.5*signal_buffer)

        # each mode will be a sinusoid up to t0, then a damped sinusoid
        if mode_isel is None:
            mode_isel = slice(None)
        margs = {k: np.array(pars[k][mode_isel], ndmin=2)
                 for k in cls._MODE_PARS}
        if modes:
            if len(modes) > len(margs[cls._MODE_PARS[0]][0]):
                raise ValueError("insufficient parameters provided")
        signal[mpost] = np.sum(cls.complex_mode(t[mpost]-t0, *margs.values()),
                               axis=1)

        # add a damping to the amplitude near t0 for t < t0
        if two_sided:
            margs['omega'] = margs['omega']*np.exp(df_pre)
            margs['gamma'] = -margs['gamma']*np.exp(-dtau_pre)
            mpre = (time < t0) & (time > t0 - 0.5*signal_buffer)
            signal[mpre] = \
                np.sum(cls.complex_mode(t[mpre]-t0, *margs.values()), axis=1)
        return cls(signal, index=time, parameters=pars, modes=modes)

    def get_parameter(self, k, *args, **kwargs):
        k = k.lower()
        if k == 'f':
            return self.get_parameter('omega', *args, **kwargs) / (2*np.pi)
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
        n = self.modes.index(mode)
        pars = {k: self.parameters[k][n] for k in self._MODE_PARS}
        return pars


Signal._register_model(Ringdown)
