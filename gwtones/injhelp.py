from pylab import *
import os
import arviz as az
from . import qnms as uq
from . import data as ud
from . import sampling as us
from . import reconstructwf as rwf
import scipy.linalg as sl
import dask

def get_analytic_data(truth, time_dict, noise_dict, rho_dict, **kws):
    # get signal parameters
    truth = truth.copy()
    defaults = kws.get('defaults', {})
    snr_requested = truth.pop('snr', False)
    for k, v in defaults.items():
        if k not in truth:
            truth[k] = v
    
    M = truth.pop('M', None)
    chi = truth.pop('chi', None)
    if not ('f' in truth and 'tau' in truth):
        truth.update({'f': [], 'tau': []})
        for n in range(len(truth['A'])):
            f, tau = uq.get_ftau(M, chi, n)
            truth['f'].append(f)
            truth['tau'].append(tau)

    truth_1d = {k: np.array(v, ndmin=2) for k,v in truth.items()}
    
    # perturb frequencies
    truth_1d['f'] *= (1. + truth_1d.pop('df', 0))
    truth_1d['tau'] *= (1. + truth_1d.pop('dtau', 0))
    
    # get antenna patterns and trigger times
    ifos = list(time_dict.keys())
    tgps_dict = kws.pop('tgps_dict', None)
    ap_dict = kws.pop('ap_dict', None)
    if not tgps_dict or not ap_dict:
        ra, dec, psi = [kws.pop(k) for k in ['ra', 'dec', 'psi']]
        tgps_dict, ap_dict = ud.get_tgps_aps(tgps_geocent, ra, dec, psi, ifos)

    # produce injection
    template = kws.get('template', uq.ellip_template)
    signal_dict = {i: template(t, fpfc=ap_dict[i], t0=tgps_dict[i], **truth_1d)
                   for i, t in time_dict.items()}
    
    # compute matched-filter SNRs
    def compute_snr_net():
        snrs = [ud.compute_snr(s, noise_dict[i] + s, rho_dict[i])
                for i, s in signal_dict.items()]
        return linalg.norm(snrs)
    
    # rescale to requested SNR
    snr = compute_snr_net()
    if snr_requested:
        while abs(snr_requested - snr) > 0.1:
            snr_ratio = snr_requested / snr
            for ifo in ifos:
                signal_dict[ifo] *= snr_ratio
            truth['A'] = array(truth['A'])*snr_ratio
            snr = compute_snr_net()        
    # create data
    data_dict = {i: s + noise_dict[i] for i,s in signal_dict.items()}
    truth['snr'] = snr
    return data_dict, signal_dict, truth



def get_data(noise_dict=None, raw_time=None, nroll_dict=None,
             n_analyze=None, tgps_geocent=None, duration=None, ds=2,
             flow=20, mtotal=None, rho_dict=None, snr_inj=None, t0=0,
             **kws):
    """ Synthesize data with injection.
    
    Arguments
    ---------
    phase: float
        orbital phase for injection (def. 0)
    nroll_dict: dict
        roll noise for each IFO
    noise_dict: dict
        conditioned noise for each IFO
    n_analyze: int
        length of analysis segment (def. len(noise))
    i0_ref: dict
        noise reference index
    raw_time: array
        raw times
    duration: float
        length of array (in seconds) to generate waveform
    """
    # get noise
    noise_dict = noise_dict.copy()
    ifos = list(noise_dict.keys())
    if not rho_dict and 'L_dict' in kws:
        rho_dict = {i: dot(L, L)[:,0] for i, L in kws['L_dict'].items()}
    Nanalyze = len(rho_dict[ifos[0]]) if rho_dict else len(noise_dict[ifos[0]])
    Nanalyze = n_analyze or Nanalyze
    i0_ref = kws.pop('i0_ref', {i: 0 for i in ifos})
    nroll_dict = nroll_dict or {i: 0 for i in ifos}
    for i, i0 in i0_ref.items():
        noise_dict[i] = roll(noise_dict[i], nroll_dict[i])[i0:i0+Nanalyze]
    
    # get antenna patterns and trigger times
    tgps_dict = kws.pop('tgps_dict', None)
    ap_dict = kws.pop('ap_dict', None)
    if not tgps_dict or not ap_dict:
        ra, dec, psi = [kws.pop(k) for k in ['ra', 'dec', 'psi']]
        tgps_dict, ap_dict = ud.get_tgps_aps(tgps_geocent, ra, dec, psi, ifos)
    
    # get raw signal
    phase = kws.pop('phase', 0)
    inc = kws.pop('theta_jn', 0)
    duration = duration or raw_time[-1] - raw_time[0]
    raw_time_dict = {i: ud.get_raw_time_ifo(tgps, raw_time, duration, ds)
                     for i, tgps in tgps_dict.items()}
    raw_signal_dict = rwf.get_signal_dict(raw_time_dict, 1760, phase=phase, 
                                          mtot_msun=mtotal, approx='NR_hdf5',
                                          nr_path=kws.pop('nr_path'), inclination=inc,
                                          tgps_dict=tgps_dict, ap_dict=ap_dict)[0]
    # condition signal
    cond_signal_dict = {}
    cond_time_dict = {}
    i0_dict = {}
    for i, s in raw_signal_dict.items():
        t = raw_time_dict[i]
        cond_time_dict[i], cond_signal_dict[i] = ud.condition(t, s, ds=ds, flow=flow)
        # the truncation point for the time need not be the same as noise
        i0_dict[i] = argmin(abs(cond_time_dict[i] - tgps_dict[i] - t0))

    if snr_inj:
        # compute matched-filter SNRs
        def compute_snr_net():
            snrs = []
            for ifo, i0 in i0_dict.items():
                s = cond_signal_dict[ifo][i0:i0+Nanalyze]
                n = noise_dict[ifo]
                snrs.append(ud.compute_snr(s, n+s, rho_dict[ifo]))
            return linalg.norm(snrs)

        # rescale to requested SNR
        snr_net = compute_snr_net()
        while abs(snr_inj - snr_net) > 0.1:
            for ifo in ifos:
                cond_signal_dict[ifo] *= (snr_inj / snr_net)
            snr_net = compute_snr_net()

    # crop data to specified duration
    time_dict = {}
    data_dict = {}
    signal_dict = {}
    for ifo, i0 in i0_dict.items():
        time_dict[ifo] = cond_time_dict[ifo][i0:i0+Nanalyze]
        signal_dict[ifo] = cond_signal_dict[ifo][i0:i0+Nanalyze]
        data_dict[ifo] = signal_dict[ifo] + noise_dict[ifo]
    return time_dict, signal_dict, data_dict

def run(model=None, **kws):
    if kws.get('use_dask', False):
        delayed = dask.delayed
    else:
        delayed = lambda x: x
    # sampler settings
    L_dict = kws.get('L_dict', None)
    if not L_dict:
        rhos = kws['rho_dict']
        L_dict = {i: delayed(linalg.cholesky)(delayed(sl.toeplitz)(r)) 
                  for i,r in rhos.items()}
    n = kws.pop('thin', 1)
    n_chains = kws.pop('n_chains', 4)
    n_jobs = kws.pop('n_jobs', n_chains)
    n_iter = kws.pop('n_iter', 2000*n)
    stan_kws = {
        'iter': n_iter,
        'thin': n,
        'init': (kws.pop('init_dict', None),)*n_chains,
        'n_jobs': n_jobs,
        'chains': n_chains,
    }
    stan_kws.update(kws.pop('stan_kws', {}))
    if stan_kws['init'][0]:
        stan_kws['init'] = (stan_kws['init'][0],)*n_chains
    else:
        del stan_kws['init']
    
    # data settings
    nmode = kws.pop('nmode', 2)
    prior = kws.pop('prior', {})
    ifos = list(L_dict.keys())
    tgps_dict = kws.pop('tgps_dict', None)
    ap_dict = kws.pop('ap_dict', None)
    if not tgps_dict or not ap_dict:
        ra, dec, psi = [kws.pop(k) for k in ['ra', 'dec', 'psi']]
        tgps_dict_ap_dict = delayed(ud.get_tgps_aps)(kws['tgps_geocent'], ra, dec, psi, ifos)
        tgps_dict, ap_dict = tgps_dict_ap_dict[0], tgps_dict_ap_dict[1]
    tsd = delayed(get_data)(tgps_dict=tgps_dict, ap_dict=ap_dict, **kws)
    time_dict, signal_dict, data_dict = tsd[0], tsd[1], tsd[2]
    data = {
        'nobs': delayed(len)(tgps_dict),
        'nsamp': delayed(len)(delayed(list)(time_dict.values())[0]),
        'nmode': nmode,

        't0': delayed(list)(tgps_dict.values()),
        'times': delayed(list)(time_dict.values()),
        'strain': delayed(list)(data_dict.values()),

        'L': delayed(list)(L_dict.values()),

        'MMin': 50.0,
        'MMax': 100.0,

        'FpFc': delayed(list)(ap_dict.values()),

        'dt_min': -1e-5,
        'dt_max': 1e-5,

        'Amax': 1e-20,

        'df_max': 0.99,
        'dtau_max': 0.99,

        'perturb_f': zeros(nmode),
        'perturb_tau': zeros(nmode),
        'only_prior': 0,
    }
    data.update(prior)
    fit = delayed(model.sampling)(data=data, **stan_kws)
    return delayed(az.convert_to_inference_data)(fit)

def run_many(pars, nrolls=None, fits_dict=None, ijs=None, run_name=None,
             **kwargs):
    if kwargs.get('use_dask', False):
        delayed = dask.delayed
    else:
        delayed = lambda x: x
    rerun = kwargs.pop('rerun', False)
    retry = kwargs.pop('retry', False)
    nmax = kwargs.pop('nmax', 1)
    # select run name
    name = 'gr'
    prior = kwargs.get('prior', {})
    perts = [k for k in prior if 'perturb' in k]
    for k in perts:
        if sum(kwargs['prior'][k]) > 0:
            name = 'df1_dtau1'
    name = kwargs.pop('name', name)
    # initialize output
    fits_dict = fits_dict or {}
    nrolls = [None] if nrolls is None else nrolls
    niter = len(nrolls) * len(pars)
    # iterate over sky locs and noise instances
    for j, pd in enumerate(pars):
        if j not in fits_dict:
            fits_dict[j] = {}
        for i, nr in enumerate(nrolls):
            # attempt to load cache
            fit_path = run_name.format(nmax=nmax, model=name, M=kwargs['mtotal'],
                                       snr=kwargs.get('snr_inj', 0), i=i, **pd)
            # define condition for loading cache
            c = [os.path.isfile(fit_path), i not in fits_dict[j], not rerun]
            # load cache or run
            if ijs is None or (i, j) in ijs:
                if all(c):
                    fits_dict[j][i] = delayed(az.from_json)(fit_path)
                else:
                    # print('{}/{}'.format(j*len(nrolls)+i+1, niter))
                    fits_dict[j][i] = run(nmode=nmax+1, nroll_dict=nr, 
                                          **kwargs, **pd)
    
    # retry and store
    def retry_fit(i, j):
        fit = fits_dict[j][i]
        pd = pars[j]
        # check effective number of samples
        neff = us.get_neff(fit, relative=False)
        if neff < kwargs['nsamp']:
            print('Retrying {}/{}'.format(j*len(nrolls)+i+1, niter))
            kws = kwargs.copy()
            ni = 2*fit.posterior.M.values.shape[1]  # x2 to cut warmup
            kws.update({'stan_kws': {
                'control': {'metric': 'dense_e', 'adapt_delta': 0.9},
            }, 'n_iter': min(ni*int(kwargs['nsamp']/neff), 4000)})
            return run(nmode=nmax+1, nroll_dict=nr, **kws, **pd)
        
    if retry:
        for j, pd in enumerate(pars):
            for i, nr in enumerate(nrolls):
                fits_dict[j][i] = delayed(retry_fit)(i, j)

    # cache
    def save(i, j, fit):
        fit_path = run_name.format(nmax=nmax, model=name, M=kwargs['mtotal'],
                                   snr=kwargs.get('snr_inj', 0), i=i, **pars[j])
        if rerun or not os.path.isfile(fit_path):
            fit.to_json(fit_path)
        return fit
    
    for j, pd in enumerate(pars):
        for i, nr in enumerate(nrolls):
            fits_dict[j][i] = delayed(save)(i, j, fits_dict[j][i])
    
    return dask.compute(fits_dict)[0]