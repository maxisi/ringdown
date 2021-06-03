from pylab import *
import arviz as az

DEF_KEYS = ('M', 'chi', 'A', 'ellip', 'theta', 'phi0', 'df', 'dtau')

def get_neff(fit, keys=DEF_KEYS, **kws):
    keys = [k for k in keys if k in fit.posterior]
    kws['relative'] = kws.get('relative', True)
    # compute effective number of samples for each parameter
    esss = az.stats.diagnostics.ess(fit, var_names=list(keys), **kws)
    # find minimum number of effective samples for this fit
    return min([min(atleast_1d(esss[k])) for k in keys])

def get_thin(*args, **kwargs):
    return int(round(1/get_neff(*args, **kwargs)))

def get_neff_dict(all_fits, **kws):
    neffs = {k: [] for k in all_fits}
    for i, fits in all_fits.items():
        for j, fit in fits.items():
            neffs[i].append(get_neff(fit, **kws))
    return neffs

def get_thin_dict(all_fits, **kws):
    thins = {k: [] for k in all_fits}
    for i, fits in all_fits.items():
        for j, fit in fits.items():
            thins[i].append(get_thin(fit, **kws))
    return thins

# def run_2d(pars_dict, fits_dict=None, ijs=None, **kwargs):
#     rerun = kwargs.pop('rerun', False)
#     retry = kwargs.pop('retry', False)
#     nmax = kwargs.pop('nmax', 1)
#     # initialize output
#     if fits_dict is None:
#         fits_dict = {}
#     # iterate over sky locs and noise instances
#     for i, fits in :
#         if j not in fits_dict:
#             fits_dict[j] = {}
#         for i, nr in enumerate(nrolls):
#             # attempt to load cache
#             kws = dict(zip(['ra', 'dec', 'psi'], rdp))
#             fit_path = run_name.format(ifos=ifostr, nmax=nmax, model=name,
#                                        snr=snr_inj, M=mtotal, i=i, **kws)
#             # define condition for loading cache
#             c = [os.path.isfile(fit_path), i not in fits_dict[j], not rerun]
#             kws['nroll_dict'] = dict(zip(ifos, nr))
#             kwargs.update(kws)
#             # load cache or run
#             if ijs is None or (i, j) in ijs:
#                 if all(c):
#                     print(fit_path)
#                     fits_dict[j][i] = fit = az.from_json(fit_path)
#                 else:
#                     print('{}/{}'.format(j*nnoise+i+1, niter))
#                     fits_dict[j][i] = fit = run(nmode=nmax+1, **kwargs)
#                     fit.to_json(fit_path)
#             if retry:
#                 # check effective number of samples
#                 neff = us.get_neff(fit, relative=True)
#                 if neff < 0.1:
#                     print('Retrying {}/{}'.format(j*nnoise+i+1, niter))
#                     kwargs.update({'stan_kws': {
#                         'chains': 8,
#                         'n_jobs': 8,
#                         'adapt_delta': 0.9
#                     }})
#                     fits_dict[j][i] = fit = run(nmode=nmax+1, **kwargs)
#                     fit.to_json(fit_path)
#     return fits_dict
