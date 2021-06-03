from .__init__ import *
from pylab import *

def get_sym_ci(x, cl=.90):
    m = median(x)
    u = percentile(x, 50*(1 + cl))
    d = percentile(x, 50*(1 - cl))
    return m, u-m, m-d

def get_quantile(samples, value):
    return len(samples[samples < value]) / len(samples)

def null_cdf(nobs, nsamp, nbins=None, ndraws=100000):
    nbins = nbins or nsamp/4
    ks = linspace(0, 1, nbins+1)
    hs = []
    for i in range(ndraws):
        qs = random.uniform(0, nsamp+1, size=nobs)/nsamp
        hs.append(histogram(qs, bins=ks)[0])

    chs = cumsum(hs, axis=1)/nobs
    return chs

def pp_plot(qdf, chs=None):
    fig, ax = subplots()
    N = min(inf, len(qdf))
    ks = linspace(0, 1, len(chs))

    m = percentile(chs, 50, axis=0)
    for ci in [99.73, 95.45, 68.27]:
        fill_between(ks[:-1], percentile(chs, 50+0.5*ci, axis=0)-m, 
                     percentile(chs, 50-0.5*ci, axis=0)-m, step='post',
                     color='gray', alpha=0.15)
    axhline(0, c='k');

    for k, v in qdf.items():
        y, _ = histogram(v, bins=ks)
        step(ks[:-1], cumsum(y)/len(qdf)-m, label=k)

    ncol = 2 if len(qdf) > 6 else 1
    legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
           frameon=False, ncol=ncol)
    xlabel(r'$p$');
    ylabel(r'$p-p$');
    title(r'$N = %i$' % len(q_df));
    return figs