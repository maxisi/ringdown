from .__init__ import *
from pylab import *
import warnings
import pandas as pd
from .sns_patch import *

def get_label(par, i=0, unit=None):
    ks = {
        'tau': r'\tau',
        'phi0': r'\phi',
    }
    s = r'$%s_%i$' % (ks.get(par, par), i)
    if unit:
        s += ' (%s)' % unit
    return s

def corner(df, truths=None, levels=[0.9, 1-exp(-0.5)],
           lims=None, scale=None, crosshairs=None, diag_hist=True, 
           truth_kw=None, upper_kw=None, lower_kw=None, ci_kw=None,
           diag_kw=None, plot_points=True, ci=None, **kwargs):
    with sns.plotting_context('paper', font_scale=1.5, rc=rcParams):
        
        hue = kwargs.pop('hue', 'varied')
        g = sns.PairGrid(df, vars=[k for k in df.columns if k != hue],
                         diag_sharey=False, hue=hue, **kwargs)

        # diagonal
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")
            if diag_hist:
                kw = dict(lw=1.5, element='step', stat='density', fill=False)
                kw.update(diag_kw or {})
                g.map_diag(sns.histplot, **kw)
            else:
                kw = dict(lw=1.5)
                kw.update(diag_kw or {})
                g.map_diag(sns.kdeplot, **kw)
        
        # lower
        kw = dict(alpha=0.1, marker='.')
        kw.update(lower_kw or {})
        if plot_points:
            g.map_lower(scatter, **kw)
        
        # upper
        kw = dict(alpha=0.7, levels=levels, use_map=False)
        kw.update(upper_kw or {})
        if kwargs.get('corner', False):
            g.map_lower(kdeplot_2d_clevels, **kw)
        else:
            g.map_upper(kdeplot_2d_clevels, **kw)

    #     for i in range(2):
    #         g.axes[i,0].set_xlim(g.axes[0,0].get_ylim())

        if truths is not None:
            kw = dict(ls=':', c='gray', alpha=0.9)
            kw.update(truth_kw or {})
            for i, (row, ti) in enumerate(zip(g.axes, truths)):
                for j, (ax, tj) in enumerate(zip(row, truths)):
                    if ax:
                        if tj is not None: 
                            ax.axvline(tj, **kw)
                        if i != j and ti is not None:
                            ax.axhline(ti, **kw)
        
        if crosshairs is not None:
            for i, row in enumerate(g.axes):
                for j, ax in enumerate(row):
                    if i != j and ax:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        ax.scatter([crosshairs[i]], [crosshairs[j]], marker='+', c='k')
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
        
        if lims is not None:
            for i, row in enumerate(g.axes):
                for j, ax in enumerate(row):
                    if ax:
                        if lims[j] is not None:
                            ax.set_xlim(lims[j])
                        if lims[i] is not None:
                            ax.set_ylim(lims[i])
        
        if ci:
            keys = df[hue].unique()
            p = kwargs.get('palette', sns.color_palette(n_colors=len(keys)))
            kw = dict(alpha=0.5, ls='--')
            kw.update(ci_kw or {})
            for key, c in zip(keys, p):
                df_i = df[df[hue] == key]
                pars = [k for k in df_i.columns if k != hue]
                for i, k in enumerate(pars):
                    if not df_i[k].empty:
                        for cl in [50*(1-ci), 100-50*(1-ci)]:
                            g.axes[i,i].axvline(percentile(df_i[k], cl), c=c, **kw)
        return g
    

def corner_par(par, fits, truths=None, nsamples=np.inf, truth_palette=None,
               truth_kw=None, **kwargs):
    scale = kwargs.get('scale', 1)
    u = None if scale == 1 else r'$\times 10^{%i}$' % log10(scale)  
    df = pd.DataFrame()
    for i, fit in enumerate(fits):
        nmodes = fit.posterior.A.values.shape[2]
        d = {get_label(par, n, u): fit.posterior[par].values[:,:,n].flatten() / scale
             for n in range(nmodes)}
        d.update({'varied': i})
        df_fit = pd.DataFrame(data=d)
        # throw out points outside specified limits
        lims = kwargs.get('lims')
        if lims is not None:
            ks = [k for k in df_fit.columns if k != 'varied']
            m = all([(l[0] < df_fit[k]) & (l[1] > df_fit[k]) for k,l in zip(ks, lims)], axis=0)
            df_fit = df_fit[m]
        # downselect to specific number of samples
        ns = min(len(df_fit), nsamples)
        idxs = np.random.choice(range(len(df_fit)), ns, replace=False)
        df = df.append(df_fit.iloc[idxs])    
    g = corner(df, **kwargs)
    
    if truths is not None:
        tc = sns.color_palette(truth_palette or kwargs.get('palette'))
        if scale is not None:
            try:
                scale[0]
            except TypeError:
                scale = [scale]*len(g.axes)
        for c, truth in zip(tc, truths):
            if truth:
                n = min(len(g.axes), len(truth[par]))
                for i, row in enumerate(g.axes[:n]):
                    for j, ax in enumerate(row[:n]):
                        kw = dict(ls=':', c=c, alpha=0.9)
                        kw.update(truth_kw or {})
                        ax.axvline(truth[par][j]/scale[j], **kw)
                        if i != j:
                            ax.axhline(truth[par][i]/scale[i], **kw)
    return g, df

def corner_fit(fits, pars=('f', 'tau'), scales=None, nsamples=np.inf,
               truths=None, **kwargs):
    scales = scales or {}
    truths = truths or {}
    df = pd.DataFrame()
    for i, fit in enumerate(fits):
        d, s, t = {}, [], []
        for par in pars:
            scale = scales.get(par, 1)
            u = None if scale==1 else str(scale)
            p = fit.posterior[par]
            if len(p.shape) > 2:
                par_n = p.shape[2]
                for n in range(par_n):
                    d[get_label(par, n, u)] = p.values[:,:,n].flatten() / scale
                    s.append(scale)
                    t.append(truths.get(par, [None]*par_n)[n])
            else:
                d[get_label(par)] = p.values.flatten() / scale
                s.append(scale)
                t.append(truths.get(par, None))
        d.update({'varied': i})
        df_fit = pd.DataFrame(data=d)
        # throw out points outside specified limits
        lims = kwargs.get('lims')
        if lims is not None:
            ks = [k for k in df_fit.columns if k != 'varied']
            m = all([(l[0] < df_fit[k]) & (l[1] > df_fit[k]) for k,l in zip(ks, lims)], axis=0)
            df_fit = df_fit[m]
        # downselect to specific number of samples
        ns = min(len(df_fit), nsamples)
        idxs = np.random.choice(range(len(df_fit)), ns, replace=False)
        df = df.append(df_fit.iloc[idxs])
    g = corner(df, scale=s, truths=t, **kwargs)
    return g, df

def joint_ftau(fit, truth=None, palette=None, levels=[0.9, 1-exp(-0.5)],
               cis=[16, 84], truth_kw=None, ci_kw=None, label=None,
               crosshairs=True, joint_kw=None, marg_kw=None, **kwargs):
    cs = sns.color_palette(palette)
    label = label or '{n}'
    with sns.plotting_context('paper', font_scale=1.5, rc=rcParams):
        g = sns.JointGrid([], [], **kwargs)
        for i in fit.posterior.A_dim_0.values:
            g.x = fit.posterior.f.values[:,:,i].flatten()
            g.y = fit.posterior.tau.values[:,:,i].flatten()*1E3
            l, = g.ax_joint.plot([], [], label=label.format(n=i), c=cs[i])
            c = l.get_color()
            kw = dict(colors=[c,], cmap=None, levels=levels)
            kw.update(joint_kw or {})
            g.plot_joint(kdeplot_2d_clevels, **kw)
            kw = dict(c=c)
            kw.update(marg_kw or {})
            g.plot_marginals(sns.kdeplot, **kw)

            if truth:
                kw = dict(ls=':', c=c, alpha=0.9)
                kw.update(truth_kw or {})
                g.ax_joint.axvline(truth['f'][i], **kw)
                g.ax_joint.axhline(truth['tau'][i]*1E3, **kw)
                if crosshairs:
                    g.ax_joint.plot([truth['f'][i]], [truth['tau'][i]*1E3], 
                                       color=c, marker='+', mew=1.5, ms=10, zorder=0)

            if cis is not None:
                kw = dict(c=c, ls='--')
                kw.update(ci_kw or {})
                for ci in cis:
                    g.ax_marg_x.axvline(percentile(g.x, ci), **kw)
                    g.ax_marg_y.axhline(percentile(g.y, ci), **kw)
        g.set_axis_labels(r'$f_n$ (Hz)', r'$\tau_n$ (ms)')
        return g
    
def plot_mchi(fit, truth=None, d=1, g=None, levels=[0.9, 0.5, 0.1], points=True,
              truth_kws=None, xlim=(50, 100), ylim=(0, 1), **kws):
    with sns.plotting_context('paper', font_scale=1.5, rc=rcParams):
        g = g or sns.JointGrid([], [], xlim=xlim, ylim=ylim)
        g.x = fit.posterior.M[:,::d].values.flatten()
        g.y = fit.posterior.chi[:,::d].values.flatten()
        l, = g.ax_joint.plot([], [], c=kws.pop('c', kws.pop('color', None)))
        c = l.get_color()
        if points:
            g.plot_joint(scatter, color=c, alpha=0.03, marker='.')
        g.plot_joint(kdeplot_2d_clevels, colors=[c,], cmap=None, levels=levels,
                    linewidths=linspace(1, 2, 3), **kws)
        g.plot_marginals(sns.kdeplot, c=c)
        
        if truth:
            tkws = dict(c=c, ls='--')
            tkws.update(truth_kws or {})
            
            g.ax_joint.axvline(truth['M'], **tkws)
            g.ax_joint.axhline(truth['chi'], **tkws)
            g.ax_joint.plot(truth['M'], truth['chi'], marker='+', markersize=10,
                            markeredgewidth=1.5, **tkws)
        g.set_axis_labels(r'$M\, (M_\odot)$', r'$\chi$');
    return g

def plot_ftau(fit, truth=None, d=1, n=0, xlim=(200, 300), ylim=(2, 6), g=None, 
              levels=[0.9, 0.5, 0.1], points=True, truth_kws=None, **kws):
    with sns.plotting_context('paper', font_scale=1.5, rc=rcParams):
        g = g or sns.JointGrid([], [], xlim=xlim, ylim=ylim)
        g.x = fit.posterior.f[:,::d,n].values.flatten()
        g.y = fit.posterior.tau[:,::d,n].values.flatten() * 1E3
        l, = g.ax_joint.plot([], [], c=kws.pop('c', kws.pop('color', None)))
        c = l.get_color()
        if points:
            g.plot_joint(scatter, alpha=0.05, marker='.', color=c)
        g.plot_joint(kdeplot_2d_clevels, colors=[c,], cmap=None, levels=levels,
                    linewidths=linspace(1, 1.5, 3))
        g.plot_marginals(sns.kdeplot, c=c)
    if truth:
        tkws = dict(c=c, ls='--')
        tkws.update(truth_kws or {})
        
        g.ax_joint.axvline(truth['f'][n], **tkws);
        g.ax_joint.axhline(truth['tau'][n]*1E3, **tkws);
        g.ax_joint.plot(truth['f'][n], truth['tau'][n]*1E3, marker='+', markersize=10,
                        markeredgewidth=1.5, **tkws);
    return g

def plot_modes_single(fit, truth=None, thin=1, time=None, maxp=False, data=False, 
                      defaults=None, ixs=None, get_data=None):
    d = thin
    defaults = defaults or {}
    if truth:
        data, signal, _  = get_data(truth, defaults)
    lp = fit.sample_stats.lp.values
    imax = unravel_index(argmax(lp), lp.shape)
    
    if time is None:
        t = arange(fit.posterior.h_det.shape[-1])
    else:
        t = time
        
    fig, ax = subplots(figsize=(fig_width, fig_height))
    if data is not None:
        ax.plot(t, data, label='data', c='gray', lw=1, alpha=0.5)
    if truth:
        ax.plot(t, signal, label='truth', c='gray', lw=1)

    hsamps = fit.posterior.h_det[:,::d,:]
    a = (0, 1)
    if ixs is not None:
        hsamps = hsamps.values[ixs]
        a = 0
    if maxp:
        h = fit.posterior.h_det[imax[0],imax[1],:]
    else:
        h = mean(hsamps, axis=a)
    l, = ax.plot(t, h, c='gray', lw=0, alpha=0.3)
    ys = [percentile(hsamps, p, axis=a) for p in [5, 95]]
    ax.fill_between(t, ys[0], ys[1], lw=0, color=l.get_color(), alpha=0.2, label='rec.')

    if maxp:
        hmodes = fit.posterior.h_det_mode[imax[0],imax[1],:,:]
    else:
        hmodes = mean(fit.posterior.h_det_mode[:,::d,:,:], axis=(0,1))
    # s = zeros(len(time))
    n_colors = {}
    for n, h in enumerate(hmodes):
        l, = ax.plot(t, h, lw=0, ls='--')

        hsamps = fit.posterior.h_det_mode[:,::d,n,:]
        if ixs is not None:
            hsamps = hsamps.values[ixs]
        ys = [percentile(hsamps, p, axis=a) for p in [5, 95]]
        ax.fill_between(t, ys[0], ys[1], lw=0, color=l.get_color(), alpha=0.1)
        n_colors[n] = l.get_color()

    if truth:
        for n in range(len(truth['A'])):
            truth_n = {k: v.copy() for k, v in truth.items() if k != 'snr'}
            truth_n['A'] = [v if na==n else 0 for na,v in enumerate(truth_n['A'])]
            _, s, _ = get_data(truth_n, defaults)
            kws = {}
            if n in n_colors: kws['c'] = n_colors[n]
            ax.plot(t, s, lw=1, **kws)
            # s += sdict_n[ifo]
    # add extra modes that may have been present in the injection
    
    # if any(s):
    #     ax.plot(time, s, ls=':', c='k')

    ax.legend(loc='upper right', fontsize=10);
    ax.set_ylabel(r'$h(t)$');
    ax.set_xlabel(r'$t - t_0$ (s)');
    return fig

def plot_modes(fit, truth=None, thin=1, plot_mean=False, plot_maxp=False, data=False,
               defaults=None, ixs=None, time_dict=None, tgps_dict=None, 
               get_data=None, ifo_ixs=None, figsize=None):
    d = thin
    defaults = defaults or {}
    if truth or data:
        data_dict, signal_dict, _  = get_data(truth, defaults)
    lp = fit.sample_stats.lp.values
    imax = unravel_index(argmax(lp), lp.shape)
    
    time_dict = time_dict or {}
    tgps_dict = tgps_dict or {}
    
    nifo = fit.posterior.h_det.shape[2]
    if ifo_ixs is None:
        ifo_ixs = range(nifo)
    else:
        nifo = len(ifo_ixs)
        
    if figsize is None:
        figsize = (fig_width*nifo, fig_height)
        
    fig, axs = subplots(1, nifo, figsize=figsize)
    for i in ifo_ixs:
        try:
            ax = axs[i]
        except TypeError:
            ax = axs
        if time_dict:
            ifo = list(time_dict.keys())[i]
        else:
            ifo = 'IFO %i' % i
        time = array(time_dict.get(ifo, arange(fit.posterior.h_det.shape[-1])))
        time -= tgps_dict.get(ifo, 0)
        if data:
            ifo = list(data_dict.keys())[i]
            ax.plot(time, data_dict[ifo], label='data', c='gray', lw=1, alpha=0.5)
        if truth:
            ifo = list(signal_dict.keys())[i]
            ax.plot(time, signal_dict[ifo], label='truth', c='gray', lw=1)

        hsamps = fit.posterior.h_det[:,::d,i,:]
        a = (0, 1)
        if ixs is not None:
            hsamps = hsamps.values[ixs]
            a = 0
        if plot_maxp:
            h = fit.posterior.h_det[imax[0],imax[1],i,:]
        else:
            h = mean(hsamps, axis=a)
        lw = 1 if plot_mean or plot_maxp else 0
        l, = ax.plot(time, h, c='gray', lw=lw, ls=':', alpha=1)
        ys = [percentile(hsamps, p, axis=a) for p in [5, 95]]
        ax.fill_between(time, ys[0], ys[1], lw=0, color=l.get_color(), alpha=0.2, label='rec.')

        # s = zeros(len(time))
        nmodes = fit.posterior.h_det_mode.shape[3]
        n_colors = {}
        for n in range(nmodes):
            if plot_maxp:
                h = fit.posterior.h_det_mode[imax[0],imax[1],i,n,:]
            elif ixs is None:
                h = mean(fit.posterior.h_det_mode[:,::d,i,n,:], axis=(0, 1))
            else:
                h = mean(fit.posterior.h_det_mode[:,::d,i,n,:].values[ixs], axis=0)
            l, = ax.plot(time, h, lw=lw, ls=':')
            hsamps = fit.posterior.h_det_mode[:,::d,i,n,:]
            if ixs is not None:
                hsamps = hsamps.values[ixs]
            ys = [percentile(hsamps, p, axis=a) for p in [5, 95]]
            ax.fill_between(time, ys[0], ys[1], lw=0, color=l.get_color(), alpha=0.1)
            n_colors[n] = l.get_color()

        if truth:
            for n in range(len(truth['A'])):
                t = {k: v.copy() for k, v in truth.items() if k != 'snr'}
                t['A'] = [v if na==n else 0 for na,v in enumerate(t['A'])]
                _, sdict_n, _ = get_data(t, defaults)
                kws = {'lw': 1}
                if n in n_colors: kws['c'] = n_colors[n]
                ax.plot(time, sdict_n[ifo], **kws)
                # s += sdict_n[ifo]

        ax.legend(loc='upper right', fontsize=10);
        ax.set_ylabel(r'$h(t)$');
        ax.set_xlabel(r'$t - t_0$ (s) [%s]' % ifo);
    return fig


def plot_dfdtau(fits, truths=None, thins=None, cs=None, g=None, levels=[0.9,],
                plot_points=False, lws=linspace(1.5, 1.5, 3), lw=None,
                lss=['-'], ls=None,
                truth_kw=None, labels=None):
    truths = [None for f in fits] if truths is None else truths
    thins = [1 for f in fits] if thins is None else thins
    cs = cs or sns.color_palette(n_colors=len(fits))
    labels = range(len(fits)) if labels is None else labels
    with sns.plotting_context('paper', font_scale=1.5, rc=rcParams):
        g = g or sns.JointGrid([], [], xlim=(-1,1), ylim=(-1,1))
        for c, l, fit, truth, d in zip(cs, labels, fits, truths, thins):
            g.x = fit.posterior.df.values[:,::d,1].flatten()
            g.y = fit.posterior.dtau.values[:,::d,1].flatten()
            c = g.ax_joint.plot([], [], c=c, label=l)[0].get_color()
            if plot_points:
                g.plot_joint(scatter, color=c, marker='.', alpha=0.2)
            g.plot_joint(kdeplot_2d_clevels, colors=[c,], cmap=None, levels=levels,
                        linewidths=lws, xlow=-1, xhigh=1, ylow=-1, yhigh=1, linestyles=lss)
            g.plot_marginals(sns.kdeplot, c=c, lw=lw or lws[0], ls=ls or lss[0])
            
            if truth:
                tkw = dict(ls=':', c=c, alpha=0.5, marker='+', markersize=10,
                           markeredgewidth=1.5)
                tkw.update(truth_kw or {})
                mkw = {k: tkw.pop(k) for k in list(tkw.keys()) if 'marker' in k}
                g.ax_joint.axvline(truth['df'][1], **tkw)
                g.ax_joint.axhline(truth['dtau'][1], **tkw)
                g.ax_joint.plot(truth['df'][1], truth['dtau'][1], **tkw, **mkw)

#             g.x = fits['_df'][n].posterior.df.values[:,:,1].flatten()
#             g.y = fits['_dtau'][n].posterior.dtau.values[:,:,1].flatten()
#             g.plot_marginals(sns.kdeplot, c=c, alpha=0, lw=2, ls='--')

        g.set_axis_labels(r'$\delta f_{221}$', r'$\delta\tau_{221}$')
    return g