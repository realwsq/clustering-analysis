import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from utils.dr_algo import dr

def hierarchy_trend(results, area2H, quantity, ax=None):
    H = []
    keys = list(area2H.keys())
    for region in results['region']:
        if region in keys:
            H.append(area2H[region])
        else:
            H.append(np.nan)
    H = np.asarray(H)
    
    y = results[quantity][np.isnan(H) == 0]
    x = H[np.isnan(H) == 0]
    reg = results['region'][np.isnan(H) == 0]

    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4))
    else:
        f = None # placeholder
    for i in range(len(reg)):
        ax.text(x[i], y[i], '%s' % (reg[i]),
                size=8, zorder=2, color='w', fontweight='bold')
        ax.text(x[i], y[i], '%s' % (reg[i]),
                size=9, zorder=3, color='k')

    sns.regplot(x=x, y=y, ax=ax, color="C3", scatter_kws={"edgecolors": "k"})
    mask = (np.isnan(H) == 0) & (np.isnan(results[quantity]) == 0)
    y = results[quantity][mask]
    x = H[mask]
    r, p = ss.pearsonr(x, y)
    print("Pearson R: %.2f, %s" % (r, p_to_text(p)))
    rs, ps = ss.spearmanr(x, y)
    print("Spearman R: %.2f, %s" % (rs, p_to_text(ps)))
    
    ax.set_xlabel('Position in the hierarchy')
    ax.set_ylabel(quantity)
    ax.set_title(r"$\rho$"+f'={rs:.2f} {p_to_text(ps)}', fontsize=10)
    sns.despine(ax=ax)
    return f, ax

def p_to_text(p):
    if p < 0.0001:
        return '*** P=%.1e' % p
    if p < 0.001:
        return '*** P=%.4f' % p
    if p < 0.01:
        return '** P=%.3f' % p
    if p < 0.05:
        return '* P=%.3f' % p
    if p >= 0.05:
        return 'ns P=%.2f' % p


def clustered_data_mat_plot(data_mat_sorted, labels, vs, ax, fig, cmap='bwr', _vmax=None, _vmin=None, colorbar=True, xticklabel=True, ylabel=True):
    if _vmax is None:
        _vmax = np.percentile(np.abs(data_mat_sorted), 95)
    
    if _vmin is None:
        _vmin = np.percentile(np.abs(data_mat_sorted), 5)
    
    if cmap=='bwr':
        improp = dict(aspect='auto', cmap='bwr', interpolation='nearest', vmax=_vmax, vmin=-_vmax)
    elif cmap=='Greys':
        improp = dict(aspect='auto', cmap='Greys', interpolation='nearest', vmax=_vmax, vmin=_vmin)
    else:
        assert False, "invalid cmap"
    im = ax.imshow(data_mat_sorted, **improp)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.6)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('selectivity', fontsize=10)  

    for li in range(np.max(labels)):
        ax.axhline(y=np.sum(labels<=li)-0.5, c='k', linewidth=3)

    ax.set_xticks(np.linspace(0, data_mat_sorted.shape[1], len(vs), endpoint=False))
    if xticklabel:
        ax.set_xticklabels(vs, rotation = 90, ha="right") 
    else: 
        ax.set_xticklabels([""]*len(vs)) 
    if ylabel:  
        ax.set_ylabel("neuron", fontsize=10)
        

def dr_scatterplots(data_lowd, labels, sizes, axes, cmap='tab20'):
    ncolor = int(cmap[-2:])
    _min = np.min(data_lowd)
    _max = np.max(data_lowd)
    for ax, label, size in zip(axes, labels, sizes):
        label = label.astype(int)
        label_m1 = label==-1
        ax.scatter(*data_lowd[label_m1].T, s=size[label_m1], color='gray', alpha=0.1)
        ax.scatter(*data_lowd[~label_m1].T, s=size[~label_m1], c=plt.get_cmap(cmap)(label[~label_m1]%ncolor), alpha=1.0)
        ax.set_xlim([_min*1.1, _max*1.1])
        ax.set_ylim([_min*1.1, _max*1.1])
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

def dr_scatterplots_main(data, labels, dr_kwargs, plot_kwargs):
    if (dr_kwargs['method'] == 'lda') and len(np.unique(labels))<3: return
    if dr_kwargs['method'] == 'lda': dr_kwargs['labels'] = labels
    dr_kwargs['ncomp'] = 2
    data_lowd = dr(data, dr_kwargs)
    
    s = plot_kwargs['size'] if 'size' in plot_kwargs else 2
    ax = plot_kwargs['ax']
    dr_scatterplots(data_lowd, 
                    [labels],
                    [s*np.ones(len(labels))],
                    [ax], cmap='tab10')
    ax.set_title(dr_kwargs['method'])
   