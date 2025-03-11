import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def get_area_anatomical_info_Allen():

    ## Hierarchy 
    conn_area_list = pd.read_csv('./example_selectivity/utils/area_list.csv', header=None).values[:,0]
    conn_i2a = {i: a for i, a in enumerate(conn_area_list)}
    conn_a2i = {a: i for i, a in enumerate(conn_area_list)}

    ## Module 
    mod2area = {
        "prefrontal": ["FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl"], 
        "lateral": ["AId", "AIv", "AIp", "GU", "VISC", "TEa", "PERI", "ECT"], 
        "somatomotor": ["SSs", "SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-un", "SSp-n", "SSp-m", "MOp", "MOs", ], 
        "visual": ["VISal", "VISl", "VISp", "VISpl", "VISli", "VISpor", "VISrl"], 
        "medial": ["VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"], 
        "auditory": ["AUDd", "AUDp", "AUDpo", "AUDv"], 
    }
    area2mod = {}
    for mod in mod2area:
        for a in mod2area[mod]:
            area2mod[a] = mod

    return {"cortical_area_list": conn_area_list, 
            "area2H": conn_a2i,
            "H2area": conn_i2a,
            "mod2area": mod2area,
            "area2mod": area2mod}


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


def hierarchy_trend(results, area2H, quantity, meanfunc=np.nanmean, errfunc=ss.sem, title='',
                    plot_trend=True, plot_regression=False, ax=None):
    pltcolors = [plt.get_cmap('tab10')(i) for i in range(10)]

    if title == '':
        title = quantity

    Hvalues = list(area2H.values())

    # create H as a sync array with result
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

    # scatter
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4))
    else:
        f = None # placeholder
    sns.despine(ax=ax)
    ax.scatter(x, y, color='k', s=25, alpha=0.5)
    for i in range(len(reg)):
        ax.text(x[i], y[i], '%s' % (reg[i]),
                size=8, zorder=2, color='w', fontweight='bold')
        ax.text(x[i], y[i], '%s' % (reg[i]),
                size=9, zorder=3, color='k')

    # trend
    ymean = []
    yerr = []

    for h in Hvalues:
        ys = np.asarray(y[x == h])
        ys = ys[np.isnan(ys) == 0]
        ymean.append(meanfunc(ys))
        yerr.append(errfunc(ys))
    if plot_trend:
        ax.errorbar(Hvalues, ymean, yerr, capsize=8, linewidth=3, alpha=0.5, color=pltcolors[0])
        ax.set_xticks(Hvalues)
    ax.set_xlabel('Position in the hierarchy')
    ax.set_ylabel(quantity)

    if plot_regression:
        sns.regplot(x=x, y=y, ax=ax)
    # quantify statistics?
    mask = (np.isnan(H) == 0) & (np.isnan(results[quantity]) == 0)
    y = results[quantity][mask]
    x = H[mask]
    r, p = ss.pearsonr(x, y)
    print("Pearson R: %.2f, %s" % (r, p_to_text(p)))
    rs, ps = ss.spearmanr(x, y)
    print("Spearman R: %.2f, %s" % (rs, p_to_text(ps)))
    ax.set_title(f'{title}\nPearson R: {r:.2f} {p_to_text(p)}\nSpearman R: {rs:.2f} {p_to_text(ps)}', fontsize=10)
    return f, ax
