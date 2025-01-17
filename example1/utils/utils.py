import pandas as pd
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils import make_folder, remove_space




def import_areagroup():

    ## Hierarchy for ranking
    # Hierarchy XJW
    H1 = pd.read_csv('./example1/utils/area_list.csv', header=None).to_dict()[0]
    for key in H1:
        H1[key] = [H1[key]]

    ## area2area connectivity
    conn_mat = pd.read_csv('./example1/utils/conn_cxcx.csv', header=None)
    conn_mat = conn_mat.values
    conn_area_list = pd.read_csv('./example1/utils/area_list.csv', header=None).values[:,0]
    conn_i2a = {i: a for i, a in enumerate(conn_area_list)}
    conn_a2i = {a: i for i, a in enumerate(conn_area_list)}


    hierarchy_byHarris = {
        "prefrontal": ["FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl"], 
        "lateral": ["AId", "AIv", "AIp", "GU", "VISC", "TEa", "PERI", "ECT"], 
        "somatomotor": ["SSs", "SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-un", "SSp-n", "SSp-m", "MOp", "MOs", ], 
        "visual": ["VISal", "VISl", "VISp", "VISpl", "VISli", "VISpor", "VISrl"], 
        "medial": ["VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"], 
        "auditory": ["AUDd", "AUDp", "AUDpo", "AUDv"], 
    }
    area2ci_byHarris = {}
    for hi in hierarchy_byHarris:
        for a in hierarchy_byHarris[hi]:
            area2ci_byHarris[a] = hi

    return {"cortical_area_list": conn_area_list, 
            "hierarchy":[H1], 
            "conn_mat": [conn_mat, conn_i2a, conn_a2i],
            "hierarchy_byHarris": [hierarchy_byHarris, area2ci_byHarris]}




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


def hierarchy_trend(results, hierarchy, quantity, meanfunc=np.nanmean, errfunc=ss.sem, title='',
                    plot_trend=True, plot_regression=False, ax=None):
    pltcolors = [plt.get_cmap('tab10')(i) for i in range(10)]

    if title == '':
        title = quantity

    # invert dict
    hvalues = list(hierarchy.keys())
    hierarchy_inverted = {}
    for key in hierarchy:
        for region in hierarchy[key]:
            hierarchy_inverted[region] = key

    # create H as a sync array with result
    H = []
    keys = list(hierarchy_inverted.keys())
    for region in results['region']:
        if region in keys:
            H.append(hierarchy_inverted[region])
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

    for h in hvalues:
        ys = np.asarray(y[x == h])
        ys = ys[np.isnan(ys) == 0]
        ymean.append(meanfunc(ys))
        yerr.append(errfunc(ys))
    if plot_trend:
        ax.errorbar(hvalues, ymean, yerr, capsize=8, linewidth=3, alpha=0.5, color=pltcolors[0])
        ax.set_xticks(hvalues)
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
