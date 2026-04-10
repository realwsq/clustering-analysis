###########################################################################
### Supplementary code for the paper:
### "Rarely categorical, highly separable representations along the cortical hierarchy"
### L. Posani*, S. Wang*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026).
### Relevant: Fig.3e-f, Suppl. Fig.7 (cluster analysis for groups of areas, selectivity space)
###########################################################################

### Import packages
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("../single_area/")
from utils.clustering_analysis import cluster_analysis_selectivity_space, area_identity_analysis

# load the trained RRR encoding model for clustering analysis in the selectivity space
RRR_res_df = pd.read_json("../data/RRR_selectivity.json")

# only include neurons 
#   1. whose R2 pass the minimum \Delta R2 threshold (0.015)
#   2. that are located in cortical areas
ctx_area_list = pd.read_csv('../data/area_list.csv', header=None).values[:,0]  # list of cortical areas, ordered by hierarchy
RRR_res_df['RRR_deltaR2'] = RRR_res_df["RRR_r2"] - RRR_res_df['null_r2']
nis_incmask = (RRR_res_df['RRR_deltaR2'] > 0.015)

# module information
mod2area = {
        "prefrontal": ["FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl"], 
        "lateral": ["AId", "AIv", "AIp", "GU", "VISC", "TEa", "PERI", "ECT"], 
        "somatomotor": ["SSs", "SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-un", "SSp-n", "SSp-m", "MOp", "MOs", ], 
        "visual": ["VISal", "VISl", "VISp", "VISpl", "VISli", "VISpor", "VISrl"], 
        "medial": ["VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"], 
        "auditory": ["AUDd", "AUDp", "AUDpo", "AUDv"], 
    }
area2mod = {a: m for m, al in mod2area.items() for a in al}

## clustering analysis
# setup the clustering analysis
area_inc_param=dict(areas2remove=["VISp", "AUDp", "SSp-ul"],
                sample=dict(strategy="top", n_samples=100, seed=42)) 
null_kwargs = {"N_null": 50, "null_dist": "Gaussian"}  
sus_clus_kwargs = {"remove_sus_clus": True, # remove the SUSpicious clusters if any and redo the clustering
                   "sessions_orig": None,   # the original sessions of each neuron: values will be set later
                   "sus_clus_thres": 0.9,   # a cluster whose total SScore summed over all neurons is mainly contributed by neurons from one single session (>0.9) is defined as SUSpicious cluster
                   }
algo_kwargs = {"min_N": 50, 
               "algo": "kmeans", "n_init": 50, "n_clus_lim": [3,20],
               "dis_metric": "euclidean"}
res_folder = f"./results_fig3f/{area_inc_param['sample']['strategy']}{area_inc_param['sample']['n_samples']}" # folder to save the clustering analysis results
os.makedirs(res_folder, exist_ok=True)
plot_kwargs = dict(plot=True, folder=res_folder, vs=['block', 'side', 'contrast', 'choice', "outcome", "wheel", "whisker", "lick"])

# helper functions for loading neurons
def load_neurons_from_arealist(area_list, area_inc_param):
    nis_incmask_all = np.zeros(len(RRR_res_df), dtype=bool)
    area_list = [a for a in area_list if np.sum(nis_incmask & (RRR_res_df.acronym == a)) >= 50] # remove areas with less than 50 neurons
    if "areas2remove" in area_inc_param: # remove areas that are already clustered
        area_list = [a for a in area_list if not (a in area_inc_param['areas2remove'])] # remove clustered area
    if len(area_list) < 2: 
        return nis_incmask_all # skip if there are not enough areas
    for area in area_list:
        _area_nis = np.where((nis_incmask) & (RRR_res_df.acronym == area))[0]
        if "sample" in area_inc_param:
            n_samples = min(area_inc_param['sample']['n_samples'], len(_area_nis))
            if area_inc_param['sample']['strategy'] == "random":
                np.random.seed(area_inc_param['sample']['seed'])
                _area_nis = np.random.choice(_area_nis, size=n_samples, replace=False)
            elif area_inc_param['sample']['strategy'] == "top":
                _area_nis = _area_nis[np.argsort(RRR_res_df.loc[_area_nis, "RRR_deltaR2"])[-n_samples:]]
            else: assert False
        else: pass
        nis_incmask_all[_area_nis] = True # include the neurons in the area
    return nis_incmask_all

res_module = dict(module=[], ss_z=[], ri_z=[])
for module, area_pair in [["ctx", ctx_area_list]]+list(mod2area.items()):
    if module == "ctx": 
        # change the neuron inclusion strategy according to the module label instead of the area label
        nis_incmask_area = np.zeros(len(RRR_res_df), dtype=bool)
        for _mod, _ap in mod2area.items():
            _nis_incmask_area = load_neurons_from_arealist(_ap, {})
            _area_nis = np.where(_nis_incmask_area)[0]
            if "sample" in area_inc_param:
                n_samples = min(area_inc_param['sample']['n_samples'], len(_area_nis))
                if area_inc_param['sample']['strategy'] == "random":
                    np.random.seed(area_inc_param['sample']['seed'])
                    _area_nis = np.random.choice(_area_nis, size=n_samples, replace=False)
                elif area_inc_param['sample']['strategy'] == "top":
                    _area_nis = _area_nis[np.argsort(RRR_res_df.loc[_area_nis, "RRR_deltaR2"])[-n_samples:]]
                else: assert False
            else: pass
            nis_incmask_area[_area_nis] = True
    else:
        nis_incmask_area = load_neurons_from_arealist(area_pair, area_inc_param)
    if not np.any(nis_incmask_area): continue

    algo_kwargs['save_id'] = plot_kwargs['save_id'] = module
    # get the sessions of the neurons in the area
    eids = RRR_res_df.loc[nis_incmask_area,"eid"]
    sus_clus_kwargs['sessions_orig'] = eids
    # get the selectivity of the neurons in the area
    beta_neurons = RRR_res_df.loc[nis_incmask_area,"RRR_beta"] # (n_neurons, n_vars+1, n_times)
    sel_neurons = np.array([_[:-1] for _ in beta_neurons]) # remove the intercept
    sel_neurons = np.sum(sel_neurons, axis=2) # sum beta over time as the selectivity (n_neurons, n_vars)

    clus_res = cluster_analysis_selectivity_space(sel_neurons,
                                algo_kwargs, 
                                sus_clus_kwargs, 
                                null_kwargs, 
                                plot_kwargs)
    
    area_labels = RRR_res_df.loc[nis_incmask_area,"acronym"].values[clus_res['good_nismask']]
    if module == "ctx": 
        area_labels = [area2mod[a] for a in area_labels] # use the module labels instead of area labels
    else: pass
    ri_res = area_identity_analysis(clus_res['clus_labels'], area_labels,
                                    plot_kwargs)
    
    res_module['module'].append(module)
    res_module['ss_z'].append(clus_res['sscore_z'])
    res_module['ri_z'].append(ri_res['ri_z'])

pickle.dump(res_module, open(os.path.join(res_folder, "results.pk"), 'wb'))

fig, axes = plt.subplots(1, 2, figsize=(3.5*2,3.5))
ax = axes[0]
sns.scatterplot(x=res_module['ss_z'], y=res_module['ri_z'], cmap='viridis', s=100, ax=ax)
for i in range(len(res_module['module'])):
    ax.text(res_module['ss_z'][i], res_module['ri_z'][i], res_module['module'][i],
             fontsize=9, ha='center', va='center')
ax.set_xlabel('Silhouette Score (z-scored)')
ax.set_ylabel('Rand Index (z-scored)')
ax = axes[1]
sns.barplot(x=res_module['module'], y=res_module['ss_z'], color='C0', ax=ax)
ax.set_xticklabels(res_module['module'], rotation=90)
ax.set_xlabel('Module')
ax.set_ylabel('Silhouette Score (z-scored)')
sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(res_folder, 'result.pdf')); plt.close()