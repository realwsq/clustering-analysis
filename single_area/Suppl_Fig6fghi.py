###########################################################################
### Supplementary code for the paper:
### "Rarely categorical, highly separable representations along the cortical hierarchy"
### L. Posani*, S. Wang*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026).
### Relevant: Suppl. Fig.6 b (single-region clustering analysis, condition space)
###########################################################################

### Import packages
import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from utils.clustering_analysis import cluster_analysis_condition_space, epairs_main
from utils.plot import hierarchy_trend


## load the mean firing rate data in the condition space
independent_condition = False
# either considering mfr of only independent conditions (IC=True) or all conditions (IC=False)
# only selective neurons (delta_R2>0.015) of cortical areas are included in the file
if independent_condition:
    mfr_data = pickle.load(open("../data/mfr_independent_conditions.pck", 'rb'))
else:
    mfr_data = pickle.load(open("../data/mfr_all_conditions.pck", 'rb'))

# load the cortical area list
ctx_area_list = pd.read_csv('../data/area_list.csv', header=None).values[:,0] 

## clustering analysis
# setup the clustering analysis
null_kwargs = {"N_null": 100, "null_dist": "multi-log-normal"}
sus_clus_kwargs = {"remove_sus_clus": True, # remove the SUSpicious clusters if any and redo the clustering
                   "sessions_orig": None,   # the original sessions of each neuron: values will be set later
                   "sus_clus_thres": 0.9,   # a cluster whose total SScore summed over all neurons is mainly contributed by neurons from one single session (>0.9) is defined as SUSpicious cluster
                   }
algo_kwargs = {"min_N": 40, 
               "algo": "kmeans", "n_init": 50, "n_clus_lim": [3,20],
               "dis_metric": "euclidean"}
epairs_kwargs = {"n_neigh": 1, "N_null": 5000}
res_folder = f"./results_supplfig6/condition_space/IC{independent_condition}" # folder to save the clustering analysis results
os.makedirs(res_folder, exist_ok=True)
plot_kwargs = dict(plot=True, folder=res_folder)

if os.path.isfile(os.path.join(res_folder, "results.pk")):
    results = pickle.load(open(os.path.join(res_folder, "results.pk"), 'rb'))
else:
    # start clustering analysis for each area
    results = dict(region=[],   # brain region name
                ss_z=[],     # z-scored Silhouette Score
                epairs_z=[], # z-scored ePAIRS statistic
                )
    for area in ctx_area_list:
        if area not in mfr_data: 
            print(f"area {area} not in the data, skip!")
            continue # skip if no data for this area (not enough selective neurons)
        algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
        
        conditions = list(mfr_data[area][0].keys())
        
        # get the mfrs and sessions of the neurons in the area
        mfr_neurons = []; session_area = []
        for eid, data_eid in enumerate(mfr_data[area]):
            # average mfr over trials for each condition
            mfr = np.stack([data_eid[c].mean(0) for c in conditions], axis=1) # (n_neurons, n_conds)
            nis_inc = np.all(mfr>0, axis=1) # only include neurons with positive mfr
            print(f"area: {area}, N%: {np.mean(nis_inc)}, N: {len(nis_inc)}")
            mfr = mfr[nis_inc]
            mfr_neurons.append(mfr)
            session_area.append([eid]*np.sum(nis_inc))
        sus_clus_kwargs['sessions_orig'] = np.concatenate(session_area)
        mfr_neurons = np.concatenate(mfr_neurons, 0)

        clus_res = cluster_analysis_condition_space(mfr_neurons,
                                    algo_kwargs, 
                                    sus_clus_kwargs, 
                                    null_kwargs, 
                                    plot_kwargs)

        if clus_res['clus_success']: 
            results['region'].append(area)
            results['ss_z'].append(clus_res['sscore_z'])

            ## also perform the ePAIRS analysis
            epairs_res = epairs_main(clus_res['X_orig'], epairs_kwargs, null_kwargs)
            results['epairs_z'].append(epairs_res['epairs_z'])
            print(f"area: {area}, Silhouette Score (z): {clus_res['sscore_z']:.2f}, ePAIRS (z): {epairs_res['epairs_z']:.2f}")

    results['region'] = np.asarray(results['region'])
    results['Silhouette Score (z)'] = np.asarray(results['ss_z'])
    results['ePAIRS (z)'] = np.asarray(results['epairs_z'])
    pickle.dump(results, open(os.path.join(res_folder, "results.pk"), 'wb'))


## plot
area2i = {a: i for i, a in enumerate(ctx_area_list)} # convert region name to hierarchy position
fig, ax = plt.subplots(1, 1, figsize=(3,3.5))
hierarchy_trend(results, area2i, f"Silhouette Score (z)", ax=ax)
ax.axhline(y=ss.norm.ppf(1-0.05/len(results['region'])), c='gray', linestyle='--', linewidth=3)
if independent_condition:
    ax.set_ylabel("Silhouette Score (z, IC)")
else:
    ax.set_ylabel("Silhouette Score (z, firing rate)")
ax.set_ylim(-2.5, 12.5)
sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(res_folder, f"result.pdf")); plt.close('all')

fig, ax = plt.subplots(1, 1, figsize=(3,3.5))
hierarchy_trend(results, area2i, f"ePAIRS (z)", ax=ax)
ax.axhline(y=-ss.norm.ppf(1-0.05/len(results['region'])), c='gray', linestyle='--')
if independent_condition:
    ax.set_ylabel("ePAIRS (z, IC)")
else:
    ax.set_ylabel("ePAIRS (z, firing rate)")
sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(res_folder, f"results_epairs.pdf")); plt.close('all')
