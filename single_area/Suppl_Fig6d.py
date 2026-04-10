###########################################################################
### Supplementary code for the paper:
### "Rarely categorical, highly separable representations along the cortical hierarchy"
### L. Posani*, S. Wang*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026).
### Relevant: Suppl. Fig.6 b (single-region clustering analysis, selectivity space, including time profiles)
###########################################################################

### Import packages
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from utils.clustering_analysis import cluster_analysis_selectivity_space
from utils.plot import hierarchy_trend

# load the trained RRR encoding model for clustering analysis in the selectivity space
RRR_res_df = pd.read_json("../data/RRR_selectivity.json")

# only include neurons 
#   1. whose R2 pass the minimum \Delta R2 threshold 
#   2. that are located in cortical areas
delta_R2_thres = 0.015
ctx_area_list = pd.read_csv('../data/area_list.csv', header=None).values[:,0]  # list of cortical areas, ordered by hierarchy
RRR_res_df['RRR_deltaR2'] = RRR_res_df["RRR_r2"] - RRR_res_df['null_r2']
nis_incmask_ctx = (RRR_res_df['RRR_deltaR2'] > delta_R2_thres) & (RRR_res_df.acronym.isin(ctx_area_list))

# only include cortical areas with >= 50 selective neurons
ctx_area_included = np.array([a for a in ctx_area_list if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=50]) 

## clustering analysis
# setup the clustering analysis
null_kwargs = {"N_null": 100, "null_dist": "Gaussian"} 
sus_clus_kwargs = {"remove_sus_clus": True, # remove the SUSpicious clusters if any and redo the clustering
                   "sessions_orig": None,   # the original sessions of each neuron: values will be set later
                   "sus_clus_thres": 0.9,   # a cluster whose total SScore summed over all neurons is mainly contributed by neurons from one single session (>0.9) is defined as SUSpicious cluster
                   }
algo_kwargs = {"min_N": 50, 
               "algo": "kmeans", "n_init": 50, "n_clus_lim": [3,20],
               "dis_metric": "euclidean"}
res_folder = f"./results_supplfig6/selectivity_space/inc_time_profiles" # folder to save the clustering analysis results
os.makedirs(res_folder, exist_ok=True)
plot_kwargs = dict(plot=True, folder=res_folder)
vs=['block', 'side', 'contrast', 'choice', "outcome", "wheel", "whisker", "lick"]

if os.path.isfile(os.path.join(res_folder, "results.pk")):
    results = pickle.load(open(os.path.join(res_folder, "results.pk"), 'rb'))
else:
    results = dict(region=[],   # brain region name
                ss_z=[],     # z-scored Silhouette Score
                )
    for area in ctx_area_included:
        nis_incmask_area = (nis_incmask_ctx) & (RRR_res_df.acronym == area)
        print(np.sum(nis_incmask_area), "neurons included in area", area)
        print(np.sum(RRR_res_df.acronym == area), "total neurons in area", area)

        algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
        # get the sessions of the neurons in the area
        eids = RRR_res_df.loc[nis_incmask_area,"eid"]
        sus_clus_kwargs['sessions_orig'] = eids
        # get the selectivity of the neurons in the area
        beta_neurons = RRR_res_df.loc[nis_incmask_area,"RRR_beta"] # (n_neurons, n_vars+1, n_times)
        beta_neurons = np.array([_[:-1] for _ in beta_neurons]) # remove the intercept
        sel_neurons = []; _vs = []
        for i in range(beta_neurons.shape[1]):
            pca = PCA().fit(beta_neurons[:, i])
            n_comp = np.where(np.cumsum(pca.explained_variance_ratio_)>0.8)[0][0]+1
            _X_lowd = PCA(n_components=n_comp).fit_transform(beta_neurons[:, i])
            sel_neurons.append(_X_lowd)
            _vs += [vs[i]+"_PC"+str(j) for j in range(n_comp)]
        sel_neurons = np.concatenate(sel_neurons,axis=1)
        plot_kwargs['vs'] = _vs

        clus_res = cluster_analysis_selectivity_space(sel_neurons,
                                    algo_kwargs, 
                                    sus_clus_kwargs, 
                                    null_kwargs, 
                                    plot_kwargs)

        if clus_res['clus_success']: 
            results['region'].append(area)
            results['ss_z'].append(clus_res['sscore_z'])
            
    results['region'] = np.asarray(results['region'])
    results['Silhouette Score (z)'] = np.asarray(results['ss_z'])
    pickle.dump(results, open(os.path.join(res_folder, "results.pk"), 'wb'))


## plot
area2i = {a: i for i, a in enumerate(ctx_area_list)} # convert region name to hierarchy position
fig, ax = plt.subplots(1, 1, figsize=(3,3.5))
hierarchy_trend(results, area2i, f"Silhouette Score (z)", ax=ax)
ax.axhline(y=ss.norm.ppf(1-0.05/len(results['region'])), c='gray', linestyle='--', linewidth=3)
ax.set_ylabel("Silhouette Score (z, time profiles)")
sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(res_folder, f"result.pdf")); plt.close('all')