from utils import make_folder, remove_space, log_kv
from example_selectivity.utils.utils import get_area_anatomical_info_Allen, hierarchy_trend
from clustering_analysis import cluster_analysis, epairs_main

import pdb, os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


# folder where the intermediate outputs of the clustering analysis will be saved
local_folder_lf = f"/burg/stats/users/sw3894/bwm_original/cortexbwm/mfr"
# folder where the final results will be saved
resgood_folder = make_folder("./example_mfr/results")

# load the anatomical information that will be used later
area_atm_info = get_area_anatomical_info_Allen()
conn_area_list_byH = area_atm_info['cortical_area_list'] # list of cortical areas, ordered by hierarchy
area2H = area_atm_info['area2H'] # area to hierarchy mapping



### setup the clustering analysis
clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
                sel="mfr>0", )  
beta_preprocess_steps = [["normalize", 1], ["pca", 0.9]]
null_params = {"null_dist": "multi-log-normal", "preprocess_null": True,}


# only include neurons whose R2 pass the minimmum \Delta R2 threshold
#                       and of the cortical areas
inc_param=dict(min_N=40, min_r2=0.015, IC=True) # slightly lower min_N to include more areas in the analysis
# either considering mfr of only independent conditions (IC=True) or all conditions (IC=False)
if "IC" in inc_param and inc_param['IC']:
    data = pickle.load(open("./example_mfr/neural_mfr_data/CTreduced_for_shuqi.pck", 'rb'))
else:
    data = pickle.load(open("./example_mfr/neural_mfr_data/CT_for_shuqi.pck", 'rb'))

clusfig_folder = make_folder(os.path.join(resgood_folder, f"clus_mfr",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess_steps}_{null_params}")))
clus_folder = make_folder(os.path.join(local_folder_lf, f"clus_mfr",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess_steps}_{null_params}")))




## clustering analysis
N_nullG = 100; 
null_kwargs = {"N_null": 100, **null_params}
sus_clus_kwargs = {"remove_sus_clus": True, # remove the SUSpicious clusters if any and redo the clustering
                   "sessions_orig": None,   # the original sessions of each neuron: values will be set later
                   "sus_clus_thres": 0.9,   # a cluster whose total SScore summed over all neurons is mainly contributed by neurons from one single session (>0.9) is defined as SUSpicious cluster
                   }
algo_kwargs = {"min_N": inc_param['min_N'], 'clus_folder': clus_folder, 'save_id': None, 
               **clus_param}
plot_kwargs = dict(plot=True, folder=clusfig_folder)

results = dict(region=[],   # brain region name
               ss_z=[],     # z-scored Silhouette Score
               ss=[],       # mean Silhouette Score
               ss_k=[],     # number of clusters
               N=[],        # number of neurons used in the final round of clustering (after removing suspicious clusters)
               dispersion=[],   # dispersion of the clusters
               )
if os.path.exists(os.path.join(clusfig_folder, "results.pk")):
    results = pickle.load(open(os.path.join(clusfig_folder, "results.pk"), 'rb'))
    results_df = pd.DataFrame.from_dict(results)
else:
    for area in tqdm(conn_area_list_byH):
        log_kv(area=area)
        if area not in data: continue # skip if no data for this area
        algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
        conditions = list(data[area][0].keys())
        
        # get the mfrs and sessions of the neurons in the area
        mfr_neurons = []; session_area = []
        for eid, data_eid in enumerate(data[area]):
            mfr = np.stack([data_eid[c].mean(0) for c in conditions], axis=1) # (n_neurons, n_conds)
            nis_inc = mfr.std(1) > 1e-3 # only include neurons with non-zero variance
            if clus_param['sel'].endswith(">0"): # only include neurons with positive mfr
                nis_inc = nis_inc & np.all(mfr>0, axis=1)
            print(f"area: {area}, N%: {np.mean(nis_inc)}, N: {len(nis_inc)}")
            mfr = mfr[nis_inc]
            mfr_neurons.append(mfr)
            session_area.append([eid]*np.sum(nis_inc))
        sus_clus_kwargs['sessions_orig'] = np.concatenate(session_area)
        mfr_neurons = np.concatenate(mfr_neurons, 0)

        clus_res = cluster_analysis(mfr_neurons, # (n_neurons, n_conds)
                                    beta_preprocess_steps, 
                                    algo_kwargs, 
                                    sus_clus_kwargs, 
                                    null_kwargs, 
                                    plot_kwargs)

        if clus_res['final_clus_res']['clus_success']: 
            log_kv(area=area, sscore_z=clus_res['final_clus_res']['sscore_z'], sscore=clus_res['final_clus_res']['sscore_mean'])
            results['region'].append(area)
            results['ss_z'].append(clus_res['final_clus_res']['sscore_z'])
            results['ss'].append(clus_res['final_clus_res']['sscore_mean'])
            results['ss_k'].append(len(np.unique(clus_res['final_clus_res']['clus_labels'])))
            results['N'].append(clus_res['final_clus_res']['X'].shape[0])
            sigmas = []
            _labels = clus_res['final_clus_res']['clus_labels']
            for li in np.unique(_labels):
                Xs = clus_res['final_clus_res']['X']
                Xs_li = Xs[_labels==li]
                centroids = np.asarray([np.mean(Xs[_labels==li], 0) for li in np.unique(_labels)])
                dis = np.mean(np.linalg.norm(Xs_li-np.mean(Xs_li,0), axis=1))
                _mask = np.zeros(len(centroids), dtype=bool)
                _mask[li] = True
                dis_cent = np.mean(np.linalg.norm(centroids[~_mask]-centroids[_mask], axis=1))
                sigmas.append(dis/dis_cent)
            results['dispersion'].append(np.mean(sigmas))

            if null_kwargs['preprocess_null']:
                X_4_epairs = clus_res['data_clus_hist'][-1]['X_orig']
            else:
                X_4_epairs = clus_res['data_clus_hist'][-1]['X']
            np.save(os.path.join(clus_folder, f"{area}_X4epairs.npy"), X_4_epairs)

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(os.path.join(clusfig_folder, "results.csv"))
    pickle.dump(results, open(os.path.join(clusfig_folder, "results.pk"), 'wb'))


results['region'] = np.asarray(results['region'])
results['H'] = np.asarray([area2H[a] for a in results['region']])
results['Silhouette Score (z)'] = np.asarray(results['ss_z'])
results['Silhouette Score'] = np.asarray(results['ss'])
results['N'] = np.asarray(results['N'])

fig, axes = plt.subplots(2, 1,figsize=(5*1, 4*2))
ax = axes[0]
hierarchy_trend(results, area2H, f"Silhouette Score (z)", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=ss.norm.ppf(1-0.05/len(results)), c='gray', linestyle='--')
ax = axes[1]
hierarchy_trend(results, area2H, f"Silhouette Score", 
                    plot_trend=False, plot_regression=True, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result.pdf")); plt.close('all')




## clustering analysis - ePAIRS
epairs_kwargs = {"n_neigh": 1, "N_null": 5000}
results_epairs = dict(region=[], epairs_z=[], epairs_p=[])
if os.path.isfile(os.path.join(clusfig_folder, "results_epairs.pk")):
    results_epairs = pickle.load(open(os.path.join(clusfig_folder, "results_epairs.pk"), 'rb'))
    results_df = pd.DataFrame.from_dict(results_epairs)
else:
    for area in tqdm(conn_area_list_byH):
        _fname = os.path.join(clus_folder, f"{area}_X4epairs.npy")
        if os.path.isfile(_fname):
            X_4_epairs = np.load(_fname)
        else:
            continue
        epairs_res = epairs_main(X_4_epairs, epairs_kwargs, null_kwargs, beta_preprocess_steps)
        log_kv(area=area, epairs_z=epairs_res['epairs_z'], epairs_p=epairs_res['epairs_p'])
        results_epairs['region'].append(area)
        results_epairs['epairs_z'].append(epairs_res['epairs_z'])
        results_epairs['epairs_p'].append(epairs_res['epairs_p'])
    results_df = pd.DataFrame.from_dict(results_epairs)
    results_df.to_csv(os.path.join(clusfig_folder, "results_epairs.csv"))
    pickle.dump(results_epairs, open(os.path.join(clusfig_folder, "results_epairs.pk"), 'wb'))

results_epairs['region'] = np.asarray(results_epairs['region'])
results_epairs['epairs_z'] = np.asarray(results_epairs['epairs_z'])
results_epairs['epairs_p'] = np.asarray(results_epairs['epairs_p'])
fig, ax = plt.subplots(1, 1,figsize=(5*1, 4*1))
hierarchy_trend(results_epairs, area2H, f"epairs_z", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=-ss.norm.ppf(1-0.05/len(results_epairs)), c='gray', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"results_epairs.pdf")); plt.close('all')


pdb.set_trace()
pdb.set_trace()
pdb.set_trace()

    
    