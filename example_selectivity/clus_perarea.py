from utils import make_folder, remove_space, log_kv
from example_selectivity.utils.utils import get_area_anatomical_info_Allen, hierarchy_trend
from clustering_analysis import cluster_analysis

import pdb, os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as ss


# load the neural selectivity data
RRR_res_df = pd.read_json("./example_selectivity/neural_selectivity_data/RRRglobal_full.json")
# folder where the intermediate outputs of the clustering analysis will be saved
local_folder_lf = f"/burg/stats/users/sw3894/bwm_original/cortexbwm/RRR_local_folder_original"
# folder where the final results will be saved
resgood_folder = make_folder("./example_selectivity/results")


# load the anatomical information that will be used later
area_atm_info = get_area_anatomical_info_Allen()
conn_area_list_byH = area_atm_info['cortical_area_list'] # list of cortical areas, ordered by hierarchy
area2H = area_atm_info['area2H'] # area to hierarchy mapping


# only include neurons whose R2 pass the minimmum \Delta R2 threshold
#                       and of the cortical areas
inc_param=dict(min_N=50, min_Deltar2=0.015) 
# inc_param=dict(min_N=50, min_Deltar2=0.01) # could try to vary this threshold 
# inc_param=dict(min_N=50, min_Deltar2=0.02) 
RRR_res_df['RRRglobal_Deltar2'] = RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']
nis_incmask = RRR_res_df['RRRglobal_Deltar2'] > inc_param['min_Deltar2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas by hierarchy
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])



### setup the clustering analysis
null_params = {"null_dist": "Gaussian", "preprocess_null": False,}
clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
                sel="beta_sum", ) 
beta_preprocess_steps = [['sum', 2]]
# # could try to include time profiles
# clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
#                 sel="beta_PCA", ) 
# beta_preprocess_steps = [['pca_temporal', [False, 0.8]]]
# # could try a different clustering algorithm
# clus_param=dict(algo="leiden", dis_metric="euclidean", k_list=range(3,100), n_iterations=-1, eval_metrics=["sscore", "modularity"], ms_metric="sscore",
#                 sel="beta_sum", ) 
# beta_preprocess_steps = [['sum', 2]]

# folder for saving the figures of the clustering analysis
clusfig_folder = make_folder(os.path.join(resgood_folder, f"clus",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess_steps}_{null_params}")))
# folder for saving the intermediate outputs of the clustering analysis
clus_folder = make_folder(os.path.join(local_folder_lf, f"clus",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess_steps}_{null_params}")))



## clustering analysis
null_kwargs = {"N_null": 100, **null_params}
sus_clus_kwargs = {"remove_sus_clus": True, # remove the SUSpicious clusters if any and redo the clustering
                   "sessions_orig": None,   # the original sessions of each neuron: values will be set later
                   "sus_clus_thres": 0.9,   # a cluster whose total SScore summed over all neurons is mainly contributed by neurons from one single session (>0.9) is defined as SUSpicious cluster
                   }
algo_kwargs = {"min_N": inc_param['min_N'], 'clus_folder': clus_folder, 'save_id': None, 
               **clus_param}
plot_kwargs = dict(plot=True, folder=clusfig_folder, vs=['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker", "lick"])

results = dict(region=[],   # brain region name
               ss_z=[],     # z-scored Silhouette Score
               ss=[],       # mean Silhouette Score
               ss_k=[],     # number of clusters
               ss_null=[],  # Silhouette Scores of the null datasets
               ss_null_mean=[], # mean Silhouette Score of the null datasets
               ss_null_std=[],  # std of the Silhouette Scores of the null datasets
               N=[],        # number of neurons used in the final round of clustering (after removing suspicious clusters)
               )
if os.path.exists(os.path.join(clusfig_folder, "results.pk")):
    results = pickle.load(open(os.path.join(clusfig_folder, "results.pk"), 'rb'))
    results_df = pd.DataFrame.from_dict(results)
else:
    for area in tqdm(area_order_H):
        nis_incmask_area = (nis_incmask) & (RRR_res_df.acronym == area)
        log_kv(area=area, total_N=np.sum(nis_incmask_area))

        algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
        # get the sessions of the neurons in the area
        eids = RRR_res_df.loc[nis_incmask_area,"eid"]
        sus_clus_kwargs['sessions_orig'] = eids
        # get the selectivity of the neurons in the area
        beta_neurons = RRR_res_df.loc[nis_incmask_area,"RRRglobal_beta"] # (n_neurons, n_vars+1, n_times)
        sel_neurons = np.array([_[:-1] for _ in beta_neurons]) # remove the intercept

        clus_res = cluster_analysis(sel_neurons, # (n_neurons, n_vars, n_times)
                                    beta_preprocess_steps, 
                                    algo_kwargs, 
                                    sus_clus_kwargs, 
                                    null_kwargs, 
                                    plot_kwargs)

        if clus_res['final_clus_res']['clus_success']: 
            log_kv(area=area, sscore_z=clus_res['final_clus_res']['sscore_z'])
            results['region'].append(area)
            results['ss_z'].append(clus_res['final_clus_res']['sscore_z'])
            results['ss'].append(clus_res['final_clus_res']['sscore_mean'])
            results['ss_k'].append(len(np.unique(clus_res['final_clus_res']['clus_labels'])))
            results['ss_null'].append(clus_res['final_clus_res']['sscore_nulls'])
            results['ss_null_mean'].append(np.mean(clus_res['final_clus_res']['sscore_nulls']))
            results['ss_null_std'].append(np.std(clus_res['final_clus_res']['sscore_nulls']))
            results['N'].append(clus_res['final_clus_res']['X'].shape[0])

        
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(os.path.join(clusfig_folder, "results.csv"))
    pickle.dump(results, open(os.path.join(clusfig_folder, "results.pk"), 'wb'))



results['region'] = np.asarray(results['region'])
results['H'] = np.asarray([area2H[a] for a in results['region']])
results['Silhouette Score (z)'] = np.asarray(results['ss_z'])
results['ss_null_mean'] = np.asarray(results['ss_null_mean'])
results['ss_null_std'] = np.asarray(results['ss_null_std'])
results['ss'] = np.asarray(results['ss'])
results['N'] = np.asarray(results['N'])

fig, ax = plt.subplots(1, 1,figsize=(5*1, 4*1))
hierarchy_trend(results, area2H, f"Silhouette Score (z)", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=ss.norm.ppf(1-0.05/len(results)), c='gray', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result.pdf")); plt.close('all')


pdb.set_trace()
pdb.set_trace()
pdb.set_trace()

    