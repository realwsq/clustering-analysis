from utils import make_folder, remove_space, log_kv
from example1.utils.utils import import_areagroup, hierarchy_trend
from clustering_analysis import cluster_analysis, epairs_main

import pdb, os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

local_folder_lf = f"/burg/stats/users/sw3894/bwm_original/cortexbwm/mfr"
resgood_folder = make_folder("./example_mfr/res_good")
data = data = pickle.load(open("example_mfr/utils/CTreduced_for_shuqi.pck", 'rb'))


inc_param=dict(min_N=40, min_r2=0.015, IC=True, inc_nowhisking=True)
## parameters for clustering
## old version
# setup 1
clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
                sel="mfr_GaussianCoupla_>0", )  
beta_preprocess = [["normalize", 1], ["pca", 0.9]]
# beta_preprocess = [["uniform", 1], ["pca", 0.9]]
sus_clus_thres = 0.9
# null_params = {"null_dist": "Gaussian", "preprocess_null": False,}
# null_params = {"null_dist": "GaussianCoupla_ms", "preprocess_null": True,}
null_params = {"null_dist": "multi-log-normal", "preprocess_null": True,}
# null_params = {"null_dist": "GaussianCoupla", "preprocess_null": True,}



clusfig_folder = make_folder(os.path.join(resgood_folder, f"clus_mfr",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess}_{sus_clus_thres}_{null_params}")))
clus_folder = make_folder(os.path.join(local_folder_lf, f"clus_mfr",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess}_{sus_clus_thres}_{null_params}")))
print(clusfig_folder)
print(clusfig_folder)
print(clus_folder)
print(clus_folder)


res = import_areagroup()
conn_area_list_byH = res['cortical_area_list']
H1 = res['hierarchy'][0]
area2H = {H1[_H][0]: _H for _H in H1}


## clustering analysis
N_nullG = 100; 
null_kwargs = {"N_null": N_nullG, **null_params}
sus_clus_kwargs = {"remove_sus_clus": False}
algo_kwargs = {"min_N": inc_param['min_N'], 'clus_folder': clus_folder, 'save_id': None, 
               **clus_param}
beta_preprocess_kwargs = {"preprocess": beta_preprocess,}
plot_kwargs = dict(plot=True, folder=clusfig_folder) 

results = dict(region=[], ss_z=[], ss=[], ss_k=[], ss_null=[], ss_null_mean=[], ss_null_std=[], N=[], dispersion=[])
for area in tqdm(conn_area_list_byH):
    log_kv(area=area)
    if area not in data: continue ### only for mfr
    algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
    
    coef_vs_area = data[area].T  # (N, conditions)
    nis_inc = coef_vs_area.std(1) > 1e-3
    if clus_param['sel'].endswith(">0"):
        nis_inc = nis_inc & np.all(coef_vs_area>0, axis=1)
    print(f"area: {area}, N%: {np.mean(nis_inc)}, N: {len(nis_inc)}")
    coef_vs_area = coef_vs_area[nis_inc]
    sus_clus_kwargs['sessions_orig'] = [0]*np.sum(nis_inc)
    plot_kwargs['vs'] = [f"{i}" for i in range(coef_vs_area.shape[1])] # add vs for plot

    clus_res = cluster_analysis(coef_vs_area, N_nullG, beta_preprocess_kwargs, algo_kwargs, sus_clus_kwargs, null_kwargs, plot_kwargs)

    if clus_res['final_clus_res']['clus_success']: 
        log_kv(area=area, sscore_z=clus_res['final_clus_res']['sscore_z'], sscore=clus_res['final_clus_res']['sscore_mean'])
        results['region'].append(area)
        results['ss_z'].append(clus_res['final_clus_res']['sscore_z'])
        results['ss'].append(clus_res['final_clus_res']['sscore_mean'])
        results['ss_k'].append(len(np.unique(clus_res['final_clus_res']['clus_labels'])))
        results['ss_null'].append(clus_res['final_clus_res']['sscore_nulls'])
        results['ss_null_mean'].append(np.mean(clus_res['final_clus_res']['sscore_nulls']))
        results['ss_null_std'].append(np.std(clus_res['final_clus_res']['sscore_nulls']))
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

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv(os.path.join(clusfig_folder, "results.csv"))
pickle.dump(results, open(os.path.join(clusfig_folder, "results.pk"), 'wb'))


# results = pickle.load(open(os.path.join(clusfig_folder, "results.pk"), 'rb'))
# results_df = pd.DataFrame.from_dict(results)
results['region'] = np.asarray(results['region'])
results['H'] = np.asarray([area2H[a] for a in results['region']])
results['ss'] = np.asarray(results['ss'])
results['Silhouette Score (z)'] = np.asarray(results['ss_z'])
results['N'] = np.asarray(results['N'])


fig, axes = plt.subplots(3, 1,figsize=(5*1, 4*3))
hierarchy_trend(results, H1, f"Silhouette Score (z)", 
                    plot_trend=False, plot_regression=True, ax=axes[0])
hierarchy_trend(results, H1, f"ss", 
                    plot_trend=False, plot_regression=True, ax=axes[1])
hierarchy_trend(results, H1, f"N", 
                    plot_trend=False, plot_regression=True, ax=axes[-1])
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result_{N_nullG}.pdf")); plt.close('all')
pdb.set_trace()

