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
data = data = pickle.load(open("example_mfr/utils/CT_for_shuqi.pck", 'rb'))


inc_param=dict(min_N=40, min_r2=0.015, inc_nowhisking=True)
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

if inc_param['inc_nowhisking']:
    conditions = [_ for _ in list(data['MOp'][0].keys())]
else:
    conditions = [_ for _ in list(data['MOp'][0].keys()) if _[0] == 'W']


res = import_areagroup()
conn_area_list_byH = res['cortical_area_list']
H1 = res['hierarchy'][0]
area2H = {H1[_H][0]: _H for _H in H1}


## clustering analysis
N_nullG = 100; 
null_kwargs = {"N_null": N_nullG, **null_params}
sus_clus_kwargs = {"remove_sus_clus": True, "sessions_orig": None, "sus_clus_thres": sus_clus_thres}
algo_kwargs = {"min_N": inc_param['min_N'], 'clus_folder': clus_folder, 'save_id': None, 
               **clus_param}
beta_preprocess_kwargs = {"preprocess": beta_preprocess,}
plot_kwargs = dict(plot=True, folder=clusfig_folder, vs=[f"{c}" for c in conditions])

results = dict(region=[], ss_z=[], ss=[], ss_k=[], ss_null=[], ss_null_mean=[], ss_null_std=[], N=[], dispersion=[])
for area in tqdm(conn_area_list_byH):
    log_kv(area=area)
    if area not in data: continue ### only for mfr
    algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
    
    coef_vs_area = []; session_area = []
    for eid, data_eid in enumerate(data[area]):
        mfr = np.stack([data_eid[c].mean(0) for c in conditions], axis=1) # (N, conditions)
        nis_inc = mfr.std(1) > 1e-3
        if clus_param['sel'].endswith(">0"):
            nis_inc = nis_inc & np.all(mfr>0, axis=1)
        print(f"area: {area}, N%: {np.mean(nis_inc)}, N: {len(nis_inc)}")
        mfr = mfr[nis_inc]
        coef_vs_area.append(mfr)
        session_area.append([eid]*np.sum(nis_inc))
    sus_clus_kwargs['sessions_orig'] = np.concatenate(session_area)
    coef_vs_area = np.concatenate(coef_vs_area, 0)

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

        if null_kwargs['preprocess_null']:
            X_4_epairs = clus_res['data_clus_hist'][-1]['X_orig']
        else:
            X_4_epairs = clus_res['data_clus_hist'][-1]['X']
        np.save(os.path.join(clusfig_folder, f"{area}_X4epairs.npy"), X_4_epairs)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv(os.path.join(clusfig_folder, "results.csv"))
pickle.dump(results, open(os.path.join(clusfig_folder, "results.pk"), 'wb'))


# results = pickle.load(open(os.path.join(clusfig_folder, "results.pk"), 'rb'))
# results_df = pd.DataFrame.from_dict(results)
results['region'] = np.asarray(results['region'])
results['H'] = np.asarray([area2H[a] for a in results['region']])
results['Silhouette Score (z)'] = np.asarray(results['ss_z'])
results['ss_null_mean'] = np.asarray(results['ss_null_mean'])
results['ss_null_std'] = np.asarray(results['ss_null_std'])
results['ss'] = np.asarray(results['ss'])
results['N'] = np.asarray(results['N'])

results_epairs = pickle.load(open(os.path.join(clusfig_folder, "results_epairs.pk"), 'rb'))
results_epairs['epairs_p_clipped'] = np.clip(results_epairs['epairs_p'], 0.0, 0.05)
results_epairs['n_neigh'] = np.asarray(results_epairs['n_neigh'])
results_epairs_part = dict(); n_neigh = 1
for k in results_epairs:
    results_epairs_part[k] = np.asarray(results_epairs[k])[results_epairs['n_neigh']==n_neigh]
results_epairs_df = pd.DataFrame.from_dict(results_epairs)


fig, axes = plt.subplots(5, 1,figsize=(5*1, 4*5))
hierarchy_trend(results, H1, f"Silhouette Score (z)", 
                    plot_trend=False, plot_regression=True, ax=axes[0])
axes[0].axhline(y=ss.norm.ppf(1-0.01/len(results)), c='gray', linestyle='--')
axes[0].axhline(y=ss.norm.ppf(1-0.05/len(results)), c='gray', linestyle='--')
ax = axes[1]
hierarchy_trend(results, H1, f"ss", 
                    plot_trend=False, plot_regression=True, ax=ax)
x_hi = [hi for hi in H1 if H1[hi][0] in results['region']]
x_area = [H1[hi][0] for hi in x_hi]
y = [results_df['ss_null_mean'][np.where(results_df['region']==a)[0][0]] for a in x_area]
yerr = [results_df['ss_null_std'][np.where(results_df['region']==a)[0][0]]*2 for a in x_area]
ax.errorbar(x_hi, y=y, yerr=yerr, fmt='o', c='gray', label='null')
ax = axes[2]
hierarchy_trend(results_epairs_part, H1, f"epairs_p_clipped", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=0.01/len(results_epairs_part), c='gray', linestyle='--')
ax.axhline(y=0.05/len(results_epairs_part), c='gray', linestyle='--')
ax = axes[3]
hierarchy_trend(results_epairs_part, H1, f"epairs_z", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=-ss.norm.ppf(1-0.01/len(results_epairs_part)), c='gray', linestyle='--')
ax.axhline(y=-ss.norm.ppf(1-0.05/len(results_epairs_part)), c='gray', linestyle='--')
hierarchy_trend(results, H1, f"N", 
                    plot_trend=False, plot_regression=True, ax=axes[-1])
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result_{N_nullG}.pdf")); plt.close('all')
pdb.set_trace()


results_epairs = dict(region=[], epairs_z=[], epairs_p=[], n_neigh=[])
for n_neigh in [1, 3, 5]:
    epairs_kwargs = {"n_neigh": n_neigh, "N_null": 5000}
    for area in tqdm(conn_area_list_byH):
        if os.path.isfile(os.path.join(clusfig_folder, f"{area}_X4epairs.npy")):
            X_4_epairs = np.load(os.path.join(clusfig_folder, f"{area}_X4epairs.npy"))
        else:
            continue
        epairs_res = epairs_main(X_4_epairs, epairs_kwargs, null_kwargs, beta_preprocess_kwargs)
        log_kv(area=area, epairs_z=epairs_res['epairs_z'], epairs_p=epairs_res['epairs_p'])
        results_epairs['region'].append(area)
        results_epairs['n_neigh'].append(n_neigh)
        results_epairs['epairs_z'].append(epairs_res['epairs_z'])
        results_epairs['epairs_p'].append(epairs_res['epairs_p'])

    results_epairs_df = pd.DataFrame.from_dict(results_epairs)
    results_epairs_df.to_csv(os.path.join(clusfig_folder, "results_epairs.csv"))
    pickle.dump(results_epairs, open(os.path.join(clusfig_folder, "results_epairs.pk"), 'wb'))

results_epairs['epairs_p_clipped'] = np.clip(results_epairs['epairs_p'], 0.0, 0.05)
results_epairs['n_neigh'] = np.asarray(results_epairs['n_neigh'])
results_epairs_part = dict(); n_neigh = 1
for k in results_epairs:
    results_epairs_part[k] = np.asarray(results_epairs[k])[results_epairs['n_neigh']==n_neigh]
results_epairs_df = pd.DataFrame.from_dict(results_epairs)

plt.figure(figsize=(3.5, 3.5))
sns.swarmplot(data=results_epairs_df, x="n_neigh", y="epairs_z")
plt.axhline(y=-ss.norm.ppf(1-0.01/len(results_epairs_part)), c='gray', linestyle='--')
plt.axhline(y=-ss.norm.ppf(1-0.05/len(results_epairs_part)), c='gray', linestyle='--')
for ri, row in results_epairs_df[results_epairs_df.epairs_z < -ss.norm.ppf(1-0.01/len(results_epairs_part))].iterrows():
    plt.annotate(row.region, xy=(np.where(np.unique(results_epairs_df.n_neigh)==row.n_neigh)[0][0],row.epairs_z), 
                 fontsize=8, ha='left', va='center',)
                
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result_epairs_{N_nullG}.pdf")); plt.close('all')



fig, axes = plt.subplots(2, 1,figsize=(5*1, 4*2))
ax = axes[0]
hierarchy_trend(results_epairs_part, H1, f"epairs_p_clipped", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=0.01/len(results_epairs_part), c='gray', linestyle='--')
ax.axhline(y=0.05/len(results_epairs_part), c='gray', linestyle='--')
ax = axes[1]
hierarchy_trend(results_epairs_part, H1, f"epairs_z", 
                    plot_trend=False, plot_regression=True, ax=ax)
ax.axhline(y=-ss.norm.ppf(1-0.01/len(results_epairs_part)), c='gray', linestyle='--')
ax.axhline(y=-ss.norm.ppf(1-0.05/len(results_epairs_part)), c='gray', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result_epairs_{N_nullG}_{n_neigh}.pdf")); plt.close('all')

pdb.set_trace()
pdb.set_trace()
pdb.set_trace()

    
    