from utils import make_folder, remove_space, log_kv
from example1.utils.utils import import_areagroup
from clustering_analysis import cluster_analysis

import pdb, os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


local_folder_lf = f"/burg/stats/users/sw3894/bwm_original/cortexbwm/RRR_local_folder_original"
RRR_res_df = pd.read_json(os.path.join(local_folder_lf, "RRRglobal_full.json"))
resgood_folder = make_folder("./example1/res_good")

null_params = {"null_dist": "Gaussian", "preprocess_null": False,}
# parameters for clustering
clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
                sel="beta_sum", ) 
beta_preprocess = [['sum', 2]]
inc_param=dict(min_N=50, min_r2=0.015, rem_clus=["VISp", "AUDp", "SSp-ul", "SSp-ll"]) 



sus_clus_thres = 0.9
clusfig_folder = make_folder(os.path.join(resgood_folder, f"clus_module",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess}_{sus_clus_thres}_{null_params}")))
clus_folder = make_folder(os.path.join(local_folder_lf, f"clus_module",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess}_{sus_clus_thres}_{null_params}")))
print(clusfig_folder)
print(clusfig_folder)
print(clus_folder)
print(clus_folder)



res = import_areagroup()
conn_area_list_byH = res['cortical_area_list']
hierarchy_byHarris = res['hierarchy_byHarris'][0]

nis_incmask = (RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']) > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])

## clustering analysis
N_nullG = 50; 
null_kwargs = {"N_null": N_nullG, **null_params}
sus_clus_kwargs = {"remove_sus_clus": True, "sessions_orig": None, "sus_clus_thres": sus_clus_thres}
algo_kwargs = {"min_N": inc_param['min_N'], 'clus_folder': clus_folder, 'save_id': None, 
               **clus_param}
beta_preprocess_kwargs = {"preprocess": beta_preprocess}
plot_kwargs = dict(plot=True, folder=clusfig_folder, vs=['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker", "lick"])


results = dict(region=[], 
            ss_z=[], ss=[], ss_k=[], ss_null=[], ss_null_mean=[], ss_null_std=[], 
            mfr=[], N=[])
for ap_name, ap in tqdm((list(hierarchy_byHarris.items())+[["cortex", conn_area_list_byH]])):
    log_kv(ap_name=ap_name)
    if ap_name in ['visual', 'auditory']: continue
    ap = [a for a in ap if a in area_order_H] # remove non-sufficient area
    if "rem_clus" in inc_param:
        ap = [a for a in ap if not (a in inc_param['rem_clus'])] # remove clustered area
    if len(ap) == 0: continue

    nis = []; area_labels = []
    for i in range(len(ap)):
        nis_i = np.where((nis_incmask_ctx)&(RRR_res_df.acronym==ap[i]))[0]
        nis.append(nis_i)
        area_labels.append([i]*len(nis_i))
    nis_incmask_area = np.zeros(len(RRR_res_df), dtype=bool)
    nis_incmask_area[np.concatenate(nis)] = True
    area_labels = np.concatenate(area_labels)
    log_kv(total_N=np.sum(nis_incmask_area))


    algo_kwargs['save_id'] = plot_kwargs['save_id'] = ap_name
    eids = RRR_res_df.loc[nis_incmask_area,"eid"]
    betas = RRR_res_df.loc[nis_incmask_area,"RRRglobal_beta"]
    sus_clus_kwargs['sessions_orig'] = eids
    coef_vs_area = np.array([_[:-1] for _ in betas])
    sel_vs_area = coef_vs_area.sum(-1)

    plot_kwargs['plot_others'] = [coef_vs_area, sel_vs_area]
    plot_kwargs['area_labels'] = area_labels
    clus_res = cluster_analysis(coef_vs_area, N_nullG, beta_preprocess_kwargs, algo_kwargs, sus_clus_kwargs, null_kwargs, plot_kwargs)

    if clus_res['final_clus_res']['clus_success']: 
        log_kv(ap_name=ap_name, sscore_z=clus_res['final_clus_res']['sscore_z'], epairs_p=clus_res['final_clus_res']['epairs_p'])
        results['region'].append(ap_name)
        results['ss_z'].append(clus_res['final_clus_res']['sscore_z'])
        results['ss'].append(clus_res['final_clus_res']['sscore_mean'])
        results['ss_k'].append(len(np.unique(clus_res['final_clus_res']['clus_labels'])))
        results['ss_null'].append(clus_res['final_clus_res']['sscore_nulls'])
        results['ss_null_mean'].append(np.mean(clus_res['final_clus_res']['sscore_nulls']))
        results['ss_null_std'].append(np.std(clus_res['final_clus_res']['sscore_nulls']))
        results['N'].append(clus_res['final_clus_res']['X'].shape[0])
        mfr = RRR_res_df.loc[nis_incmask_area, 'mfr_task'][clus_res['final_clus_res']['nismask']].mean()
        results['mfr'].append(mfr)

results_df = pd.DataFrame.from_dict(results)
results_df.to_csv(os.path.join(clusfig_folder, "results.csv"))
pickle.dump(results, open(os.path.join(clusfig_folder, "results.pk"), 'wb'))



fig, axes = plt.subplots(1, 2, sharey=True,
                         gridspec_kw={'width_ratios': [1, 4]}, figsize=(4, 3.5))
ax = axes[0]
sns.barplot(data=results_df[results_df.region=='cortex'], x="region", y="ss_z", ax=ax)

ax = axes[1]
sns.barplot(data=results_df[~(results_df.region=='cortex')], x="region", y="ss_z", ax=ax,
            order=['somatomotor', 'medial', 'lateral', 'prefrontal'])

# Rotate x-axis tick labels
for ax in axes:
    ax.set_xlabel('')
    ax.set_ylabel('Silhouette Score (z)')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
plt.tight_layout(); sns.despine()
plt.savefig(os.path.join(clusfig_folder, remove_space(f"result_{N_nullG}.pdf"))); plt.close('all')

