from utils import make_folder, remove_space, log_kv
from example1.utils.utils import import_areagroup, hierarchy_trend
from clustering_analysis import cluster_analysis

import pdb, os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as ss


local_folder_lf = f"/burg/stats/users/sw3894/bwm_original/cortexbwm/RRR_local_folder_original"
RRR_res_df = pd.read_json(os.path.join(local_folder_lf, "RRRglobal_full.json"))
resgood_folder = make_folder("./example1/res_good")

inc_param=dict(min_N=50, min_r2=0.015) 
## parameters for clustering
# setup 1
clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
                sel="beta_sum", ) 
beta_preprocess = [['sum', 2]]
# # setup 2
# clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
#                 sel="beta_PCA", ) 
# beta_preprocess = [['pca_temporal', [False, 0.8], False]]
# # setup 3
# clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
#                 sel="beta_PCA", ) 
# beta_preprocess = [['pca_temporal', [True, 1], False]]
# # setup 4
# clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
#                 sel="beta", ) 
# beta_preprocess = [['pca', 0.5]]
# # setup 5
# clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
#                 sel="beta_sum", ) 
# beta_preprocess = [['sum', 2]]
# inc_param['min_r2']=0.01 
# # setup 6
# clus_param=dict(algo="kmeans",dis_metric="euclidean",n_clus_lim=[3,20], n_init=50, ms_metric='sscore',
#                 sel="beta_sum", ) 
# beta_preprocess = [['sum', 2]]
# inc_param['min_r2']=0.02 
# # setup 7 
# clus_param=dict(algo="leiden", dis_metric="euclidean", k_list=range(3,100), n_iterations=-1, eval_metrics=["sscore", "modularity"], ms_metric="sscore",
# # clus_param=dict(algo="spectral",k_list=[10,20,30],n_clus_lim=[3,20], n_init=50, ms_metric='sscore',dis_metric="euclidean",
#                 sel="beta_sum", ) 
# beta_preprocess = [['sum', 2]]


sus_clus_thres = 0.9
clusfig_folder = make_folder(os.path.join(resgood_folder, f"clus",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess}_{sus_clus_thres}")))
clus_folder = make_folder(os.path.join(local_folder_lf, f"clus",remove_space(f"{inc_param.values()}"), remove_space(f"{clus_param.values()}_{beta_preprocess}_{sus_clus_thres}")))
print(clusfig_folder)
print(clusfig_folder)
print(clus_folder)
print(clus_folder)



res = import_areagroup()
conn_area_list_byH = res['cortical_area_list']
H1 = res['hierarchy'][0]
area2H = {H1[_H][0]: _H for _H in H1}

nis_incmask = (RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']) > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])


## clustering analysis
N_nullG = 100; 
sus_clus_kwargs = {"remove_sus_clus": True, "sessions_orig": None, "sus_clus_thres": sus_clus_thres}
algo_kwargs = {"min_N": inc_param['min_N'], 'clus_folder': clus_folder, 'save_id': None, 
               **clus_param}
beta_preprocess_kwargs = {"preprocess": beta_preprocess, "preprocess_null": False}
plot_kwargs = dict(plot=True, folder=clusfig_folder, vs=['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick"])

results = dict(region=[], ss_z=[], ss=[], ss_k=[], ss_null=[], ss_null_mean=[], ss_null_std=[], N=[])
for area in tqdm(area_order_H):
    nis_incmask_area = (nis_incmask) & (RRR_res_df.acronym == area)
    log_kv(area=area, total_N=np.sum(nis_incmask_area))

    algo_kwargs['save_id'] = plot_kwargs['save_id'] = area
    eids = RRR_res_df.loc[nis_incmask_area,"eid"]
    betas = RRR_res_df.loc[nis_incmask_area,"RRRglobal_beta"]
    sus_clus_kwargs['sessions_orig'] = eids
    coef_vs_area = np.array([_[:-1] for _ in betas])
    sel_vs_area = coef_vs_area.sum(-1)

    plot_kwargs['plot_others'] = [coef_vs_area, sel_vs_area]
    clus_res = cluster_analysis(coef_vs_area, N_nullG, beta_preprocess_kwargs, algo_kwargs, sus_clus_kwargs, plot_kwargs)

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

fig, axes = plt.subplots(3, 1,figsize=(5*1, 4*3))
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
hierarchy_trend(results, H1, f"N", 
                    plot_trend=False, plot_regression=True, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(clusfig_folder, f"result_{N_nullG}.pdf")); plt.close('all')


pdb.set_trace()
pdb.set_trace()
pdb.set_trace()

    