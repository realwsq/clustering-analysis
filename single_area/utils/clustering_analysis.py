import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from utils.clustering_algo import clustering
from utils.dr_algo import dr
from utils.plot import clustered_data_mat_plot, dr_scatterplots_main

def cluster_analysis_selectivity_space(beta_all, algo_kwargs, sus_clus_kwargs, null_kwargs, plot_kwargs):
    ### step 1-4: clustering analysis of data
    # initialization
    roundi = 0; good_nismask = np.ones(len(beta_all), dtype=bool)
    clus_success = True
    X_orig_all = beta_all.copy()

    # while loop, until there is no SUSpicious cluster or the number of neurons is too small
    while True:
        # step 1: Check whether there are more than min_N neurons and only continue if so.
        if np.sum(good_nismask) < algo_kwargs['min_N']: 
            print(f"clus failed because only {np.sum(good_nismask)} neurons left at round {roundi}!")
            clus_success = False
            break
        
        # step 2-3: perform clustering analysis
        X4clus = X_orig_all[good_nismask] # (n_neurons, n_vars)
        clus_res_roundi = _clus_main(X4clus, algo_kwargs)
        
        # step 4: detect suspicious clusters if any
        sus_clus_res_roundi = _check_sus_clus(clus_res_roundi['clus_labels'], clus_res_roundi['sscores'], good_nismask, sus_clus_kwargs)

        # post-process and prepare for the next round
        roundi += 1
        good_nismask &= (~sus_clus_res_roundi['suspicious_nismask'])
        if not sus_clus_res_roundi['has_sus_clus']:
            break
    
    final_result = {"clus_success": None, 
                    "sscore_z": None, "sscore_mean": None, "sscore_nulls": None,
                    "X": None, "clus_labels": None, "good_nismask": None, "neuron_sort": None, 
                    "example_null_data": None}
    if not clus_success:
        final_result["clus_success"] = False
    else:
        ### step 5-6: clustering analysis of null data
        sscore_nulls = []
        _X_nulls = _construct_null_model(X4clus, null_kwargs)
        for i in tqdm(range(null_kwargs['N_null']-1, -1, -1), desc="null model clustering"):
            _X_null = _X_nulls[i].copy()
            clus_res_nulli = _clus_main(_X_null, algo_kwargs)
            sscore_nulls.append(clus_res_nulli['sscore_mean'])

        final_result["clus_success"] = True
        final_result["sscore_z"] = (clus_res_roundi['sscore_mean'] - np.mean(sscore_nulls)) / np.std(sscore_nulls)
        final_result["sscore_mean"] = clus_res_roundi['sscore_mean']
        final_result["sscore_nulls"] = sscore_nulls
        final_result["X"] = clus_res_roundi['X']
        final_result["clus_labels"] = clus_res_roundi['clus_labels']
        final_result["good_nismask"] = good_nismask
        final_result['neuron_sort'] = clus_res_roundi['neuron_sort']
        final_result["example_null_data"] = dict(X=clus_res_nulli['X'], 
                                                 clus_labels=clus_res_nulli['clus_labels'], 
                                                 neuron_sort=clus_res_nulli['neuron_sort'])

    # plot the result
    if (plot_kwargs['plot']) and (final_result['clus_success']): 
        fig, axes = plt.subplots(3, 2, figsize=(3*2, 4*3))
        vs = plot_kwargs['vs'] if 'vs' in plot_kwargs else np.arange(X4clus.shape[1])
        # plot the data selectivity matrix
        clustered_data_mat_plot(clus_res_roundi['X'][clus_res_roundi['neuron_sort']], 
                                clus_res_roundi['clus_labels'][clus_res_roundi['neuron_sort']], 
                                vs, axes[0][0], fig, 'bwr')
        axes[0][0].set_title(plot_kwargs['save_id'])
        dr_scatterplots_main(clus_res_roundi['X'], clus_res_roundi['clus_labels'], 
                                dict(method='lda'), 
                                dict(ax=axes[0][1]))
        
        # plot an example null data set
        clustered_data_mat_plot(clus_res_nulli['X'][clus_res_nulli['neuron_sort']], 
                                clus_res_nulli['clus_labels'][clus_res_nulli['neuron_sort']], 
                                vs, axes[1][0], fig, 'bwr')
        axes[1][0].set_title("null data")
        dr_scatterplots_main(clus_res_nulli['X'], clus_res_nulli['clus_labels'], 
                                dict(method='lda'), 
                                dict(ax=axes[1][1]))

        # distribution of null SS and the mean SS of the real data
        ax = axes[2][0]
        ax.hist(final_result["sscore_nulls"], bins=20)
        _mean = np.mean(final_result["sscore_nulls"]); _std = np.std(final_result["sscore_nulls"])
        for _stdi in [-2,-1,0,1,2]:
            ax.axvline(x=_mean+_stdi*_std, color='k')
        ax.axvline(x=final_result['sscore_mean'], color='r', linewidth=3)
        ax.set_xlabel("sscore", fontsize=10)
        ax.set_ylabel("# null models", fontsize=10)
        ax.set_title(f"ss_z: {final_result['sscore_z']:.2f}")
        
        # compute and plot the influence of each variable to the clustering result (footprint) 
        #   by the decrease in SS after removing the variable
        _sscore_mean_orig = final_result['sscore_mean']
        coef_vs = final_result['X']
        clus_labels = final_result['clus_labels']
        footprint = np.zeros((coef_vs.shape[1]))
        for vi in range(coef_vs.shape[1]):
            sel_rm = np.asarray([coef_vs[:, _vi] for _vi in range(coef_vs.shape[1]) if not (_vi ==vi)]).T 
            _sscore_mean_rm = silhouette_score(sel_rm, clus_labels, metric="euclidean")
            footprint[vi] = _sscore_mean_orig-_sscore_mean_rm
        ax = axes[2][1]
        ax.bar(vs, footprint)
        ax.set_xticklabels(vs, rotation=90)
        ax.set_ylabel("decrease in SS \n (by removing the variable)")
        
        sns.despine(); plt.tight_layout()
        plt.savefig(os.path.join(plot_kwargs['folder'], f"{plot_kwargs['save_id']}.pdf")); 
        plt.close()
        
    return final_result

def cluster_analysis_condition_space(mfr_all, algo_kwargs, sus_clus_kwargs, null_kwargs, plot_kwargs):
    ### step 1-4: clustering analysis of data
    # initialization
    roundi = 0; good_nismask = np.ones(len(mfr_all), dtype=bool)
    clus_success = True
    X_orig_all = mfr_all.copy()

    # while loop, until there is no SUSpicious cluster or the number of neurons is too small
    while True:
        # step 1: Check whether there are more than min_N neurons and only continue if so.
        if np.sum(good_nismask) < algo_kwargs['min_N']: 
            print(f"clus failed because only {np.sum(good_nismask)} neurons left at round {roundi}!")
            clus_success = False
            break
        
        # step 2-3: perform clustering analysis
        X_orig = X_orig_all[good_nismask] # (n_neurons, n_conds)
        #   preprocess the data for clustering
        X4clus = (X_orig-X_orig.mean(1, keepdims=True))/X_orig.std(1, keepdims=True)
        X4clus = dr(X4clus, dict(method="pca", exp_var=0.9)) # (n_neurons, n_features4clus)
        #   cluster
        clus_res_roundi = _clus_main(X4clus, algo_kwargs)
        
        # step 4: detect suspicious clusters if any
        sus_clus_res_roundi = _check_sus_clus(clus_res_roundi['clus_labels'], clus_res_roundi['sscores'], good_nismask, sus_clus_kwargs)

        # post-process and prepare for the next round
        roundi += 1
        good_nismask &= (~sus_clus_res_roundi['suspicious_nismask'])
        if not sus_clus_res_roundi['has_sus_clus']:
            break
    
    final_result = {"clus_success": None, 
                    "sscore_z": None, "sscore_mean": None, "sscore_nulls": None,
                    "X": None, "clus_labels": None, "X_orig": None, "good_nismask": None}
    if not clus_success:
        final_result["clus_success"] = False
    else:
        ### step 5-6: clustering analysis of null data
        sscore_nulls = []
        _X_nulls_orig = _construct_null_model(X_orig, null_kwargs)
        for i in tqdm(range(null_kwargs['N_null']-1, -1, -1), desc="null model clustering"):
            _X_null_orig = _X_nulls_orig[i].copy()
            #   preprocess the data for clustering the same way
            _X_null = (_X_null_orig-_X_null_orig.mean(1, keepdims=True))/_X_null_orig.std(1, keepdims=True)
            _X_null = dr(_X_null, dict(method="pca", exp_var=0.9)) # (n_neurons, n_features4clus)
            clus_res_nulli = _clus_main(_X_null, algo_kwargs)
            sscore_nulls.append(clus_res_nulli['sscore_mean'])

        final_result["clus_success"] = True
        final_result["sscore_z"] = (clus_res_roundi['sscore_mean'] - np.mean(sscore_nulls)) / np.std(sscore_nulls)
        final_result["sscore_mean"] = clus_res_roundi['sscore_mean']
        final_result["sscore_nulls"] = sscore_nulls
        final_result["X"] = clus_res_roundi['X']
        final_result["clus_labels"] = clus_res_roundi['clus_labels']
        final_result["X_orig"] = X_orig
        final_result["good_nismask"] = good_nismask

    # plot the result
    if (plot_kwargs['plot']) and (final_result['clus_success']): 
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        # distribution of null SS and the mean SS of the real data
        ax.hist(final_result["sscore_nulls"], bins=20)
        _mean = np.mean(final_result["sscore_nulls"]); _std = np.std(final_result["sscore_nulls"])
        for _stdi in [-2,-1,0,1,2]:
            ax.axvline(x=_mean+_stdi*_std, color='k')
        ax.axvline(x=final_result['sscore_mean'], color='r', linewidth=3)
        ax.set_xlabel("sscore", fontsize=10)
        ax.set_ylabel("# null models", fontsize=10)
        ax.set_title(f"ss_z: {final_result['sscore_z']:.2f}")
        
        sns.despine(); plt.tight_layout()
        plt.savefig(os.path.join(plot_kwargs['folder'], f"{plot_kwargs['save_id']}.pdf")); 
        plt.close()
        
    return final_result

"""
Function for calculating the SS of the 2D data matrix. 
Output: dictionary of
    - sscore_mean: float
    - X: 2d matrix
    - neuron_sort: np.array([index to sort the neurons (first by cluster label, then by sscore)])
    - clus_labels: np.array([cluster label for each neuron])
    - sscores: np.array([sscore for each neuron])
"""
def _clus_main(X_lowd, algo_kwargs):
    if len(X_lowd) >= algo_kwargs['min_N']: pass
    else: return None

    clus_res = clustering(X_lowd, algo_kwargs['algo'],
                                order_label=True, 
                                plot_res=False,
                                **algo_kwargs)

    return dict(sscore_mean=clus_res['sscores_mean'], 
                X=X_lowd,
                neuron_sort=clus_res['y_sort'], 
                clus_labels=clus_res['res'], 
                sscores=clus_res['sscores_sample'])

"""
Function for checking whether there is SUSpicious clusters.
A cluster whose total SScore summed over all neurons is mainly contributed by 
    neurons from one single session (>sus_clus_kwargs['sus_clus_thres']) is 
    defined as SUSpicious cluster.
Output (dictionary of ...):
    - has_sus_clus: True/ False
    - suspicious_nismask: np.array([whether suspicious for each neuron])
"""
def _check_sus_clus(clus_labels, sscores, good_nismask, sus_clus_kwargs):
    suspicious_nismask = np.zeros_like(good_nismask)
    if sus_clus_kwargs['remove_sus_clus']:
        sessions = sus_clus_kwargs['sessions_orig'][good_nismask]
        for _li in np.unique(clus_labels):
            _li_mask = clus_labels==_li
            if np.sum(_li_mask)==1: 
                suspicious_nismask[good_nismask] |= _li_mask
            else:
                for _eid in np.unique(sessions[_li_mask]):
                    _li_session_mask = (clus_labels==_li) & (sessions==_eid)
                    if np.sum(sscores[_li_session_mask])/np.sum(sscores[_li_mask]) >= sus_clus_kwargs['sus_clus_thres']:
                        print(f"label {_li} is suspicious because of session {_eid}!")
                        suspicious_nismask[good_nismask] |= _li_session_mask
    return {"has_sus_clus": np.sum(suspicious_nismask)>0,
            "suspicious_nismask": suspicious_nismask}

"""
Function for constructing N_null null datasets 
    with specified single-mode distribution
    that has the same mean and covariance as the original data.
Output: np.array of shape (N_null, n_neurons, n_features)
"""
def _construct_null_model(X, null_kwargs):
    if (null_kwargs['null_dist'] == "Gaussian"):
        np.random.seed(42)
        X_null = np.random.multivariate_normal(np.mean(X, 0), np.cov(X.T), null_kwargs['N_null']*len(X))
    elif (null_kwargs['null_dist'] == "multi-log-normal"):
        log_X = np.log(X) # Only include neurons whose mfr > 0 for all conditions
        np.random.seed(42)
        log_X_null = np.random.multivariate_normal(np.mean(log_X, 0), np.cov(log_X.T), len(log_X)*null_kwargs['N_null'])
        X_null = np.exp(log_X_null)
    else:
        assert False, f"Unknown null distribution: {null_kwargs['null_dist']}"
    return X_null.reshape((null_kwargs['N_null'], len(X), -1))

"""
ePAIRS analysis with the proposed pipeline.
"""
def epairs_main(X_4_epairs, epairs_kwargs, null_kwargs):
    null_kwargs_epairs = null_kwargs.copy()
    null_kwargs_epairs['N_null'] = epairs_kwargs['N_null'] # rewrite the N_null
    epairs_res = epairs(X_4_epairs, epairs_kwargs['n_neigh'], null_kwargs_epairs)
    return epairs_res

def epairs(beta, n_neigh, null_kwargs):
    def _mean_dis(X):
        # preprocess the data for clustering the same way
        X = (X-X.mean(1, keepdims=True))/X.std(1, keepdims=True)
        X = dr(X, dict(method="pca", exp_var=0.9)) # (n_neurons, n_features4clus)
        # epairs analysis
        nbrs = NearestNeighbors(n_neighbors=n_neigh+1,  metric='cosine').fit(X)
        dist, inds = nbrs.kneighbors(X, n_neigh+1, return_distance=True)
        angs = np.arccos(1-dist)[:,1:].mean(1)
        ang_median = np.median(angs)
        return ang_median, angs

    ang_median_data, angs_data = _mean_dis(beta)

    ang_nulls = []
    _X_nulls = _construct_null_model(beta, null_kwargs)
    for i in tqdm(range(len(_X_nulls))):
        ang_median_null, angs_null_example = _mean_dis(_X_nulls[i])
        ang_nulls.append(ang_median_null)
    
    c = (ang_median_data-np.mean(ang_nulls))/np.std(ang_nulls)
    p = np.mean(ang_nulls<ang_median_data)
    return {'epairs_p': p, 'epairs_z': c} 

"""
Rand index measure of how well the neurons' 
    functional clustering labels matches with the anatomical acronym labels.
"""
def area_identity_analysis(clus_labels, area_labels, plot_kwargs):
    # transform area labels from text to 0,1,2,... for calculating the Rand Index
    area_labels_unique = np.unique(area_labels)
    area_labels_01 = np.array([np.where(area_labels_unique==a)[0][0] for a in area_labels])
    # calculate the Rand Index and its z-score by comparing with shuffled area labels
    ri = rand_score(area_labels_01, clus_labels)
    ri_shuffled = []
    np.random.seed(42)
    for si in range(10000):
        area_labels_01_shuffled = np.random.permutation(area_labels_01)
        ri_shuffled.append(rand_score(area_labels_01_shuffled, clus_labels))
    ri_shuffled = np.array(ri_shuffled)
    ri_z = (ri-np.mean(ri_shuffled))/np.std(ri_shuffled)
    
    # plot the result
    if plot_kwargs['plot']:
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        ax.hist(ri_shuffled, bins=20, color='gray', label='shuffled RI', density=True)
        ax.axvline(x=np.mean(ri_shuffled), color='gray', linestyle='--', linewidth=2, label='shuffled mean')
        ax.axvline(x=ri, color='r', linewidth=3, label='RI')
        ax.set_xlabel("RI")
        ax.set_ylabel("density")
        ax.legend()
        ax.set_title(f"{plot_kwargs['save_id']} RI: {ri:.2f} (z={ri_z:.1f})")
        sns.despine(); plt.tight_layout()
        plt.savefig(os.path.join(plot_kwargs['folder'], f"{plot_kwargs['save_id']}_ri.pdf")); plt.close()
        
    return dict(ri=ri, ri_z=ri_z)