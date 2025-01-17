import numpy as np
import pdb, os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from cluster_algo import clustering, clustered_data_mat_plot
from dr_algo import dr, dr_scatterplots_main

#### script for clustering analysis


"""
Function for calculating the z-scored SS of the 2D data matrix. 
Output: None (clustering failed) or dictionary of
    - sscore_mean: float
    - X: 2d matrix
    - vs: np.array([name for each dim])
    - neuron_sort: index to sort the neurons
    - clus_labels: np.array([cluster label for each neuron])
    - sscores: np.array([sscore for each neuron])
"""
def _clus_main(X_lowd, vs, algo_kwargs):
    if len(X_lowd) >= algo_kwargs['min_N']: pass
    else: return None

    _clus_res_area = clustering(X_lowd, algo_kwargs['algo'],
                                order_label=True, 
                                save_prefix=os.path.join(algo_kwargs['clus_folder'], algo_kwargs['save_label']),
                                plot_res=False,
                                **algo_kwargs)
    
    clus_labels = _clus_res_area['res']
    sscore_samples = _clus_res_area['sscores_sample']

    return dict(sscore_mean=np.mean(sscore_samples), 
                X=X_lowd,
                vs=vs,
                neuron_sort=_clus_res_area['y_sort'], 
                clus_labels=clus_labels, sscores=sscore_samples)

"""
Function for checking whether there is suspicious clusters.
Suspicious cluster is defined as the cluster whose neurons largely belong to one same session.
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
                        print(f"label {_li} is suspicious!")
                        suspicious_nismask[good_nismask] |= _li_session_mask
    
        return {"has_sus_clus": np.sum(suspicious_nismask)>0,
                "suspicious_nismask": suspicious_nismask,} 
    else:
        # skip checking
        return {"has_sus_clus": False,
                "suspicious_nismask": suspicious_nismask}
    


def PCA_dr(X, n_comp_fixed, normalize=False):
    vs = np.array([f"{i}" for i in range(X.shape[1])])

    X_lowd = []; vs_new = []
    for i in range(X.shape[1]):
        _X = X[:, i]
        if n_comp_fixed[0] == False:
            pca = PCA().fit(_X)
            n_comp = np.where(np.cumsum(pca.explained_variance_ratio_)>n_comp_fixed[1])[0][0]+1
        else:
            n_comp = n_comp_fixed[1]
        _X_lowd = PCA(n_components=n_comp).fit_transform(_X)
        X_lowd.append(_X_lowd)
        vs_new.append([f"{vs[_]}_{_}" for _ in range(n_comp)])
    X_lowd = np.concatenate(X_lowd,axis=1)
    vs_new = np.concatenate(vs_new,axis=0)
    if normalize:
        X_lowd = (X_lowd-np.mean(X_lowd,0))/np.std(X_lowd,0)
    return X_lowd, vs_new


def _plot_clus_res(X, vs, X_orig, y_sort, clus_labels, plot_kwargs, others=None):
    Nothers_toplot = 0 if others is None else len(others)
    nrow = 2+Nothers_toplot
    fig = plt.figure(figsize=(3*3, 4*nrow))
    gs = GridSpec(nrow, 3, height_ratios=[1]*nrow, width_ratios=[1, 1, 1])
    ax = fig.add_subplot(gs[0, 0])
    clustered_data_mat_plot(X[y_sort], 
                            clus_labels[y_sort], 
                            vs, 
                            ax, fig,
                            'bwr' if np.min(X)<0 else 'Greys')
    ax2 = fig.add_subplot(gs[0, 1:]) 
    clustered_data_mat_plot(X_orig[y_sort].reshape((len(X_orig), -1)), 
                            clus_labels[y_sort], 
                            vs,
                            ax2, fig, 
                            'bwr' if np.min(X_orig)<0 else 'Greys')
    ax3 = fig.add_subplot(gs[1, 0])  # First subplot in the second row
    ax4 = fig.add_subplot(gs[1, 1])  # Second subplot in the second row
    ax5 = fig.add_subplot(gs[1, 2])  # Third subplot in the second row
    dr_scatterplots_main(X, clus_labels, 
                            dict(method='tsne', perplexity=5), 
                            dict(ax=ax3))
    dr_scatterplots_main(X, clus_labels, 
                            dict(method='pca'), 
                            dict(ax=ax4))
    dr_scatterplots_main(X, clus_labels, 
                            dict(method='lda'), 
                            dict(ax=ax5))
    for i in range(Nothers_toplot):
        _X = others[i]
        _X = _X.reshape((len(_X), -1))
        ax = fig.add_subplot(gs[2+i, :2])
        clustered_data_mat_plot(_X[y_sort], 
                            clus_labels[y_sort], 
                            vs, 
                            ax, fig,
                            'bwr' if np.mean(_X>=0)<0.9 else 'Greys')
        ax = fig.add_subplot(gs[2+i, 2])
        dr_scatterplots_main(_X, clus_labels, 
                            dict(method='tsne'), 
                            dict(ax=ax))
    plt.tight_layout(); sns.despine();
    fname = os.path.join(plot_kwargs['folder'], f"{plot_kwargs['save_name']}.pdf")
    plt.savefig(fname); 
    plt.close()


"""
Main function for calculating the z-scored SS of the 2D data matrix. 
Input:
    - beta: [N, nv], model time-varying coefficients
    - N_null: # of null datasetes
    - beta_preprocess_kwargs: steps for preprocessing the beta, specified as list of tuples
    - algo_kwargs: clustering algorithm and the accompanying kwargs
        - min_N: int
        - clus_folder: string
        - save_id: string 
            -> save_file will be os.path.join(clus_folder, f"{save_id}_xxx_.pkl")
        - algo: string
            - kmeans:
                - n_clus_lim: [a, b]
                - dis_metric: "euclidean"
                - n_init: int
    - whether remove the suspicious clusters
        - remove_sus_clus: True/ False
        - sessions_orig: np.array([session id for each neuron])
        - sus_clus_thres: float
Output:
    - final_clus_res:
        - clus_success: True/ False
        - sscore_z: float
        - sscore_mean: float
        - sscore_nulls: list of ss_nulls
        - X: 2d matrix
        - vs: np.array([name for each dim])
        - neuron_sort: index to sort the neurons
        - nismask: np.array([whether used in clustering for each neuron])
        - clus_labels: np.array([cluster label for each neuron])
        - sscore_samples: np.array([ss for each neuron])
        - clus_res_null_example: clustering example of one null dataset
    - data_clus_hist: list of history:
        - clus_success: True/ False
        - sscore_mean: float
        - nismask: np.array([whether used in clustering for each neuron])
        - X: 2d matrix
        - X_orig_all: 2d matrix, before preprocessing
        - X_orig: 2d matrix, before preprocessing
        - vs: np.array([name for each dim])
        - neuron_sort: index to sort the neurons
        - clus_labels: np.array([cluster label for each neuron])
        - sscore_samples: np.array([ss for each neuron])
        - has_sus_clus: whether has sus clus this round
"""
def cluster_analysis(beta, N_null, beta_preprocess_kwargs, algo_kwargs, sus_clus_kwargs, plot_kwargs):
    # initialization
    roundi = 0; data_clus_hist = []; good_nismask = np.ones(len(beta), dtype=bool)
    X_orig_all = beta.copy()
    # while loop
    while True:
        algo_kwargs['save_label'] = f"{algo_kwargs['save_id']}_round{roundi}"
        if np.sum(good_nismask) < algo_kwargs['min_N']: 
            hist = {"clus_success": False}
            data_clus_hist.append(hist)
            break

        X_lowd = X_orig = X_orig_all[good_nismask]
        if beta_preprocess_kwargs['preprocess'] is None:
            pass
        elif type(beta_preprocess_kwargs['preprocess']) is list:
            for p in beta_preprocess_kwargs['preprocess']:
                if p[0] == "normalize":
                    _axis = p[1]
                    X_lowd = (X_lowd-X_lowd.mean(_axis, keepdims=True))/X_lowd.std(_axis, keepdims=True)
                elif p[0] == "pca":
                    X_lowd = X_lowd.reshape((len(X_lowd), -1))
                    
                    if type(p[1]) == int:
                        dr_kwargs=dict(method=p[0], ncomp=p[1])
                    elif type(p[1]) == float:
                        dr_kwargs=dict(method=p[0], exp_var=p[1])
                    X_lowd = dr(X_lowd, dr_kwargs)
                    plot_kwargs['vs'] = np.array([f"PC{i}" for i in range(X_lowd.shape[1])])

                elif p[0] == "pca_temporal":
                    X_lowd, vs_new = PCA_dr(X_lowd, p[1], normalize=p[2])
                    plot_kwargs['vs'] = vs_new

                elif p[0] == "sum":
                    _axis = p[1]
                    X_lowd = np.sum(X_lowd, _axis)
                
        else:
            assert False
        vs = plot_kwargs['vs']
        clus_res_roundi = _clus_main(X_lowd, vs, algo_kwargs)
            
        
        sus_clus_res_roundi = _check_sus_clus(clus_res_roundi['clus_labels'], clus_res_roundi['sscores'], good_nismask, sus_clus_kwargs)

        # collect result
        hist = {"clus_success": True,
                "sscore_mean": clus_res_roundi['sscore_mean'],
                "nismask": good_nismask,
                "X": clus_res_roundi['X'],
                "X_orig_all": X_orig_all,
                "X_orig": X_orig,
                "vs": clus_res_roundi['vs'],
                "neuron_sort": clus_res_roundi['neuron_sort'],
                "clus_labels": clus_res_roundi['clus_labels'],
                "sscore_samples": clus_res_roundi['sscores'],
                "has_sus_clus": sus_clus_res_roundi['has_sus_clus']
                } 
        data_clus_hist.append(hist)

        # plot data if needed
        if plot_kwargs['plot']:
            _plot_kwargs = dict(folder=plot_kwargs['folder'], save_name=f"{plot_kwargs['save_id']}_data_hist_{roundi}")
            _plot_clus_res(hist['X'], 
                           hist['vs'], 
                           hist['X_orig'], 
                           hist['neuron_sort'], 
                           hist['clus_labels'], 
                           _plot_kwargs,
                           others=[_[hist['nismask']] for _ in plot_kwargs['plot_others']] if 'plot_others' in plot_kwargs else None,
                           )
            # ## temporary for plotting areas
            if "area_labels" in plot_kwargs:
                fig, axes = plt.subplots(2,2,figsize=(3.5*2, 3.5*2))
                for ri, method in enumerate(["tsne", "pca",]):
                    dr_scatterplots_main(hist['X'], hist['clus_labels'], 
                                    dict(method=method), 
                                    dict(ax=axes[ri][0]))
                    dr_scatterplots_main(hist['X'], plot_kwargs['area_labels'][hist['nismask']], 
                                    dict(method=method), 
                                    dict(ax=axes[ri][1]))
                plt.tight_layout(); sns.despine()
                fname = os.path.join(plot_kwargs['folder'], f"{_plot_kwargs['save_name']}_area.pdf")
                plt.savefig(fname); 
                plt.close()

        # post-process and prepare for the next round
        roundi += 1
        good_nismask &= (~sus_clus_res_roundi['suspicious_nismask'])
        if not sus_clus_res_roundi['has_sus_clus']:
            break
    
    final_result = {"clus_success": None, "sscore_z": None, "clus_labels": None}
    if data_clus_hist[-1]['clus_success'] == False:
        final_result["clus_success"] = False
    else:
        _X_orig = data_clus_hist[-1]['X_orig']
        _X = data_clus_hist[-1]['X']
        _vs = data_clus_hist[-1]['vs']
        sscore_nulls = []
        for i in tqdm(range(N_null)):
            np.random.seed(42+i)

            if beta_preprocess_kwargs['preprocess_null']:
                _X_null = np.random.multivariate_normal(np.mean(_X_orig, 0), np.cov(_X_orig.T), len(_X_orig))
                _X_null_orig = _X_null.copy()
                # preprocess the null data
                if beta_preprocess_kwargs['preprocess'] is None:
                    pass
                elif type(beta_preprocess_kwargs['preprocess']) is list:
                    for p in beta_preprocess_kwargs['preprocess']:
                        if p[0] == "normalize":
                            _axis = p[1]
                            _X_null = (_X_null-_X_null.mean(_axis, keepdims=True))/_X_null.std(_axis, keepdims=True)
                        elif p[0] == "pca":
                            if type(p[1]) == int:
                                dr_kwargs=dict(method=p[0], ncomp=p[1])
                            elif type(p[1]) == float:
                                dr_kwargs=dict(method=p[0], exp_var=p[1])
                            _X_null = dr(_X_null, dr_kwargs)
                else:
                    assert False
            else:
                _X_null = np.random.multivariate_normal(np.mean(_X, 0), np.cov(_X.T), len(_X))
                _X_null_orig = _X_null.copy()

            algo_kwargs['save_label'] = f"{algo_kwargs['save_id']}_nullG{i}"
            clus_res_nulli = _clus_main(_X_null, _vs, algo_kwargs)

            sscore_nulls.append(clus_res_nulli['sscore_mean'])

            # plot a null dataset if needed
            if (plot_kwargs['plot']) and ((i==0)): 
                # plot the result of the last null dataset
                _plot_kwargs = dict(folder=plot_kwargs['folder'], save_name=f"{plot_kwargs['save_id']}_nullG{i}")
                _plot_clus_res(clus_res_nulli['X'], 
                            clus_res_nulli['vs'], 
                            clus_res_nulli['X'], ## placeholder
                            clus_res_nulli['neuron_sort'], 
                            clus_res_nulli['clus_labels'], 
                            _plot_kwargs)

        
        final_result["clus_success"] = True
        final_result["sscore_z"] = (data_clus_hist[-1]['sscore_mean'] - np.mean(sscore_nulls)) / np.std(sscore_nulls)
        final_result["sscore_mean"] = data_clus_hist[-1]['sscore_mean']
        final_result["sscore_nulls"] = sscore_nulls
        final_result["X"] = data_clus_hist[-1]['X']
        final_result["vs"] = data_clus_hist[-1]['vs']
        final_result["neuron_sort"] = data_clus_hist[-1]['neuron_sort']
        final_result["nismask"] = data_clus_hist[-1]['nismask']
        final_result["clus_labels"] = data_clus_hist[-1]['clus_labels']
        final_result["sscore_samples"] = data_clus_hist[-1]['sscore_samples']
        final_result["clus_res_null_example"] = dict(X=clus_res_nulli['X'], X_orig=_X_null_orig, clus_labels=clus_res_nulli['clus_labels'],
                                                     neuron_sort=clus_res_nulli['neuron_sort'],
                                                     sscore_mean=clus_res_roundi['sscore_mean'], 
                                                     sscore_samples=clus_res_roundi['sscores'], 
                                                     vs=data_clus_hist[-1]['vs'])

        # plot the overall result if needed
        if (plot_kwargs['plot']): 
            fig, axes = plt.subplots(1,2,figsize=(3.5*2,3.5))

            # sscores
            ax = axes[0]
            ax.hist(final_result["sscore_nulls"], bins=20)
            _mean = np.mean(final_result["sscore_nulls"]); _std = np.std(final_result["sscore_nulls"])
            for _stdi in [-2,-1,0,1,2]:
                ax.axvline(x=_mean+_stdi*_std, color='k')
            ax.axvline(x=final_result['sscore_mean'], color='r', linewidth=3)
            ax.set_xlabel("sscore"); ax.set_ylabel("# null models")
            ax.set_title(f"z: {final_result['sscore_z']:.2f}")
            
            # compute and plot the footprint
            _sscore_mean_orig = final_result['sscore_mean']
            coef_vs = final_result['X']
            clus_labels = final_result['clus_labels']
            N_clus = np.max(clus_labels)+1
            footprints = np.zeros((coef_vs.shape[1]))
            for vi in range(coef_vs.shape[1]):
                sel_rm = np.asarray([coef_vs[:, _vi] for _vi in range(coef_vs.shape[1]) if not (_vi ==vi)]).T 
                _sscore_mean_rm = silhouette_score(sel_rm, clus_labels, metric="euclidean")
                footprints[vi] = _sscore_mean_orig-_sscore_mean_rm
            ax = axes[1]
            ax.plot(footprints, '-x')
            ax.set_ylabel("decrease in SS \n (by removing v)")
            ax.set_xticks(np.arange(len(final_result['vs'])))
            ax.set_xticklabels(final_result['vs'],rotation = 90, ha="right")
            ax.tick_params(axis='both', which='major', labelsize=15)

            plt.tight_layout()
            plt.savefig(os.path.join(plot_kwargs['folder'], f"{plot_kwargs['save_id']}.pdf")); 
            plt.close()

    return dict(final_clus_res=final_result, data_clus_hist=data_clus_hist)
    



