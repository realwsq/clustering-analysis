import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
import leidenalg as la
import igraph as ig

from utils import log_kv, load_or_save_dict, remove_space


def _get_best(ms_hist, _train_and_eval, **kwargs):
    _ = np.argmax(ms_hist['ms_metric'])
    best_param = ms_hist['param'][_]
    best_metric = ms_hist['ms_metric'][_]
    res = _train_and_eval(**kwargs, **best_param)
    return res


def kmeans_sort(data_mat, n_clus_lim=None, dis_metric='euclidean',
                ms_metric="sscore", n_init=10, kmeans_plus=False,
                save_prefix=None, **others):
    assert dis_metric == "euclidean", "kmeans only take euclidean distance"
    def _train_and_eval(**kwargs):
        if kmeans_plus == False:
            clustering = KMeans(random_state=42, 
                                init='random',
                                max_iter=1000,
                                n_init=kwargs['n_init'],
                                n_clusters=kwargs['n_clusters']).fit(data_mat) # n_init=200
        else:
            clustering = KMeans(random_state=42, 
                                init='k-means++',
                                max_iter=1000,
                                n_init=1,
                                n_clusters=kwargs['n_clusters']).fit(data_mat)
        labels = clustering.labels_
        sscore = silhouette_score(data_mat, labels)
        sscore_samples = silhouette_samples(data_mat, labels)
        sscore_macro = np.mean([np.mean(sscore_samples[labels==li]) for li in range(np.max(labels)+1)])
        if ms_metric == "sscore":
            ms_score = sscore
        elif ms_metric == "sscore_macro":
            ms_score = sscore_macro
        else:
            assert False, "invalid ms_metric"
        return dict(eval_metrics={"sscore":sscore, "sscore_macro":sscore_macro}, 
                    ms_metric=ms_score, res=labels,
                    model=clustering, n_clus=len(np.unique(labels)),
                    **kwargs)
    def _main(n_clus_lim):
        ms_hist = {'param': [], "eval_metrics": [], "ms_metric": []}
        for n_clus in tqdm(range(n_clus_lim[0], n_clus_lim[1]), desc="kmeans_sort"):
            if n_clus >= len(data_mat): continue
            param = dict(n_clusters=n_clus, n_init=n_init, kmeans_plus=kmeans_plus)
            sc_clus_res = _train_and_eval(**param)
            ms_hist['param'].append(param)
            ms_hist['eval_metrics'].append(sc_clus_res['eval_metrics'])
            ms_hist['ms_metric'].append(sc_clus_res['ms_metric'])
        
        best_res = _get_best(ms_hist, _train_and_eval)
        best_res['ms_hist'] = ms_hist
        return best_res

    save_midfix = remove_space(f"{n_clus_lim}")
    if save_prefix is not None:
        best_res = load_or_save_dict(f"{save_prefix}_{save_midfix}_best.pkl", _main, n_clus_lim=n_clus_lim)
    else:
        best_res = _main(n_clus_lim)

    return best_res         



def leiden_sort(data_mat, k_list, dis_metric='minkowski', 
                n_iterations=2, partition_type='modularity', weighted=False,
                eval_metrics=["sscore", "modularity"], ms_metric="modularity", 
                save_prefix=None, **others):
    def eval_leiden_partition(data_mat, labels, G, partition, eval_metrics, dis_metric):
        eval_funs = dict(sscore=lambda: silhouette_score(data_mat, labels, metric=dis_metric), 
                        modularity=lambda: G.modularity(partition) )
        res = {}
        for e in eval_metrics:
            res[e] = eval_funs[e]()
        return res

    def _train_and_eval(data_mat, **kwargs):
        nbrs = NearestNeighbors(n_neighbors=kwargs['n_neighbors'], metric=kwargs['metric']).fit(data_mat)
        aff_mat = nbrs.kneighbors_graph(data_mat).toarray()
        aff_mat = aff_mat + aff_mat.T ## to be symmatric
        aff_mat = aff_mat > 0
        G = ig.Graph.Adjacency(aff_mat.tolist())
        if weighted:
            dis_mat = pairwise_distances(data_mat, metric=kwargs['metric'])
            dis_mat = dis_mat[aff_mat]
            weights = np.exp(-dis_mat)
            G.es['weight'] = weights
            weights = G.es['weight']
        else:
            weights = None
        if kwargs['partition_type'] == 'modularity':
            partition_type = la.ModularityVertexPartition
        elif kwargs['partition_type'] == 'CPM':
            partition_type = la.CPMVertexPartition
        partition = la.find_partition(G, 
                                      partition_type, 
                                      weights=weights,
                                      n_iterations=kwargs['n_iterations'],
                                      seed=42)
        n_clus = len(partition)
        labels = np.zeros(len(data_mat), dtype=int)
        for i in range(n_clus):
            labels[partition[i]] = i
        _res = eval_leiden_partition(data_mat, labels, G, partition, eval_metrics, dis_metric)

        return dict(eval_metrics=_res, ms_metric=_res[ms_metric], res=labels, 
                    aff_mat=aff_mat, n_clus=n_clus, model=None, # not able to save the model
                    **kwargs)
    
    def _main(k_list, data_mat):
        if k_list is None:
            k_list = range(2, len(data_mat))
        ms_hist = {'param': [], "eval_metrics": [], "ms_metric": []}
        for k in tqdm(k_list, desc="leiden_sort"):
            param = dict(n_neighbors=k, metric=dis_metric, 
                         n_iterations=n_iterations, partition_type=partition_type)
            if k >= len(data_mat):
                continue
            # if save_prefix is not None:
            #     res = load_or_save_dict(f"{save_prefix}_{save_midfix}_{k}.pkl", _train_and_eval, **param)
            # else:
            #     res = _train_and_eval(**param)
            res = _train_and_eval(data_mat, **param)
            ms_hist['param'].append(param)
            ms_hist['eval_metrics'].append(res['eval_metrics'])
            ms_hist['ms_metric'].append(res['ms_metric'])

        best_res = _get_best(ms_hist, _train_and_eval, data_mat=data_mat)
        best_res['ms_hist'] = ms_hist
        return best_res
    
    save_midfix = remove_space(f"{k_list}")
    if save_prefix is not None:
        best_res = load_or_save_dict(f"{save_prefix}_{save_midfix}_best.pkl", _main, k_list=k_list, data_mat=data_mat)
    else:
        best_res = _main(k_list, data_mat)

    return best_res


# clustering
def clustering(data_mat, clus_algo, 
               order_label=False, # whether to order label
               **clus_kwargs):
    if clus_algo == "kmeans":
        clus_res = kmeans_sort(data_mat, **clus_kwargs)
    elif clus_algo == "kmeans++":
        clus_res = kmeans_sort(data_mat, kmeans_plus=True, **clus_kwargs)
    elif clus_algo == "leiden":
        clus_res = leiden_sort(data_mat, **clus_kwargs)
    else:
        assert False, "invalid clus_algo"

    n_clus = clus_res['n_clus']; 
    clus_labels = clus_res['res']
    if order_label:
        n_clus = len(np.unique(clus_labels))
        _sscores = silhouette_samples(data_mat, clus_labels, metric=clus_kwargs['dis_metric'])
        _sscores_mean = [np.mean(_sscores[clus_labels==li]) for li in range(n_clus)]
        labels_sorted = np.argsort(_sscores_mean)[::-1]
        labels_sorted = {lidx: li for li, lidx, in enumerate(labels_sorted)}
        clus_labels = np.array([labels_sorted[lidx] for lidx in clus_labels])
        clus_res['res'] = clus_labels # rewrite
    n_clus = len(np.unique(clus_labels))
    _sscores = silhouette_samples(data_mat, clus_labels, metric=clus_kwargs['dis_metric'])
    y_sort = np.concatenate([np.where(clus_labels==li)[0][np.argsort(_sscores[clus_labels==li])[::-1]] for li in range(n_clus)])
    clus_res['sscores_sample'] = _sscores
    clus_res['y_sort'] = y_sort

    return clus_res



def clustered_data_mat_plot(data_mat_sorted, labels, vs, ax, fig, cmap='bwr', _vmax=None, _vmin=None, xticklabel=True, ylabel=True):
    if _vmax is None:
        _vmax = np.percentile(np.abs(data_mat_sorted), 95)
    
    if _vmin is None:
        _vmin = np.percentile(np.abs(data_mat_sorted), 5)
    
    if cmap=='bwr':
        improp = dict(aspect='auto', cmap='bwr', interpolation='nearest', vmax=_vmax, vmin=-_vmax)
    elif cmap=='Greys':
        improp = dict(aspect='auto', cmap='Greys', interpolation='nearest', vmax=_vmax, vmin=_vmin)
    else:
        assert False, "invalid cmap"
    im = ax.imshow(data_mat_sorted, **improp)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('selectivity', fontsize=10)  

    for li in range(np.max(labels)):
        ax.axhline(y=np.sum(labels<=li)-0.5, c='k', linewidth=3)

    ax.set_xticks(np.linspace(0, data_mat_sorted.shape[1], len(vs), endpoint=False))
    if xticklabel:
        ax.set_xticklabels(vs, rotation = 90, ha="right") 
    else: 
        ax.set_xticklabels([""]*len(vs)) 
    if ylabel:  
        ax.set_ylabel("neuron",) 
