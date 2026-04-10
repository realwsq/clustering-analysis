import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def clustering(data_mat, clus_algo, 
               order_label=False, 
               **clus_kwargs):
    if clus_algo == "kmeans":
        clus_res = kmeans_sort(data_mat, **clus_kwargs)
    elif clus_algo == "leiden":
        clus_res = leiden_sort(data_mat, **clus_kwargs)
    else:
        assert False, "invalid clus_algo"

    clus_labels = clus_res['res']
    if order_label:
        # order the cluster labels so that the cluster with smaller labels has higher silhouette score
        n_clus = len(np.unique(clus_labels))
        _sscores = silhouette_samples(data_mat, clus_labels, metric=clus_kwargs['dis_metric'])
        _sscores_mean = [np.mean(_sscores[clus_labels==li]) for li in range(n_clus)]
        labels_sorted = np.argsort(_sscores_mean)[::-1]
        labels_sorted = {lidx: li for li, lidx, in enumerate(labels_sorted)}
        clus_labels = np.array([labels_sorted[lidx] for lidx in clus_labels])
        clus_res['res'] = clus_labels # rewrite
    # sort the neurons (y_sort) first by their cluster labels and then by their sample silhouette scores
    n_clus = len(np.unique(clus_labels))
    _sscores = silhouette_samples(data_mat, clus_labels, metric=clus_kwargs['dis_metric'])
    y_sort = np.concatenate([np.where(clus_labels==li)[0][np.argsort(_sscores[clus_labels==li])[::-1]] for li in range(n_clus)])
    clus_res['sscores_mean'] = np.mean(_sscores)
    clus_res['sscores_sample'] = _sscores
    clus_res['y_sort'] = y_sort

    return clus_res

def _get_best(ms_hist, _clus_and_eval, **kwargs):
    _ = np.argmax(ms_hist['sscore'])
    best_param = ms_hist['param'][_]
    res = _clus_and_eval(**kwargs, **best_param)
    return res

def kmeans_sort(data_mat, n_clus_lim=None, dis_metric='euclidean', n_init=50, 
                **others):
    assert dis_metric in ["euclidean", "manhattan"], "kmeans only take euclidean or manhattan distance"
    def _clus_and_eval(**kwargs):
        clustering = KMeans(random_state=42, 
                            init='random',
                            max_iter=1000,
                            n_init=kwargs['n_init'],
                            n_clusters=kwargs['n_clusters']).fit(data_mat) 
        labels = clustering.labels_
        sscore = silhouette_score(data_mat, labels, metric=dis_metric)

        return dict(sscore=sscore, res=labels,
                    model=clustering, n_clus=len(np.unique(labels)),
                    **kwargs)
        
    ms_hist = {'param': [], "sscore": []}
    for n_clus in tqdm(range(n_clus_lim[0], n_clus_lim[1]), desc="kmeans_sort"):
        if n_clus >= len(data_mat): continue
        param = dict(n_clusters=n_clus, n_init=n_init)
        sc_clus_res = _clus_and_eval(**param)
        ms_hist['param'].append(param)
        ms_hist['sscore'].append(sc_clus_res['sscore'])
    
    best_res = _get_best(ms_hist, _clus_and_eval)
    best_res['ms_hist'] = ms_hist
    return best_res


def leiden_sort(data_mat, n_neighbor_list, dis_metric='euclidean', 
                **others):
    print("="*60)
    print("install `leidenalg` to use Leiden algorithm for clustering")
    print("="*60)
    import leidenalg as la
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors
    def eval_leiden_partition(data_mat, labels, G, partition, dis_metric):
        eval_funs = dict(sscore=lambda: silhouette_score(data_mat, labels, metric=dis_metric), 
                        modularity=lambda: G.modularity(partition) )
        res = {}
        for e in eval_funs:
            res[e] = eval_funs[e]()
        return res

    def _clus_and_eval(data_mat, **kwargs):
        nbrs = NearestNeighbors(n_neighbors=kwargs['n_neighbors'], metric=kwargs['metric']).fit(data_mat)
        aff_mat = nbrs.kneighbors_graph(data_mat).toarray()
        aff_mat = aff_mat + aff_mat.T ## to be symmetric
        aff_mat = aff_mat > 0
        G = ig.Graph.Adjacency(aff_mat.tolist())
        partition = la.find_partition(G, 
                                      partition_type=la.ModularityVertexPartition, 
                                      weights=None,
                                      n_iterations=-1,
                                      seed=42)
        n_clus = len(partition)
        labels = np.zeros(len(data_mat), dtype=int)
        for i in range(n_clus): labels[partition[i]] = i
        _res = eval_leiden_partition(data_mat, labels, G, partition, dis_metric)

        return dict(sscore=_res['sscore'], res=labels, 
                    modularity=_res['modularity'], 
                    aff_mat=aff_mat, n_clus=n_clus, 
                    **kwargs)
    
    if n_neighbor_list is None: n_neighbor_list = range(2, len(data_mat)) 
    ms_hist = {'param': [], "sscore": [], "modularity": []}
    for k in tqdm(n_neighbor_list, desc="leiden_sort"):
        param = dict(n_neighbors=k, metric=dis_metric)
        if k >= len(data_mat):
            continue
        res = _clus_and_eval(data_mat, **param)
        ms_hist['param'].append(param)
        ms_hist['sscore'].append(res['sscore'])
        ms_hist['modularity'].append(res['modularity'])

    best_res = _get_best(ms_hist, _clus_and_eval, data_mat=data_mat)
    best_res['ms_hist'] = ms_hist
    return best_res
    
    

