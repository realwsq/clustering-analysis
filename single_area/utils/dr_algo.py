import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def dr(data, dr_kwargs):
    if data.shape[1] > 2:
        seed = dr_kwargs['seed'] if 'seed' in dr_kwargs else 0
        if dr_kwargs['method'] == 'tsne':
            pp = dr_kwargs['perplexity'] if 'perplexity' in dr_kwargs else 15
            data_lowd = TSNE(n_components=dr_kwargs['ncomp'], random_state=seed, 
                             learning_rate='auto', method='barnes_hut' if dr_kwargs['ncomp']<4 else 'exact', init='random', 
                             perplexity=pp).fit_transform(data)
        elif dr_kwargs['method'] == 'pca':
            if ("ncomp" not in dr_kwargs) and ("exp_var" in dr_kwargs):
                pca = PCA().fit(data)
                ncomp = np.where(np.cumsum(pca.explained_variance_ratio_)>dr_kwargs['exp_var'])[0][0]+1
                ncomp = np.clip(ncomp, 2, None)
            elif ("ncomp" in dr_kwargs):
                ncomp = dr_kwargs['ncomp']
            else:
                assert False
            data_lowd = PCA(n_components=ncomp).fit_transform(data)
        elif dr_kwargs['method'] == 'mds':
            embedding = MDS(n_components=dr_kwargs['ncomp'], metric=False, random_state=seed)
            data_lowd = embedding.fit_transform(data)
        elif dr_kwargs['method'] == 'lda':
            data_lowd = LinearDiscriminantAnalysis(n_components=dr_kwargs['ncomp']).fit_transform(data, dr_kwargs['labels'])
    else:
        data_lowd = data.copy()
    return data_lowd

