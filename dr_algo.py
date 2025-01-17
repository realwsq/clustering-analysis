import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt



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


def dr_scatterplots(data_lowd, labels, sizes, axes, cmap='tab20'):
    ncolor = int(cmap[-2:])
    _min = np.min(data_lowd)
    _max = np.max(data_lowd)
    for ax, label, size in zip(axes, labels, sizes):
        label = label.astype(int)
        label_m1 = label==-1
        ax.scatter(*data_lowd[label_m1].T, s=size[label_m1], color='gray', alpha=0.1)
        ax.scatter(*data_lowd[~label_m1].T, s=size[~label_m1], c=plt.get_cmap(cmap)(label[~label_m1]%ncolor), alpha=0.5)
        ax.set_xlim([_min*1.1, _max*1.1])
        ax.set_ylim([_min*1.1, _max*1.1])
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        

def dr_scatterplots_main(data, labels, dr_kwargs, plot_kwargs):
    if (dr_kwargs['method'] == 'lda') and len(np.unique(labels))<3: return
    if dr_kwargs['method'] == 'lda': dr_kwargs['labels'] = labels
    dr_kwargs['ncomp'] = 2
    data_lowd = dr(data, dr_kwargs)
    
    s = plot_kwargs['size'] if 'size' in dr_kwargs else 2
    ax = plot_kwargs['ax']
    dr_scatterplots(data_lowd, 
                    [labels],
                    [s*np.ones(len(labels))],
                    [ax], cmap='tab10')
    ax.set_title(dr_kwargs['method'])
   