import pandas as pd
import seaborn as sns
sns.set_style("white")

from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances


def timeline_scatter_plot(X, time_index, method='MDS', metric='cosine', **kwargs):
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex(time_index)
    dm = pairwise_distances(X, metric=metric)
    if method.upper() == 'MDS':
        decomposer = MDS(n_components=2, dissimilarity='precomputed', verbose=1, **kwargs)
        decomposer.fit(dm)
    elif method.upper() == 'TSNE':
        decomposer = TSNE(n_components=2, metric='precomputed', verbose=1, **kwargs)
        decomposer.fit(dm)
    else:
        raise ValueError("Method %s is not supported..." % method)
    X, Y = decomposer.embedding_[:,0], decomposer.embedding_[:,1]
    unique_index = time_index.unique().order()
    colormap = {time_stamp: color for time_stamp, color in zip(
        unique_index, sns.cubehelix_palette(unique_index.shape[0]))}
    colors = [colormap[time_stamp] for time_stamp in time_index]
    sns.plt.scatter(X, Y, s=40, color=colors, alpha=0.7)
    sns.plt.axis('off')


