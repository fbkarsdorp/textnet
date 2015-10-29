import igraph

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from .bootstrap_network import evolving_graphs, to_graph
from .utils import nx2igraph


def effective_diameter(G, mode="ALL", q=90):
    if mode == 'OUT':
        return np.percentile(G.eccentricity(), q=q)
    distance_matrix = np.array(G.shortest_paths(mode=mode), dtype=np.float64)
    distance_matrix[distance_matrix == np.inf] = np.nan
    return np.percentile(np.nanmax(distance_matrix, axis=1), q)


def graph_statistics(graph, lower_degree_bounds=0):
    """
    Function used to compute some topological properties of a graph.

    Parameters
    ----------

    graph : igraph.Graph(directed=True) or nx.DiGraph
        The input graph, can be constructed using to_graph or one of the
        other graph creating functions.
    lower_degree_bounds : integer, default 0
        Set lower_degree_bounds to 0 if you don't want to include unconnected
        nodes in the degree computations, -1 otherwise.
    """
    if not isinstance(graph, igraph.Graph) and isinstance(graph, nx.DiGraph):
        graph = nx2igraph(graph)
    degree_distribution = np.array(graph.degree())
    in_degree_distribution = np.array(graph.indegree())
    largest_component = max(map(len, graph.components(mode="WEAK"))) / len(graph.vs)
    return {
        'n': graph.vcount(), 
        'm': graph.ecount(), 
        'D': graph.diameter(directed=True),
        "ED": effective_diameter(graph, mode="OUT", q=90),
        'APL': graph.average_path_length(directed=False), 
        'CC': graph.transitivity_undirected(), 
        'k': degree_distribution[degree_distribution > lower_degree_bounds].mean(),
        'k_var': degree_distribution.var(), 
        'k_in': in_degree_distribution[in_degree_distribution > lower_degree_bounds].mean(), 
        'k_in_var': in_degree_distribution.var(), 
        'density': graph.density(),
        'gini_d': gini_coeff(degree_distribution[degree_distribution > lower_degree_bounds]),
        'gini_d_in': gini_coeff(in_degree_distribution[in_degree_distribution > lower_degree_bounds]),
        'comp_f': largest_component
    }


def fit_densification(statistics, ax=None):
    """
    Fit densification of nodes and edges according to x ** alpha as described by
    Leskovec et al. (2007) Graph evolution. Densification and shrinking diameters

    Parameters
    ----------
    statistics : output of graph_statistics
    """
    def densification(x, alpha):
        return x ** alpha
    popt, pcov = curve_fit(densification, statistics.n, statistics.m)
    sns.plt.plot(statistics.n, statistics.m, 'o', markeredgewidth=1, markeredgecolor='k', markerfacecolor='None', ax=ax)
    sns.plt.plot(statistics.n, densification(statistics.n, *popt), '-k', ax=ax)
    return r2_score(densification(statistics.n, *popt), statistics.m), popt[0]


def evolving_graph_statistics(choices, time_index, groupby=lambda x: x, sigma=0.5, lower_degree_bounds=0):
    """
    Utility function to compute the topological properties of graphs created 
    at different points in time. The function expects a time_index with Timestamp 
    objects. These Timestamps are grouped using the specified groupby function.
    The function return as pandas DataFrame with topological statistics for each
    (grouped) point in time. Examples of the groupby function:

    >>> groupby = lambda x: x.year # create graphs for each year
    >>> groupby = lambda x: x.year // 10 * 10 # create graphs for each decade
    >>> groupby = pd.TimeGrouper(freq='M') # create graphs for each month

    Parameters
    ----------
    Choices : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : ndarray of Timestamps or pandas DatetimeIndex, shape: (n_samples_X), 
        Index corresponding to time points of each sample in X. If supplied,
        neighbors for each item x in X will only consist of samples that occur 
        before or at the time point corresponding with x. Default is None.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    sigma : float, default 0.5
        The threshold percentage of how often a data point must be 
        assigned as nearest neighbor.
    lower_degree_bounds : integer, default 0
        Set lower_degree_bounds to 0 if you don't want to include unconnected
        nodes in the degree computations, -1 otherwise.        
    """
    statistics = []
    for time_step, graph in evolving_graphs(choices, time_index, groupby=groupby, sigma=sigma):
        graph_stats = graph_statistics(graph, lower_degree_bounds=lower_degree_bounds)
        graph_stats['time'] = time_step
        statistics.append(graph_stats)
    statistics = pd.DataFrame(statistics).set_index('time')
    return statistics


def eval_sigmas(neighbors, min_sigma=0, max_sigma=1, step_size=0.01):
    """
    Utility function that computes topological properties of graphs created
    at different thresholds of sigma.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    min_sigma : float, default 0
        the minimum value of sigma at which the topological statistics are computed.
    max_sigma : float, default 1
        the maximum value of sigma at which the topological statistics are computed.
    step_size : float, default 0.01
        increase sigma with step_size
    """
    statistics = []
    for sigma in np.arange(min_sigma + step_size, max_sigma + step_size, step_size):
        G = to_graph(neighbors, sigma=sigma)
        stats = graph_statistics(G)
        stats['sigma'] = sigma
        statistics.append(stats)
    return pd.DataFrame(statistics).set_index('sigma')    


def cdf(x, survival=False):
    "Return the cumulative distribution function of x."
    x = np.array(x)
    x = x[x > 0]
    x = np.sort(np.array(x))
    cdf = np.searchsorted(x, x, side='left') / x.shape[0]
    unique_data, unique_indices = np.unique(x, return_index=True)
    x = unique_data
    cdf = cdf[unique_indices]
    return x, 1 - cdf if survival else cdf


def ccdf(x):
    "Return the complementary cumulative distribution function of x."
    return cdf(x, survival=True)


def lorenz(data):
    "Compute a lorenz curve for the data."
    d = sorted(data, reverse=True)
    n, s, p = len(d), sum(d), np.arange(0.0, 1.01, 0.01)
    c = np.zeros(p.shape[0])
    items = np.zeros(p.shape[0])
    i = 0
    for x in p:
        if x == 0:
            items[i] = 0
            c[i] = 0
        else:
            items[i] = int(np.floor(n * x));
            c[i] = sum(d[:int(items[i])]) / float(s)
        i += 1
    return p, c


def gini_coeff(data):
    "Compute the gini coefficient of the data."
    d = sorted(data, reverse=True)
    n = len(d)
    sq = 0.0
    for i in range(n):
        if i == 0:
            q0 = d[i]
        else:
            q1 = q0 + d[i]
            q0 = q1
        sq = sq + q0
    try:
        s = 2 * sq / sum(d) - 1
        R = n / (n - 1.) * (1. / n * s - 1.)
    except ZeroDivisionError:
        R = np.nan
    return R