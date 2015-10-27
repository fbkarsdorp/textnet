import igraph

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

from bootstrap_network import evolving_graphs, to_graph
from utils import nx2igraph


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
        'n': len(graph.vs), 
        'm': len(graph.es), 
        'D': graph.diameter(directed=False), 
        'ED': np.percentile(graph.eccentricity(), q=90),
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


def fit_densification(statistics):
    """
    Fit densification of nodes and edges according to x ** alpha. 
    """
    def densification(x, alpha):
        return x ** alpha
    popt, pcov = curve_fit(densification, statistics.n, statistics.m)
    print("alpha = %.3f" % popt[0])
    sns.plt.plot(statistics.n, statistics.m, 'o', markeredgewidth=1, markeredgecolor='k', markerfacecolor='None')
    sns.plt.plot(statistics.n, densification(statistics.n, *popt), '-k')
    return r2_score(densification(statistics.n, *popt), statistics.m)


def evolving_graph_statistics(choices, time_index, groupby=lambda x: x, sigma=0.5, lower_degree_bounds=0):
    """
    Utility function to compute the topological properties of graphs created 
    at different points in time.
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
    with different thresholds of sigma.
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