# -*- coding: utf-8 -*-

from collections import deque

import networkx as nx
import numpy as np
import pandas as pd

from .statistics import evolving_graph_statistics
from .utils import node_counter, nx2igraph
from .network import to_graph, evolving_graphs


def empirical_growth(choices, time_index, groupby=lambda x: x, sigma=0.5):
    """
    Utility function that returns a dataframe consisting of the number of
    added nodes and edges per time step as specified by groupby.
    """
    statistics = []
    for time_step, graph in evolving_graphs(choices, time_index, groupby=groupby, sigma=sigma):
        statistics.append({'n': len(graph), 'time': time_step})
    return pd.DataFrame(statistics).set_index('time')


def randomized_dynamic_time_graph(neighbors, time_index, m=1, groupby=lambda x: x):
    """
    Returns a generator of random graphs at each time step t in time_index 
    according to the Barabási–Albert preferential attachment model. 

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : int, default 1
        Number of edges to attach from a new node to existing nodes
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
    G = nx.DiGraph()
    repeated_nodes = np.zeros(stats.n.max(), dtype=np.float64)
    all_nodes = np.arange(stats.n.max())
    for i in range(1, stats.shape[0]):
        new_nodes = np.arange(len(G), stats.n.iat[i])
        repeated_nodes[new_nodes] += m
        for new_node in new_nodes:
            p_vals = repeated_nodes / repeated_nodes.sum()
            targets = np.random.choice(all_nodes, size=m, replace=False, p=p_vals)
            G.add_node(new_node, date=stats.index[i])
            G.add_edges_from(zip([new_node] * m, targets))
            repeated_nodes[targets] += 1
        yield stats.index[i], G

import random
def _random_subset(seq,m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets=set()
    while len(targets) < m:
        x = random.choice(seq)
        targets.add(x)
    return targets

# def barabasi_albert_graph(neighbors, time_index, m=1, groupby=lambda x: x):
#     stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
#     G = nx.DiGraph()
#     for n in range(m):
#         G.add_node(n, date=stats.index[0])
#     targets = list(range(m))
#     repeated_nodes = []
#     source = m
#     i = 0
#     while source < stats.n.max():
#         G.add_node(source, date=stats.index[i])
#         G.add_edges_from(zip([source] * m, targets))
#         repeated_nodes.extend(targets)
#         repeated_nodes.extend([source] * m)
#         targets = _random_subset(repeated_nodes, m)
#         source += 1
#         if source > stats.n.iat[i]:
#             yield stats.index[i], G
#             i += 1

def barabasi_albert_graph(neighbors, time_index, m=1, groupby=lambda x: x):
    stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
    G = nx.DiGraph()
    for n in range(m):
        G.add_node(n, date=stats.index[0])
    targets = list(range(m))
    repeated_nodes = np.zeros(stats.n.max(), dtype=np.float64)
    all_nodes = np.arange(stats.n.max())
    source = m
    i = 0
    while source < stats.n.max():
        print(stats.index[i])
        G.add_node(source, date=stats.index[i])
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes[targets] += 1
        repeated_nodes[source] += m
        p = repeated_nodes / repeated_nodes.sum()
        targets = np.random.choice(all_nodes, size=m, p=p)
        source += 1
        if source >= stats.n.iat[i]:
            yield stats.index[i], G
            i += 1


def randomized_time_graph(neighbors, time_index, m=1, groupby=lambda x: x):
    """
    Returns a random graphs based on the empirical data 
    according to the Barabási–Albert preferential attachment model. 

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : int, default 1
        Number of edges to attach from a new node to existing nodes
    groupby : callable
        Function specifying the time steps at which the graphs should be created   
    """ 
    return deque(randomized_dynamic_time_graph(
        neighbors, time_index, m=m, groupby=groupby), maxlen=1)[0][1]


def chronological_attachment_model(neighbors, time_index, m=1, gamma=0.1, groupby=lambda x: x):
    """TODO: update documentation.
    Returns a generator of random graphs at each time step t in time_index 
    according to the Barabási–Albert preferential attachment model. 

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : int, default 1
        Number of edges to attach from a new node to existing nodes
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
    G = nx.DiGraph()
    for n in range(m):
        G.add_node(n, date=stats.index[0])
    targets = list(range(m))
    repeated_nodes = np.zeros(stats.n.max(), dtype=np.float64)
    all_nodes = np.arange(stats.n.max())
    source = m
    i = 0
    time_steps = np.array([t.year for t in time_index.order()])
    while source < stats.n.max():
        G.add_node(source, date=stats.index[i])
        G.add_edges_from(zip([source] * m, targets))
        repeated_nodes[targets] += 1
        repeated_nodes[source] += m
        weights = (time_steps - stats.index[0] + 1) ** gamma
        weights[time_steps > stats.index[i]] = 0
        p = repeated_nodes * weights
        p = p / p.sum()
        targets = np.random.choice(all_nodes, size=m, p=p)
        source += 1
        if source > stats.n.iat[i]:
            yield stats.index[i], G
            i += 1
    # stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
    # G = nx.DiGraph()
    # repeated_nodes = np.zeros(stats.n.max(), dtype=np.float64)
    # all_nodes = np.arange(stats.n.max())
    # time_steps = np.array([t.year for t in time_index.order()])
    # for i in range(1, stats.shape[0]):
    #     weights = (time_steps - stats.index[0] + 1) ** gamma
    #     weights[time_steps > stats.index[i]] = 0
    #     new_nodes = np.arange(len(G), stats.n.iat[i])
    #     repeated_nodes[new_nodes] += m
    #     for new_node in new_nodes:
    #         vals = repeated_nodes * weights
    #         p_vals = vals / vals.sum()
    #         targets = np.random.choice(all_nodes, size=m, replace=False, p=p_vals)
    #         G.add_node(new_node, date=stats.index[i])
    #         G.add_edges_from(zip([new_node] * m, targets))
    #         repeated_nodes[targets] += 1
    #     yield stats.index[i], G


def aging_model(neighbors, time_index, m=1, gamma=0.1, groupby=lambda x: x):
    """TODO: update documentation.
    Returns a generator of random graphs at each time step t in time_index 
    according to the Barabási–Albert preferential attachment model. 

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : int, default 1
        Number of edges to attach from a new node to existing nodes
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
    G = nx.DiGraph()
    for n in range(m):
        G.add_node(n, date=stats.index[0])
    targets = list(range(m))
    all_nodes = np.arange(stats.n.max())
    source = m
    i = 0
    time_steps = np.array([t.year for t in time_index.order()])
    while source < stats.n.max():
        G.add_node(source, date=stats.index[i])
        G.add_edges_from(zip([source] * m, targets))
        weights = (time_steps - stats.index[0] + 1) ** gamma
        weights[time_steps > stats.index[i]] = 0
        p = weights / weights.sum()
        targets = np.random.choice(all_nodes, size=m, p=p)
        source += 1
        if source > stats.n.iat[i]:
            yield stats.index[i], G
            i += 1    
    # stats = empirical_growth(neighbors, time_index, groupby=groupby).sort_index()
    # G = nx.DiGraph()
    # all_nodes = np.arange(stats.n.max())
    # time_steps = np.array([t.year for t in time_index.order()])
    # for i in range(1, stats.shape[0]):
    #     weights = (time_steps - stats.index[0] + 1) ** gamma
    #     weights[time_steps > stats.index[i]] = 0
    #     p_vals = weights / weights.sum()
    #     new_nodes = np.arange(len(G), stats.n.iat[i])
    #     for new_node in new_nodes:
    #         targets = np.random.choice(all_nodes, size=m, replace=False, p=p_vals)
    #         G.add_node(new_node, date=stats.index[i])
    #         G.add_edges_from(zip([new_node] * m, targets))
    #     yield stats.index[i], G


def gnp_random_dynamic_time_graph(neighbors, time_index, p=0.3, groupby=lambda x: x): 
    """
    Returns a generator of random graphs at each time step t according to the Erdős-Rényi
    graph model. Possible edges are defined by the time steps and probability p.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    p : float, default 0.3
        probability of creating an edge.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    index_series = pd.Series(sorted(neighbors.keys()), index=time_index)
    G = nx.DiGraph()
    _nodes = node_counter()
    for group_id, story_ids in index_series.groupby(groupby):
        for story_id in story_ids:
            G.add_node(_nodes[story_id])
            neighbors = np.where(time_index[story_id] <= time_index)[0]
            ps = np.random.rand(neighbors.shape[0]) < p
            for neighbor in neighbors[ps]:
                if not neighbor == story_id:
                    G.add_edge(_nodes[story_id], _nodes[neighbor])
        yield group_id, G


def gnp_random_time_graph(neighbors, time_index, p=0.3, groupby=lambda x: x):
    """
    Returns a random graph based on the empirical data according to the Erdős-Rényi
    graph model. Possible edges are defined by the time steps and probability p.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    p : float, default 0.3
        probability of creating an edge.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    return deque(gnp_random_dynamic_time_graph(
        neighbors, time_index, p, groupby=groupby), maxlen=1)[0][1]


def uniform_random_dynamic_time_graph(neighbors, time_index, m=1, groupby=lambda x: x): 
    """
    Returns a generator of random graphs at each time step t where for each story m
    edges are created at random.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : integer, default 1
        number of edges per node.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    index_series = pd.Series(sorted(neighbors.keys()), index=time_index)
    G = nx.DiGraph()
    _nodes = node_counter()
    for group_id, story_ids in index_series.groupby(groupby):
        for story_id in story_ids:
            G.add_node(_nodes[story_id])
            neighbors = np.where(time_index[story_id] <= time_index)[0]
            for neighbor in np.random.choice(neighbors, size=m):
                if not neighbor == story_id:
                    G.add_edge(_nodes[story_id], _nodes[neighbor])
        yield group_id, G


def uniform_random_time_graph(neighbors, time_index, m=1, groupby=lambda x: x):
    """
    Returns a random graph based on the empirical data according to the Erdős-Rényi
    graph model. Possible edges are defined by the time steps and probability p.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : integer, default 1
        number of edges per node.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    return deque(uniform_random_dynamic_time_graph(
        neighbors, time_index, m=m, groupby=groupby), maxlen=1)[0][1]


def rewire_dynamic_time_graph(choices, time_index, sigma=0.5, groupby=lambda x: x):
    """
    Returns a generator of random graphs. Each graph follows the development of the empirical
    graph with respect to the number of nodes per time step and the number of edges created 
    for each node. Edges, however, are randomly formed between nodes and potential neighbors.

    Parameters
    ----------
    choices : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    sigma : float, default 0.5
        The threshold percentage of how often a data point must be 
        assigned as nearest neighbor.  
    groupby : callable
        Function specifying the time steps at which the graphs should be created                  
    """
    index_series = pd.Series(sorted(choices.keys()), index=time_index)
    G = nx.DiGraph()
    _nodes = node_counter()
    for group_id, story_ids in index_series.groupby(groupby):
        for story_id in story_ids:
            G.add_node(_nodes[story_id])
            n_neighbors = sum(score >= sigma for score in choices[story_id].values())
            neighbors = np.where(time_index[story_id] <= time_index)[0]
            if n_neighbors > 0:
                neighbors = np.random.choice(neighbors, replace=False, size=n_neighbors)
                for neighbor in neighbors:
                    G.add_edge(_nodes[story_id], _nodes[neighbor])
        yield group_id, G


def rewired_time_graph(neighbors, time_index, sigma=0.5, groupby=lambda x: x):
    """
    Returns a rewired graph based on the development of the empirical graph with respect 
    to the number of nodes per time step and the number of edges created for each node. 
    Edges, however, are randomly formed between nodes and potential neighbors.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : numpy.ndarray of Timestamps or pandas.DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    sigma : float, default 0.5
        The threshold percentage of how often a data point must be 
        assigned as nearest neighbor.  
    groupby : callable
        Function specifying the time steps at which the graphs should be created.
    """     
    return deque(rewire_dynamic_time_graph(
        neighbors, time_index, sigma=sigma, groupby=groupby), maxlen=1)[0][1]


def small_world_index(neighbors, time_index, sigma=0.5):
    """
    Compute the small-worldness index as introduced by Humphries et al. which
    is defined as

              C_e / C_r
        swi = ---------
              L_e / L_r

    where C is the clustering coefficient of the empirical and random network 
    and L represents the average path length in both networks. swi >= 1 indicates
    a small world network.

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : ndarray of Timestamps or pandas DatetimeIndex, shape: (n_samples_X), 
        Index corresponding to time points of each sample in X. If supplied,
        neighbors for each item x in X will only consist of samples that occur 
        before or at the time point corresponding with x. Default is None.
    sigma : float, default 0.5
        The threshold percentage of how often a data point must be 
        assigned as nearest neighbor.    
    """
    G_r = nx2igraph(rewired_time_graph(neighbors, time_index, sigma=sigma))
    G_e = nx2igraph(to_graph(neighbors, time_index, sigma=sigma))
    # empirical APL and clustering coefficient
    L_e = G_e.average_path_length(directed=False)
    C_e = G_e.transitivity_undirected()
    # random APL and clustering coefficient
    L_r = G_r.average_path_length(directed=False)
    C_r = G_r.transitivity_undirected()
    return (C_e / (C_r + 1e-10)) / (L_e / (L_r + 1e-10))
