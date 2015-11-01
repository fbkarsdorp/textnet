# -*- coding: utf-8 -*-
import random

from collections import deque

import networkx as nx
import numpy as np
import pandas as pd

from .statistics import evolving_graph_statistics
from .utils import node_counter


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
    stats = evolving_graph_statistics(neighbors, time_index, groupby=groupby).sort_index()
    G = nx.DiGraph()
    # create an empty graph with the nodes at the first time step 
    G.add_nodes_from(range(stats.n[0]))
    repeated_nodes = np.zeros(stats.n.max(), dtype=np.float64)
    repeated_nodes[range(stats.n[0])] = 1
    all_nodes = np.arange(stats.n.max())
    for i in range(1, stats.shape[0]):
        p_vals = repeated_nodes / repeated_nodes.sum()
        for j in range(len(G), stats.n[i]):
            # sample m target nodes without replacement for j
            targets = np.random.choice(all_nodes, size=m, replace=False, p=p_vals)
            G.add_edges_from(zip([j] * m, targets))
            repeated_nodes[targets] += 1
            repeated_nodes[j] += m
        yield stats.index[i], G


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
    return deque(randomized_dynamic_time_graph(neighbors, time_index, m=m, groupby=groupby))[0][1]


def gnp_random_dynamic_time_graph(neighbors, time_index, p, groupby=lambda x: x): 
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
    p : float
        probability of creating an edge.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    index_series = pd.Series(sorted(neighbors.keys()), index=time_index)
    G = nx.DiGraph()
    potential_neighbors = time_index <= time_index[np.newaxis].T
    _nodes = node_counter()
    for group_id, story_ids in index_series.groupby(groupby):
        for story_id in story_ids:
            G.add_node(_nodes[story_id], name=story_id, date=time_index[story_id])
            neighbors = np.where(potential_neighbors[story_id])[0]
            for neighbor in neighbors:
                if random.random() < p:
                    G.add_node(_nodes[neighbor], name=neighbor, date=time_index[neighbor])
                    G.add_edge(_nodes[story_id], _nodes[neighbor])
        yield group_id, G

def gnp_random_time_graph(neighbors, time_index, p, groupby=lambda x: x):
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
    p : float
        probability of creating an edge.
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    """
    return deque(gnp_random_dynamic_time_graph(neighbors, time_index, p, groupby=groupby), maxlen=1)[0][1]
