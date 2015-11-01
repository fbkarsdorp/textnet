import networkx as nx
import numpy as np

from .statistics import evolving_graph_statistics
from .utils import nx2igraph


def rewire_dynamic_time_graph(G, time_index, rewire_prop=1.0):
    """
    This function rewires the edges in a time-indexed graph given a
    rewire probability. This is the most naive randomization presented
    here, because the time-dependent anchoring of nodes is lost.

    Parameters
    ----------
    G : graph : igraph.Graph(directed=True) or nx.DiGraph
        The input graph, can be constructed using textnet.to_graph or one of the
        other graph creating functions.
    time_index : ndarray of Timestamps or pandas DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    rewire_prop : float, default 1.0
        rewiring probability
    """
    G = nx2igraph(G) if isinstance(G, (nx.Graph, nx.DiGraph)) else G.copy()
    G.rewire_edges(rewire_prop)
    for time_stamp in time_index.unique().order():
        yield time_stamp, G.subgraph(G.vs.select(date_le=time_stamp))


def randomized_dynamic_time_graph(neighbors, time_index, m=1, groupby=lambda x: x, sigma=0.5):
    """
    Returns a generator of random graphs at each time step t in time_index 
    according to the Barabási–Albert preferential attachment model. 

    Parameters
    ----------
    neighbors : output of textnet.bootstrap_neighbors or textnet.bootstrap_neighbors_sparse_batch
    time_index : ndarray of Timestamps or pandas DatetimeIndex, shape: (n_nodes) 
        Index corresponding to time points of each sample in G. If supplied,
        neighbors for each node n in G will only consist of samples that occur 
        before or at the time point corresponding with x.
    m : int, default 1
        Number of edges to attach from a new node to existing nodes
    groupby : callable
        Function specifying the time steps at which the graphs should be created
    sigma : float, default 0.5
        The threshold percentage of how often a data point must be 
        assigned as nearest neighbor.        
    """
    stats = evolving_graph_statistics(neighbors, time_index, groupby=groupby, sigma=sigma).sort_index()
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



