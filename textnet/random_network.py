import networkx as nx
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

