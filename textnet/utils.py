from collections import defaultdict
import networkx as nx
import igraph

def igraph2nx(iG):
    nx_G = nx.DiGraph()
    nx_G.add_nodes_from(node.index for node in iG.vs)
    nx_G.add_edges_from(iG.get_edgelist())
    return nx_G

def nx2igraph(nx_G):
    iG = igraph.Graph(directed=True)
    iG.add_vertices(nx_G.nodes())
    iG.add_edges(nx_G.edges())
    return iG

def node_counter():
    counter = defaultdict()
    counter.default_factory = lambda: len(counter)
    return counter