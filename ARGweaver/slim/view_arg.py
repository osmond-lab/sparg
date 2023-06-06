import msprime
import tskit
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import chain
import pandas as pd


def ts_to_nx(ts, connect_recombination_nodes=False, recomb_nodes=[]):
    """
    Converts tskit tree sequence to networkx graph.
    """
    topology = defaultdict(list)
    for tree in ts.trees():
        for k, v in chain(tree.parent_dict.items()):
            if connect_recombination_nodes:
                if recomb_nodes == []:
                    recomb_nodes = list(np.where(ts.tables.nodes.flags == 131072)[0])
                if v in recomb_nodes and recomb_nodes.index(v)%2 == 1:
                    v -= 1
                if k in recomb_nodes and recomb_nodes.index(k)%2 == 1:
                    k -= 1
                if v not in topology[k]:
                    topology[k].append(v)
            else:
                if v not in topology[k]:
                    topology[k].append(v)
    nx_graph = nx.MultiDiGraph(topology)
    return nx_graph

def simplify_graph(G):
    ''' Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused 
    Adapted from https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
    '''

    g = G.copy()
    while any(degree==2 for _, degree in g.degree):
        g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
        for node, degree in g.degree():
            if degree==2:
                if g.is_directed(): #<-for directed graphs
                    a0,b0 = list(g0.in_edges(node))[0]
                    a1,b1 = list(g0.out_edges(node))[0]
                else:
                    edges = g0.edges(node)
                    edges = list(edges.__iter__())
                    a0,b0 = edges[0]
                    a1,b1 = edges[1]
                e0 = a0 if a0!=node else b0
                e1 = a1 if a1!=node else b1
                g0.remove_node(node)
                g0.add_edge(e0, e1)
        g = g0
    return g



ts = tskit.load("run1/slim_0.25rep0sigma.trees")
keep_nodes = list(np.random.choice(ts.samples(), 10, replace=False))
subset_ts = ts.simplify(samples=keep_nodes, keep_input_roots=True, keep_unary=True)
nx_arg = ts_to_nx(ts=subset_ts)
simple_arg = simplify_graph(G=nx_arg)
first_unique_bounds = pd.DataFrame({"left":subset_ts.tables.edges.left, "right":subset_ts.tables.edges.right}).drop_duplicates()
breakpoint_edges = subset_ts.tables.edges[first_unique_bounds.index.tolist()[1:]]
recomb_nodes = set(breakpoint_edges.child)

tables = tskit.TableCollection(sequence_length=subset_ts.sequence_length)
tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()

node_lookup = {}

node_table = tables.nodes
counter = 0
for node in simple_arg.nodes:
    orig_row = subset_ts.tables.nodes[node]
    flag = orig_row.flags
    if node in recomb_nodes:
        flag = msprime.NODE_IS_RE_EVENT
    metadata = orig_row.metadata
    metadata["subset_ts_id"] = node
    node_table.add_row(
        flags=flag,
        time=orig_row.time,
        #population=orig_row.population,
        #individual=orig_row.individual,
        metadata=metadata
    )
    node_lookup[node] = counter
    counter += 1
    if node in recomb_nodes:
        flag = flag
        node_table.add_row(
            flags=flag,
            time=orig_row.time,
            #population=orig_row.population,
            #individual=orig_row.individual,
            metadata=metadata
        )
        counter += 1

edge_table = tables.edges
previously_found_children = []
for edge in simple_arg.edges:
    left = 0
    right = subset_ts.sequence_length
    child = node_lookup[edge[0]]
    if edge[0] in previously_found_children:
        child += 1
    found_edges = edge_table[np.where(edge_table.parent == child)[0]]
    if len(found_edges) > 0:
        left = min(found_edges.left)
        right = max(found_edges.right)
    #relevant_subset_ts_edges = subset_ts.tables.edges[np.where(subset_ts.tables.edges.child == edge[0])[0]]
    if edge[1] in recomb_nodes:
        relevant_edges = breakpoint_edges[np.where(breakpoint_edges.child == edge[1])[0]]
        edge_table.add_row(left=relevant_edges[0].left, right=relevant_edges[0].right, parent=node_lookup[edge[1]], child=child)
        edge_table.add_row(left=relevant_edges[1].left, right=relevant_edges[1].right, parent=node_lookup[edge[1]]+1, child=child)
    else:
        edge_table.add_row(left=left, right=right, parent=node_lookup[edge[1]], child=child)
    previously_found_children.append(edge[0])

        
tables.sort()
final_ts = tables.tree_sequence()

print(final_ts.tables.nodes)
print(final_ts.tables.edges)
print(final_ts.tables.nodes.time)

print(final_ts.first().draw_text())
        


"""
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/tskit_arg_visualizer/visualizer")
import visualizer

d3arg = visualizer.D3ARG(ts=final_ts)
d3arg.draw(width=1000, height=750, line_type="ortho", y_axis_scale="rank")
"""