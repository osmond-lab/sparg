import msprime
import tskit
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import chain
import pandas as pd
import matplotlib.pyplot as plt


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

def ts_to_nx_updated(ts):
    topology = defaultdict(list)
    for edge in ts.tables.edges:
        topology[edge.parent].append(edge.child)
    nx_graph = nx.MultiDiGraph(topology)
    return nx_graph

def simplify_graph(G, root=-1):
    ''' Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused 
    Adapted from https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
    '''

    g = G.copy()
    while any(degree==2 for _, degree in g.degree):
        g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
        for node, degree in g.degree():
            if degree==2 and node!=root:
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





ts = tskit.load("run1/slim_0.25rep3sigma.trees")
np.random.seed(1)
keep_nodes = list(np.random.choice(ts.samples(), 20, replace=False))
subset_ts = ts.simplify(samples=keep_nodes, keep_input_roots=True, keep_unary=True)
nx_arg = ts_to_nx_updated(ts=subset_ts)
simple_arg = simplify_graph(G=nx_arg, root=subset_ts.node(subset_ts.num_nodes-1).id)

recomb_nodes = []
for i in simple_arg.nodes:
    is_parent = subset_ts.tables.edges[np.where(subset_ts.tables.edges.parent==i)[0]]
    is_child = subset_ts.tables.edges[np.where(subset_ts.tables.edges.child==i)[0]]
    if (len(set(is_parent.child)) == 1) & (len(set(is_child.parent)) > 1) & ((len(is_parent) + len(is_child)) % 2 != 0):
        recomb_nodes.append(i)

tables = tskit.TableCollection(sequence_length=subset_ts.sequence_length)
tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()

node_lookup = {}

node_table = tables.nodes
simple_recomb_nodes = []
subset_ts_id = []
simple_id = []
counter = 0
for node in sorted(simple_arg.nodes):
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
    subset_ts_id.append(node)
    simple_id.append(counter)
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
        simple_recomb_nodes.append(counter-1)
        counter += 1

print(simple_recomb_nodes)

node_lookup = pd.DataFrame({"subset_ts_id":subset_ts_id, "simple_id":simple_id})


def check_range_overlap(start1, stop1, start2, stop2):
    """Checks if two ranges overlap. Returns corresponding boolean"""
    return (start1 <= start2 < stop1) or (start2 <= start1 < stop2)

def mergeIntervals(intervals):
    """Adapted from https://www.geeksforgeeks.org/merging-intervals/"""
    # Sort the array on the basis of start values of intervals.
    intervals.sort()
    stack = []
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)
    return stack

children = []
parents = []
left = []
right = []
already_found = []
for edge in simple_arg.edges:
    child_edges = subset_ts.tables.edges[np.where(subset_ts.tables.edges.child==edge[0])[0]]
    parent_edges = subset_ts.tables.edges[np.where(subset_ts.tables.edges.parent==edge[1])[0]]
    found = False
    for ce in child_edges:
        for pe in parent_edges:
            if (ce not in already_found) and (ce.left == pe.left) and (ce.right == pe.right):
                already_found.append(ce)
                found = True
                break
        if found:
            break
    children.append(node_lookup["simple_id"].values[(node_lookup["subset_ts_id"].values==edge[0]).argmax()])
    parents.append(node_lookup["simple_id"].values[(node_lookup["subset_ts_id"].values==edge[1]).argmax()])
    left.append(ce.left)
    right.append(ce.right)
    #if edge[0] not in recomb_nodes:
    #    for alt_ce in child_edges:
    #        if (alt_ce.parent == ce.parent) and (alt_ce.child == ce.child) and (alt_ce.left != ce.left) and (alt_ce.right != ce.right):
    #            children.append(node_lookup["simple_id"].values[(node_lookup["subset_ts_id"].values==edge[0]).argmax()])
    #            parents.append(node_lookup["simple_id"].values[(node_lookup["subset_ts_id"].values==edge[1]).argmax()])
    #            left.append(alt_ce.left)
    #            right.append(alt_ce.right)
simple_edges = pd.DataFrame({"parent":parents, "child":children, "left":left, "right":right})

for recomb in simple_recomb_nodes:
    is_parent = simple_edges.loc[simple_edges["parent"]==recomb]
    is_child = simple_edges.loc[simple_edges["child"]==recomb]
    simple_edges.at[is_parent.index.values[0],"left"] = is_child["left"].iloc[0]
    simple_edges.at[is_parent.index.values[0],"right"] = is_child["right"].iloc[0]
    simple_edges.at[is_child.index.values[1],"child"] = recomb + 1
    new_row = pd.DataFrame({
            "parent":[recomb+1],
            "child":[is_parent["child"].iloc[0]],
            "left":[is_child["left"].iloc[1]],
            "right":[is_child["right"].iloc[1]]
        })
    simple_edges = pd.concat([simple_edges, new_row], ignore_index=True)
simple_edges = simple_edges.sort_values("parent").reset_index(drop=True)

edge_table = tables.edges
for index, edge in simple_edges.iterrows():
    edge_table.add_row(
        left=edge["left"],
        right=edge["right"],
        parent=int(edge["parent"]),
        child=int(edge["child"])
    )

tables.sort()
final_ts = tables.tree_sequence()

print(node_lookup["subset_ts_id"].values[(node_lookup["simple_id"].values==54).argmax()])
print(edge_table[np.where(edge_table.parent==54)[0]])
print(edge_table[np.where(edge_table.child==54)[0]])
print(subset_ts.tables.edges[np.where(subset_ts.tables.edges.parent==604)[0]])
print(subset_ts.tables.edges[np.where(subset_ts.tables.edges.child==604)[0]])

all_tree_common_ancestors = []
for tree in final_ts.trees():
    common_ancestor = []
    for node in tree.nodes():
        if tree.num_samples(node) == final_ts.num_samples:
            common_ancestor.append(node)
    if len(common_ancestor) > 0:
        all_tree_common_ancestors.append(min(common_ancestor))
    else:
        #print(tree.draw_text())
        all_tree_common_ancestors.append(final_ts.node(final_ts.num_nodes-1).id)
gmrca = max(all_tree_common_ancestors)

exit()

tables = tskit.TableCollection(sequence_length=subset_ts.sequence_length)
tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
node_table = tables.nodes
for i, node in enumerate(final_ts.tables.nodes):
    if i <= gmrca:
        node_table.add_row(
            flags=node.flags,
            time=node.time,
            metadata=node.metadata
        )
edge_table = tables.edges
for i, edge in enumerate(final_ts.tables.edges):
    if (edge.parent <= gmrca) and (edge.child <= gmrca):
        edge_table.add_row(
            left=edge.left,
            right=edge.right,
            parent=edge.parent,
            child=edge.child
        )
tables.sort()
condensed_ts = tables.tree_sequence()

import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/tskit_arg_visualizer/visualizer")
import visualizer

d3arg = visualizer.D3ARG(ts=condensed_ts)
d3arg.draw(width=1000, height=1000, line_type="line", y_axis_scale="rank")

ts_locs = subset_ts.tables.individuals.location[::3]
individual_list = subset_ts.tables.nodes[np.where(subset_ts.tables.nodes.flags==1)[0]].individual
sample_locs = ts_locs[individual_list]

import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/sparg2.0/ARGweaver/msprime")
import top_down

paths, node_times, node_locs, dispersal_rate = top_down.reconstruct_node_locations(ts=condensed_ts, sample_locs=sample_locs)

print(dispersal_rate)

for p in paths:
    p_times = []
    p_locs = []
    for n in p:
        p_times.append(node_times[n])
        p_locs.append(node_locs[n])
    plt.plot(p_locs, p_times)
plt.show()


"""
import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/tskit_arg_visualizer/visualizer")
import visualizer

d3arg = visualizer.D3ARG(ts=final_ts)
d3arg.draw(width=1000, height=750, line_type="ortho", y_axis_scale="rank")
"""