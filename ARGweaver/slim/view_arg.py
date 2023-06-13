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

ts = tskit.load("run1/slim_0.25rep3sigma.trees")
np.random.seed(1)
keep_nodes = list(np.random.choice(ts.samples(), 50, replace=False))
subset_ts = ts.simplify(samples=keep_nodes, keep_input_roots=True, keep_unary=True)
nx_arg = ts_to_nx_updated(ts=subset_ts)

critical_nodes = []
for node in subset_ts.nodes():
    is_parent = subset_ts.tables.edges[np.where(subset_ts.tables.edges.parent==node.id)[0]]
    is_child = subset_ts.tables.edges[np.where(subset_ts.tables.edges.child==node.id)[0]]
    if (len(np.unique(is_parent.child)) != len(np.unique(is_child.parent))):
        if (len(is_parent.child) != len(is_child.parent)) or (len(np.unique(is_parent.child)) > len(np.unique(is_child.parent))):
            critical_nodes.append(node.id)

def simplify_graph(G, keep):
    ''' Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused 
    Adapted from https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
    '''
    g = G.copy()
    while any((node not in keep) for node in g.nodes):
        g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
        for node in g.nodes:
            if node not in keep:
                in_list = list(g0.in_edges(node))
                out_list = list(g0.out_edges(node))
                previously_found = []
                for in_edge in in_list:
                    a0,b0 = in_edge
                    for out_edge in out_list:
                        if (out_edge not in previously_found) and (out_edge[0] == b0):
                            a1,b1 = out_edge
                            previously_found.append(out_edge)
                            break
                    e0 = a0 if a0!=node else b0
                    e1 = a1 if a1!=node else b1
                    g0.add_edge(e0, e1)
                g0.remove_node(node)
        g = g0
    return g

simple_arg = simplify_graph(G=nx_arg, keep=critical_nodes)

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
    child = min(edge[0], edge[1])
    parent = max(edge[0], edge[1])
    child_edges = subset_ts.tables.edges[np.where(subset_ts.tables.edges.child==child)[0]]
    parent_edges = subset_ts.tables.edges[np.where(subset_ts.tables.edges.parent==parent)[0]]
    found = False
    for ce in child_edges:
        for pe in parent_edges:
            if (ce not in already_found) and (ce.left == pe.left) and (ce.right == pe.right):
                already_found.append(ce)
                found = True
                break
        if found:
            break
    children.append(node_lookup["simple_id"].values[(node_lookup["subset_ts_id"].values==child).argmax()])
    parents.append(node_lookup["simple_id"].values[(node_lookup["subset_ts_id"].values==parent).argmax()])
    left.append(ce.left)
    right.append(ce.right)

simple_edges = pd.DataFrame({"parent":parents, "child":children, "left":left, "right":right})


for recomb in simple_recomb_nodes:
    is_parent = simple_edges.loc[simple_edges["parent"]==recomb]
    is_child = simple_edges.loc[simple_edges["child"]==recomb]
    if len(is_child) > 2:
        associated_edges = defaultdict(list)
        for pi, parent_edge in is_parent.iterrows():
            for ci, child_edge in is_child.iterrows():
                if (child_edge["left"] >= parent_edge["left"]) and (child_edge["right"] <= parent_edge["right"]):
                    # child is within the region of the parent
                    associated_edges[pi].append(ci)
        for i, pi in enumerate(associated_edges):
            for j, ci in enumerate(associated_edges[pi]):
                #print(pi, ci)
                if j > 0:
                    new_row = pd.DataFrame({
                        "parent":[recomb+j],
                        "child":[is_parent["child"].loc[pi]],
                        "left":[is_child["left"].loc[ci]],
                        "right":[is_child["right"].loc[ci]]
                    })
                    simple_edges = pd.concat([simple_edges, new_row], ignore_index=True)
                else:
                    simple_edges.at[pi, "parent"] = recomb+j
                    simple_edges.at[pi, "left"] = simple_edges.at[ci, "left"]
                    simple_edges.at[pi, "right"] = simple_edges.at[ci, "right"]
                simple_edges.at[ci, "child"] = recomb+j
    else:
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

print(final_ts.tables.edges[np.where(final_ts.tables.edges.parent==89)[0]])
print(final_ts.tables.edges[np.where(final_ts.tables.edges.child==89)[0]])

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
d3arg.draw(width=2000, height=2000, line_type="line", y_axis_scale="rank")