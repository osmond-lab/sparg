import msprime
import tskit
import numpy as np
import math
import random
from collections import defaultdict
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt
import time


def ts_to_nx(ts, connect_recombination_nodes=False, recomb_nodes=[]):
    """
    Converts tskit tree sequence to networkx graph.

    Need to add a check to ensure that the list of recombination nodes is valid
    (there should always be an even number of recombination nodes if following
    tskit setup)
    """
    if recomb_nodes == []:
        recomb_nodes = list(np.where(ts.tables.nodes.flags == 131072)[0])
    recomb_nodes_to_remove = recomb_nodes[1::2]
    topology = defaultdict(list)
    for tree in ts.trees():
        for k, v in chain(tree.parent_dict.items()):
            if connect_recombination_nodes:
                if v in recomb_nodes_to_remove:
                    v -= 1
                if k in recomb_nodes_to_remove:
                    k -= 1
                if v not in topology[k]:
                    topology[k].append(v)
            else:
                if v not in topology[k]:
                    topology[k].append(v)
    nx_graph = nx.DiGraph(topology)
    node_times = {v: k for v, k in enumerate(ts.tables.nodes.time)}
    nx.set_node_attributes(nx_graph, node_times, "time")
    return nx_graph

def locate_loop_group_nodes(ARG):
    """
    First calculates the cycle basis of the graph (this utilizes the ARG with connected
    recombination nodes), then groups those loops based on whether they are interconnected.
    Returns a list of lists of nodes that are interconnected by loops in the ARG.

    THE OUTPUT HAS BEEN CHANGED FROM PREVIOUS ITERATIONS. Loops themselves are no longer
    preserved, only loop groups.
    """
    
    loop_list = nx.cycle_basis(ARG.nx_graph_connected_recomb_nodes.to_undirected())
    edges = ARG.ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    if len(loop_list) != len(ARG.recomb_nodes)/2:
        for node in ARG.recomb_nodes[::2]:
            parent = parent_list[child_list.index(node)]
            if parent == parent_list[child_list.index(node+1)]:
                loop_list.append([node, parent])
    num_loops = len(loop_list)
    loop_group_nodes = []
    if num_loops > 1:
        build_instructions = []
        for loop in loop_list:
            for n in range(len(loop)):
                if n == len(loop)-1:
                    a, b = loop[n], loop[0]
                else:
                    a, b = loop[n], loop[n+1]
                build_instructions.append([a, b])
        g = nx.Graph(build_instructions)
        loop_group_nodes = list(nx.connected_components(g))
    elif num_loops == 1:
        loop_group_nodes = [set(loop_list[0])]
    return loop_group_nodes

def generate_skeleton(ARG):
    """
    This matches Puneeth's skeleton graph but stored as a dictionary.
    """

    loop_group_nodes = locate_loop_group_nodes(ARG)
    gmrca = ARG.ts.node(ARG.ts.num_nodes-1).id
    all_loop_nodes = set([node for lg in loop_group_nodes for node in lg])
    skeleton = defaultdict(list)
    previously_found = []
    working_nodes = list(ARG.ts.samples())
    for node in working_nodes:
        no_shallower_connection = True
        for lg in loop_group_nodes:
            input_node = max(lg)
            if node < input_node:
                if nx.has_path(ARG.nx_graph_connected_recomb_nodes, source=node, target=input_node):
                    path = nx.shortest_path(ARG.nx_graph_connected_recomb_nodes, source=node, target=input_node)
                    youngest_connection_node = min(list(set(path[1:]) & all_loop_nodes))
                    #if youngest_connection_node not in lg:
                    #    continue
                    skeleton[youngest_connection_node].append(node)
                    if youngest_connection_node not in previously_found:
                        skeleton[input_node].append(youngest_connection_node)
                        previously_found.append(youngest_connection_node)
                        #if youngest_connection_node in ARG.recomb_nodes:
                        #    skeleton[input_node].append(youngest_connection_node+1)
                    if input_node != gmrca and input_node not in working_nodes:
                        working_nodes.append(input_node)
                    no_shallower_connection = False
                    break
        if no_shallower_connection:
            skeleton[gmrca].append(node)
    for group in skeleton:
        skeleton[group] = sorted(skeleton[group])
    return skeleton

def combine_cov_submatrices(skeleton, sub_cov_mats, row_column_ids, group_node, samples):
    if group_node in samples:
        return np.array([0])
    else:
        num_output_nodes = len(skeleton[group_node])
        output_mat = [list(x) for x in np.zeros((num_output_nodes, num_output_nodes))]
        for i, output_node_1 in enumerate(skeleton[group_node]):
            for j, output_node_2 in enumerate(skeleton[group_node]):
                print(group_node, output_node_1, output_node_2)
                current_cov = combine_cov_submatrices(skeleton=skeleton, sub_cov_mats=sub_cov_mats, row_column_ids=row_column_ids, group_node=output_node_2, samples=samples)
                new_cov = sub_cov_mats[group_node][row_column_ids[output_node_1]["start"]:row_column_ids[output_node_1]["stop"], row_column_ids[output_node_2]["start"]:row_column_ids[output_node_2]["stop"]]
                if i == j:
                    output_mat[i][j] = np.kron( current_cov, np.ones(new_cov.shape)) + np.kron(np.ones(current_cov.shape), new_cov)
                else:
                    output_mat[i][j] = np.kron(np.ones(current_cov.shape), new_cov)
                print(output_mat)
        print(group_node)
        print(output_mat)
        return np.bmat(output_mat)

def non_recursive_combine_cov_submatrices(skeleton, sub_cov_mats, row_column_ids, samples):
    aggregated_cov_mats = {}
    for group in sorted(skeleton.keys()):
        if all(child not in sub_cov_mats for child in skeleton[group]):
            aggregated_cov_mats[group] = sub_cov_mats[group]
        else:
            num_output_nodes = len(skeleton[group])
            output_mat = [list(x) for x in np.zeros((num_output_nodes, num_output_nodes))]
            for i, output_node_1 in enumerate(skeleton[group]):
                if output_node_1 in samples:
                    output_node_1_cov = np.array([[0]])
                else:
                    output_node_1_cov = aggregated_cov_mats[output_node_1]
                for j, output_node_2 in enumerate(skeleton[group]):
                    if output_node_2 in samples:
                        output_node_2_cov = np.array([[0]])
                    else:
                        output_node_2_cov = aggregated_cov_mats[output_node_2]
                    new_cov = sub_cov_mats[group][row_column_ids[output_node_1]["start"]:row_column_ids[output_node_1]["stop"], row_column_ids[output_node_2]["start"]:row_column_ids[output_node_2]["stop"]]
                    if output_node_1 == output_node_2:
                        output_mat[i][j] = np.kron(output_node_1_cov, np.ones(new_cov.shape)) + np.kron(np.ones(output_node_1_cov.shape), new_cov)
                    else:
                        output_mat[i][j] = np.kron(np.ones((output_node_1_cov.shape[0],output_node_2_cov.shape[1])), new_cov)
            aggregated_cov_mats[group] = np.bmat(output_mat)
    return aggregated_cov_mats[max(aggregated_cov_mats.keys())]

def calc_cov_matrix(ARG):
    """
    THIS METHOD DOES NOT MATCH UP PUNEETH'S, I misremembered it so will have to redo.
    
    Strips the ARG to its skeleton graph, where a group is a set of nodes which share the same parent
    in the skeleton graph. Calculates all of the paths between each node and the parent of the group.
    Calculates the covariance matrix between these paths.

    This function should then merge the covariance matrices of each group to build the full covariance
    matrix of the whole ARG, which is when I realized I was going about this process in a different
    way to Puneeth.
    """

    skeleton = generate_skeleton(ARG)
    sub_cov_mats = {}
    number_of_paths_per_node = {}
    for group in skeleton:
        all_paths = []
        path_counter = 0
        for output_node in skeleton[group]:
            node_paths = list(nx.all_simple_paths(ARG.nx_graph, source=output_node, target=group))
            if output_node in ARG.recomb_nodes:
                node_paths.extend(nx.all_simple_paths(ARG.nx_graph, source=output_node+1, target=group))
            number_of_paths_per_node[output_node] = {"start":path_counter,"stop":path_counter+len(node_paths)}
            path_counter += len(node_paths)
            all_paths.extend(node_paths)
        edges = ARG.ts.tables.edges
        parent_list = list(edges.parent)
        child_list = list(edges.child)
        times = np.empty((len(all_paths),len(all_paths)))
        for i, p in enumerate(all_paths):
            for j in range(i+1):
                intersect = list(set(p).intersection(all_paths[j]))
                edges = []
                for child in intersect:
                    if child != group:
                        edges.append(ARG.ts.node(parent_list[child_list.index(child)]).time - ARG.ts.node(child).time)
                times[i,j] = np.sum(edges)
                times[j,i] = times[i,j]
        sub_cov_mats[group] = times
    return non_recursive_combine_cov_submatrices(skeleton=skeleton, sub_cov_mats=sub_cov_mats, row_column_ids=number_of_paths_per_node, samples=ARG.ts.samples())


class ARGObject:
    """
    A ARGObject contains various methods for reference an ARG:
        
        ts - tskit tree sequence
        nx_graph - directed networkx graph built from tree sequence instructions
        nx_graph_connected_recomb_nodes - directed networkx graph where
            recombination nodes have been merged

    There are also an attributes that is often used with this class:
        recomb_nodes - recombination nodes (very commonly referenced)
    """

    def __init__(self, ts):
        self.ts = ts
        self.recomb_nodes = list(np.where(self.ts.tables.nodes.flags == 131072)[0])
        self.nx_graph = ts_to_nx(ts=ts, recomb_nodes=self.recomb_nodes)
        self.nx_graph_connected_recomb_nodes = ts_to_nx(ts=ts, connect_recombination_nodes=True, recomb_nodes=self.recomb_nodes)


rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=200,
    recombination_rate=1e-8,
    sequence_length=2000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=rs #7483,8131
)

#print(ts.draw_text())
#print(ts.num_trees)

start = time.time()
arg = ARGObject(ts=ts)
cov_mat = calc_cov_matrix(ARG=arg)
end = time.time()
print("HYBRID - Total Execution Time:", round((end - start)/60, 2), "minutes")
print(cov_mat.shape)