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
                    if youngest_connection_node not in lg:
                        continue
                    skeleton[youngest_connection_node].append(node)
                    # if youngest_connection_node in ARG.recomb_nodes:
                    #     skeleton[youngest_connection_node+1].append(node)
                        
                    if youngest_connection_node not in previously_found:
                        skeleton[input_node].append(youngest_connection_node)
                        previously_found.append(youngest_connection_node)
                        # if youngest_connection_node in ARG.recomb_nodes:
                        #     skeleton[input_node].append(youngest_connection_node+1)
                    
                    if input_node != gmrca and input_node not in working_nodes:
                        working_nodes.append(input_node)
                    no_shallower_connection = False
                    break
        if no_shallower_connection:
            skeleton[gmrca].append(node)
    return skeleton

def calc_matrix(skeleton, sub_cov_matrix, group_node, start_nodes):  
    if group_node in start_nodes:
        return np.array([0])
    else:
        C_mats = { pi: calc_matrix(skeleton, sub_cov_matrix, pi, start_nodes) for pi in skeleton[group_node] }
        num = len(skeleton[group_node])
        output_mat = [list(x) for x in np.empty( (num,num) )]
        for i, pi in enumerate( skeleton[group_node] ): 
            for j in range(i+1): 
                pj = skeleton[group_node][j]
                ci = C_mats[pi].shape[0]
                cj = C_mats[pj].shape[0]
                
                if i == j  :
                    Ci = C_mats[pi]
                    Dii = sub_cov_matrix[group_node][ (pi,pi) ]
                    d = Dii.shape
                    output_mat[i][j] = np.kron( Ci, np.matrix(np.ones(d)) ) + np.kron( np.matrix(np.ones((ci,cj))), Dii)
                else: 
                    Dij = sub_cov_matrix[group_node][ (pi,pj) ]
                    d = Dij.shape
                    output_mat[i][j] = np.kron( np.matrix(np.ones((ci,cj))), Dij)
                    output_mat[j][i] = np.transpose( output_mat[i][j] )
        return np.bmat( output_mat )
 

def calc_subcov_matrix(ARG):
    skeleton = generate_skeleton(ARG)
    sub_cov_mats = {}
    for group in skeleton:
        paths = defaultdict(list)
        for output_node in skeleton[group]:
            paths[output_node].extend(nx.all_simple_paths(ARG.nx_graph, source=output_node, target=group))
            if output_node in ARG.recomb_nodes :
                paths[output_node].extend(nx.all_simple_paths(ARG.nx_graph, source=output_node+1, target=group))
                
        edges = ARG.ts.tables.edges
        parent_list = list(edges.parent)
        child_list = list(edges.child)
        times = defaultdict(list)
        
        for pi in skeleton[group]:
            for pj in skeleton[group]: 
                times_pipj = np.empty( (len(paths[pi]), len(paths[pj])) )
                
                for i, pth_i in enumerate(paths[pi]):
                    for j, pth_j in enumerate(paths[pj]):
                        
                        intersect = list(set(pth_i).intersection(pth_j ))
                        edges = []
                        for child in intersect:
                            if child != group:
                                edges.append(ARG.ts.node(parent_list[child_list.index(child)]).time - ARG.ts.node(child).time)
                        times_pipj[i,j] = np.sum(edges) # Do I need np.unique()? Ask Matt, because it was previously in his
                        # times_pipj[j,i] = times_pipj[i,j]
                times[ (pi,pj) ] = np.matrix( times_pipj )
                times[ (pj,pi) ] = np.matrix( np.transpose(times_pipj) )
        sub_cov_mats[group] = times
    return sub_cov_mats
        
def calc_cov_matrix(ARG):       
    gmrca = ARG.ts.node(ARG.ts.num_nodes-1).id
    sub_cov_mats = calc_subcov_matrix(ARG)
    skeleton = generate_skeleton(ARG)
    return calc_matrix(skeleton, sub_cov_mats, gmrca, ARG.ts.samples() )


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
        self.loop_list = locate_loop_group_nodes(self)


def benchmark(ts):
    start = time.time()
    arg = ARGObject(ts=ts)
    cov_mat = calc_cov_matrix(ARG=arg)
    end = time.time()
    return end-start, cov_mat.sum(), len(arg.loop_list)


if __name__ == "__main__":
    rs = random.randint(0,10000)
    print(rs)
    ts = msprime.sim_ancestry(
        samples=2,
        recombination_rate=1e-8,
        sequence_length=2_000,
        population_size=10_000,
        record_full_arg=True,
        random_seed=rs
    )
    print(ts.draw_text())
    arg = ARGObject(ts=ts)
    cov_mat = calc_cov_matrix(ARG=arg)
