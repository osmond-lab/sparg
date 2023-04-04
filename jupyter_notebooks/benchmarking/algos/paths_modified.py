import random
import msprime
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import chain, combinations
import time


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

def identify_unique_paths(ts):
    """
    Finds all of the paths within the incomplete ARG, stored as a tskit tree sequence
    
    Input:
    - ts: tskit tree sequence
    
    Output:
    - all_paths: list, unique paths within the ARG
    """
    G = ts_to_nx(ts=ts)
    # originally had grmca but there are instances of multiple roots,
    # so this should handle that.
    roots = [i.id for i in list(ts.nodes()) if i.id not in list(ts.tables.edges.child)]
    all_paths = []
    for sample in ts.samples():
        for root in roots:
            paths = nx.all_simple_paths(G, source=sample, target=root)
            all_paths.extend(paths)
    return all_paths

def calc_covariance_matrix(ts):
    edges = ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    gmrca = ts.node(ts.num_nodes-1).id
    paths = identify_unique_paths(ts=ts)
    cov_mat = np.zeros((len(paths), len(paths)))
    for node in ts.nodes():
        if node.id != gmrca:
            path_indices = []
            for i, path in enumerate(paths):
                if node.id in path:
                    path_indices.append(i)
            if node.id in child_list:
                shared_time = ts.node(parent_list[child_list.index(node.id)]).time - node.time
                for a in path_indices:
                    for b in path_indices:
                        cov_mat[a, b] += shared_time
    return cov_mat, paths

def benchmark(ts):
    start = time.time()
    cov_mat, paths = calc_covariance_matrix(ts=ts)
    end = time.time()
    np.savetxt("paths_modified.csv", cov_mat, delimiter=",")
    return end-start, cov_mat.sum(), "NA"


if __name__ == "__main__":
    rs = random.randint(0,10000)
    print(rs)
    ts = msprime.sim_ancestry(
        samples=2,#30
        recombination_rate=1e-8,
        sequence_length=2_000,#1_000
        population_size=10_000,
        record_full_arg=True,
        random_seed=9203#9080
    )
    print(ts.draw_text())
    
    cov_mat, paths = calc_covariance_matrix(ts=ts)
    #np.savetxt("paths_modified.csv", cov_mat, delimiter=",")