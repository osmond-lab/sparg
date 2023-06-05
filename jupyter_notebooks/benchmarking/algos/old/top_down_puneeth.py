#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:07:19 2023

@author: puneeth

Top-down Algorithm
"""
import numpy as np
from collections import defaultdict
from itertools import chain
import networkx as nx
import time

import msprime
import random

import paths_modified


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

def calc_covariance_matrix(ts):
    G = ts_to_nx(ts=ts, connect_recombination_nodes=True)
    S = list(ts.samples())
    Current_Nodes = { max(G.nodes) }
    CovMat = np.matrix([[0.0]])
    Indices = defaultdict(list)
    Indices[max(G.nodes)] = [0]
    while set(Current_Nodes) != set(S):
        current_node = max(Current_Nodes)
        path_ind = Indices[current_node] #the indicies of the paths that currently have the end point as the current node 
        npaths = len(path_ind)
        child_nodes = list(G.predecessors(current_node))
        nchild = len(child_nodes)
        #print(Current_Nodes, current_node,child_nodes)
        #print(CovMat, CovMat.shape)
        #print('------------')
        
        if nchild == 1 : 
            child = child_nodes[0]
            edge_len = G.nodes()[current_node]['time'] - G.nodes()[child]['time']
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_len*np.ones((npaths,npaths))
            Indices[child] += path_ind
        else :
            edge_lens = [ G.nodes()[current_node]['time'] - G.nodes()[child]['time'] for child in child_nodes ]
            existing_paths = CovMat.shape[0]
            # print(CovMat,npaths)
            CovMat = np.hstack(  (CovMat,) + tuple( ( CovMat[:,path_ind] for j in range(nchild-1) ) ) ) #Duplicate the rows
            # print(CovMat)
            CovMat = np.vstack(  (CovMat,) + tuple( ( CovMat[path_ind,:] for j in range(nchild-1) ) ) ) #Duplicate the columns
            # print(CovMat)
            
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_lens[0]*np.ones((npaths,npaths))
            Indices[ child_nodes[0] ] += path_ind
            for child_ind in range(1,nchild):
                mod_ind = range(existing_paths+ npaths*(child_ind-1),existing_paths + npaths*child_ind) #indices of the entries that will be modified
                CovMat[ np.ix_( mod_ind , mod_ind  ) ] += edge_lens[child_ind]*np.ones( (npaths,npaths) )
                Indices[ child_nodes[child_ind] ] += mod_ind
                
        Current_Nodes.remove(current_node)
        Current_Nodes.update(child_nodes)

    return CovMat

def benchmark(ts):
    start = time.time()
    cov_mat = calc_covariance_matrix(ts=ts)
    end = time.time()
    return end-start, cov_mat.sum(), "NA"


if __name__ == "__main__":
    rs = random.randint(0,10000)
    print(6857)
    ts = msprime.sim_ancestry(
        samples=2,#30
        recombination_rate=1e-8,
        sequence_length=2_000,#1_000
        population_size=10_000,
        record_full_arg=True,
        random_seed=7960#9080
    )
    #print(ts.draw_text())
    cov_mat = calc_covariance_matrix(ts=ts)
    print(cov_mat.sum())

    true_cov_mat, paths = paths_modified.calc_covariance_matrix(ts=ts)
    print(true_cov_mat.sum())
    np.savetxt("topdown_puneeth_true.csv", cov_mat, delimiter=",")

    print(cov_mat.shape[0], true_cov_mat.shape[0])