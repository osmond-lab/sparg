#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:07:19 2023

@author: puneeth

Top-down Algorithm
"""
import numpy as np
import math
import random
from collections import defaultdict
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt
import graphviz

G = nx.DiGraph()

S = [1,2] #Sample nodes 
n_nodes = 13 
for nd in range(1,n_nodes+1) :
    if nd in S : 
        G.add_node(nd, time = 0)
    else : 
        G.add_node(nd, time = nd)

G.add_edges_from([(1,3),(2,6),(3,4),(3,6),(4,5),(4,7),(5,8),(5,10),(6,13),(7,8),(7,11),(8,9),(9,10),(9,11),(10,12),(11,12),(12,13)])

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
    
    print(Current_Nodes, current_node,child_nodes)
    print(CovMat, CovMat.shape)
    print('------------')
    
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
    
    
    
