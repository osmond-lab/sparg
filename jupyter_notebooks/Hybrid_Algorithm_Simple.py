#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 23:46:30 2023

@author: puneeth

Hybrid Algorithm - Simple Version 
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 00:09:15 2023

@author: puneeth

Hybrid Method 
"""

import numpy as np
import math
import random
from collections import defaultdict
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt
import graphviz

       

def locate_loops(g, cycle_root=-1):
    """
    Finds loops within the ARG. I thought that it would be easiest to utilize functions from
    networkx package. Identifies recombination events, converts the tree sequence into a networkx
    graph. The paired recombination nodes are merged together in this graph. Converts graph to 
    undirected, then calculates cycle basis. This does not identify 'bubbles', so we need to add
    an extra step to this.
    """
    
    if cycle_root < 0:
        cycle_root = list(g.nodes())[0]
    recomb_nodes = [ x for x in g.nodes() if len(list(g.successors(x))) == 2 ]
    # print(recomb_nodes)
    g_un = g.to_undirected()
    loop_list = nx.cycle_basis(g_un, root=cycle_root)
    if len(loop_list) != len(recomb_nodes):
        for node in recomb_nodes:
            parent = list(g.successors(node))[0]
            loop_list.append([node, parent])  
    return loop_list

def group_loops(loops, plot=False):
    """
    Groups intersecting loops in list. Builds networkx graph based on the loop list. Determines
    if the nodes are connected through the graph. Returns a list of lists of loops.
    """
    
    num_loops = len(loops)
    if num_loops == 0:
        return []
    else:
        if num_loops > 1:
            build_instructions = []
            for loop in loops:
                for n in range(len(loop)):
                    if n == len(loop)-1:
                        a, b = loop[n], loop[0]
                    else:
                        a, b = loop[n], loop[n+1]
                    build_instructions.append([a, b])
            g = nx.Graph(build_instructions)
            grouped_nodes = list(nx.connected_components(g))
            if plot:
                nx.draw(g, with_labels=True)
            grouped_loops = [[] for i in range(len(grouped_nodes))]
            for loop in loops:
                for i in range(len(grouped_loops)):
                    if loop[0] in grouped_nodes[i]:
                        grouped_loops[i].append(loop)
            return grouped_loops
        else:
            return [loops]

    
def Cov(edge_path1,edge_path2,G):
    """ 
    edge_path1 : A path in the graph G as a list of edges 
    edge_path2 : A path in the graph G as a list of edges 
    G : A graph G in which the paths exist and the nodes have an attribute time
    returns the Covariance between the two paths = the shared time between the two paths in G.
    """
    # edges_path1 = set(path_to_edges(path1))
    # edges_path2 = set(path_to_edges(path2))
    # print('check',edge_path2)
    common_edges = set(edge_path1).intersection(set(edge_path2))
    cov = 0 
    for edge in common_edges: 
        t_edge = G.nodes()[edge[1]]['time'] - G.nodes()[edge[0]]['time']
        cov += t_edge 
    return cov 


G = nx.DiGraph() #This will record the real ARG
G_skeleton = nx.DiGraph() #This will record connection nodes after we break down the ARG into simple components (trees and loops)

#Example 1
Sample_Nodes = [1,2,3,4,5,6,7,8] 
n_nodes = 23 # The number of nodes
edgelist = [(1,9),(2,9),(3,11),(4,12),(5,10),(6,10),(7,13),(8,20),(9,11),(10,12),(11,14),(12,13),(13,15),(14,16),(14,18),(15,16),(15,17),(16,17),(17,18),(18,19),(19,21),(19,23),(20,21),(20,22),(21,22),(22,23)]

#Example 2
# Sample_Nodes = [1,2,3,4]
# n_nodes = 12
# edgelist = [(1,5),(2,5),(3,6),(4,6),(5,7),(6,8),(7,9),(7,11),(8,9),(8,10),(9,10),(10,11)]

"""Building the ARG as a Directed Graph"""
for nd in range(1,n_nodes+1) :
    if nd in Sample_Nodes : 
        G.add_node(nd, time = 0)
        G_skeleton.add_node(nd)
    else : 
        G.add_node(nd, time = nd)
G.add_edges_from(edgelist)

"""Locating the loops using James' Method""" 
loops = locate_loops(g=G) #Identify each loop as a list of nodes 
grouped_loops = group_loops(loops=loops) #Group the loops if they shared edges

""" Finding the GMRCA """
vGMRCA = max(list(G.nodes()), key = lambda nd:G.nodes()[nd]['time'] ) #This is the GMRCA of the entire ARG identified as the oldest node in the ARG 

""" Storing some information about the loops we will repeatedly use """
nodes_in_all_groups = [] #This will store all the nodes present in any loop in the Graph 
Grp_details = {} #This will store information about each group of nodes including the index of the group, the nodes in the group and the oldest node in the group
ind = 0 #This will keep track of the number of groups of loops present 
for Lp_grp in grouped_loops: 
    nodes_in_group = [] #To keep track of nodes in a given group of loops 
    for lp in Lp_grp: 
        nodes_in_group += lp
        nodes_in_all_groups += lp
    Grp_details[ind] = {'ind':ind, 'nodes': nodes_in_group, 'maximum': max(nodes_in_group, key = lambda nd: G.nodes()[nd]['time']) }
    ind += 1 
nodes_in_all_groups = set(nodes_in_all_groups)

""" Break the graph in "irreducible" components of trees and loops """
S = Sample_Nodes 
while len(S) > 0 :
    sample_connects = {} #For each s in S, this will record the earliest ancestor of s that is part of any loop
    for s in S: 
        if s == vGMRCA : 
            continue 
        
        group_connections = [vGMRCA] #This list will containt all the nodes in any loop that s is connected to. vGMRCA is included here for algorithmic purposes. 
        for g in nodes_in_all_groups : 
            if nx.has_path(G,s,g) and s != g:  
                group_connections += [g]
        
        earliest_connection = min(group_connections, key = lambda nd: G.nodes()[nd]['time']  )#Choose the earliest in group_connection
        sample_connects[s] = earliest_connection
        G_skeleton.add_node(earliest_connection, time= G.nodes()[earliest_connection]['time'] )
        G_skeleton.add_edge(s,earliest_connection, typ ='tree', paths = list(nx.all_simple_edge_paths(G,s,earliest_connection)) ) #To each edge in the skeleton, we all attribute all the paths between the two nodes in the real Graph. The attribute typ keeps track of whether this edge is part of a loop or a tree. typ attribute was mainly included for verification purposes and can eventually be removed. 
    
    loop_st_nds = list(sample_connects.values()) #This is the list of nodes in the loops which are connected to S 
    loop_st_nds_grpwise = { ind: list( set(Grp_details[ind]['nodes']).intersection(set(loop_st_nds))) for ind in Grp_details if len(list( set(Grp_details[ind]['nodes']).intersection(set(loop_st_nds)))) != 0 } #We divide and label these nodes according the group of loops they belong to.
    
    S_new = [] #The next set of S will be the the MRCA of each loop involved in this step
    for ind in loop_st_nds_grpwise : 
        loop_MRCA = Grp_details[ind]['maximum']
        S_new += [loop_MRCA]
        G_skeleton.add_node( loop_MRCA )
        for v in loop_st_nds_grpwise[ind]: 
            G_skeleton.add_edge(v,loop_MRCA, typ='loop', paths = list(nx.all_simple_edge_paths(G,v,loop_MRCA)))
    S = S_new
    # print('S',S)

""" Compute the Matrix """
Gske_nodes = sorted(G_skeleton.nodes, key = lambda nd:G.nodes()[nd]['time']) #Ordering the nodes in the skeleton according to their times

#To each node in G_skeleton we will associcate two matrices. CovMatrix of a nodes will be the covariance matrix of all the paths from its immediate predecessor to itself where the CovFullMatrix will be the covariance matrix from the samples to itself. 

for v in Gske_nodes: 
    if v in Sample_Nodes:
        G_skeleton.add_node(v, CovMatrix = np.matrix([0]), CovMatrixFull = np.matrix([0]))
    else:    
        v_pred  = list(sorted(G_skeleton.predecessors(v), key = lambda nd:G.nodes()[nd]['time'])) #The predecessors nodes of v
        n_pred = len(v_pred)
        CovMat = {} #This will record the CovMatrix between all the paths from the predecessors to v. This is stored as a dictionary of dictonary. The [v1][v2] entry is the covariance between paths from v1 to v and v2 to v. 
        CovMatFull = [] #This will record the CovMatrixFull which the covariance matrix from all paths from the samples to v
        for v1 in v_pred:
            Covrow = {}
            CovFullrow = []
            
            v1_paths = G_skeleton.edges()[(v1,v)]['paths']
            n1 = len(v1_paths)
            v1FullMat = G_skeleton.nodes()[v1]['CovMatrixFull'] #The covariance matrix between all paths from the samples to v1
            m1 = len(v1FullMat)
            
            for v2 in v_pred : 
                print('v2',v2)
                v2_paths = G_skeleton.edges()[(v2,v)]['paths']
                n2 = len(v2_paths)
                v2FullMat = G_skeleton.nodes()[v2]['CovMatrixFull'] #The covariance matrix between all paths from the samples to v2
                m2 = len(v2FullMat)
                
                Matv1v2 = np.matrix( [ [ Cov(path1,path2,G) for path2 in v2_paths] for path1 in v1_paths ] )
                MatFullv1v2 = np.kron(np.matrix(np.ones( (m1,m2) )), Matv1v2 )
                print('Matv1v2',Matv1v2)
                print('MatFullv1v2',MatFullv1v2)
                if v1 == v2 :
                    MatFullv1v2 = MatFullv1v2 + np.kron( v1FullMat, np.matrix(np.ones( (n1,n1) )) ) 
                Covrow[v2] = Matv1v2
                CovFullrow += [ MatFullv1v2 ]
            CovMat[v1] = Covrow
            CovMatFull += [CovFullrow ]
        
        G_skeleton.add_node(v, CovMatrix = CovMat, CovMatrixFull = np.bmat(CovMatFull))
CMFull = G_skeleton.nodes()[vGMRCA]['CovMatrixFull'] 
    
""" Verification """ 

All_Paths = [] 
for s in Sample_Nodes:
    All_Paths +=  list(nx.all_simple_edge_paths(G, s, vGMRCA)) 

CM = [] 
for pathi in All_Paths :
    CMrow = []
    for pathj in All_Paths: 
        cij = Cov(pathi,pathj,G)
        CMrow += [ cij ]
    CM += [CMrow]
        
CM = np.matrix(CM)