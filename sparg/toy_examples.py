#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:44:55 2023

@author: puneeth
"""
import tskit 
import numpy as np 


def ts_bubble( start=0.01, x = 0.25, t=1 ):     
    ts_bubble = tskit.TableCollection(sequence_length=1e3)
    node_table = ts_bubble.nodes
    node_table.set_columns(
        flags = np.array([1, 1, 131072, 131072, 0, 0], dtype=np.uint32),
        time = np.array([ 0.0, 0.0, start*t, start*t, (start+x)*t, t ]),
        individual = np.array(list(range(2)) + [-1,-1,-1,-1], dtype = np.int32)
    )
    
    individual_table = ts_bubble.individuals 
    individual_table.set_columns( 
        flags = np.array([0,0], dtype = np.uint32),
        location = np.array([-0.5,0.5]),
        location_offset = np.array([0,1,2], dtype = np.uint64) 
    )
    
    edge_table = ts_bubble.edges
    edge_table.set_columns(
        left=np.array([0.0, 5e2, 0.0, 0.0, 5e2, 0.0]),
        right=np.array([5e2, 1e3, 1e3, 5e2, 1e3, 1e3]),
        parent=np.array([2, 3, 5, 4, 4, 5], dtype=np.int32),  # References IDs in the node table
        child=np.array([0, 0, 1, 2, 3, 4], dtype=np.int32),  # References IDs in the node table
    )
    ts_bubble.sort()
    return ts_bubble.tree_sequence()


def ts_singlecompound(x=0.25, y=0.125, t=1 ):     
    ts_singlecompound = tskit.TableCollection(sequence_length=1e3)
    node_table = ts_singlecompound.nodes
    node_table.set_columns(
        flags = np.array([1, 1, 131072, 131072, 0, 0], dtype=np.uint32),
        time = np.array([ 0.0, 0.0, (1-x)*t, (1-x)*t, (1-y)*t, t ]),
        individual = np.array(list(range(2)) + [-1,-1,-1,-1], dtype = np.int32)
    )
    
    individual_table = ts_singlecompound.individuals 
    individual_table.set_columns( 
        flags = np.array([0,0], dtype = np.uint32),
        location = np.array([-0.5,0.5]),
        location_offset = np.array([0,1,2], dtype = np.uint64) 
    )
    
    edge_table = ts_singlecompound.edges
    edge_table.set_columns(
        left=np.array([0.0, 5e2, 0.0, 0.0, 5e2, 5e2]),
        right=np.array([5e2, 1e3, 1e3, 5e2, 1e3, 1e3]),
        parent=np.array([2, 3, 4, 4, 5, 5], dtype=np.int32),  # References IDs in the node table
        child=np.array([0, 0, 1, 2, 3, 4], dtype=np.int32),  # References IDs in the node table
    )
    ts_singlecompound.sort()
    return ts_singlecompound.tree_sequence()

def ts_doublecompound(x=0.75, y=0.5, z = 0.4, w = 0.2, t=1): 
    ts_doublecompound = tskit.TableCollection(sequence_length=1e3)
    node_table = ts_doublecompound.nodes
    node_table.set_columns(
        # flags = np.array([1, 1, 131072, 131072,0, 131072, 131072, 0 ,0], dtype=np.uint32),
        flags = np.array([1, 1, 0, 0,0, 0, 0, 0 ,0], dtype=np.uint32),
        time = np.array([ 0.0, 0.0, (1-x)*t, (1-x)*t, (1-y)*t, (1-z)*t, (1-z)*t, (1-w)*t, t ]),
        individual = np.array(list(range(2)) + [-1 for i in range(7)], dtype = np.int32)
    )
    
    individual_table = ts_doublecompound.individuals 
    individual_table.set_columns( 
        flags = np.array([0,0], dtype = np.uint32),
        location = np.array([-0.5,0.5]),
        location_offset = np.array([0,1,2], dtype = np.uint64) 
    )
    
    edge_table = ts_doublecompound.edges
    edge_table.set_columns(
        right=np.array([3e2, 1e3, 1e3, 3e2, 6e2, 1e3, 1e3, 6e2, 1e3, 6e2]),
        left=np.array([0.0, 3e2, 0.0, 0.0, 3e2, 6e2, 3e2, 3e2, 6e2, 3e2]),
        parent=np.array([2, 3, 4, 4, 5, 6, 7, 8, 7, 8], dtype=np.int32),  # References IDs in the node table
        child=np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7], dtype=np.int32),  # References IDs in the node table
    )
    ts_doublecompound.sort()
    return ts_doublecompound.tree_sequence()

def ts_singlecompound_3sam(x=0.25, y=0.125, z=0.25, t=1 ):     
    ts_singlecompound_3sam = tskit.TableCollection(sequence_length=1e3)
    node_table = ts_singlecompound_3sam.nodes
    node_table.set_columns(
        flags = np.array([1, 1, 1, 0, 131072, 131072, 0, 0], dtype=np.uint32),
        time = np.array([ 0.0, 0.0, 0.0, z*t ,(1-x)*t, (1-x)*t, (1-y)*t, t ]),
        individual = np.array(list(range(3)) + [-1,-1,-1,-1,-1], dtype = np.int32)
    )
    
    individual_table = ts_singlecompound_3sam.individuals 
    individual_table.set_columns( 
        flags = np.array([0,0,0], dtype = np.uint32),
        location = np.array([-20,-30,50]),
        location_offset = np.array([0,1,2,3], dtype = np.uint64) 
    )
    
    edge_table = ts_singlecompound_3sam.edges
    edge_table.set_columns(
        left=np.array([0.0, 0.0, 5e2, 0.0, 0.0, 5e2, 0, 5e2]),
        right=np.array([1e3, 1e3, 1e3, 5e2, 1e3, 1e3, 5e2, 1e3]),
        parent=np.array([3, 3, 4, 5, 6, 7, 6, 7], dtype=np.int32),  # References IDs in the node table
        child=np.array([0, 1, 3, 3, 2, 4, 5, 6], dtype=np.int32),  # References IDs in the node table
    )
    ts_singlecompound_3sam.sort()
    return ts_singlecompound_3sam.tree_sequence()

def ts_stacked(x=0.25, n=1, seq_len = 1000 ):     
    ts_stacked = tskit.TableCollection(sequence_length=1e3)
    n_nodes = 3 + n*3
    # n_steps = 3 + 2*n 
    
    node_table = ts_stacked.nodes
    flag_list = np.zeros((n_nodes,))
    flag_list[0] = 1
    flag_list[1] = 1
    time_list = np.zeros((n_nodes,))
    time_list[-1] = (3 +2*n )*x
    
    for i in range(n):
        flag_list[2 +3*i] = 131072
        flag_list[2 +3*i + 1] = 131072
        time_list[2 +3*i] = (3 + 2*i)*x
        time_list[2 +3*i + 1] = (3 + 2*i)*x
        time_list[2 +3*i + 2] = (3 + 2*i + 1)*x
    
    node_table.set_columns(
        flags = np.array(flag_list, dtype=np.uint32),
        time = np.array(time_list),
        individual = np.array(list(range(2)) + [-1 for i in range(n_nodes-2)], dtype = np.int32)
    )
    
    individual_table = ts_stacked.individuals 
    individual_table.set_columns( 
        flags = np.array([0,0], dtype = np.uint32),
        location = np.array([-0.5,0.5]),
        location_offset = np.array([0,1,2], dtype = np.uint64) 
    )
    
    edge_table = ts_stacked.edges
    
    seqlen = seq_len/float(n+1)
    child = [0,0,1]
    parent = [2,3,4]
    left = [ seqlen, 0, 0  ]
    right = [ (n+1)*seqlen, seqlen, (n+1)*seqlen ] 
    for i in range(n-1): 
        child += [ 2 + 3*i, 2+3*i, 2+3*i+1, 2+3*i+2 ]
        parent += [ 2 + 3*(i+1), 2+3*(i+1)+1, 2+3*i+2, 2+3*(i+1)+2 ]
        left += [(i+2)*seqlen , (i+1)*seqlen , i*seqlen , (i+1)*seqlen ]
        right += [(n+1)*seqlen , (i+2)*seqlen , (i+1)*seqlen  , (n+1)*seqlen ]
    child += [2+3*(n-1), 3*n, 3*n+1  ]
    parent += [2+3*n, 3*n+1, 3*n+2 ]
    left += [n*seqlen, (n-1)*seqlen, n*seqlen ]
    right += [(n+1)*seqlen, n*seqlen, (n+1)*seqlen]
    
    # print(child)
    # print(parent)
    # print(left)
    # print(right)
    
    edge_table.set_columns(
        left=np.array(left),
        right=np.array(right),
        parent=np.array(parent, dtype=np.int32),  # References IDs in the node table
        child=np.array(child, dtype=np.int32),  # References IDs in the node table
    )
    ts_stacked.sort()
    return ts_stacked.tree_sequence()

