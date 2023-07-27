#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:44:08 2023

@author: puneeth
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:25:29 2023

@author: puneeth
"""

import msprime
import tskit 
import top_down
import numpy as np 
import pyslim
import matplotlib.pyplot as plt
import random 

def ts_to_ARG(ts):
    ts_tables = ts.dump_tables()
    
    node_table = ts_tables.nodes
    flags = node_table.flags
    
    recomb_nodes = []
    coal_nodes = []
    
    # for nd in ts_sim.nodes(): 
    #     parents = np.unique(ts_tables.edges.parent[np.where(ts_tables.edges.child == nd.id)[0]])
    #     children = np.unique(ts_tables.edges.child[np.where(ts_tables.edges.parent == nd.id)[0]])
    #     if len(parents) == 2: 
    #         recomb_nodes += list(parents)
    #         flags[parents] = [131072,131072]
            
    #     if len(children) == 2 : 
    #         coal_nodes += [nd.id]
    #     if len(parents) > 2:
    #         raise TypeError('Error',nd.id)
    #     if len(children) > 2:
    #         print('Multiple Children', nd.id)
    
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has. 
    multiple_parents = nd[count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    flags[recomb_nodes] = msprime.NODE_IS_RE_EVENT
    
    node_table.flags = flags
    ts_tables.sort() 
    ts_new = ts_tables.tree_sequence()
    
    keep_nodes = list(ts_new.samples()) + list(np.unique(list(np.unique(recomb_nodes))  + list(np.unique(coal_nodes))))
    ts_final, maps = ts_new.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)

    return ts_final, maps, recomb_nodes

# def ts_to_ARG(ts):
#     ts_tables = ts.dump_tables()
#     node_table = ts_tables.nodes
#     flags = node_table.flags
#     recomb_nodes = []
    
#     for nd in ts_sim.nodes(): 
#         parents = np.unique(ts_tables.edges.parent[np.where(ts_tables.edges.child == nd.id)[0]])
#         children = np.unique(ts_tables.edges.child[np.where(ts_tables.edges.parent == nd.id)[0]])
#         if len(parents) == 2: 
#             recomb_nodes += list(parents)
#             flags[parents] = [131072,131072]
            
#         if len(parents) > 2:
#             raise TypeError('Error',nd.id)
        
#     node_table.flags = flags
#     ts_tables.sort() 
#     ts_new = ts_tables.tree_sequence()
    
#     keep_nodes = list(ts_new.samples()) + list(np.unique(recomb_nodes))
#     ts_final, maps = ts_new.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)

#     return ts_final, maps, recomb_nodes


if __name__ == "__main__":    
    # # ts = tskit.load('/home/puneeth/UofT/SpARG Project/sparg2.0/SLiM/slim_0.25rep1sigma.trees')
    # fname = 'slim_'+str(0.25)+'rep'+str(1)+'sigma.trees'
    # ts = tskit.load('../../SLiM/'+fname)
    # keep_nodes = list(np.random.choice(ts.samples(), 30, replace=False))
    
    
    random.seed(5)
    ts = msprime.sim_ancestry(samples=100, recombination_rate=1e-6, sequence_length=500, population_size=10_000, record_full_arg=True)
    keep_nodes = list(np.random.choice(ts.samples(), 10, replace=False))
    
    # ts = ts.decapitate(time=10000)
    ts_sim, maps = ts.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=True)
    re_nodes = np.where(ts_sim.nodes_flags & msprime.NODE_IS_RE_EVENT)[0]
    
    
    ts_final, maps_final, recomb_nodes= ts_to_ARG(ts_sim)
    
    
    
            
    