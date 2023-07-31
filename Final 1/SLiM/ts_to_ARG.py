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
import numpy as np 
import random 
def merge_roots(ts): 
    ts_tables = ts.dump_tables() 
    edge_table = ts_tables.edges 
    parent = edge_table.parent 
    
    roots = np.where(ts_tables.nodes.time == ts.max_time)[0]
    root_children = []
    for root in roots:
        root_children += list(ts.tables.edges.child[np.where(ts.tables.edges.parent == root)[0]])
    for root_child in root_children: 
        pts = np.unique(ts.tables.edges.parent[np.where(ts.tables.edges.child == root_child)[0]])
        if len(pts) > 2 : 
            for i,pt in enumerate(pts): 
                parent[np.where(ts.tables.edges.parent == pt)[0]] = pts[0] 
    edge_table.parent = parent 
    ts_tables.sort() 
    ts_new = ts_tables.tree_sequence() 
    return ts_new 
                
def ts_to_ARG(ts):
    ts_tables = ts.dump_tables()
    node_table = ts_tables.nodes
    flags = node_table.flags
    
    recomb_nodes = []
    coal_nodes = []
    
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has. 
    multiple_parents = nd[count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    flags[recomb_nodes] = msprime.NODE_IS_RE_EVENT
    
    nd, count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has. 
    coal_nodes = nd[count > 1] #Find parent who have more than 1 children. 
    
    node_table.flags = flags
    ts_tables.sort() 
    ts_new = ts_tables.tree_sequence()
    
    keep_nodes = list(np.unique( list(ts_new.samples()) + list(np.unique(recomb_nodes)) + list(np.unique(coal_nodes)) )) 
    ts_final, maps = ts_new.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
    
    return ts_final, maps
    

if __name__ == "__main__":    
    
    random.seed(5)
    ts = msprime.sim_ancestry(samples=100, recombination_rate=1e-6, sequence_length=1000, population_size=10_000, record_full_arg=True)
    keep_nodes = list(np.random.choice(ts.samples(), 30, replace=False))
    # ts_sim1, maps1 = ts.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=True)
    ts = ts.decapitate(time=15000)
    ts_sim, maps = ts.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=True)
    re_nodes = np.where(ts_sim.nodes_flags & msprime.NODE_IS_RE_EVENT)[0]
    
    ts_final, maps_final = ts_to_ARG(ts_sim)
    ts_final = merge_roots(ts_final)
    
    
    uniq_child_parent_final = np.unique(np.column_stack((ts_final.edges_child, ts_final.edges_parent)), axis=0)
    nd_chd_final, cnt_chd_final = np.unique(uniq_child_parent_final[:,1],return_counts=True)
    multiple_chd_final = nd_chd_final[cnt_chd_final > 2]
    nd_pt_final, cnt_pt_final = np.unique(uniq_child_parent_final[:,0],return_counts=True)
    multiple_pt_final = nd_pt_final[cnt_pt_final > 2]
    
    uniq_child_parent_sim = np.unique(np.column_stack((ts_sim.edges_child, ts_sim.edges_parent)), axis=0)
    nd_chd_sim, cnt_chd_sim = np.unique(uniq_child_parent_sim[:,1],return_counts=True)
    multiple_chd_sim = nd_chd_sim[cnt_chd_sim > 2]
    nd_pt_sim, cnt_pt_sim = np.unique(uniq_child_parent_sim[:,0],return_counts=True)
    multiple_pt_sim = nd_pt_sim[cnt_pt_sim > 2]
    
    # uniq_child_parent_sim1 = np.unique(np.column_stack((ts_sim1.edges_child, ts_sim1.edges_parent)), axis=0)
    # nd_chd_sim1, cnt_chd_sim1 = np.unique(uniq_child_parent_sim1[:,1],return_counts=True)
    # multiple_chd_sim1 = nd_chd_sim1[cnt_chd_sim1 > 2]
    # nd_pt_sim1, cnt_pt_sim1 = np.unique(uniq_child_parent_sim1[:,0],return_counts=True)
    # multiple_pt_sim1 = nd_pt_sim1[cnt_pt_sim1 > 2]
    
    # print(multiple_pt_sim1, multiple_chd_sim1)
    print(multiple_pt_sim, multiple_chd_sim)
    print(multiple_pt_final, multiple_chd_final)
    
    
    
    
            
    