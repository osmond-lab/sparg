#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:25:29 2023

@author: puneeth
"""

import msprime
import tskit 
import Toy_Examples
import numpy as np 
import pyslim

if __name__ == "__main__":    
    # ts = tskit.load('/home/puneeth/UofT/SpARG Project/sparg2.0/SLiM/slim_0.25rep1sigma.trees')
    for sigma in np.arange(0.25,1.25,0.25):
        print(sigma)
        for rep in range(1,11):
            fname = 'slim_'+str(round(sigma,2))+'rep'+str(rep)+'sigma.trees'
            
            ts = tskit.load('../../SLiM/'+fname)
            ts = pyslim.generate_nucleotides(ts) #generate random nucleotides for slim mutations, https://github.com/tskit-dev/pyslim/pull/174
            ts = pyslim.convert_alleles(ts) #convert slim alleles (0,1) to nucleotides
            ts_mut = msprime.sim_mutations(ts, rate=float(1e-7), model=msprime.JC69(), keep=True) 
            
            keep_nodes = list(np.random.choice(ts.samples(), 50, replace=False))
            
            sts = ts_mut.simplify(samples=keep_nodes, keep_input_roots=True, keep_unary=False)
            sts.write_fasta(fname+".fa")
    
    
    # ts = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True)
    # print(len(list(ts.edges())), len(list(ts.samples())), len(list( ts.nodes() )))
    # print(len(list(ts.edges())), len(list(ts.samples())), len(list( ts.nodes() )))
    
    # ts_tables = ts.dump_tables()
    
    # node_table = ts_tables.nodes
    # flags = node_table.flags
    
    # recomb_nodes = []
    
    # for nd in ts.nodes(): 
    #     parents = np.unique(ts_tables.edges.parent[np.where(ts.tables.edges.child == nd.id)[0]])
        
    #     # print(nd, parent_ids)
    #     if len(parents) == 2: 
    #         # parents = ts_tables.edges.parent[parent_ids[0]]
    #         # print(nd.id, parents)
    #         recomb_nodes += list(parents)
    #         flags[parents] = [131072,131072]
    #         # print(flags)
    
    # node_table.flags = flags
    # ts_tables.sort() 
    # ts = ts_tables.tree_sequence()
        
    # keep_nodes = list(ts.samples()) + recomb_nodes #For Error Debug keep_nodes = array([130,  59, 185, 155, 164,  82,  92, 200, 202, 243], dtype=int32)
    # ts = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
    # ts = ts.decapitate(time=500)
    
    # print('Done')
    # print(len(list(ts.edges())), len(list(ts.samples())), len(list( ts.nodes() )))
    # mu, sigma = Toy_Examples.ARG_estimate(ts)
    # print(sigma)
    
    