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

if __name__ == "__main__":    
    # ts = tskit.load('/home/puneeth/UofT/SpARG Project/sparg2.0/SLiM/slim_0.25rep1sigma.trees')
    data = []
    for sigma in np.arange(0.25,1.25,0.25):
        sigmas_true = []
        sigmas_ARG = []
        for rep in range(1,11):
            print(sigma, rep)
            
            sigmas_true += [sigma]
            
            fname = 'slim_'+str(round(sigma,2))+'rep'+str(rep)+'sigma.trees'
            
            ts = tskit.load('../../SLiM/'+fname)
            keep_nodes = list(np.random.choice(ts.samples(), 30, replace=False))
            
            # ts = pyslim.generate_nucleotides(ts) #generate random nucleotides for slim mutations, https://github.com/tskit-dev/pyslim/pull/174
            # ts = pyslim.convert_alleles(ts) #convert slim alleles (0,1) to nucleotides
            # ts_mut = msprime.sim_mutations(ts, rate=float(1e-7), model=msprime.JC69(), keep=True) 
            # sts = ts_mut.simplify(samples=keep_nodes, keep_input_roots=True, keep_unary=False)
            # sts.write_fasta(fname+".fa")
    
            ts = ts.decapitate(time=500)
            # ts_sim = ts
            ts_sim = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True)
            # print(len(list(ts.edges())), n(list(ts.samples())), len(list( ts.nodes() )))
            # print(len(list(ts.edges())), len(list(ts.samples())), len(list( ts.nodes() )))
            
            ts_tables = ts_sim.dump_tables()
            
            node_table = ts_tables.nodes
            flags = node_table.flags
            
            recomb_nodes = []
            coal_nodes = []
            
            for nd in ts_sim.nodes(): 
                parents = np.unique(ts_tables.edges.parent[np.where(ts_tables.edges.child == nd.id)[0]])
                children = np.unique(ts_tables.edges.child[np.where(ts_tables.edges.parent == nd.id)[0]])
                # print(nd, parent_ids)
                if len(parents) == 2: 
                    # parents = ts_tables.edges.parent[parent_ids[0]]
                    # print(nd.id, parents)
                    recomb_nodes += list(parents)
                    flags[parents] = [131072,131072]
                    # print(flags)
                if len(children) == 2 : 
                    coal_nodes += [nd.id]
                if len(parents) > 2:
                    raise TypeError('Error',nd.id)
                if len(children) > 2:
                    print('Multiple Children', nd.id)
                # if len(set(recomb_nodes).intersection(set(coal_nodes))) > 0 :
                #     raise TypeError('Recombination and Coalescent Node', set(recomb_nodes).intersection(set(coal_nodes)))
            node_table.flags = flags
            ts_tables.sort() 
            ts_new = ts_tables.tree_sequence()
            
            keep_nodes = list(ts_new.samples()) + list(np.unique(list(np.unique(recomb_nodes))  + list(np.unique(coal_nodes))))
            ts_final, maps = ts_new.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
            
            
            # print('Done')
            # print(len(list(ts.edges())), len(list(ts.samples())), len(list( ts.nodes() )))
            mu, sigma_ARG = top_down.ARG_estimate(ts_final)
            print(sigma_ARG)
            sigmas_ARG += [sigma_ARG[0][0]]
            plt.scatter(sigmas_true, sigmas_ARG)
            
            data += [ [ sigma,sigma_ARG ] ]
    plt.plot([0,1.25],[0,1.25],color='black')
            
    