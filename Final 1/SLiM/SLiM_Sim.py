#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:25:29 2023

@author: puneeth
"""

import msprime
import tskit 
import ts_to_ARG
import SpARG 
import numpy as np 
import pyslim
import matplotlib.pyplot as plt

if __name__ == "__main__":    
    
    ts = tskit.load('/home/puneeth/UofT/SpARG Project/SLiM/trial.trees')
    data = []
    
    fig_dispersal, ax_dispersal = plt.subplots() 
    fig_intloc, ax_intloc = plt.subplots() 
    
    intloc_average_absdiff = [] 
    intloc_average_cov = [] 
    sigma_rng = np.arange(0.25,0.76,0.25)
    for sigma in sigma_rng:
        sigmas_true = []
        sigmas_ARG = []
        intloc_absdiff = []
        intloc_cov = []
        for rep in range(1,11):
            print(sigma, rep)            
            fname = 'slim_'+str(round(sigma,2))+'rep'+str(rep)+'sigma.trees'
            ts = tskit.load(fname)
            keep_nodes = list(np.random.choice(ts.samples(), 4, replace=False))
            
            # ts = pyslim.generate_nucleotides(ts) #generate random nucleotides for slim mutations, https://github.com/tskit-dev/pyslim/pull/174
            # ts = pyslim.convert_alleles(ts) #convert slim alleles (0,1) to nucleotides
            # ts_mut = msprime.sim_mutations(ts, rate=float(1e-7), model=msprime.JC69(), keep=True) 
            # sts = ts_mut.simplify(samples=keep_nodes, keep_input_roots=True, keep_unary=False)
            # sts.write_fasta(fname+".fa")
    
            ts = ts.decapitate(time=3000)
            ts_sim = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True)
            
            
            ts_final, maps =  ts_to_ARG.ts_to_ARG(ts_sim) 
            ts_final = ts_to_ARG.merge_roots(ts_final)
            
            print('Done')
            # print(len(list(ts.edges())), len(list(ts.samples())), len(list( ts.nodes() )))
            internal_nodes = np.unique(ts_final.tables.edges.parent)
            mu, sigma_ARG, internal_locations = SpARG.ARG_estimate(ts_final, internal_nodes = list(internal_nodes) )
            print(sigma_ARG)
            sigmas_ARG += [sigma_ARG]
            sigmas_true += [sigma]
            data += [ [ sigma,sigma_ARG ] ]
            
            """Comparing Internal Locations """
            true_locs = [ts.tables.individuals[ ts.tables.nodes[nd].indvidual ].location[0] for nd in internal_nodes  ]
            
            #Average absolute difference between true and estimated locations 
            abs_diff = 0 
            for i in range(len(internal_nodes)): 
                abs_diff += np.abs( internal_locations[i] - true_locs[i] ) 
            abs_diff = abs_diff/float(len(internal_nodes))
            intloc_absdiff += [abs_diff]
            
            #Covariance between true and inferred values 
            true_mean = np.mean(true_locs)
            estimated_mean = np.mean(internal_locations)
            cov = sum((x - true_mean)*(y - estimated_mean) for x, y in zip(true_locs, internal_locations)) / len(true_locs)
            intloc_cov += [cov]
            
        ax_dispersal.scatter(sigmas_true, sigmas_ARG)
        ax_intloc.scatter(sigmas_true, intloc_absdiff, color = 'blue')
        ax_intloc.scatter(sigmas_true, intloc_cov, color = 'red')
            
        intloc_average_absdiff += [np.mean(intloc_absdiff) ]
        intloc_average_cov += [np.mean(intloc_cov) ]
        
    ax_dispersal.plot([0,1.25],[0,1.25],color='black')
    
    ax_intloc.plot(sigma_rng, intloc_average_absdiff, color = 'blue', label = 'Internal Locationn - Abs Diff')
    ax_intloc.plot(sigma_rng, intloc_average_cov, color = 'red', label = 'Internal Locationn - Covariance')
    
    