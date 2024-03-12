#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:51:36 2024

@author: puneeth
"""
import sparg_with_inf as sparg
import sys
import tskit
import msprime
import numpy as np
import warnings
import matplotlib.pyplot as plt
from importlib import reload

ts = tskit.load("slim_0.25rep1sigma.trees")

for sd in range(1):
    print(sd)
    np.random.seed(sd)
    keep_nodes = list(np.random.choice(ts.samples(), 10, replace=False))
    
    ts_sim, sim_map = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True, map_nodes=True)
    ts_filtered, filtered_maps = sparg.simplify_with_recombination(ts=ts_sim)
    ts_chopped = sparg.chop_arg(ts_filtered,10000)
    #FullARG = sparg.SpatialARG(ts=ts_chopped, dimensions=2, verbose=False)
    print("Step")
    
    outfile = open("slim_0.25rep1sigma_inf_dispersalrateestimates_seqlen_100_chopped_10000_rep_" + str(sd) +".txt", "w")
    i = 0
    ts_breaks = ts_chopped.breakpoints(as_array=True)
    print(len(ts_breaks))
    FI1_trees = []
    FI2_trees = []
    sigma_trees = [] 
    for (bp_i,bp) in enumerate(ts_breaks):
        if bp_i > 0:
             if bp_i%100 == 0 or bp_i == 1 :
                print(sd, bp_i)
                if bp_i in range(0,1551,100):
                    print(sd, bp_i)
                ts_short = ts_chopped.keep_intervals(intervals=[(0,bp)], simplify=False).trim()
                ts_short_sim, maps_short_sim = sparg.simplify_with_recombination(ts=ts_short)
                ts_short_attached = sparg.chop_arg(ts=ts_short, time = ts_short.max_time )
                
                ts_tree = ts_short_attached.keep_intervals(intervals=[(ts_breaks[bp_i-1],bp)], simplify=False).trim() 
                ts_tree = sparg.chop_arg(ts=ts_tree,time= ts_tree.max_time)
                
                PartialARG = sparg.SpatialARG(ts=ts_short_attached,dimensions=2, verbose=False)
                inf_dispersal_rate = PartialARG.inf_dispersal_rate_matrix
                FI1 = PartialARG.inf_fishers_information_1
                FI2 = PartialARG.inf_fishers_information_2 
                
                Tree = sparg.SpatialARG(ts=ts_tree, dimensions=2, verbose=False)
                
                inf_dispersal_rate_tree = Tree.inf_dispersal_rate_matrix
                FI1_tree = Tree.inf_fishers_information_1 
                FI2_tree = Tree.inf_fishers_information_2
                               
                FI1_trees += [ inf_dispersal_rate_tree[0][0]**2*FI1 ]
                FI2_trees += [ inf_dispersal_rate_tree[0][0]**3*FI2 ]
                sigma_trees += [ inf_dispersal_rate_tree[0][0] ]

                sigma_avg = np.average(sigma_trees)
                FI1_avg = np.sum(FI1_trees)/(sigma_avg**2)
                FI2_avg = np.sum(FI2_trees)/(sigma_avg**3)
                
                outfile.write(str(ts_short_attached.num_trees) + " " + str(bp) + " " + str(inf_dispersal_rate[0][0]) + " " + str(inf_dispersal_rate[1][1]) + " " + str(FI1 + FI2)+ " " + str(sigma_avg) + " " + str(FI1_avg + FI2_avg)  + "\n")
                
                
    outfile.close()