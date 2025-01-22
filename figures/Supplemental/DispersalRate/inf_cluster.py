#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:51:36 2024

@author: puneeth
"""
import sparg_with_inf as spinf
import sparg
import tskit
import numpy as np

ts = tskit.load("slim_0.25rep1sigma.trees")
sd = 0
print(sd)
np.random.seed(sd)

outfile = open("slim_0.25rep0_Supl2_DispersalRate.txt", "w")

keep_nodes = list(np.random.choice(ts.samples(), 10, replace=False))
ts_sim, sim_map = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True, map_nodes=True)
ts_filtered, filtered_maps = sparg.simplify_with_recombination(ts=ts_sim)
ts_chopped = sparg.chop_arg(ts_filtered,10000)
i = 0
ts_breaks = ts_chopped.breakpoints(as_array=True)
print(len(ts_breaks))
sigma_trees_x = [] 
sigma_trees_y = [] 

for (bp_i,bp) in enumerate(ts_breaks):
    if (bp_i > 99 and bp_i<110) or bp_i == len(ts_breaks)-1:
        print(sd, bp_i,flush=True)
        ts_short = ts_chopped.keep_intervals(intervals=[(0,bp)], simplify=False).trim()
        ts_short_sim, maps_short_sim = sparg.simplify_with_recombination(ts=ts_short)
        ts_short_attached = sparg.chop_arg(ts=ts_short, time = ts_short.max_time )

        
        PartialARG_inf = spinf.SpatialARG(ts=ts_short_attached,dimensions=2, verbose=False)
        inf_dispersal_rate = PartialARG_inf.inf_dispersal_rate_matrix
        
        PartialARG = sparg.SpatialARG(ts=ts_short_attached,dimensions=2, verbose=False)
        dispersal_rate = PartialARG.dispersal_rate_matrix

        ts_tree = ts_short_attached.keep_intervals(intervals=[(ts_breaks[bp_i-1],bp)], simplify=False).trim() 
        ts_tree = sparg.chop_arg(ts=ts_tree,time= ts_tree.max_time)
        Tree = sparg.SpatialARG(ts=ts_tree, dimensions=2, verbose=False)
        dispersal_rate_tree = Tree.dispersal_rate_matrix
        sigma_trees_x += [ dispersal_rate_tree[0][0] ]
        sigma_trees_y += [ dispersal_rate_tree[1][1] ]
        sigma_avg_x = np.average(sigma_trees_x)
        sigma_avg_y = np.average(sigma_trees_y)
        
        if bp_i > 21 and bp_i < len(ts_breaks)-21: 
            W20_ts = ts_chopped.keep_intervals(intervals=[(ts_breaks[bp_i-21],ts_breaks[bp_i+20])], simplify=False).trim() 
            W20_ts = sparg.chop_arg(ts=W20_ts,time= W20_ts.max_time)
            W20_ARG = sparg.SpatialARG(ts=W20_ts, dimensions=2,verbose=False)
            W20_disp = W20_ARG.dispersal_rate_matrix
            outfile.write(str(ts_short_attached.num_trees) + " " + str(bp) + " " + str(dispersal_rate[0][0]) + " " + str(dispersal_rate[1][1]) + " " + str(sigma_avg_x) + " " + str(sigma_avg_y) + " " + str(inf_dispersal_rate[0][0]) + " " + str(inf_dispersal_rate[1][1]) + str(W20_disp[0][0]) + " " +str(W20_disp[1][1]) + "\n")
        else: 
            outfile.write(str(ts_short_attached.num_trees) + " " + str(bp) + " " + str(dispersal_rate[0][0]) + " " + str(dispersal_rate[1][1]) + " " + str(sigma_avg_x) + " " + str(sigma_avg_y) + " " + str(inf_dispersal_rate[0][0]) + " " + str(inf_dispersal_rate[1][1]) + "\n")

outfile.close()
