#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:03:45 2023

@author: puneeth
"""

import msprime
import random
import tskit_arg_visualizer

# Generate a random tree sequence with record_full_arg=True so that you get marked recombination nodes
ts_rs = random.randint(0,10000)   
ts = msprime.sim_ancestry(
    samples=3,
    recombination_rate=1e-8,
    sequence_length=3_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=ts_rs
)
ts_sim = ts.simplify(samples = [0,1,2], keep_unary=True, keep_input_roots=True)
print(ts_sim.draw_text())

# d3arg = tskit_arg_visualizer.D3ARG(ts=ts)
# d3arg.draw(
#     width=750,
#     height=750,
#     y_axis_labels=True,
#     y_axis_scale="rank",
#     tree_highlighting=True,
#     edge_type="ortho"
# )