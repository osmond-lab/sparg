import sparg
import tskit
import numpy as np
import tskit_arg_visualizer as viz
import matplotlib.pyplot as plt
import math


ts = tskit.load("QE/slim/slim_0.25rep1sigma.trees")
keep_nodes = list(np.random.choice(ts.samples(), 4, replace=False))
ts_cut = ts.decapitate(time=3000)
ts_sim = ts_cut.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True)
ts_final, maps =  sparg.remove_excess_nodes(ts_sim) 
ts_final = sparg.merge_roots(ts_final)
paths = sparg.identify_unique_paths(ts=ts_final)
