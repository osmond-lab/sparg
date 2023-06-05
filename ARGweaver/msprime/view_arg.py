import msprime
import random

# Generate a random tree sequence with record_full_arg=True so that you get marked recombination nodes
ts_rs = 5804 #random.randint(0,10000)   
ts = msprime.sim_ancestry(
    samples=100,
    recombination_rate=1.5e-8,
    sequence_length=10_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=5804
)
ts.dump("run6/5804.trees")






import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/tskit_arg_visualizer/visualizer")
import visualizer

d3arg = visualizer.D3ARG(ts=ts)
d3arg.draw(width=1000, height=750)