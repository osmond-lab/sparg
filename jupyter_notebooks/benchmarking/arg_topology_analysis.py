import random
import msprime
import time
import numpy as np
import algos.paths
import algos.hybrid_nonrecursive


topology_file = open("topology.csv", "w")
topology_file.write("num_samples,sequence_length,seed,num_trees,nodes,num_paths,num_loop_groups,max_loop_size\n")

for i in range(5):
    for samples in range(50,550,50):
        for seq_len in range(1000, 11000, 1000):
            print(i, samples, seq_len)
            rs = random.randint(0,10000)
            topology_file.write(str(samples) + "," + str(seq_len) + "," + str(rs) + ",")

            ts = msprime.sim_ancestry(
                samples=samples,
                recombination_rate=1e-8,
                sequence_length=seq_len,
                population_size=10_000,
                record_full_arg=True,
                random_seed=rs
            )

            topology_file.write(str(ts.num_trees) + "," + str(len(ts.nodes())) + "," + str(len(algos.paths.identify_unique_paths(ts=ts))) + ",")

            loop_list = algos.hybrid_nonrecursive.ARGObject(ts=ts).loop_list
            num_nodes_in_loop = []
            for loop in loop_list:
                num_nodes_in_loop.append(len(loop))
            if len(num_nodes_in_loop) == 0:
                max_size = 0
            else:
                max_size = max(num_nodes_in_loop)
            topology_file.write(str(len(loop_list)) + "," + str(max_size) + "\n")

topology_file.close()