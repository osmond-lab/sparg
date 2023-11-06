import random
import msprime
import time
import numpy as np
import algos.paths
import algos.bottom_up
import algos.minimal
from itertools import permutations



benchmark_file = open("benchmarking_final.csv", "w")
benchmark_file.write("num_samples,sequence_length,seed,num_trees,nodes,paths,algo_order,paths_time,bottom_up_time,minimal_time,minimal_paths\n")

perms = list(permutations([0, 1, 2]))

for samples in range(10,110,10):
    print(samples)
    for seq_len in range(1000, 6000, 1000):
        for repeat in range(3):
            benchmark_file.write(str(samples) + ",")
            benchmark_file.write(str(seq_len) + ",")

            rs = random.randint(0,10000)
            benchmark_file.write(str(rs) + ",")

            ts = msprime.sim_ancestry(
                samples=samples,
                recombination_rate=1e-8,
                sequence_length=seq_len,
                population_size=10_000,
                record_full_arg=True,
                random_seed=rs
            )

            #print(ts.draw_text())

            benchmark_file.write(str(ts.num_trees) + ",")
            benchmark_file.write(str(len(ts.nodes())) + ",")
            benchmark_file.write(str(len(algos.paths.identify_unique_paths(ts=ts))) + ",")

            specific_perm = perms[random.randint(0, len(perms)-1)]
            string_perm = str(specific_perm).replace(" ", "").replace(",", "").replace(")", "").replace("(", "")
            benchmark_file.write(string_perm + ",")
            for value in specific_perm:
                if value == 0:
                    paths_bench = algos.paths.benchmark(ts=ts)
                elif value == 1:
                    bottom_up_bench = algos.bottom_up.benchmark(ts=ts)
                else:
                    minimal_bench = algos.minimal.benchmark(ts=ts)
            benchmark_file.write(str(paths_bench[0]) + "," + str(bottom_up_bench[0]) + "," + str(minimal_bench[0]) + "," + str(minimal_bench[2]) + "\n")
    
benchmark_file.close()

