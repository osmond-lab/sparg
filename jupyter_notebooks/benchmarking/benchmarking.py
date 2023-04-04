import random
import msprime
import time
import numpy as np
import algos.paths
import algos.hybrid_nonrecursive
import algos.hybrid_recursive
import algos.top_down_puneeth
import algos.paths_modified
from itertools import permutations



benchmark_file = open("benchmarking_with_td.csv", "w")
benchmark_file.write("num_samples,sequence_length,seed,num_trees,nodes,paths,algo_order,num_loop_groups,paths_time,hybrid_r_time,hybrid_nr_time,top_down_time,paths_modified_time,paths_sum,hybrid_r_sum,hybrid_nr_sum,top_down_sum,paths_modified_sum\n")

perms = list(permutations([0, 1, 2, 3, 4]))

for samples in range(10,110,10):
    print(samples)
    for seq_len in range(1000, 2000, 1000):
        for repeat in range(5):
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
                random_seed=rs#6334 #7483,8131
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
                    hybrid_r_bench = algos.hybrid_recursive.benchmark(ts=ts)
                elif value == 2:
                    hybrid_nr_bench = algos.hybrid_nonrecursive.benchmark(ts=ts)
                elif value == 3:
                    top_down = algos.top_down_puneeth.benchmark(ts=ts)
                else:
                    paths_modified = algos.paths_modified.benchmark(ts=ts)
            benchmark_file.write(str(hybrid_nr_bench[2]) + "," + str(paths_bench[0]) + "," + str(hybrid_r_bench[0]) + "," + str(hybrid_nr_bench[0]) + "," + str(top_down[0]) + "," + str(paths_modified[0]) + "," + str(paths_bench[1]) + "," + str(hybrid_r_bench[1]) + "," + str(hybrid_nr_bench[1]) + "," + str(top_down[1]) + "," + str(paths_modified[1]) + "\n")
    
benchmark_file.close()

