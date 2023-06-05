import msprime
import random
import numpy as np
from collections import defaultdict


rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=2,
    recombination_rate=1e-8,
    sequence_length=2_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=5802 #9818
)

print(ts.draw_text())
print(ts.tables.edges)

recomb_nodes = np.where(ts.tables.nodes.flags == 131072)[0]
recomb_nodes_to_convert = dict(zip(recomb_nodes[1::2], recomb_nodes[::2]))
edges = ts.tables.edges

cov_mat = np.zeros((ts.num_samples,ts.num_samples))
node_indices = dict(zip(ts.samples(), [[i] for i in range(ts.num_samples)]))
print(node_indices)

max_index = ts.num_samples - 1
for node in ts.nodes():
    if node.id in recomb_nodes_to_convert or node.flags == 1:
        continue
    children = edges.child[np.where(edges.parent == node.id)]
    for child in children:
        if child in recomb_nodes_to_convert:
            child = recomb_nodes_to_convert[child]
        cov_mat[node_indices[child], node_indices[child]] += node.time - ts.node(child).time
        node_indices[node.id] = node_indices.get(node.id, []) + node_indices[child]
    if node.id in recomb_nodes:
        to_duplicate = cov_mat[node_indices[node.id], node_indices[node.id]]
        print(to_duplicate)
        num_new_paths = to_duplicate.shape[0]

        max_index += num_new_paths
        
    print(node_indices)



        

    
    


    if len(children) == 1:
        pass
        #cov_mat += (node.time - ts.node(children[0]).time)
    else:
        pass
    #if len(children) > 1:
    #    cov_mats[node.id] = np.array([0])
    #else:
    #    cov_mats[node.id] = cov_mats[children[0]] + (node.time - ts.node(children[0]).time)
print(cov_mat)