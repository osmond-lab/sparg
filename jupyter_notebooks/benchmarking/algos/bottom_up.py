import msprime
import random
import numpy as np


rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=2,
    recombination_rate=1e-8,
    sequence_length=2_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=rs #9818
)

print(ts.draw_text())
print(ts.tables.edges)

recomb_nodes = np.where(ts.tables.nodes.flags == 131072)[0]
recomb_nodes_to_convert = dict(zip(recomb_nodes[1::2], recomb_nodes[::2]))

cov_mats = {}
edges = ts.tables.edges
for node in ts.nodes():
    if node.id in recomb_nodes_to_convert:
        continue
    if node.flags == 1:
        cov_mats[node.id] = np.array([0])
    else:
        children = edges.child[np.where(edges.parent == node.id)]
        for child in children:
            if child in recomb_nodes_to_convert:
                children[children == child] = recomb_nodes_to_convert[child]
        print(node.id, children)
        if len(children) > 1:
            cov_mats[node.id] = np.array([0])
        else:
            cov_mats[node.id] = cov_mats[children[0]] + (node.time - ts.node(children[0]).time)
print(cov_mats)