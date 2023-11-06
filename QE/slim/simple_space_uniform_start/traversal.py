import sys
sys.path.append("/Users/jameskitchens/Documents/GitHub/sparg2.0")

import sparg
import tskit
import msprime
import math
import numpy as np
import tskit_arg_visualizer as viz
import warnings
import matplotlib.pyplot as plt
import scipy.stats as st
warnings.simplefilter("always")


def simplify_with_recombination(ts, flag_recomb=False, keep_nodes_below=0):
    """Simplifies a tree sequence while keeping recombination nodes

    Removes unary nodes that are not recombination nodes. Does not remove non-genetic ancestors.
    Edges intervals are not updated. This differs from how tskit's TreeSequence.simplify() works.

    Parameters
    ----------
    ts : tskit.TreeSequence
    flag_recomb : bool
        Whether to add msprime node flags

    Returns
    -------
    ts_sim : tskit.TreeSequence
        Simplified tree sequence
    maps_sim : numpy.ndarray
        Mapping for nodes in the simplified tree sequence versus the original
    """

    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0)
    child_node, parents_count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has.
    parent_node, children_count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has.
    multiple_parents = child_node[parents_count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    
    if flag_recomb:
        ts_tables = ts.dump_tables()
        node_table = ts_tables.nodes
        flags = node_table.flags
        flags[recomb_nodes] = msprime.NODE_IS_RE_EVENT
        node_table.flags = flags
        ts_tables.sort() 
        ts = ts_tables.tree_sequence()
    
    keep_nodes = np.unique(np.concatenate((np.where(ts.tables.nodes.time <= keep_nodes_below)[0], recomb_nodes)))
    potentially_uninformative = np.intersect1d(child_node[np.where(parents_count!=0)[0]], parent_node[np.where(children_count==1)[0]])
    truly_uninformative = np.delete(potentially_uninformative, np.where(np.isin(potentially_uninformative, keep_nodes)))
    all_nodes = np.array(range(ts.num_nodes))
    important = np.delete(all_nodes, np.where(np.isin(all_nodes, truly_uninformative)))
    ts_sim, maps_sim = ts.simplify(samples=important, map_nodes=True, keep_input_roots=False, keep_unary=False, update_sample_flags=False)
    return ts_sim, maps_sim

def identify_all_nodes_above(ts, nodes):
    """Traverses all nodes above provided list of nodes

    Parameters
    ----------
    ts : tskit.TreeSequence
    nodes : list or numpy.ndarray
        Nodes to traverse above in the ARG. Do not need to be connected

    Returns
    -------
    above_samples : numpy.ndarray
        Sorted array of node IDs above the provided list of nodes
    """

    edges = ts.tables.edges
    above_samples = []
    while len(nodes) > 0:
        above_samples.append(nodes[0])
        parents = list(np.unique(edges[np.where(edges.child==nodes[0])[0]].parent))
        new_parents = []
        for p in parents:
            if (p not in nodes) and (p not in above_samples):
                new_parents.append(p)
        nodes = nodes[1:] + new_parents
    return np.sort(above_samples)

def chop_arg(ts, time):
    decap = ts.decapitate(time)
    subset = decap.subset(nodes=np.where(decap.tables.nodes.time <= time)[0])
    merged = sparg.merge_unnecessary_roots(ts=subset)
    return merged


ts = tskit.load("rep1/simple_space_uniform_start_rep1.trees")

np.random.seed(1)
samples = list(np.random.choice(ts.samples(), 1000, replace=False))

#ts_sim, maps_sim = simplify_with_recombination(ts=ts)
#above_samples = identify_all_nodes_above(ts=ts_sim, nodes=samples)
#ts_sub = ts_sim.subset(nodes=above_samples)
#ts_final, maps_final = simplify_with_recombination(ts=ts_sub, flag_recomb=True)
#ts_chopped = chop_arg(ts=ts_final, time=10000)
#viz.D3ARG(ts=ts_chopped).draw(width=1000, height=1000)

ts_sim, map_sim = ts.simplify(samples=samples, map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
ts_final, maps_final = simplify_with_recombination(ts=ts_sim, flag_recomb=True, keep_nodes_below=1000)
ts_chopped = chop_arg(ts=ts_final, time=40000)


sample_of_interest = 512
viz.D3ARG(ts=chop_arg(ts=simplify_with_recombination(ts=ts_chopped)[0], time=1000).simplify(samples=[sample_of_interest], map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)[0]).draw(width=500, height=500)

nodes_above = identify_all_nodes_above(ts=ts_chopped, nodes=[sample_of_interest])
dispersal_rate, cov_mat, paths, node_locations, node_variances = sparg.estimate_minimal_spatial_parameters(ts=ts_chopped, return_ancestral_node_positions=nodes_above)




"""
follow_path = None
for path in paths:
    if path[0] == sample_of_interest:
        follow_path = path
        break

true_xs = []
true_ys = []
predicted_xs = []
predicted_ys = []
variance_xs = []
variance_ys = []

for node in follow_path:
    indiv = ts_chopped.node(node).individual
    if indiv != -1:
        true_location = ts_chopped.individual(indiv).location
        true_xs.append(true_location[0])
        true_ys.append(true_location[1])
        predicted_xs.append(node_locations[node][0])
        predicted_ys.append(node_locations[node][1])
        if node_variances[node][0][0] < 0:
            variance_xs.append(0)
        else:
            variance_xs.append(1.96*math.sqrt(node_variances[node][0][0]))
        if node_variances[node][1][1] < 0:
            variance_ys.append(0)
        else:
            variance_ys.append(1.96*math.sqrt(node_variances[node][1][1]))



fig = plt.figure()
ax1 = fig.add_subplot() # regular resolution color map
cm = plt.get_cmap("winter")

i = 20

ax1.set_prop_cycle(color=[cm(1.*j/i) for j in range(i)])
for n in range(i):
    plt.plot(true_xs[n:n+2], true_ys[n:n+2], zorder=0)

plt.errorbar(predicted_xs[i], predicted_ys[i], xerr=variance_xs[i], yerr=variance_ys[i], label="estimated", fmt="none", color="red", zorder=1)
plt.scatter(true_xs[i], true_ys[i], color="black")
plt.xlim(0,100)
plt.ylim(0,100)
plt.show()
"""

ax = plt.figure().add_subplot(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

for path in paths:
    if path[0] == sample_of_interest:
        true_xs = []
        true_ys = []
        predicted_xs = []
        predicted_ys = []
        variance_xs = []
        variance_ys = []
        times = []
        for node in path:
            indiv = ts_chopped.node(node).individual
            if indiv != -1:
                if ts_chopped.node(node).time <= 1000:
                    true_location = ts_chopped.individual(indiv).location
                    true_xs.append(true_location[0])
                    true_ys.append(true_location[1])
                    predicted_xs.append(node_locations[node][0])
                    predicted_ys.append(node_locations[node][1])
                    if node_variances[node][0][0] < 0:
                        variance_xs.append(0)
                    else:
                        variance_xs.append(1.96*math.sqrt(node_variances[node][0][0]))
                    if node_variances[node][1][1] < 0:
                        variance_ys.append(0)
                    else:
                        variance_ys.append(1.96*math.sqrt(node_variances[node][1][1]))
                    times.append(ts_chopped.node(node).time)
        ax.plot(true_xs, true_ys, times)
plt.xlim(50,)
plt.ylim(25,75)
ax.set_zlim(0,1000)
plt.show()
        











exit()

bps = []
num_trees = []
dispersal_rate_x = []
dispersal_rate_y = []
for bp in ts_chopped.breakpoints():
    if bp > 0:
        ts_trimmed = ts_chopped.keep_intervals(intervals=[[0,bp]], simplify=False).trim()
        ts_filtered = sparg.remove_unattached_nodes(ts=ts_trimmed)
        dispersal_rate, cov_mat, paths = sparg.estimate_minimal_spatial_parameters(ts=ts_filtered)
        bps.append(bp)
        num_trees.append(ts_filtered.num_trees)
        dispersal_rate_x.append(dispersal_rate[0][0])
        dispersal_rate_y.append(dispersal_rate[1][1])


plt.plot(num_trees, dispersal_rate_x)
plt.show()
        






