import glob
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

with open("coverage.csv", "w") as file:
    file.write("file,epoch_low,epoch_high,num_std,within,total\n")

for arg in glob.glob("orig_args/*"):
    print(arg)

    ts = tskit.load(arg)

    np.random.seed(1)
    cutoff = 40000
    samples = list(np.random.choice(ts.samples(), 50, replace=False))
    ts_sim, map_sim = ts.simplify(samples=samples, map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
    ts_final, maps_final = sparg.simplify_with_recombination(ts=ts_sim, flag_recomb=True)
    ts_chopped = chop_arg(ts=ts_final, time=cutoff)

    ts_chopped.dump(arg.replace("orig_args", "simplified_args"))

    dispersal_rate, cov_mat, paths, node_locations, node_variances = sparg.estimate_minimal_spatial_parameters(ts=ts_chopped, return_ancestral_node_positions=range(ts_chopped.num_nodes), verbose=True)

    recombination_nodes_to_merge = np.where(ts_chopped.tables.nodes.flags==131072)[0][1::2]

    epoch_step = 1000

    times = [[] for i in range(0,cutoff,epoch_step)]
    ranks = [[] for i in range(0,cutoff,epoch_step)]
    locations = [[] for i in range(0,cutoff,epoch_step)]
    true_locations = [[] for i in range(0,cutoff,epoch_step)]
    confidence_interval = [[] for i in range(0,cutoff,epoch_step)]
    residuals = [[] for i in range(0,cutoff,epoch_step)]
    within_95 = [[] for i in range(0,cutoff,epoch_step)]
    within = [[[] for i in range(0,300)] for i in range(0,cutoff,epoch_step)]

    indiv = ts_chopped.tables.individuals
    r = 0
    for node in ts_chopped.nodes():
        if node.flags != 1:
            ranks[int((node.time-1)//epoch_step)].append(r)
            times[int((node.time-1)//epoch_step)].append(node.time)
            locations[int((node.time-1)//epoch_step)].append(node_locations[node.id][0])
            true_locations[int((node.time-1)//epoch_step)].append(indiv[node.individual].location[0])
            confidence_interval[int((node.time-1)//epoch_step)].append(1.96*math.sqrt(round(node_variances[node.id][0][0])))
            if abs(node_locations[node.id][0] - indiv[node.individual].location[0]) <= 1.96*math.sqrt(round(node_variances[node.id][0][0])):
                within_95[int((node.time-1)//epoch_step)].append("gray")
            else:
                within_95[int((node.time-1)//epoch_step)].append("red")
            for i,z in enumerate(range(0,300)):
                interval = (z/100)*math.sqrt(round(node_variances[node.id][0][0]))
                within[int((node.time-1)//epoch_step)][i].append(abs(node_locations[node.id][0] - indiv[node.individual].location[0]) <= interval)
            residuals[int((node.time-1)//epoch_step)].append(node_locations[node.id][0] - indiv[node.individual].location[0])
            if node.flags != 1 and node.flags != 131072:
                r += 1
            elif node.id in recombination_nodes_to_merge:
                r += 1

    outfile = open("coverage.csv", "a")
    for e in range(len(within)):
        for i in range(300):
            outfile.write(arg.split("/")[-1] + "," + str(e*epoch_step) + "," + str((e+1)*epoch_step) + "," + str(i/100) + "," + str(sum(within[e][i])) + "," + str(len(within[e][i])) + "\n")
    outfile.close()

exit()

plt.errorbar(locations[0], times[0], xerr=confidence_interval[0], label="estimated", fmt="o", zorder=1)
plt.scatter(true_locations[0], times[0], color="orange", label="truth", zorder=2)
plt.yscale("log")
plt.xlim(0,100)
plt.xlabel("Location (x-dimension)")
plt.ylabel("Time (generations)")
plt.savefig("rep1/figures/location_of_ancestors.png")
plt.clf()

plt.errorbar(locations[0], ranks[0], xerr=confidence_interval[0], ecolor=within_95[0], label="estimated", fmt="none", zorder=1)
plt.scatter(true_locations[0], ranks[0], color=within_95[0], label="truth", zorder=2)
plt.xlim(0,100)
plt.xlabel("Location (x-dimension)")
plt.ylabel("Time (rank)")
plt.savefig("rep1/figures/location_of_ancestors_colored.png")
plt.clf()

plt.scatter(times[0], residuals[0])
plt.xscale("log")
plt.xlabel("Time (generations)")
plt.ylabel("Residual")
plt.savefig("rep1/figures/residuals.png")
plt.clf()


expected_zs = []
expected_percent_within = []
for i in range(0,300):
    value = st.norm.cdf(i/100)
    expected_zs.append(i/100)
    expected_percent_within.append(value - (1-value))
plt.plot(expected_zs, expected_percent_within, "--", color="gray")

for c in range(len(within)):
    zs = []
    percent_within = []
    for i in range(0,300):
        zs.append(i/100)
        percent_within.append(sum(within[c][i])/len(within[c][i]))
    plt.plot(zs, percent_within, label=str(c*epoch_step) + " to " + str((c+1)*epoch_step))

plt.xlabel("Number of Standard Deviations")
plt.ylabel("Percent Within Confidence Interval")
plt.ylim(0,1)
plt.xlim(0,3)
plt.legend(loc="lower right")
plt.savefig("rep1/figures/percent_within.png")
plt.clf()


plt.plot(expected_percent_within, expected_percent_within, "--", color="gray")

for c in range(len(within)):
    zs = []
    percent_within = []
    for i in range(0,300):
        zs.append(i/100)
        percent_within.append(sum(within[c][i])/len(within[c][i]))
    plt.plot(expected_percent_within, percent_within, label=str(c*epoch_step) + " to " + str((c+1)*epoch_step))

plt.xlabel("Expected")
plt.ylabel("Observed")
plt.ylim(0,1)
plt.xlim(0,1)
plt.legend(loc="lower right")
plt.savefig("rep1/figures/percent_within_evo.png")
plt.clf()