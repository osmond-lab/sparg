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


ts = tskit.load("rep1/simple_space_uniform_start_rep1.trees")

print(ts)

np.random.seed(1)
keep_nodes = list(np.random.choice(ts.samples(), 50, replace=False))
ts_sim, maps_sim = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True, map_nodes=True)
ts_flagged = sparg.add_recomb_node_flags_to_ts(ts=ts_sim)

def chop_arg(ts, time):
    decapped = ts.decapitate(time)
    cut = ts.num_nodes
    for node in decapped.nodes(order="timeasc"):
        if node.time > time:
            cut = node.id
            break
    return ts.subset(nodes=range(cut))

cutoff = 40000
ts_chopped_40k = chop_arg(ts=ts_flagged, time=cutoff)
ts_filtered_40k, maps_filtered_40k = sparg.remove_uninformative_nodes(ts=ts_chopped_40k)#, keep_young_nodes={"below": 1000, "step": 10})
ts_merged_40k = sparg.merge_unnecessary_roots(ts=ts_filtered_40k)
ts_attached_40k, maps_attached_40k = sparg.remove_unattached_nodes(ts=ts_merged_40k)

dispersal_rate, cov_mat, paths, node_locations, node_variances = sparg.estimate_minimal_spatial_parameters(ts=ts_attached_40k, return_ancestral_node_positions=range(ts_attached_40k.num_nodes))

recombination_nodes_to_merge = np.where(ts_attached_40k.tables.nodes.flags==131072)[0][1::2]

epoch_step = 10000

times = [[] for i in range(0,cutoff,epoch_step)]
ranks = [[] for i in range(0,cutoff,epoch_step)]
locations = [[] for i in range(0,cutoff,epoch_step)]
true_locations = [[] for i in range(0,cutoff,epoch_step)]
confidence_interval = [[] for i in range(0,cutoff,epoch_step)]
residuals = [[] for i in range(0,cutoff,epoch_step)]
within_95 = [[] for i in range(0,cutoff,epoch_step)]
within = [[[] for i in range(0,300)] for i in range(0,cutoff,epoch_step)]

indiv = ts_attached_40k.tables.individuals
r = 0
for node in ts_attached_40k.nodes():
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
    for i,z in enumerate(range(0,300)):
        zs.append(z/100)
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
    for i,z in enumerate(range(0,300)):
        zs.append(z/100)
        percent_within.append(sum(within[c][i])/len(within[c][i]))
    plt.plot(expected_percent_within, percent_within, label=str(c*epoch_step) + " to " + str((c+1)*epoch_step))

plt.xlabel("Expected")
plt.ylabel("Observed")
plt.ylim(0,1)
plt.xlim(0,1)
plt.legend(loc="lower right")
plt.savefig("rep1/figures/percent_within_evo.png")
plt.clf()