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


ts = tskit.load("simple_space_uniform_start_d0.4.trees")



np.random.seed(1)
keep_nodes = list(np.random.choice(ts.samples(), 50, replace=False))
ts_sim, maps_sim = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True, map_nodes=True)
ts_flagged = sparg.add_recomb_node_flags_to_ts(ts=ts_sim)

def chop_arg(ts, time):
    decapped = ts.decapitate(time)
    cut = ts.num_nodes
    for node in decapped.nodes():
        if node.time > time:
            cut = node.id
            break
    return ts.subset(nodes=range(cut))


print(ts_flagged.tables.edges)
exit()

ts_chopped_40k = chop_arg(ts=ts_flagged, time=25000)
ts_filtered_40k, maps_filtered_40k = sparg.remove_uninformative_nodes(ts=ts_chopped_40k)#, keep_young_nodes={"below": 1000, "step": 10})
ts_merged_40k = sparg.merge_unnecessary_roots(ts=ts_filtered_40k)
ts_attached_40k, maps_attached_40k = sparg.remove_unattached_nodes(ts=ts_merged_40k)

dispersal_rate, cov_mat, paths, node_locations, node_variances = sparg.estimate_minimal_spatial_parameters(ts=ts_attached_40k, return_ancestral_node_positions=range(ts_attached_40k.num_nodes))

recombination_nodes_to_merge = np.where(ts_attached_40k.tables.nodes.flags==131072)[0][1::2]

#print(ts_attached_40k.tables.nodes[recombination_nodes_to_merge])



times = []
ranks = []
locations = []
true_locations = []
confidence_interval = []
within_95 = []
within = [[] for i in range(0,300)]
residuals = []

indiv = ts_attached_40k.tables.individuals
r = 0
for node in ts_attached_40k.nodes():
    if node.time <= 2000 and node.time > 0:
        ranks.append(r)
        times.append(node.time)
        locations.append(node_locations[node.id][0])
        true_locations.append(indiv[node.individual].location[0])
        confidence_interval.append(1.96*math.sqrt(round(node_variances[node.id][0][0])))
        if abs(node_locations[node.id][0] - indiv[node.individual].location[0]) <= 1.96*math.sqrt(round(node_variances[node.id][0][0])):
            within_95.append("gray")
        else:
            within_95.append("red")
        for i,z in enumerate(range(0,300)):
            interval = (z/100)*math.sqrt(round(node_variances[node.id][0][0]))
            within[i].append(abs(node_locations[node.id][0] - indiv[node.individual].location[0]) <= interval)
        residuals.append(node_locations[node.id][0] - indiv[node.individual].location[0])
        if node.flags != 1 and node.flags != 131072:
            r += 1
        elif node.id in recombination_nodes_to_merge:
            r += 1
        

plt.errorbar(locations, times, xerr=confidence_interval, label="estimated", fmt="o", zorder=1)
plt.scatter(true_locations, times, color="orange", label="truth", zorder=2)
plt.yscale("log")
plt.xlim(0,100)
plt.xlabel("Location (x-dimension)")
plt.ylabel("Time (generations)")
plt.savefig("figures/location_of_ancestors.png")
plt.clf()

plt.errorbar(locations[50:], ranks[50:], xerr=confidence_interval[50:], ecolor=within_95[50:], label="estimated", fmt="none", zorder=1)
plt.scatter(true_locations[50:], ranks[50:], color=within_95[50:], label="truth", zorder=2)
plt.xlim(0,100)
plt.xlabel("Location (x-dimension)")
plt.ylabel("Time (rank)")
plt.savefig("figures/location_of_ancestors_colored.png")
plt.clf()

plt.scatter(times, residuals)
plt.xscale("log")
plt.xlabel("Time (generations)")
plt.ylabel("Residual")
plt.savefig("figures/residuals.png")
plt.clf()

zs = []
percent_within = []

for i,z in enumerate(range(0,300)):
    zs.append(z/100)
    percent_within.append(sum(within[i])/len(within[i]))

expected_zs = []
expected_percent_within = []

for i in range(0,300):
    value = st.norm.cdf(i/100)
    expected_zs.append(i/100)
    expected_percent_within.append(value - (1-value))

plt.plot(expected_zs, expected_percent_within, "--", color="gray")
plt.plot(zs, percent_within)
plt.xlabel("Z-Score")
plt.ylabel("Percent Within Confidence Interval")
plt.ylim(0,1)
plt.xlim(0,3)
plt.savefig("figures/percent_within.png")
plt.clf()


plt.plot(expected_percent_within, expected_percent_within, "--", color="gray")
plt.plot(expected_percent_within, percent_within)
plt.xlabel("Expected")
plt.ylabel("Observed")
plt.ylim(0,1)
plt.xlim(0,1)
plt.savefig("figures/percent_within_evo.png")
plt.clf()