import tskit
import numpy as np
import glob
import top_down
import matplotlib.pyplot as plt


ts = tskit.load("run6/5804.trees")
sample_locs = np.linspace(0, 1, ts.num_samples)
paths, node_times, node_locs, dispersal_rate = top_down.reconstruct_node_locations(ts=ts, sample_locs=sample_locs)
print("truth")

args = glob.glob("run6/ARGweaver_output/ts/*")
for arg in args:
    print(arg)
    inferred_ts = tskit.load(arg)
    inferred_paths, inferred_node_times, inferred_node_locs, inferred_dispersal_rate = top_down.reconstruct_node_locations(ts=inferred_ts, sample_locs=sample_locs)
    for p in inferred_paths:
        if p[0] == 0:
            p_times = []
            p_locs = []
            for n in p:
                p_times.append(inferred_node_times[n])
                p_locs.append(inferred_node_locs[n])
            plt.plot(p_locs, p_times, color="grey")
    plt.pause(0.05)

for p in paths:
    if p[0] == 0:
        p_times = []
        p_locs = []
        for n in p:
            p_times.append(node_times[n])
            p_locs.append(node_locs[n])
        plt.plot(p_locs, p_times)

plt.show()

exit()



infered_dispersal_rates = []
gmrca_locs = []

args = glob.glob("run5/ARGweaver_output/ts/*")

for arg in args:
    results = top_down.get_dispersal_and_gmrca(arg)
    dispersal_rates.append(results[0][0])
    gmrca_locs.append(results[1])

plt.hist(dispersal_rates, bins=30)
#plt.axvline(true_values[0][0], color="red")
plt.show()

plt.hist(gmrca_locs, bins=30)
plt.axvline(true_values[1], color="red")
plt.show()
