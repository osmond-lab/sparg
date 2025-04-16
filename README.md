# sparg

Using ancestral recombination graphs to estimate dispersal rates and locate genetic ancestors. See our [manuscript](https://www.biorxiv.org/content/10.1101/2024.04.10.588900v1) for details.

## Installation

Use Python's package installer to install sparg and its dependencies on the command line:

```
pip install "git+https://github.com/osmond-lab/sparg.git"
```

## Inputs

- Ancestral recombination graph (ARG): sparg is intended to be used with a "full ARG" stored as a tskit.TreeSequence. This matches the format output by `msprime.sim_ancestry(..., record_full_arg=True)`.

- Individual locations: locations can either be provided within the tskit.TreeSequence.Individuals table using the locations column or as a separate dictionary which maps each individual ID to a list or numpy.array of coordinates.

## Usage

### Preparing SLiM simulations

We've provided an example SLiM code (spatial.slim) to run a spatially explicit simulation and output a full ARG tskit.TreeSequence. Importantly, `initializeTreeSeq(retainCoalescentOnly=F);` prevents SLiM from simplifying the ARG; you will apply our custom simplification steps which preserve necessary unary nodes (recombination nodes and coalescent nodes from other trees). SLiM stores location information in the tskit.TreeSequence.Individuals table so you do not need to keep track of this separately. To run the simulation, download [SLiM4](https://messerlab.org/slim/) and enter `slim spatial.slim` on the command line. Now we can use the output in the following Python code.

Load the tree sequence simplify to a subset of 100 samples:

```
import tskit
import numpy as np
import sparg
ts = tskit.load("slim.trees")
samples = list(np.random.choice(ts.samples(), 100, replace=False))
ts_sim, map_sim = ts.simplify(samples=samples, map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
ts_final, maps_final = sparg.simplify_with_recombination(ts=ts_sim, flag_recomb=True)
```

We recommend that you chop the ARG at a sufficiently recent point in the past to reduce the effects of reflecting boundaries in the simulation. Here we cut the ARG off at 2000 generations:

```
ts_chopped = sparg.chop_arg(ts=ts_final, time=2000)
```

### Calculating spatial parameters

Now comes the main step, which could take minutes/hours to complete depending on the size of your ARG (seconds for the SLiM example). The sparg.SpatialARG object calculates all of the necessary spatial parameters needed to estimate the dispersal rate and locations of genetic ancestors:

```
spatial_arg = sparg.SpatialARG(ts=ts_chopped, verbose=True)
```

#### Dispersal Rate

The dispersal rate matrix is stored as an attribute of the sparg.SpatialARG object and can be accessed with 

```
spatial_arg.dispersal_rate_matrix
```

#### Locations of genetic ancestors

You can also locate genetic ancestors within the ARG. Each genetic ancestor is uniquely identified with the following three pieces of information: descendant sample, genome position, and time (generations ago). We first create a dataframe of the ancestors we want to locate:

```
import pandas as pd
samples = [0]
genome_positions = [0]
times = [10,100,1000]
data = []
for sample in samples:
  for pos in genome_positions:
    for time in times:
      data.append([sample,pos,time])
ancestors = pd.DataFrame(data, columns = ['sample','genome_position','time']) 
```

And then we locate them:

```
ancestor_locations = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(df=ancestors, spatial_arg=spatial_arg, window_size=0)
```

The `window_size` parameter allows you to set the number of neighboring trees on either side of the local tree that sparg will use (0 - local tree only).

We may be interested in locating all the ancestral nodes of a sample, which we can get with:

```
sample = 0
spatial_arg = sparg.SpatialARG(ts=ts_chopped, verbose=True)
ancestors = sparg.create_ancestors_dataframe(ts=ts_chopped, samples=[sample])
ancestor_locations = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(df=ancestors, spatial_arg=spatial_arg, window_size=0)
```

and then plot:

```
import matplotlib.pyplot as plt
# draw the lineages
for label, df in ancestor_locations.groupby('genome_position'):
    plt.plot(df.window_0_estimated_location_0, df.window_0_estimated_location_1, '-k', linewidth=1, zorder=0)
# color nodes by time
plt.scatter(data=ancestor_locations, x='window_0_estimated_location_0', y='window_0_estimated_location_1', c='time', edgecolors='k')
# aesthetics
plt.xlabel('x position')
plt.ylabel('y position')
plt.colorbar(label='generations ago')
plt.show()
```



