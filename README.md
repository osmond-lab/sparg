# sparg
Using the ancestral recombination graphs to estimate dispersal rates and track the locations of genetic ancestors. See our [manuscript](https://www.biorxiv.org/content/10.1101/2024.04.10.588900v1) for details about these methods.

## Installation

```
pip install sparg
```

## Inputs

- Ancestral recombination graph (ARG): sparg is intended to be used with a "full ARG" stored as a tskit.TreeSequence. This matches the format output by `msprime.sim_ancestry(..., record_full_arg=True)`.

- Individual locations: locations can either be provided within the tskit.TreeSequence.Individuals table using the locations column or as a separate dictionary which maps each individual ID to a list or numpy.array of coordinates.

## Usage

### Preparing SLiM simulations

We've provided an example SLiM code (spatial.slim) which runs a spatially explicit simulation and outputs a tskit.TreeSequence. Importantly, `initializeTreeSeq(retainCoalescentOnly=F);` prevents SLiM from simplifying the ARG; you will apply our custom simplification steps which preserve necessary unary nodes (recombination nodes and coalescent nodes from other trees). SLiM stores location information in the tskit.TreeSequence.Individuals table, so you do not need to keep track of this separately.

```
ts = tskit.load("slim.trees")
samples = list(np.random.choice(ts.samples(), 1000, replace=False))
ts_sim, map_sim = ts.simplify(samples=samples, map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
ts_final, maps_final = sparg.simplify_with_recombination(ts=ts_sim, flag_recomb=True)
```

The above code loads the tree sequence into Python and simplifies to the full ARG of a subset of 1000 samples.

```
ts_chopped = sparg.chop_arg(ts=ts_final, time=2000)
```

We recommend that you chop the ARG at some point in the past as this helps to reduce the effects from the reflecting boundaries in the simulation. Here, we are only interest in the most recent 2000 generations.


### Calculating spatial parameters

```
spatial_arg = sparg.SpatialARG(ts=ts_chopped, verbose=True)
```

This is the main step and could take minutes/hours to complete depending on the size of your ARG. The sparg.SpatialARG object calculates all of the necessary spatial parameters needed to estimate the dispersal rate and locations of genetic ancestors.

#### Dispersal Rate

The dispersal rate matrix is stored as an attribute of the sparg.SpatialARG object and can be accessed with `spatial_arg.dispersal_rate_matrix`.


#### Locations of genetic ancestors

You can track the locations of genetic ancestors within the ARG. Each genetic ancestor is uniquely identified with the following three pieces of information: sample, genome position, and time. Create a pandas.DataFrame like the following:

If you are working with the SLiM simulation, we've provided a method for creating a dataframe to track the ancestors of a set of samples.

```
genetic_ancestors = sparg.create_ancestors_dataframe(ts=ts_sim, samples=[0], include_locations=True)
```

Note here, that we are using `ts_sim` rather than `ts_chopped` because `ts_sim` includes the true locations of all of the genetic ancestors. This is important if you want to measure the error in your estimates.

```
genetic_ancestors = sparg.estimate_locations_of_ancestors_in_dataframe_using_window(df=genetic_ancestors, spatial_arg=spatial_arg, window_size=0)
```

The `window_size` parameter allows you to set the number of neighboring trees on either side of the local tree that sparg will use (0 - local tree only).