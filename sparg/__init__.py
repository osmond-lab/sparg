import numpy as np
from collections import defaultdict
import sympy as sym
import warnings
from tqdm import tqdm
import time
import pandas as pd
import msprime
import random


#### USED WHEN PREPARING THE tskit.TreeSequence OUTPUT BY SLiM FOR SpatialARG

def find_ancestral_node_at_time(tree, u, time):
    """Find the ancestral node of a sample within a tree at a specified time.
    
    This requires that there is a node at that position in the tree. If not,
    returns None (with a warning).

    Parameters
    ----------
    tree : tskit.Tree
    u : int
        The ID for the node of interest
    time : int or float
        timing of the ancestral node of interest

    Returns
    -------
    u : int
        Node ID of the ancestral node at specified point
    """

    u = tree.parent(u)
    while u != -1:
        node_time = tree.time(u)
        if node_time == time:
            return u
        u = tree.parent(u)
    warnings.warn(f"Sample %s does not have an ancestral node at time %s in tree. Returning None." % (u, time))

def generate_random_ancestors_dataframe(ts, number_of_ancestors, include_locations=False, dimensions=2, cutoff=-1, seed=None):
    """This is a little awkward because it needs to run on the unsimplified ARG which has all of the location information.
    This info is lost during the simplification step. We could use find_nearest_ancestral_node_at_time() and check that it is
    equal, but leaving like this for now.
    """
    
    if seed != None:
        random.seed(seed)
    samples = []
    interval_left = []
    interval_right = []
    times = []
    location = []
    for n in range(number_of_ancestors):
        sample = random.randint(0, ts.num_samples-1)
        genome_pos = random.uniform(0, ts.sequence_length)
        if cutoff == -1:
            time = random.randint(0, ts.max_root_time)
        else:
            time = random.randint(0, cutoff)
        tree = ts.at(genome_pos)
        ancestor = find_ancestral_node_at_time(tree, sample, time)
        samples.append(sample)
        interval_left.append(tree.interval.left)
        interval_right.append(tree.interval.right)
        times.append(time)
        indiv = ts.node(ancestor).individual
        if indiv != -1:
            location.append(ts.individual(indiv).location[:dimensions])
        else:
            location.append([None for d in range(dimensions)])
    df = pd.DataFrame({
        "sample":samples,
        "interval_left":interval_left,
        "interval_right":interval_right,
        "time":times,
    })
    if include_locations:
        locs = pd.DataFrame(location, columns=["true_location_"+str(d) for d in range(dimensions)])
        df = pd.concat([df, locs], axis=1)
    return df

def simplify_with_recombination(ts, flag_recomb=False, keep_nodes=[]):
    """Simplifies a tree sequence while keeping recombination nodes

    Removes unary nodes that are not recombination nodes. Does not remove non-genetic ancestors.
    Edges intervals are not updated. This differs from how tskit's TreeSequence.simplify() works.

    Parameters
    ----------
    ts : tskit.TreeSequence
    flag_recomb (optional) : bool
        Whether to add msprime node flags. Default is False.
    keep_nodes (optional) : list
        List of node IDs that should be kept. Default is empty list, so ignored.

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
    
    keep_nodes = np.unique(np.concatenate((keep_nodes, recomb_nodes)))
    potentially_uninformative = np.intersect1d(child_node[np.where(parents_count!=0)[0]], parent_node[np.where(children_count==1)[0]])
    truly_uninformative = np.delete(potentially_uninformative, np.where(np.isin(potentially_uninformative, keep_nodes)))
    all_nodes = np.array(range(ts.num_nodes))
    important = np.delete(all_nodes, np.where(np.isin(all_nodes, truly_uninformative)))
    ts_sim, maps_sim = ts.simplify(samples=important, map_nodes=True, keep_input_roots=False, keep_unary=False, update_sample_flags=False)
    return ts_sim, maps_sim

def remove_unattached_nodes(ts):
    """Removes any nodes that are not attached to any other nodes from the tree sequence
    
    Parameters
    ----------
    ts : tskit.TreeSequence

    Returns
    -------
    ts_final : tskitTreeSequence
        A tree sequence with unattached nodes removed
    """

    edge_table = ts.tables.edges
    connected_nodes = np.sort(np.unique(np.concatenate((edge_table.parent,edge_table.child))))
    ts_final = ts.subset(nodes=connected_nodes)
    return ts_final
    
def merge_unnecessary_roots(ts):
    """Merges root node IDs that are referring to the same node

    This commonly occurs as a result of decapitate(). Combines the two nodes into one and then
    removes the unattached node that is no longer important. This does not merge all roots into
    one, just those that are referring to the same root.

    Parameters
    ----------
    ts : tskit.TreeSequence

    Returns
    -------
    ts_new : tskitTreeSequence
        A tree sequence with corresponding roots merged
    """

    ts_tables = ts.dump_tables()
    edge_table = ts_tables.edges 
    parent = edge_table.parent
    roots = np.where(ts_tables.nodes.time == ts.max_time)[0]
    children = defaultdict(list)
    for root in roots:
        root_children = edge_table.child[np.where(edge_table.parent == root)[0]]
        for child in root_children:
            children[child] += [root]
    for child in children:
        pts = children[child]
        if len(pts) > 1:
            for pt in pts:
                if len(np.unique(edge_table.child[np.where(edge_table.parent == pt)[0]])) > 1:
                    print(pt, "has multiple children! Merge roots with caution.")
                parent[np.where(ts.tables.edges.parent == pt)[0]] = pts[0]
    edge_table.parent = parent 
    ts_tables.sort() 
    ts_new = remove_unattached_nodes(ts=ts_tables.tree_sequence())
    return ts_new

def chop_arg(ts, time):
    """Chops the tree sequence at a time in the past

    Parameters
    ----------
    ts : tskit.TreeSequence
    time : int
        Chops at `time` generations in the past

    Returns
    -------
    merged : tskitTreeSequence
        A tree sequence that has been decapitated and subset
    """

    decap = ts.decapitate(time)
    subset = decap.subset(nodes=np.where(decap.tables.nodes.time <= time)[0])
    merged = merge_unnecessary_roots(ts=subset)
    return merged


class SpatialARG:
    """

    Attributes
    ----------

    ts
    locations_of_individuals
    paths_shared_time_matrix
    paths
    node_paths_shared_times
    node_paths
    inverted_paths_shared_time_matrix
    root_locations
    path_dispersal_distances
    dispersal_rate_matrix
    fishers_information_1
    fishers_information_2
    """
    
    def __init__(self, ts, locations_of_individuals={}, dimensions=2, verbose=False):
        total_start_time = time.time()

        section_start_time = time.time()
        self.ts = ts
        if locations_of_individuals == {}:  # if user doesn't provide a separate locations dictionary, builds one
            self.locations_of_individuals = self.get_tskit_locations(dimensions=dimensions)
        else:
            self.locations_of_individuals = locations_of_individuals
        if verbose:
            print(f"Prepared input parameters - Section Elapsed Time: {round(time.time()-section_start_time,2)} - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")
        
        section_start_time = time.time()
        self.paths_shared_time_matrix, self.paths, self.node_paths_shared_times, self.node_paths = self.calc_minimal_covariance_matrix(verbose=verbose)
        if verbose:
            print(f"Calculated covariance matrix - Section Elapsed Time: {round(time.time()-section_start_time,2)} - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")
        
        section_start_time = time.time()
        self.inverted_paths_shared_time_matrix = np.linalg.pinv(self.paths_shared_time_matrix)
        if verbose:
            print(f"Inverted covariance matrix - Section Elapsed Time: {round(time.time()-section_start_time,2)} - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")
        
        section_start_time = time.time()
        locations_of_path_starts, locations_of_samples = self.expand_locations()
        roots_array, roots = self.build_roots_array()
        root_locations = self.locate_roots(roots_array=roots_array, locations_of_path_starts=locations_of_path_starts)
        self.root_locations = dict(zip(roots, root_locations))
        root_locations_vector = np.matmul(roots_array, root_locations)
        if verbose:
            print(f"Created root locations vector - Section Elapsed Time: {round(time.time()-section_start_time,2)} - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")
        
        section_start_time = time.time()
        self.path_dispersal_distances = locations_of_path_starts - root_locations_vector
        self.dispersal_rate_matrix = np.matmul(np.matmul(np.transpose(self.path_dispersal_distances), self.inverted_paths_shared_time_matrix), self.path_dispersal_distances)/(self.ts.num_samples)
        if verbose:
            print(f"Estimated dispersal rate - Section Elapsed Time: {round(time.time()-section_start_time,2)} - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")
        
        section_start_time = time.time()
        self.fishers_information_1 = self.ts.num_samples/(2*self.dispersal_rate_matrix[0][0]**2) 
        self.fishers_information_2 = np.matmul(np.matmul(np.transpose(root_locations_vector), self.inverted_paths_shared_time_matrix), root_locations_vector)[0][0]/self.dispersal_rate_matrix[0][0]**3
        if verbose:
            print(f"Calculated Fisher's information matrices - Section Elapsed Time: {round(time.time()-section_start_time,2)} - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")
        
        if verbose:
            print(f"Completed building SpatialARG object - Total Elapsed Time: {round(time.time()-total_start_time, 2)}")

    def __str__(self):
        return "Available object attributes: " + ", ".join(self.__dict__.keys())
    
    def dump(self, path):
        pass

    def get_tskit_locations(self, dimensions=2):
        """Converts the tskit individuals locations into a dictionary.

        Parameters
        ----------
        ts : tskit.trees.TreeSequence
            This must be a tskit Tree Sequences with marked recombination nodes, as is outputted by
            msprime.sim_ancestry(..., record_full_arg=True). Must include locations within the
            individuals table.
        dimensions (optional): int
            The number of dimensions that you are interested in looking at. Often SLiM gives
            a third dimension even though individuals can't move in that dimension. Default is 2.

        Returns
        -------
        locations_of_individuals : dictionary
            Dictionary of sample node locations where the key is the node ID and the value is a
            numpy.ndarray or list with the node's location.

        """

        if len(self.ts.tables.individuals.location) == 0:
            raise RuntimeError("Locations of individuals not provided.")
        locations = np.array_split(self.ts.tables.individuals.location, self.ts.num_individuals)
        locations_of_individuals = {}
        for i,location in enumerate(locations):
            locations_of_individuals[i] = location[:dimensions]
        return locations_of_individuals

    def calc_minimal_covariance_matrix(self, verbose=False):
        """Calculates a covariance matrix between the minimal number of paths in the the ARG. Should always produce an invertible matrix 

        Parameters
        ----------
        ts : tskit.trees.TreeSequence
            This must be a tskit Tree Sequences with marked recombination nodes, as is outputted by
            msprime.sim_ancestry(..., record_full_arg=True). The covariance matrix will not be
            correct if the recombination nodes are not marked.
        internal_nodes (optional): list 
            A list of internal nodes for which you want the shared times. Default is an empty list,
            in which case no internal nodes will be calculated.
        verbose (optional): boolean
            Print checkpoints to screen as the function calculates. Default is False.

        Returns
        -------
        cov_mat : numpy.ndarray
            An array containing the shared times between different sample paths in the ARG, ordered
            by the `paths` list.
        paths : list
            List of paths from samples to respective roots through the ARG. Each path includes the
            ID of the nodes that it passes through in order from youngest to oldest.
        
        Optional Returns
        ----------------
        If internal nodes are provided:
            internal_node_shared_times : tuple
                This tuple contains two parts:
                    - shared_time : numpy.array - an array containing the shared times between internal
                    node paths and different sample paths in the ARG, ordered by the `internal_paths` list.
                    - internal_paths : list - list of paths from internal nodes to respective roots
                    through the ARG. Each path includes the ID of the nodes that it passes through in
                    order from youngest to oldest.
        """
        
        internal_nodes = range(self.ts.num_nodes)
        edges = self.ts.tables.edges
        cov_mat = np.zeros(shape=(self.ts.num_samples, self.ts.num_samples))#, dtype=np.float64)  #Initialize the covariance matrix. Initial size = #samples. Will increase to #paths
        indices = defaultdict(list) #Keeps track of the indices of paths that enter (from bottom) a particular node.
        paths = []
        for i, sample in enumerate(self.ts.samples()):
            indices[sample] = [i]   #Initialize indices for each path which at this point also corresponds to the sample.
            paths.append([sample])  #Keeps track of different paths. To begin with, as many paths as samples.
        int_nodes = {}
        internal_paths = []
        if len(internal_nodes) != 0:
            int_nodes = {nd:i for i,nd in enumerate(internal_nodes)}
            internal_paths = [ [nd] for nd in internal_nodes ]
            shared_time = np.zeros(shape=(len(int_nodes),self.ts.num_samples))
            internal_indices = defaultdict(list) #For each path, identifies internal nodes that are using that path for shared times.
        if verbose:
            nodes = tqdm(self.ts.nodes(order="timeasc"))
        else:
            nodes = self.ts.nodes(order="timeasc")
        nodes_realized = np.concatenate((self.ts.tables.edges.parent,self.ts.tables.edges.child))
        for node in nodes:
            if node.id not in nodes_realized :
                continue
            path_ind = indices[node.id]
            parent_nodes = np.unique(edges.parent[np.where(edges.child == node.id)])
            if len(internal_nodes) != 0: 
                if node.id in int_nodes: 
                    internal_indices[path_ind[0]] += [int_nodes[node.id]]
            nparent = len(parent_nodes)
            if nparent == 0 : 
                continue
            elif nparent == 1 : 
                parent = parent_nodes[0]
                for path in path_ind:
                    paths[path].append(parent)
                    if len(internal_nodes) != 0:
                        for internal_path_ind in internal_indices[path]: 
                            internal_paths[internal_path_ind] += [parent]
                edge_len = self.ts.node(parent_nodes[0]).time - node.time
                cov_mat[ np.ix_( path_ind, path_ind ) ] += edge_len
                indices[parent] += path_ind
                if len(internal_nodes) != 0:
                    int_nodes_update = []
                    for i in path_ind: 
                        int_nodes_update += internal_indices[i]
                    shared_time[ np.ix_( int_nodes_update, path_ind) ] += edge_len  
            elif nparent == 2 :
                parent1 = parent_nodes[0]
                parent1_ind = []
                parent2 = parent_nodes[1] 
                parent2_ind = []
                for (i,path) in enumerate(path_ind):
                    if i == 0:
                        paths[path].append(parent1)
                        parent1_ind += [ path ]
                        paths.append(paths[path][:])
                        paths[-1][-1] = parent2
                        parent2_ind += [ len(cov_mat) ]
                        cov_mat = np.hstack(  (cov_mat, cov_mat[:,path].reshape(cov_mat.shape[0],1) )) #Duplicate the column
                        cov_mat = np.vstack(  (cov_mat, cov_mat[path,:].reshape(1,cov_mat.shape[1]) )) #Duplicate the row
                        if len(internal_nodes) != 0:
                            shared_time = np.hstack(  (shared_time, shared_time[:,path].reshape(shared_time.shape[0],1) )) #Duplicate the column
                    elif i%2 == 0: 
                        paths[path].append(parent1)
                        parent1_ind += [path]
                    elif i%2 == 1: 
                        paths[path].append(parent2)
                        parent2_ind += [path]
                    else: 
                        raise RuntimeError("Path index is not an integer")
                edge_len = self.ts.node(parent_nodes[0]).time - node.time
                cov_mat[ np.ix_( parent1_ind + parent2_ind, parent1_ind + parent2_ind  ) ] += edge_len 
                indices[parent1] += parent1_ind
                indices[parent2] += parent2_ind 
                if len(internal_nodes) != 0:
                    int_nodes_update = []
                    for i in path_ind: 
                        int_nodes_update += internal_indices[i]
                    shared_time[ np.ix_( int_nodes_update, parent1_ind + parent2_ind) ] += edge_len 
            else : 
                print(node, parent_nodes)
                raise RuntimeError("Nodes has more than 2 parents")       
        if len(internal_nodes) != 0:
            return cov_mat, paths, shared_time, internal_paths
        else:
            return cov_mat, paths
        
    def expand_locations(self):
        """Converts individuals' locations to sample locations to start of paths locations.

        TODO: This should handle if the samples are not organized first in the node table. Need to check.

        Parameters
        ----------
        locations_of_individuals : dict
            Geographic locations of each individual
        ts : tskit.trees.TreeSequence
        paths : list
            List of paths from samples to roots

        Returns
        -------
        locations_of_path_starts : numpy.ndarray
            Geographic locations of the tips of each path
        locations_of_samples : numpy:ndarray
            Geographic locations of each sample
        """

        locations_of_samples = {}
        for node in self.ts.nodes():
            if node.flags == 1:
                locations_of_samples[node.id] = self.locations_of_individuals[node.individual]
        locations_of_path_starts = []
        for path in self.paths:
            locations_of_path_starts.append(locations_of_samples[path[0]])
        locations_of_path_starts = np.array(locations_of_path_starts)
        if len(locations_of_path_starts.shape) == 1:
            raise RuntimeError("Path locations vector is missing number of columns. Cannot process.")
        return locations_of_path_starts, locations_of_samples

    def build_roots_array(self):
        """Builds the roots array ("R" in the manuscript)

        The roots array associates paths with roots; this is specifically important if there is not a
        grand most recent common ancestor (GMRCA) for the ARG.

        Parameters
        ----------
        paths : list
            List of paths from samples to respective roots through the ARG. Each path includes the
            ID of the nodes that it passes through in order from youngest to oldest.

        Returns
        -------
        roots_array : numpy.ndarray
            Each row is associated with a path and each column is associated with a root. R_ij will
            have a 1 if the ith path has the jth root
        unique_roots : np.ndarray
            Array of unique roots in the ARG, sorted by ID
        """

        roots = [row[-1] for row in self.paths]
        unique_roots = np.unique(roots)
        roots_array = np.zeros((len(self.paths), len(unique_roots)))#, dtype=np.float64)
        for i,root in enumerate(unique_roots): 
            for path in np.where(roots == root)[0]:
                roots_array[path][i] += 1.0
        return roots_array, unique_roots

    def locate_roots(self, roots_array, locations_of_path_starts):
        """Calculate the maximum likelihood locations of the roots of the ARG.

        TODO: Need tests for these different scenarios to ensure that this is all correct

        Parameters
        ----------
        inverted_cov_mat : numpy.ndarray
            Inverted shared time matrix between paths
        roots_array : numpy.ndarray
            Matrix that associates roots to specific paths
        locations_of_path_starts : numpy.ndarray
            Matrix that associates tip locations to specific paths
        
        Returns
        -------
        np.ndarray
            Locations of root associated with each path
        """

        A = np.matmul(np.transpose(roots_array),np.matmul(self.inverted_paths_shared_time_matrix, roots_array)) #Matrix of coefficients of the system of linear equations 
        b = np.matmul(np.transpose(roots_array),np.matmul(self.inverted_paths_shared_time_matrix, locations_of_path_starts)) #Vector of constants of the system of linear equations. 
        augmented_matrix = np.column_stack((A, b)) # Construct the augmented matrix [A|b]
        rre_form, pivots = sym.Matrix(augmented_matrix).rref() # Perform row reduction on the augmented matrix
        if int(A.shape[0]) in pivots:
            raise RuntimeError("Cannot locate roots. No solution to system of linear equations.")
        else:
            if len(pivots) != A.shape[0]:
                print("Multiple solutions to system of linear equations in root location calculation.")
                warnings.warn("Multiple solutions to system of linear equations in root location calculation.")
            return np.array(rre_form.col(range(-locations_of_path_starts.shape[1],0)), dtype=np.float64)


#### ESTIMATING LOCATIONS

def estimate_location_and_variance(sigma_squared, s_a, inverted_cov_mat, sample_locs_to_root_locs, u_a, t_a):
    """

    Parameters
    ----------
    sigma_squared : numpy.ndarray
        Dispersal rate matrix
    s_a : numpy.ndarray
        One dimensional array
    inverted_cov_mat : numpy.ndarray
        Inverted shared time matrix between paths
    sample_locs_to_root_locs : numpy.ndarray
        Array that contains the difference in location between the tip and root of a path 
    u_a : numpy.ndarray
        Location of this ancestor's root
    t_a : float
        Time of this ancestor

    Returns
    -------
    ancestor_location :
    variance_in_ancestor_location :
    """

    matmul_prod = np.matmul(s_a, inverted_cov_mat)
    ancestor_location = u_a + np.matmul(matmul_prod, sample_locs_to_root_locs)
    explained_variance = np.matmul(matmul_prod, np.transpose(s_a))
    ones = np.ones(inverted_cov_mat.shape[0])
    unexplained_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
    unexplained_numerator = (1-np.matmul(np.matmul(np.transpose(s_a),inverted_cov_mat),ones))**2
    corrected_variance_scaling_factor = t_a-explained_variance+(unexplained_numerator/unexplained_denominator)
    variance_in_ancestor_location = sigma_squared*corrected_variance_scaling_factor
    return ancestor_location, variance_in_ancestor_location

def find_nearest_ancestral_nodes_at_time(tree, u, time):
    """Find the nearest ancestral nodes of a sample within a tree at a specified time.

    Parameters
    ----------
    tree : tskit.Tree
    u : int
        The ID for the node of interest
    time : int or float
        timing of the ancestral node of interest

    Returns
    -------
    u : int
        Node ID of the ancestral node above specified point
    v : int
        Node ID of the ancestral node below specified point
    """

    v = u
    u = tree.parent(u)
    while u != -1:
        if tree.time(u) >= time:
            return u, v
        v = u
        u = tree.parent(u)
    return None, v

def track_sample_ancestor(row, spatial_arg, label=""):
    """Estimate the location of a sample's ancestor from a pandas.Series or dictionary

    This is useful when applied to each row from the pandas.DataFrame output by
    `create_ancestors_dataframe()`.

    Parameters
    ----------
    row : pandas.Series or dict
        Must have key: sample, interval_left, and time
    ts : tskit.TreeSequence
    spatial_parameters : tuple
        Output from `estimate_spatial_parameters()`. This includes the dispersal rate matrix, paths,
        root locations, etc.

    Returns
    -------
    pandas.Series
        Columns for estimated locations and variances around this estimate
    """

    above, below = find_nearest_ancestral_nodes_at_time(tree=spatial_arg.ts.at(row["interval_left"]), u=int(row["sample"]), time=row["time"])
    ancestor_specific_sharing = spatial_arg.node_paths_shared_times[above].copy()
    root_location = spatial_arg.root_locations[spatial_arg.node_paths[above][-1]]
    additional_time = spatial_arg.ts.node(above).time - row["time"]
    for i,path in enumerate(spatial_arg.paths):
        if below in path:
            ancestor_specific_sharing[i] += additional_time
    ancestor_location, variance_in_ancestor_location = estimate_location_and_variance(
        sigma_squared=spatial_arg.dispersal_rate_matrix,
        s_a=ancestor_specific_sharing,
        inverted_cov_mat=spatial_arg.inverted_paths_shared_time_matrix,
        sample_locs_to_root_locs=spatial_arg.path_dispersal_distances,
        u_a=root_location,
        t_a=spatial_arg.ts.max_root_time-row["time"]
    )
    output = []
    indices = []
    if label != "":
        label += "_"
    for i,loc in enumerate(ancestor_location):
        output.append(loc)
        output.append(variance_in_ancestor_location[i][i])
        indices.append(label + "estimated_location_"+str(i))
        indices.append(label + "variance_in_estimated_location_"+str(i))
    return pd.Series(output, index=indices)

def estimate_locations_of_ancestors_in_dataframe_using_arg(df, spatial_arg):
    """
    """
    
    df = pd.concat([df, df.apply(track_sample_ancestor, axis=1, spatial_arg=spatial_arg, label="arg")], axis=1)
    return df

def track_sample_ancestor_in_tree(row, spatial_arg):
    tree = spatial_arg.ts.keep_intervals(np.array([[row["interval_left"], row["interval_right"]]]), simplify=False).trim()
    tree = remove_unattached_nodes(ts=tree)
    spatial_tree = SpatialARG(ts=tree)
    row["interval_left"] = 0
    return track_sample_ancestor(row=row, spatial_arg=spatial_tree, label="tree")

def estimate_locations_of_ancestors_in_dataframe_using_tree(df, spatial_arg):
    df = pd.concat([df, df.apply(track_sample_ancestor_in_tree, axis=1, spatial_arg=spatial_arg)], axis=1)
    return df
 

#### Comparison with Wohns et al.

def calc_midpoint_node_locations(ts, weighted = True):
    """
    """

    node_locations = {}
    for sample in ts.samples():
        node_locations[sample] = ts.individual(ts.node(sample).individual).location
    node_times = {}
    for node in ts.nodes():
        node_times[node.id] = node.time
    for node in ts.nodes(order="timeasc"):
        if not node.is_sample():
            children = ts.tables.edges.child[np.where(ts.tables.edges.parent == node.id)]
            if len(children) > 1:
                locations = [[dimension] for dimension in node_locations[children[0]]]
                for child in children[1:]:
                    for dimension, location in enumerate(node_locations[child]):
                        locations[dimension].append(location)
                weights = [1 for child in children]
                if weighted:
                    weights =  [ 1.0/(node_times[node.id] - node_times[child]) for child in children ]
                    node_times[node.id] -= 1.0/sum(weights)
                averaged_locations = []
                for dimension in locations:
                    averaged_locations.append(np.average(dimension, weights = weights))
                node_locations[node.id] = np.array(averaged_locations)
            elif len(children) == 1:
                node_locations[node.id] = node_locations[children[0]]
            else:
                raise RuntimeError("Non-sample node doesn't have child.")
    return node_locations

def midpoint_locations(row, succinct_ts, node_locations, dimensions=2, label="midpoint"):
    """
    """

    above, below = find_nearest_ancestral_nodes_at_time(tree=succinct_ts.at(row["interval_left"]), u=int(row["sample"]), time=row["time"])
    ancestor_location = []
    if above == None:
        ancestor_location = node_locations[below][:dimensions]
    else:
        for d in range(dimensions):
            ancestor_location.append((row["time"]-succinct_ts.node(below).time)*((node_locations[below][d]-node_locations[above][d])/(succinct_ts.node(below).time-succinct_ts.node(above).time))+node_locations[below][d])
    output = []
    indices = []
    if label != "":
        label += "_"
    for i,loc in enumerate(ancestor_location):
        output.append(loc)
        indices.append(label + "estimated_location_"+str(i))
    return pd.Series(output, index=indices)

def estimate_locations_of_ancestors_in_dataframe_using_midpoint(df, spatial_arg, simplify=False):
    if simplify:
        ts = spatial_arg.ts.simplify()
    else:
        ts = spatial_arg.ts
    node_locations = calc_midpoint_node_locations(ts=ts, weighted=False)
    df = pd.concat([df, df.apply(midpoint_locations, axis=1, succinct_ts=ts, node_locations=node_locations)], axis=1)
    return df


#### UNUSED?
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

def locate_nodes(ts, spatial_parameters):
    """Locate all nodes in 
    """

    node_locations = []
    for node in range(len(spatial_parameters[9])):
        location, variance = estimate_location_and_variance(
            sigma_squared=spatial_parameters[0],
            s_a=spatial_parameters[8][node],
            inverted_cov_mat=spatial_parameters[3],
            sample_locs_to_root_locs=spatial_parameters[6],
            u_a=spatial_parameters[7][spatial_parameters[9][node][-1]],
            t_a=ts.max_root_time-ts.node(node).time
        )
        output = []
        indices = []
        output.append(node)
        indices.append("id")
        output.append(ts.node(node).time)
        indices.append("time")
        true_location = ts.individual(ts.node(node).individual).location
        for i,loc in enumerate(location):
            output.append(true_location[i])
            indices.append("true_location_"+str(i))
            output.append(loc)
            indices.append("estimated_location_"+str(i))
            output.append(variance[i][i])
            indices.append("variance_in_estimated_location_"+str(i))
        node_locations.append(pd.Series(output, index=indices))
    return pd.DataFrame(node_locations, columns=indices)

def create_ancestors_dataframe(ts, samples, timestep=1, include_locations=False, dimensions=2):
    """Creates a pandas.DataFrame with each row corresponding to an ancestor of a sample

    Each ancestor is at a specific time in the past and corresponds to a region of the chromosome.

    TODO: Test in cases where include_locations=True, but there aren't locations in the TreeSequence
    TODO: Check what happens when a non-sample node is passed into samples list

    Parameters
    ----------
    ts : tskit.TreeSequence
    samples : list
        List of sample node IDs
    timestep (optional) : int
        Determines how often ancestors are measured.
    include_locations : boolean
        Whether to include columns for the true ancestor locations. Default is False.
    dimensions (optional) : int
        Number of dimensions to include in reconstruction. This only matters is include_locations=True.
        Default is 2.

    Returns
    -------
    df : pandas.DataFrame
        Each row in the DataFrame corresponds with a sample's ancestor
    """

    sample = []
    interval_left = []
    interval_right = []
    time = []
    location = []
    for node in samples:
        just_node, map = ts.simplify(samples=[node], map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
        for tree in just_node.trees():
            path = [0] + list(ancestors(tree, 0))
            for i,n in enumerate(path):
                path[i] = np.argwhere(map==n)[0][0]
            for i,n in enumerate(path):
                node_time = ts.node(n).time
                if node_time % timestep == 0:
                    sample.append(node)
                    interval_left.append(tree.interval.left)
                    interval_right.append(tree.interval.right)
                    time.append(node_time)
                    indiv = ts.node(n).individual
                    if indiv != -1:
                        location.append(ts.individual(indiv).location[:dimensions])
                    else:
                        location.append([None for d in range(dimensions)])
    df = pd.DataFrame({
        "sample":sample,
        "interval_left":interval_left,
        "interval_right":interval_right,
        "time":time,
    })
    if include_locations:
        locs = pd.DataFrame(location, columns=["true_location_"+str(d) for d in range(dimensions)])
        df = pd.concat([df, locs], axis=1)
    return 



def ancestors(tree, u):
    """Find all of the ancestors above a node for a tree

    Taken directly from https://github.com/tskit-dev/tskit/issues/2706

    Parameters
    ----------
    tree : tskit.Tree
    u : int
        The ID for the node of interest

    Returns
    -------
    An iterator over the ancestors of u in this tree
    """

    u = tree.parent(u)
    while u != -1:
         yield u
         u = tree.parent(u)

def get_paths_for_nodes(ts, nodes):
    """Identifies all of the paths that correspond to specific regions of the chromosome for a list of nodes.

    These are the paths that are most interesting to track as they follow inherited
    material. This will not identify non-genetic ancestors (for that, use networkx method).

    Parameters
    ----------
    ts : tskit.TreeSequence
    nodes : list
        List of nodes for which you want paths above
    
    Returns
    -------
    paths : list
        List of lists that are each a path from a node to the root
    """

    paths = []
    for tree in ts.trees():
        for sample in nodes:
            path = [sample] + list(ancestors(tree, sample))
            if path not in paths:
                paths.append(path)
    return paths

def get_paths_for_node(ts, node):
    """
    Identifies all of the paths that correspond to specific regions of the chromosome for a single node.

    These are the paths that are most interesting to track as they follow inherited
    material. This will not identify non-genetic ancestors (for that, use networkx method).

    Parameters
    ----------
    ts : tskit.TreeSequence
    nodes : int
        ID of node for which you want paths above
    
    Returns
    -------
    paths : list
        List of lists that are each a path from a node to the root
    """

    just_node, map = ts.simplify(samples=[node], map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
    paths = []
    for tree in just_node.trees():
        path = [0] + list(ancestors(tree, 0))
        for i,n in enumerate(path):
            path[i] = np.argwhere(map==n)[0][0]
        paths.append(path)
    return paths
