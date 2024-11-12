import numpy as np
from collections import defaultdict
import sympy as sym
import warnings
from tqdm import tqdm
import time
import pandas as pd
import random

from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()


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
    if tree.time(u) == time:
        return u
    u = tree.parent(u)
    while u != -1:
        node_time = tree.time(u)
        if node_time == time:
            return u
        u = tree.parent(u)
    warnings.warn(f"Sample %s does not have an ancestral node at time %s in tree. Returning None." % (u, time))

def generate_random_ancestors_dataframe(ts, number_of_ancestors, include_locations=False, dimensions=2, cutoff=-1, seed=None):
    """Creates a dataframe of random genetic ancestors within an ARG
    
    This function needs to run on the unsimplified ARG which has all of the location information if you want to compared
    estimated against true values. This info is lost during the simplification step.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tskit tree sequence
    number_of_ancestors : int
        Number of random ancestors to create
    include_locations : bool
        Where to include locations column(s) in the output pandas.DataFrame. Default is False.
    dimensions : int
        Number of spatial dimensions to run on. Default is 2.
    cutoff : int
        Time cutoff for the genetic ancestors. Ancestors must be younger than the cutoff. Default is -1, and ignored.
    seed : int
        Seed for `random` package functions. Default is None, and ignored.
    
    Returns
    -------
    df : pandas.DataFrame
        Output pandas.DataFrame containing all of the random genetic ancestors, one per row.
    """
    
    if cutoff > ts.max_root_time: # check that cutoff isn't further than max_root_time
        warnings.warn(f"Provided cutoff %s is greater than the max root time %s. Using max root time instead." % (cutoff, ts.max_root_time))
        cutoff = int(ts.max_root_time)
    if seed != None:
        random.seed(seed)
    samples = []
    genome_positions = []
    times = []
    location = []
    for n in range(number_of_ancestors):
        sample_time = -1
        while (sample_time == -1) or (sample_time > cutoff):
            sample = random.randint(0, ts.num_samples-1)
            sample_time = int(ts.node(sample).time)
        genome_pos = random.uniform(0, ts.sequence_length)
        if cutoff == -1:
            time = random.randint(sample_time, ts.max_root_time)
        elif cutoff >= sample_time:
            time = random.randint(int(ts.node(sample).time), cutoff)
        else:
            raise RuntimeError(f"Sample %s is older than the cutoff set. This shouldn't be possible...")
        tree = ts.at(genome_pos)
        ancestor = find_ancestral_node_at_time(tree, sample, time)
        samples.append(sample)
        genome_positions.append(genome_pos)
        times.append(time)
        indiv = ts.node(ancestor).individual
        if indiv != -1:
            location.append(ts.individual(indiv).location[:dimensions])
        else:
            location.append([None for d in range(dimensions)])
    df = pd.DataFrame({
        "sample":samples,
        "genome_position":genome_positions,
        "time":times,
    })
    if include_locations:
        locs = pd.DataFrame(location, columns=["true_location_"+str(d) for d in range(dimensions)])
        df = pd.concat([df, locs], axis=1)
    return df

def simplify_with_recombination(ts, flag_recomb=False, keep_nodes=None):
    """Simplifies a tree sequence while keeping recombination nodes

    Removes unary nodes that are not recombination nodes. Does not remove non-genetic ancestors.
    Edges intervals are not updated. This differs from how tskit's TreeSequence.simplify() works.

    Parameters
    ----------
    ts : tskit.TreeSequence
    flag_recomb (optional) : bool
        Whether to add msprime node flags. Default is False.
    keep_nodes (optional) : list
        List of node IDs that should be kept. Default is None, so empty list.

    Returns
    -------
    ts_sim : tskit.TreeSequence
        Simplified tree sequence
    maps_sim : numpy.ndarray
        Mapping for nodes in the simplified tree sequence versus the original
    """

    if keep_nodes == None:
        keep_nodes = []

    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0)
    child_node, parents_count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has.
    parent_node, children_count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has.
    multiple_parents = child_node[parents_count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    
    if flag_recomb:
        ts_tables = ts.dump_tables()
        node_table = ts_tables.nodes
        flags = node_table.flags
        flags[recomb_nodes] = 131072 #msprime.NODE_IS_RE_EVENT
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
    A tskit.TreeSequence with individuals' locations and its corresponding attributes needed to calculate
    related spatial parameters, such as dispersal rate and location of ancestors.

    Attributes
    ----------
    ts : tskit.TreeSequence
    locations_of_individuals : dict
    paths_shared_time_matrix
    paths
    node_paths_shared_times
    node_paths
    inverted_paths_shared_time_matrix
    roots
    roots_array
    root_locations
    path_dispersal_distances
    dispersal_rate_matrix
    fishers_information_1
    fishers_information_2
    """
    
    def __init__(self, ts, locations_of_individuals=None, dimensions=2, verbose=False):
        total_start_time = time.time()

        section_start_time = time.time()
        self.ts = ts
        if locations_of_individuals == None:  # if user doesn't provide a separate locations dictionary, builds one
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
        self.roots = roots
        self.roots_array = roots_array
        self.root_covariance_matrix = np.linalg.pinv(np.matmul(np.matmul(np.transpose(roots_array), self.inverted_paths_shared_time_matrix), roots_array))
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

def estimate_location_and_variance(sigma_squared, s_a, inverted_cov_mat, sample_locs_to_root_locs, u_a, t_a, roots_array, e_ra, root_cv):
    """Estimate the location and variance of a given genetic ancestor

    Uses the shared time between that ancestor and the paths through the ARG to
    calculate the covariance in locations.

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
    s_a = s_a[:,None]
    matmul_prod = np.matmul(np.transpose(s_a), inverted_cov_mat)
    ancestor_location = (u_a + np.matmul(matmul_prod, sample_locs_to_root_locs))[0]
    explained_variance = np.matmul(matmul_prod, s_a)
    ones = np.ones(inverted_cov_mat.shape[0])
    correction_3 = e_ra - np.matmul(np.matmul(np.transpose(roots_array), inverted_cov_mat), s_a)
    correction_1 = np.transpose(correction_3)
    correction_factor = np.matmul(np.matmul(correction_1, root_cv), correction_3)
    corrected_variance_scaling_factor = t_a-explained_variance+correction_factor
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
    if (u != -1) and (tree.time(v) == time):
        return u, v
    while u != -1:
        if tree.time(u) >= time:
            return u, v
        v = u
        u = tree.parent(u)
    return None, v

def estimate_locations_of_ancestors_in_dataframe_using_arg(df, spatial_arg, verbose=False):
    """Estimates the locations of genetic ancestors in dataframe using the full chromosome ARG

    Parameters
    ----------
    df : pandas.DataFrame
    spatial_arg : sparg.SpatialARG

    Returns
    ------
    df : pandas.DataFrame
    """

    df.loc[:, "position_in_arg"] = df.loc[:, "genome_position"]
    if verbose:
        df = pd.concat([df, df.progress_apply(track_sample_ancestor, axis=1, label="arg", use_this_arg=spatial_arg)], axis=1)
    else:
        df = pd.concat([df, df.apply(track_sample_ancestor, axis=1, label="arg", use_this_arg=spatial_arg)], axis=1)
    return df

def get_window_bounds(genome_pos, spatial_arg, window_size):
    """Calculates the left and right boundaries for a window of given size

    Parameters
    ----------
    genome_pos : int
        Basepair position of the genetic ancestor
    spatial_arg : sparg.SpatialARG
        The spatial ARG of interest
    window_size : int
        Number of neighboring trees on either side of the local tree
    
    Returns
    -------
    left : int
        Basepair position for the left side of the window
    right : int
        Basepair position for the right side of the window
    """

    if isinstance(genome_pos, tuple):
        center_l = spatial_arg.ts.at(genome_pos[0]).index
        center_r = spatial_arg.ts.at(genome_pos[1]).index
    else:
        center_l = spatial_arg.ts.at(genome_pos).index
        center_r = center_l
    num_trees = spatial_arg.ts.num_trees
    if center_l - window_size > 0:
        left = spatial_arg.ts.at_index(center_l-window_size).interval.left
    else:
        left = spatial_arg.ts.at_index(0).interval.left
    if center_r + window_size < num_trees-1:
        right = spatial_arg.ts.at_index(center_r+window_size).interval.right
    else:
        right = spatial_arg.ts.at_index(num_trees-1).interval.right
    return left, right


def track_sample_ancestor(row, label="", use_this_arg="", spatial_arg="", use_theoretical_dispersal=False, duped_arg_dict=None, dimensions=2):
    """Estimate the location of a sample's ancestor from a pandas.Series or dictionary

    This is useful when applied to each row from the pandas.DataFrame output by
    `create_ancestors_dataframe()`.

    Parameters
    ----------
    row : pandas.Series or dict
        Must have key: sample, interval_left, and time
    label : str
        Label used to identify the computed columns. Default is "", and ignored.
    use_this_arg : sparg.spatialARG
        Specifies the ARG to use. Default is "", and ignored.
    spatial_arg : sparg.SpatialARG
        Base ARG that can be chopped into windows. Default is "", and ignored.
    use_theoretical_dispersal : bool
        Whether to use the expected dispersal rate for the simulation (for our simulation it is 0.25*0.25+0.5). Default is False.
    duped_arg_dict : dict
        Precomputed ARGs that are used more than once. Default is {}, empty.

    Returns
    -------
    pandas.Series
        Columns for estimated locations and variances around this estimate
    """

    if duped_arg_dict == None:
        duped_arg_dict = {}

    if use_this_arg != "":
        arg = use_this_arg
    elif spatial_arg != "":
        if row["interval"] in duped_arg_dict:
            arg = duped_arg_dict[row["interval"]]
        else:
            arg = retrieve_arg_for_window((row["interval"][0], row["interval"][1]), spatial_arg=spatial_arg, use_theoretical_dispersal=use_theoretical_dispersal, dimensions=dimensions)["arg"]
    else:
        raise RuntimeError("No ARG provided.")
    above, below = find_nearest_ancestral_nodes_at_time(tree=arg.ts.at(row["position_in_arg"]), u=int(row["sample"]), time=row["time"])
    ancestor_specific_sharing = arg.node_paths_shared_times[above].copy()
    root_location = arg.root_locations[arg.node_paths[above][-1]]
    root_index = np.where(arg.roots==arg.node_paths[above][-1])[0][0]
    ancestor_specific_root = np.zeros(shape=(len(arg.roots), 1))
    ancestor_specific_root[root_index] = 1
    additional_time = arg.ts.node(above).time - row["time"]
    for i,path in enumerate(arg.paths):
        if below in path:
            ancestor_specific_sharing[i] += additional_time
    ancestor_location, variance_in_ancestor_location = estimate_location_and_variance(
        sigma_squared=arg.dispersal_rate_matrix,
        s_a=ancestor_specific_sharing,
        inverted_cov_mat=arg.inverted_paths_shared_time_matrix,
        sample_locs_to_root_locs=arg.path_dispersal_distances,
        u_a=root_location,
        t_a=arg.ts.max_root_time-row["time"],
        roots_array=arg.roots_array,
        e_ra=ancestor_specific_root,
        root_cv=arg.root_covariance_matrix
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

def retrieve_arg_for_window(interval, spatial_arg, use_theoretical_dispersal=False, dimensions=2):
    """Calculates the sparg.SpatialARG for a specified window
    
    This is useful to avoid redundant calculations for the same window. Stored as a pandas.Series.

    Parameters
    ----------
    interval : tuple or list
        The bounds of the window (left, right)
    spatial_arg : sparg.SpatialARG
        ARG to use
    use_theoretical_dispersal : bool
        Whether to use the expected dispersal rate for the simulation (for our simulation it is 0.25*0.25+0.5). Default is False.

    Returns
    -------
    pd.Series
        Contains the window and its associated sparg.SpatialARG
    """

    tree = spatial_arg.ts.keep_intervals(np.array([[interval[0], interval[1]]]), simplify=False).trim()
    tree = remove_unattached_nodes(ts=tree)
    spatial_tree = SpatialARG(ts=tree, dimensions=dimensions)
    if use_theoretical_dispersal:
        spatial_tree.dispersal_rate_matrix = np.array([[0.25*0.25+0.5,0],[0,0.25*0.25+0.5]])
    return pd.Series({"interval": interval, "arg": spatial_tree})

def estimate_locations_of_ancestors_in_dataframe_using_window(df, spatial_arg, window_size, use_theoretical_dispersal=False, verbose=False, dimensions=2):
    """
    
    Note: There may be a way to do this without applying to pd.DataFrame twice (caching?) but this
    isn't too much of a concern for the relatively small pd.DataFrames that we are working with.

    Parameters
    ----------
    df : pandas.DataFrame
        Contains the genetic ancestors to be estimated
    spatial_arg : sparg.SpatialARG
        SpatialARG containing the covariances between paths
    window_size : int
        Number of neighboring trees on either side of the local tree
    use_theoretical_dispersal : bool
        Whether to use the expected dispersal rate for the simulation (for our simulation it is 0.25*0.25+0.5). Default is False.
    verbose : bool
    dimension : int
        Number of dimensions to investigate. Default is 2.
    
    Returns
    -------
    df : pandas.DataFrame
        Contains the genetic ancestors and their estimated windows
    """
    
    if "starting_window" in df:
        if window_size < 0:
            if window_size == -1:
                if verbose:
                    intervals = df["genome_position"].progress_apply(get_window_bounds, spatial_arg=spatial_arg, window_size=0)
                else:
                    intervals = df["genome_position"].apply(get_window_bounds, spatial_arg=spatial_arg, window_size=0)
            else:
                raise RuntimeError("Window sizes can only be >= -1 when starting window provided.")
        elif verbose:
            intervals = df["starting_window"].progress_apply(get_window_bounds, spatial_arg=spatial_arg, window_size=window_size)
        else:
            intervals = df["starting_window"].apply(get_window_bounds, spatial_arg=spatial_arg, window_size=window_size)
    elif window_size < 0:
        raise RuntimeError("Cannot provide negative window size if no starting window provided.")
    elif verbose:
        intervals = df["genome_position"].progress_apply(get_window_bounds, spatial_arg=spatial_arg, window_size=window_size)
    else:
        intervals = df["genome_position"].apply(get_window_bounds, spatial_arg=spatial_arg, window_size=window_size)
    intervals.name = "interval"
    with_windows = pd.concat([df, intervals], axis=1)
    # check if interval is used more than once...
    if verbose:
        duped_args = intervals[intervals.duplicated()].drop_duplicates().progress_apply(
            retrieve_arg_for_window,
            spatial_arg=spatial_arg,
            use_theoretical_dispersal=use_theoretical_dispersal,
            dimensions=dimensions
        )
    else:
        duped_args = intervals[intervals.duplicated()].drop_duplicates().apply(
            retrieve_arg_for_window,
            spatial_arg=spatial_arg,
            use_theoretical_dispersal=use_theoretical_dispersal,
            dimensions=dimensions
        )
    duped_dict = dict(zip(duped_args["interval"], duped_args["arg"]))
    with_windows["position_in_arg"] = with_windows["genome_position"] - with_windows["interval"].str[0]
    if verbose:
        df = pd.concat([df, with_windows.progress_apply(
            track_sample_ancestor,
            axis=1,
            label="window_"+str(window_size),
            spatial_arg=spatial_arg,
            use_theoretical_dispersal=True,
            duped_arg_dict=duped_dict,
            dimensions=dimensions
        )], axis=1)
    else:
        df = pd.concat([df, with_windows.apply(
            track_sample_ancestor,
            axis=1,
            label="window_"+str(window_size),
            spatial_arg=spatial_arg,
            use_theoretical_dispersal=True,
            duped_arg_dict=duped_dict,
            dimensions=dimensions
        )], axis=1)
    return df


#### Comparison with Wohns et al.

def calc_midpoint_node_locations(ts, weighted=True):
    """Estimates node locations using averaging-up method

    Parent node location is the average of its children, potentially weighted by the length of edges
    to each child.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Succinct tree sequence
    weighted : bool
        Whether to weight by the edge length. Default is True.

    Returns
    -------
    node_locations : dict
        Key is node ID and value is the coordinates as a list
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
    """Calculates the location of a genetic ancestor using the averaging-up (midpoint) method

    Genetic ancestors can be between nodes within the ARG.

    Parameters
    ----------
    row : pandas.Series

    succinct_ts : tskit.TreeSequence
        Simplified tree sequence. Note: this differs from the full ARG tskit.TreeSequence that is more
        commonly used.
    node_locations : dict
        Contains the locations of each node in the ARG.
    dimensions : int
        Number of dimensions to calculate. Default is 2.
    label : str
        The column label for the output coordinates. Default is "midpoint"

    Returns
    -------
    pandas.Series
        Estimated coordinates of the genetic ancestor
    """

    above, below = find_nearest_ancestral_nodes_at_time(tree=succinct_ts.at(row["genome_position"]), u=int(row["sample"]), time=row["time"])
    ancestor_location = []
    if above == None:
        ancestor_location = node_locations[below][:dimensions]
    elif below == None:
        raise RuntimeError(f"Nothing below node. %s" % (row))
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

def estimate_locations_of_ancestors_in_dataframe_using_midpoint(df, spatial_arg, simplify=False, dimensions=2):
    """Applying the averaging-up (midpoint) method to a dataframe of genetic ancestors

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of genetic ancestors
    spatial_arg : sparg.SpatialARG
    simplify : bool
        Whether to simplify the ARG or not. Default is False.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of genetic ancestors with estimated locations
    """

    if simplify:
        ts = spatial_arg.ts.simplify()
    else:
        ts = spatial_arg.ts
    node_locations = calc_midpoint_node_locations(ts=ts, weighted=False)
    df = pd.concat([df, df.apply(midpoint_locations, axis=1, succinct_ts=ts, node_locations=node_locations, dimensions=dimensions)], axis=1)
    return df



#### Two Pops

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

def create_recombination_event_dataframe(ts, breakpoint, samples, timestep=1, include_locations=False, dimensions=2):
    """Creates a dataframe of random genetic ancestors within an ARG
    
    This function needs to run on the unsimplified ARG which has all of the location information if you want to compared
    estimated against true values. This info is lost during the simplification step.

    Parameters
    ----------
    ts : tskit.TreeSequence
    breakpoint : int
        Basepair position for breakpoint of interest
    samples : list
        Samples IDs to track
    timestep : int
        Timestep between genetic ancestors tracked back in time. Default is 1.
    include_locations : bool
        Whether to include columns for the true locations of genetic ancestors.
    dimensions : int
        Number of dimensions to calculate. Default is 2.

    Returns
    -------
    df : pandas.DataFrame
        Genetic ancestors with estimated locations
    """

    sample = []
    genome_positions = []
    starting_windows = []
    time = []
    location = []
    for node in samples:
        just_node, map = ts.simplify(samples=[node], map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
        for pos in [breakpoint-1, breakpoint+1]:
            tree = just_node.at(pos)
            path = [0] + list(ancestors(tree, 0))
            for i,n in enumerate(path):
                path[i] = np.argwhere(map==n)[0][0]
            for i,n in enumerate(path):
                node_time = ts.node(n).time
                if node_time % timestep == 0:
                    sample.append(node)
                    genome_positions.append(pos)
                    starting_windows.append((breakpoint-1, breakpoint+1))
                    time.append(node_time)
                    indiv = ts.node(n).individual
                    if indiv != -1:
                        location.append(ts.individual(indiv).location[:dimensions])
                    else:
                        location.append([None for d in range(dimensions)])
    df = pd.DataFrame({
        "sample":sample,
        "genome_position":genome_positions,
        "starting_window":starting_windows,
        "time":time,
    })
    if include_locations:
        locs = pd.DataFrame(location, columns=["true_location_"+str(d) for d in range(dimensions)])
        df = pd.concat([df, locs], axis=1)
    return df