import numpy as np
from collections import defaultdict
import sympy as sym
import warnings
from tqdm import tqdm
import time
import pandas as pd
import msprime


def get_tskit_locations(ts, dimensions=2):
    """Converts the tskit individuals locations into a dictionary.

    Parameters
    ----------
    ts : tskit.trees.TreeSequence
        This must be a tskit Tree Sequences with marked recombination nodes, as is outputted by
        msprime.sim_ancestry(..., record_full_arg=True). Must include locations within the
        individuals table.

    Returns
    -------
    locations_of_individuals : dictionary
        Dictionary of sample node locations where the key is the node ID and the value is a
        numpy.ndarray or list with the node's location.

    """

    if len(ts.tables.individuals.location) == 0:
        raise RuntimeError("Locations of individuals not provided.")
    locations = np.array_split(ts.tables.individuals.location, ts.num_individuals)
    locations_of_individuals = {}
    for i,location in enumerate(locations):
        locations_of_individuals[i] = location[:dimensions]
    return locations_of_individuals

def calc_minimal_covariance_matrix(ts, internal_nodes=[], verbose=False):
    """Calculates a covariance matrix between the minimal number of paths in the the ARG. Should always produce an invertible matrix 

    Parameters
    ----------
    ts : tskit.trees.TreeSequence
        This must be a tskit Tree Sequences with marked recombination nodes, as is outputted by
        msprime.sim_ancestry(..., record_full_arg=True). The covariance matrix will not be
        correct if the recombination nodes are not marked.
    internal_nodes : list 
        A list of internal nodes for which you want the shared times. Default is an empty list,
        in which case no internal nodes will be calculated.

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
    
    edges = ts.tables.edges
    cov_mat = np.zeros(shape=(ts.num_samples, ts.num_samples))#, dtype=np.float64)  #Initialize the covariance matrix. Initial size = #samples. Will increase to #paths
    indices = defaultdict(list) #Keeps track of the indices of paths that enter (from bottom) a particular node.
    paths = []
    for i, sample in enumerate(ts.samples()):
        indices[sample] = [i]   #Initialize indices for each path which at this point also corresponds to the sample.
        paths.append([sample])  #Keeps track of different paths. To begin with, as many paths as samples.
    int_nodes = {}
    internal_paths = []
    if len(internal_nodes) != 0:
        int_nodes = {nd:i for i,nd in enumerate(internal_nodes)}
        internal_paths = [ [nd] for nd in internal_nodes ]
        shared_time = np.zeros(shape=(len(int_nodes),ts.num_samples))
        internal_indices = defaultdict(list) #For each path, identifies internal nodes that are using that path for shared times.
    if verbose:
        nodes = tqdm(ts.nodes(order="timeasc"))
    else:
        nodes = ts.nodes(order="timeasc")
    for node in nodes:
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
            edge_len = ts.node(parent_nodes[0]).time - node.time
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
            edge_len = ts.node(parent_nodes[0]).time - node.time
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
    
def expand_locations(locations_of_individuals, ts, paths):
    """Converts individuals' locations to sample locations to start of paths locations.

    This should handle if the samples are not organized first in the node table. Need to check.

    Parameters
    ----------
    locations_of_individuals :
    ts : tskit.trees.TreeSequence
    paths :

    Returns
    -------
    locations_of_path_starts
    """

    locations_of_samples = {}
    for node in ts.nodes():
        if node.flags == 1:
            locations_of_samples[node.id] = locations_of_individuals[node.individual]
    locations_of_path_starts = []
    for path in paths:
        locations_of_path_starts.append(locations_of_samples[path[0]])
    locations_of_path_starts = np.array(locations_of_path_starts)#, dtype=np.float64)
    if len(locations_of_path_starts.shape) == 1:
        raise RuntimeError("Path locations vector is missing number of columns. Cannot process.")
    return locations_of_path_starts, locations_of_samples # I don't know why reshape is needed here


def build_roots_array(paths):
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

    roots = [row[-1] for row in paths]
    unique_roots = np.unique(roots)
    roots_array = np.zeros((len(paths), len(unique_roots)))#, dtype=np.float64)
    for i,root in enumerate(unique_roots): 
        for path in np.where(roots == root)[0]:
            roots_array[path][i] += 1.0
    return roots_array, unique_roots


def locate_roots(inverted_cov_mat, roots_array, locations_of_path_starts):
    """Calculate the maximum likelihood locations of the roots of the ARG.

    TODO: Need tests for these different scenarios to ensure that this is all correct

    Parameters
    ----------
    inverted_cov_mat :
    roots_array :
    locations_of_path_starts :
    
    Returns
    -------
    """

    A = np.matmul(np.transpose(roots_array),np.matmul(inverted_cov_mat, roots_array)) #Matrix of coefficients of the system of linear equations 
    b = np.matmul(np.transpose(roots_array),np.matmul(inverted_cov_mat, locations_of_path_starts)) #Vector of constants of the system of linear equations. 
    augmented_matrix = np.column_stack((A, b)) # Construct the augmented matrix [A|b]
    rre_form, pivots = sym.Matrix(augmented_matrix).rref() # Perform row reduction on the augmented matrix
    if int(A.shape[0]) in pivots:
        raise RuntimeError("Cannot locate roots. No solution to system of linear equations.")
    else:
        if len(pivots) != A.shape[0]:
            print("Multiple solutions to system of linear equations in root location calculation.")
            warnings.warn("Multiple solutions to system of linear equations in root location calculation.")
        return np.array(rre_form.col(range(-locations_of_path_starts.shape[1],0)), dtype=np.float64)

def simplify_with_recombination(ts, flag_recomb=False, keep_nodes=[]):
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
    
    #keep_nodes = np.unique(np.concatenate((np.where((ts.tables.nodes.time <= keep_nodes_below) & (ts.tables.nodes.time % keep_nodes_step == 0))[0], recomb_nodes)))
    keep_nodes = np.unique(np.concatenate((keep_nodes, recomb_nodes)))
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

def remove_unattached_nodes(ts):
    edge_table = ts.tables.edges
    connected_nodes = np.sort(np.unique(np.concatenate((edge_table.parent,edge_table.child))))
    ts_final = ts.subset(nodes=connected_nodes)
    return ts_final
    
def merge_unnecessary_roots(ts):
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
    decap = ts.decapitate(time)
    subset = decap.subset(nodes=np.where(decap.tables.nodes.time <= time)[0])
    merged = merge_unnecessary_roots(ts=subset)
    return merged

def estimate_spatial_parameters(ts, locations_of_individuals={}, dimensions=2, verbose=False):

    total_start_time = time.time()
    
    section_start_time = time.time()
    if locations_of_individuals == {}:  # if user doesn't provide a separate locations dictionary, builds one
        locations_of_individuals = get_tskit_locations(ts=ts, dimensions=dimensions)
    if verbose:
        print(f"Prepared input parameters - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")

    section_start_time = time.time()
    cov_mat, paths, node_shared_times, node_paths = calc_minimal_covariance_matrix(ts=ts, internal_nodes=range(ts.num_nodes), verbose=verbose)
    #if return_node_positions:
    #    cov_mat, paths, node_shared_times, node_paths = calc_minimal_covariance_matrix(ts=ts, internal_nodes=range(ts.num_nodes), verbose=verbose)
    #else:
    #    cov_mat, paths = calc_minimal_covariance_matrix(ts=ts, verbose=verbose)
    if verbose:
        print(f"Calculated covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")

    section_start_time = time.time()
    inverted_cov_mat = np.linalg.pinv(cov_mat)
    if verbose:
        print(f"Inverted covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    
    section_start_time = time.time()
    locations_of_path_starts, locations_of_samples = expand_locations(locations_of_individuals=locations_of_individuals, ts=ts, paths=paths)
    roots_array, roots = build_roots_array(paths)
    root_locations = locate_roots(inverted_cov_mat=inverted_cov_mat, roots_array=roots_array, locations_of_path_starts=locations_of_path_starts)
    root_output = dict(zip(roots, root_locations))
    root_locations_vector = np.matmul(roots_array, root_locations)
    if verbose:
        print(f"Created root locations vector - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")

    section_start_time = time.time()
    # calculate dispersal rate
    # this is the uncorrected dispersal rate. (in the future we may want to change this to the corrected version)
    sample_locs_to_root_locs = locations_of_path_starts - root_locations_vector
    sigma = np.matmul(np.matmul(np.transpose(sample_locs_to_root_locs), inverted_cov_mat), sample_locs_to_root_locs)/(ts.num_samples)
    if verbose:
        print(f"Estimated dispersal rate - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")

    FI1 = ts.num_samples/(2*sigma[0][0]**2) 
    FI2 = np.matmul(np.matmul(np.transpose(root_locations_vector), inverted_cov_mat), root_locations_vector)[0][0]/sigma[0][0]**3

    return sigma, cov_mat, paths, inverted_cov_mat, FI1, FI2, sample_locs_to_root_locs, root_output, node_shared_times, node_paths

    if return_node_positions:
        section_start_time = time.time()
        node_path_roots = [path[-1] for path in node_paths]
        node_path_root_locations = np.array([root_locations[np.where(roots == rt)[0]][0] for rt in node_path_roots])
        matmul_prod = np.matmul(node_shared_times, inverted_cov_mat)
        node_locations = node_path_root_locations + np.matmul(matmul_prod, sample_locs_to_root_locs)
        locations_of_nodes = {}
        for node in range(ts.num_nodes):
            locations_of_nodes[node] = node_locations[node].tolist()
        explained_variance = np.matmul(matmul_prod, np.transpose(node_shared_times))
        ones = np.ones(inverted_cov_mat.shape[0])
        unexplained_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
        corrected_variances_in_node_locations = {}
        if verbose:
            ranp = tqdm(range(ts.num_nodes))
        else:
            ranp = range(ts.num_nodes)
        for i,node in enumerate(ranp):
            node_specific_sharing = node_shared_times[i,:]
            unexplained_numerator = (1-np.matmul(np.matmul(np.transpose(node_specific_sharing),inverted_cov_mat),ones))**2
            corrected_variance_scaling_factor = (ts.max_root_time-ts.node(node).time)-explained_variance[i, i]+(unexplained_numerator/unexplained_denominator)
            corrected_variances_in_node_locations[node] = (sigma*corrected_variance_scaling_factor)
        if verbose:
            print(f"Reconstructed ancestral locations - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
        return sigma, cov_mat, paths, inverted_cov_mat, FI1, FI2, locations_of_nodes, corrected_variances_in_node_locations, node_shared_times, node_paths
    else:
        return sigma, cov_mat, paths, inverted_cov_mat, FI1, FI2
    
def ancestors(tree, u):
    """
    Returns an iterator over the ancestors of u in this tree.
    """
    u = tree.parent(u)
    while u != -1:
         yield u
         u = tree.parent(u)

def get_paths_for_nodes(ts, nodes):
    paths = []
    for tree in ts.trees():
        for sample in nodes:
            path = [sample] + list(ancestors(tree, sample))
            if path not in paths:
                paths.append(path)
    return paths

def get_paths_for_node(ts, node):
    just_node, map = ts.simplify(samples=[node], map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
    paths = []
    for tree in just_node.trees():
        path = [0] + list(ancestors(tree, 0))
        for i,n in enumerate(path):
            path[i] = np.argwhere(map==n)[0][0]
        paths.append(path)
    return paths

def create_true_locations_dataframe(ts, nodes, dim=2):
    sample = []
    interval_left = []
    interval_right = []
    time = []
    location = []
    for node in nodes:
        just_node, map = ts.simplify(samples=[node], map_nodes=True, keep_input_roots=False, keep_unary=True, update_sample_flags=False)
        for tree in just_node.trees():
            path = [0] + list(ancestors(tree, 0))
            for i,n in enumerate(path):
                path[i] = np.argwhere(map==n)[0][0]
            for i,n in enumerate(path):
                sample.append(node)
                interval_left.append(tree.interval.left)
                interval_right.append(tree.interval.right)
                time.append(ts.node(n).time)
                indiv = ts.node(n).individual
                if indiv != -1:
                    location.append(ts.individual(indiv).location[:dim])
                else:
                    location.append(None)
    df = pd.DataFrame({
        "sample":sample,
        "interval_left":interval_left,
        "interval_right":interval_right,
        "time":time,
    })
    locs = pd.DataFrame(location, columns=["true_location_"+str(d) for d in range(dim)])
    df = pd.concat([df, locs], axis=1)
    return df

def locate_ancestor(sample, chrom_pos, time, ts, spatial_parameters):
    sigma = spatial_parameters[0]
    paths = spatial_parameters[2]
    inverted_cov_mat = spatial_parameters[3]
    sample_locs_to_root_locs = spatial_parameters[6]
    root_locations = spatial_parameters[7]
    node_shared_times = spatial_parameters[8]
    node_paths = spatial_parameters[9]
    tree = ts.at(chrom_pos)
    path = [sample] + list(ancestors(tree, sample))
    for i,node in enumerate(path):
        if ts.node(node).time >= time:
            above = node
            if i > 0:
                below = path[i-1]
            else:
                below = -1
            break
    ancestor_specific_sharing = node_shared_times[above].copy()
    root_location = root_locations[node_paths[above][-1]]
    additional_time = ts.node(above).time - time
    for i,path in enumerate(paths):
        if below in path:
            ancestor_specific_sharing[i] += additional_time
    matmul_prod = np.matmul(ancestor_specific_sharing, inverted_cov_mat)
    ancestor_location = root_location + np.matmul(matmul_prod, sample_locs_to_root_locs)
    explained_variance = np.matmul(matmul_prod, np.transpose(ancestor_specific_sharing))
    ones = np.ones(inverted_cov_mat.shape[0])
    unexplained_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
    unexplained_numerator = (1-np.matmul(np.matmul(np.transpose(ancestor_specific_sharing),inverted_cov_mat),ones))**2
    corrected_variance_scaling_factor = (ts.max_root_time-time)-explained_variance+(unexplained_numerator/unexplained_denominator)
    variance_in_ancestor_location = sigma*corrected_variance_scaling_factor
    return ancestor_location, variance_in_ancestor_location

def estimate_location(row, ts, spatial_parameters):
    location, variance = locate_ancestor(sample=int(row["sample"]), chrom_pos=row["interval_left"], time=row["time"], ts=ts, spatial_parameters=spatial_parameters)
    output = []
    indices = []
    for i,loc in enumerate(location):
        output.append(loc)
        indices.append("estimated_location_"+str(i))
        output.append(variance[i][i])
        indices.append("variance_in_estimated_location_"+str(i))
    return pd.Series(output, index=indices)