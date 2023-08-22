import numpy as np
from collections import defaultdict
import sympy as sym
import warnings


def get_tskit_locations(ts):
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
        locations_of_individuals[i] = location
    return locations_of_individuals


def calc_covariance_matrix(ts, internal_nodes=[]):
    """Calculates a covariance matrix between the paths in the the ARG.

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
    cov_mat = np.zeros(shape=(ts.num_samples, ts.num_samples))  #Initialize the covariance matrix. Initial size = #samples. Will increase to #paths
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
    for node in ts.nodes(order="timeasc"):
        path_ind = indices[node.id]
        parent_nodes = np.unique(edges.parent[np.where(edges.child == node.id)])
        if len(internal_nodes) != 0: 
            if node.id in int_nodes: 
                internal_indices[path_ind[0]] += [int_nodes[node.id]]   
        for i, parent in enumerate(parent_nodes):
            for path in path_ind:
                if i == 0:
                    paths[path].append(parent)
                    for internal_path_ind in internal_indices[path]: 
                        internal_paths[internal_path_ind] += [parent]
                else:
                    paths.append(paths[path][:])
                    paths[-1][-1] = parent         
        npaths = len(path_ind)
        nparent = len(parent_nodes)
        if nparent == 0:    # if a node doesn't have a parent, then it is a root and we can skip
            continue
        else:
            edge_len = ts.node(parent_nodes[0]).time - node.time
            cov_mat = np.hstack(  (cov_mat,) + tuple( ( cov_mat[:,path_ind] for j in range(nparent-1) ) ) ) #Duplicate the columns
            cov_mat = np.vstack(  (cov_mat,) + tuple( ( cov_mat[path_ind,:] for j in range(nparent-1) ) ) ) #Duplicate the rows
            new_ind = path_ind + [len(cov_mat) + x for x in range((-(nparent-1)*len(path_ind)),0)]
            cov_mat[ np.ix_( new_ind, new_ind ) ] += edge_len
            for i,parent in enumerate(parent_nodes): 
                indices[parent] += new_ind[i*npaths:(i+1)*npaths]
            if len(internal_nodes) != 0:
                shared_time = np.hstack( (shared_time, ) + tuple( ( shared_time[:,path_ind] for j in range(nparent-1) ) ) )
                int_nodes_update = []
                for i in path_ind: 
                    int_nodes_update += internal_indices[i]
                shared_time[ np.ix_( int_nodes_update, new_ind) ] += edge_len
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
    locations_of_path_starts = np.array(locations_of_path_starts)
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
    roots_array = np.zeros((len(paths), len(unique_roots)))
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
            warnings.warn("Multiple solutions to system of linear equations in root location calculation.")
        return np.array(rre_form.col(range(-locations_of_path_starts.shape[1],0)))
    

def estimate_spatial_parameters(ts, locations_of_individuals={}, return_ancestral_node_positions=True):
    """Calculates maximum likelihood dispersal rate and the locations of ancestral nodes.

    Parameters
    ----------
    ts : tskit.trees.TreeSequence
        For accurate estimation, this must be a tskit Tree Sequences with marked recombination nodes,
        as is outputted by msprime.sim_ancestry(..., record_full_arg=True).
    locations_of_individuals (optional) : dictionary
        The locations of individuals can be provided within the tskit Tree Sequence or as a separate
        dictionary, where the key is the node ID and the value is a numpy.ndarray or list with the node's
        location.
    return_ancestral_node_positions (optional) : boolean
        Option to calculate the position of internal nodes. This can be a computationally costly calculation
        so can be bypasses if the user is only interested in dispersal rates and the covariance matrix.

    Returns
    -------
    sigma :
    locations_of_nodes :
    """

    #viz.D3ARG(ts=ts).draw(width=1000, height=700, edge_type="line")
    #if len(tracked_samples) > 0:
    #    ts = add_nodes_along_sample_paths(ts=ts, tracked_samples=tracked_samples)
        #viz.D3ARG(ts=ts).draw(width=1000, height=700, edge_type="line")

    if locations_of_individuals == {}:  # if user doesn't provide a separate locations dictionary, builds one
        locations_of_individuals = get_tskit_locations(ts=ts)
    if return_ancestral_node_positions:
        cov_mat, paths, node_shared_times, node_paths = calc_covariance_matrix(ts=ts, internal_nodes=range(ts.num_nodes))
    else:
        cov_mat, paths = calc_covariance_matrix(ts=ts)
    locations_of_path_starts, locations_of_samples = expand_locations(locations_of_individuals=locations_of_individuals, ts=ts, paths=paths)
    inverted_cov_mat = np.linalg.pinv(cov_mat)
    roots_array, roots = build_roots_array(paths) 
    root_locations = locate_roots(inverted_cov_mat=inverted_cov_mat, roots_array=roots_array, locations_of_path_starts=locations_of_path_starts)
    root_locations_vector = np.matmul(roots_array, root_locations)
    
    # calculate dispersal rate
    sigma = np.matmul(np.matmul(np.transpose(locations_of_path_starts - root_locations_vector), inverted_cov_mat), (locations_of_path_starts - root_locations_vector))/(ts.num_samples-len(roots))
    
    # calculate locations of nodes
    if return_ancestral_node_positions:
        node_path_roots = [path[-1] for path in node_paths]
        node_path_root_locations = np.array([root_locations[np.where(roots == rt)[0]][0] for rt in node_path_roots])
        node_locations = node_path_root_locations + np.matmul(np.matmul(node_shared_times, inverted_cov_mat), locations_of_path_starts - root_locations_vector)
        locations_of_nodes = {}
        for node in ts.nodes():
            locations_of_nodes[node.id] = node_locations[node.id].tolist()
        explained_variance = np.matmul(np.matmul(node_shared_times, inverted_cov_mat), np.transpose(node_shared_times))
        ones = np.ones(inverted_cov_mat.shape[0])
        #uncorrected_variances_in_node_locations = {}
        corrected_variances_in_node_locations = {}
        for node in ts.nodes():
            #uncorrected_variance_scaling_factor = (ts.max_root_time-node.time)-explained_variance[node.id, node.id]
            node_specific_sharing = node_shared_times[node.id,:]
            unexplained_numerator = (1-np.matmul(np.matmul(np.transpose(node_specific_sharing),inverted_cov_mat),ones))**2
            unexplained_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
            corrected_variance_scaling_factor = (ts.max_root_time-node.time)-explained_variance[node.id, node.id]+(unexplained_numerator/unexplained_denominator)
            #uncorrected_variances_in_node_locations[node.id] = (sigma*uncorrected_variance_scaling_factor)
            corrected_variances_in_node_locations[node.id] = (sigma*corrected_variance_scaling_factor)
        return sigma, cov_mat, paths, locations_of_nodes, corrected_variances_in_node_locations #, uncorrected_variances_in_node_locations
    
    return sigma, cov_mat, paths