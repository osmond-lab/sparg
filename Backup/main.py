import numpy as np
from collections import defaultdict
import sympy as sym
import warnings
from tqdm import tqdm
import time
import glob
import os
import datetime


def create_output_directory(output_directory):
    """
    Checks whether specified output directory exists and if not creates it.

    Parameters
    ----------
    outputDirectory : string
        path to output directory.

    Returns
    -------
    new : boolean
        True if output directory had to be created.

    """
    new = False
    output_file_path = output_directory.split(os.sep)
    for level in range(len(output_file_path)):    # loops through each level of the path to output directory
        if output_file_path[level] == "":
            continue
        elif level == 0:
            if ":" in output_file_path[level]:
                continue
            if output_file_path[level] not in glob.glob("*"): # checks whether directory exists at that level
                os.mkdir(output_file_path[level]) # if it doesn't, creates it
                new = True
        else:
            if os.sep.join(output_file_path[:level + 1]) not in glob.glob(os.sep.join(output_file_path[:level]) + os.sep +"*"):
                os.mkdir(os.sep.join(output_file_path[:level + 1]))
                new = True
    return new


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


def calc_covariance_matrix(ts, internal_nodes=[], verbose=False):
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

        npaths = len(path_ind)
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
            
            # path_ind_unique, path_ind_count = np.unique(path_ind, return_counts=True)
            # path_ind_to_be_duplicated = []
            # if len(path_ind) == len(path_ind_unique) or len(path_ind) < nparent:
            #     path_ind_to_be_duplicated += [path_ind[0]]                

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
    

def estimate_spatial_parameters(ts, verbose=False, record_to="", locations_of_individuals={}, return_ancestral_node_positions=[], n_r=False, dimensions=2):
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


    total_start_time = time.time()

    if record_to:
        try:
            create_output_directory(record_to)
        except OSError:
            print("ERROR: Creation of the directory %s failed. Please select a different -outDirectory path and try again!" % record_to)
            exit()
        log_file = open(record_to + "/log.txt", "w")
    
    section_start_time = time.time()
    if locations_of_individuals == {}:  # if user doesn't provide a separate locations dictionary, builds one
        locations_of_individuals = get_tskit_locations(ts=ts, dimensions=dimensions)
    return_ancestral_node_positions = [x for x in return_ancestral_node_positions if x <= ts.node(ts.num_nodes-1).id]
    if verbose:
        print(f"Prepared input parameters - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Prepared input parameters - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")

    section_start_time = time.time()
    if len(return_ancestral_node_positions)>0:
        cov_mat, paths, node_shared_times, node_paths = calc_covariance_matrix(ts=ts, internal_nodes=return_ancestral_node_positions, verbose=verbose)
    else:
        cov_mat, paths = calc_covariance_matrix(ts=ts, verbose=verbose)
    if verbose:
        print(f"Calculated covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Calculated covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        np.savetxt(record_to + "/cov_mat.csv", cov_mat, delimiter=",")
        with open(record_to + "/paths.txt", "w") as f:
            for path in paths:
                f.write(f"{path}\n")

    section_start_time = time.time()
    inverted_cov_mat = np.linalg.pinv(cov_mat)
    if verbose:
        print(f"Inverted covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Inverted covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        np.savetxt(record_to + "/inv_cov_mat.csv", inverted_cov_mat, delimiter=",")

    section_start_time = time.time()
    locations_of_path_starts, locations_of_samples = expand_locations(locations_of_individuals=locations_of_individuals, ts=ts, paths=paths)
    roots_array, roots = build_roots_array(paths)
    root_locations = locate_roots(inverted_cov_mat=inverted_cov_mat, roots_array=roots_array, locations_of_path_starts=locations_of_path_starts)
    root_locations_vector = np.matmul(roots_array, root_locations)
    if verbose:
        print(f"Created root locations vector - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Created root locations vector - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        #with open(record_to + "/roots.txt", "w") as f:
        #    for start in locations_of_path_starts:
        #        f.write(f"{start}\n")
        #np.savetxt(record_to + "/roots.csv", roots, delimiter=",")
        #np.savetxt(record_to + "/root_locations.csv", root_locations_vector, delimiter=",")
    
    section_start_time = time.time()
    # calculate dispersal rate
    # this is the uncorrected dispersal rate. (in the future we may want to change this to the corrected version which takes into account the number of roots: -len(roots))
    sample_locs_to_root_locs = locations_of_path_starts - root_locations_vector
    if n_r:
        sigma = np.matmul(np.matmul(np.transpose(sample_locs_to_root_locs), inverted_cov_mat), sample_locs_to_root_locs)/(ts.num_samples-len(roots))
    else:
        sigma = np.matmul(np.matmul(np.transpose(sample_locs_to_root_locs), inverted_cov_mat), sample_locs_to_root_locs)/(ts.num_samples)
    if verbose:
        print(f"Estimated dispersal rate - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Estimated dispersal rate - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        np.savetxt(record_to + "/dispersal_rate.csv", sigma, delimiter=",")
    
    # calculate locations of nodes
    if len(return_ancestral_node_positions)>0:
        
        section_start_time = time.time()
        node_path_roots = [path[-1] for path in node_paths]
        node_path_root_locations = np.array([root_locations[np.where(roots == rt)[0]][0] for rt in node_path_roots])
        matmul_prod = np.matmul(node_shared_times, inverted_cov_mat)
        node_locations = node_path_root_locations + np.matmul(matmul_prod, sample_locs_to_root_locs)
        locations_of_nodes = {}
        for i,node in enumerate(return_ancestral_node_positions):
            locations_of_nodes[node] = node_locations[i].tolist()
        explained_variance = np.matmul(matmul_prod, np.transpose(node_shared_times))
        ones = np.ones(inverted_cov_mat.shape[0])
        unexplained_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
        corrected_variances_in_node_locations = {}
        if verbose:
            ranp = tqdm(return_ancestral_node_positions)
        else:
            ranp = return_ancestral_node_positions
        for i,node in enumerate(ranp):
            node_specific_sharing = node_shared_times[i,:]
            unexplained_numerator = (1-np.matmul(np.matmul(np.transpose(node_specific_sharing),inverted_cov_mat),ones))**2
            corrected_variance_scaling_factor = (ts.max_root_time-ts.node(node).time)-explained_variance[i, i]+(unexplained_numerator/unexplained_denominator)
            corrected_variances_in_node_locations[node] = (sigma*corrected_variance_scaling_factor)
        if verbose:
            print(f"Reconstructed ancestral locations - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
        if record_to:
            log_file.write(f"Reconstructed ancestral locations - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
            with open(record_to + "/locations_of_nodes.txt", "w") as f:
                for node in locations_of_nodes:
                    f.write(f"{node} {locations_of_nodes[node]} {corrected_variances_in_node_locations[node]}\n")
            log_file.close()
        return sigma, cov_mat, paths, locations_of_nodes, corrected_variances_in_node_locations
    
    if record_to:
        log_file.close()
    return sigma, cov_mat, paths

def estimate_minimal_spatial_parameters(ts, verbose=False, record_to="", locations_of_individuals={}, return_ancestral_node_positions=[], n_r=False, dimensions=2, for_fig=0):
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


    total_start_time = time.time()

    if record_to:
        try:
            create_output_directory(record_to)
        except OSError:
            print("ERROR: Creation of the directory %s failed. Please select a different -outDirectory path and try again!" % record_to)
            exit()
        log_file = open(record_to + "/log.txt", "w")
    
    section_start_time = time.time()
    if locations_of_individuals == {}:  # if user doesn't provide a separate locations dictionary, builds one
        locations_of_individuals = get_tskit_locations(ts=ts, dimensions=dimensions)
    return_ancestral_node_positions = [x for x in return_ancestral_node_positions if x <= ts.node(ts.num_nodes-1).id]
    if verbose:
        print(f"Prepared input parameters - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Prepared input parameters - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")

    section_start_time = time.time()
    if len(return_ancestral_node_positions)>0:
        cov_mat, paths, node_shared_times, node_paths = calc_minimal_covariance_matrix(ts=ts, internal_nodes=return_ancestral_node_positions, verbose=verbose)
    else:
        cov_mat, paths = calc_minimal_covariance_matrix(ts=ts, verbose=verbose)
    if verbose:
        print(f"Calculated covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Calculated covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        np.savetxt(record_to + "/cov_mat.csv", cov_mat, delimiter=",")
        with open(record_to + "/paths.txt", "w") as f:
            for path in paths:
                f.write(f"{path}\n")

    section_start_time = time.time()
    inverted_cov_mat = np.linalg.pinv(cov_mat)
    if verbose:
        print(f"Inverted covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Inverted covariance matrix - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        np.savetxt(record_to + "/inv_cov_mat.csv", inverted_cov_mat, delimiter=",")

    section_start_time = time.time()
    locations_of_path_starts, locations_of_samples = expand_locations(locations_of_individuals=locations_of_individuals, ts=ts, paths=paths)
    roots_array, roots = build_roots_array(paths)
    root_locations = locate_roots(inverted_cov_mat=inverted_cov_mat, roots_array=roots_array, locations_of_path_starts=locations_of_path_starts)
    root_locations_vector = np.matmul(roots_array, root_locations)
    if verbose:
        print(f"Created root locations vector - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Created root locations vector - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        #with open(record_to + "/roots.txt", "w") as f:
        #    for start in locations_of_path_starts:
        #        f.write(f"{start}\n")
        #np.savetxt(record_to + "/roots.csv", roots, delimiter=",")
        #np.savetxt(record_to + "/root_locations.csv", root_locations_vector, delimiter=",")
    
    section_start_time = time.time()
    # calculate dispersal rate
    # this is the uncorrected dispersal rate. (in the future we may want to change this to the corrected version which takes into account the number of roots: -len(roots))
    sample_locs_to_root_locs = locations_of_path_starts - root_locations_vector
    if n_r:
        sigma = np.matmul(np.matmul(np.transpose(sample_locs_to_root_locs), inverted_cov_mat), sample_locs_to_root_locs)/(ts.num_samples-len(roots))
    else:
        sigma = np.matmul(np.matmul(np.transpose(sample_locs_to_root_locs), inverted_cov_mat), sample_locs_to_root_locs)/(ts.num_samples)
    if verbose:
        print(f"Estimated dispersal rate - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
    if record_to:
        log_file.write(f"Estimated dispersal rate - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
        np.savetxt(record_to + "/dispersal_rate.csv", sigma, delimiter=",")
    
    FI1 = ts.num_samples/(2*sigma[0][0]**2) 
    FI2 = np.matmul(np.matmul(np.transpose(root_locations_vector), inverted_cov_mat), root_locations_vector)[0][0]/sigma[0][0]**3
    
    # calculate locations of nodes
    if len(return_ancestral_node_positions)>0:
        
        section_start_time = time.time()
        node_path_roots = [path[-1] for path in node_paths]
        node_path_root_locations = np.array([root_locations[np.where(roots == rt)[0]][0] for rt in node_path_roots])
        matmul_prod = np.matmul(node_shared_times, inverted_cov_mat)
        node_locations = node_path_root_locations + np.matmul(matmul_prod, sample_locs_to_root_locs)
        locations_of_nodes = {}
        for i,node in enumerate(return_ancestral_node_positions):
            locations_of_nodes[node] = node_locations[i].tolist()
        explained_variance = np.matmul(matmul_prod, np.transpose(node_shared_times))
        ones = np.ones(inverted_cov_mat.shape[0])
        unexplained_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
        corrected_variances_in_node_locations = {}
        if verbose:
            ranp = tqdm(return_ancestral_node_positions)
        else:
            ranp = return_ancestral_node_positions
        for i,node in enumerate(ranp):
            node_specific_sharing = node_shared_times[i,:]
            unexplained_numerator = (1-np.matmul(np.matmul(np.transpose(node_specific_sharing),inverted_cov_mat),ones))**2
            corrected_variance_scaling_factor = (ts.max_root_time-ts.node(node).time)-explained_variance[i, i]+(unexplained_numerator/unexplained_denominator)
            corrected_variances_in_node_locations[node] = (sigma*corrected_variance_scaling_factor)
        if verbose:
            print(f"Reconstructed ancestral locations - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}")
        if record_to: 
            log_file.write(f"Reconstructed ancestral locations - Section Elapsed Time: {time.time()-section_start_time} - Total Elapsed Time: {time.time()-total_start_time}\n")
            with open(record_to + "/locations_of_nodes.txt", "w") as f:
                for node in locations_of_nodes:
                    f.write(f"{node} {locations_of_nodes[node]} {corrected_variances_in_node_locations[node]}\n")
            log_file.close()
        if for_fig == 0: 
            return sigma, cov_mat, paths, locations_of_nodes, corrected_variances_in_node_locations
        elif for_fig == 2: 
            return sigma, cov_mat, paths, locations_of_nodes, corrected_variances_in_node_locations, node_shared_times, node_paths, inverted_cov_mat
        elif for_fig == 3: 
            return sigma, cov_mat, paths, locations_of_nodes, corrected_variances_in_node_locations, FI1, FI2 
        
    if record_to:
        log_file.close()
    if for_fig == 0: 
        return sigma, cov_mat, paths
    elif for_fig ==1: 
        return sigma, cov_mat, paths, FI1, FI2
