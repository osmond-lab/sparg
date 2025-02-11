import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time

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
        
        path_ind_unique, path_ind_count = np.unique(path_ind, return_counts=True)
        path_ind_to_be_duplicated = []
        if len(path_ind) == len(path_ind_unique) or len(path_ind) < nparent:
            path_ind_to_be_duplicated += [path_ind[0]]
        
        
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
            
                if path in path_ind_to_be_duplicated:
                    paths[path].append(parent1)
                    parent1_ind += [ path ]
                    paths.append(paths[path][:])
                    paths[-1][-1] = parent2
                    parent2_ind += [ len(cov_mat) ]
                    cov_mat = np.hstack(  (cov_mat, cov_mat[:,path_ind_to_be_duplicated[0]].reshape(cov_mat.shape[0],1) )) #Duplicate the column
                    cov_mat = np.vstack(  (cov_mat, cov_mat[path_ind_to_be_duplicated[0],:].reshape(1,cov_mat.shape[1]) )) #Duplicate the row
                    if len(internal_nodes) != 0:
                        shared_time = np.hstack(  (shared_time, shared_time[:,path_ind_to_be_duplicated[0]].reshape(shared_time.shape[0],1) )) #Duplicate the column
                    
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
    

def benchmark(ts):
    start = time.time()
    sigma, paths = calc_minimal_covariance_matrix(ts=ts)
    end = time.time()
    sigma_inv = np.linalg.pinv(sigma)
    final_end = time.time() 
    return end-start, final_end-start, sigma.sum(), len(paths)