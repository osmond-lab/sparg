import numpy as np
from collections import defaultdict
from tqdm import tqdm
import time


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

def benchmark(ts):
    start = time.time()
    sigma, paths = calc_covariance_matrix(ts=ts)
    end = time.time()
    sigma_inv = np.linalg.pinv(sigma)
    final_end = time.time() 
    return end-start, final_end-start, sigma.sum(), len(paths)