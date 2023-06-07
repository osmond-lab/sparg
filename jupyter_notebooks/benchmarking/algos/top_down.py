import tskit
import msprime
import random
import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt




def calc_paths(ts):
    gmrca = ts.node(ts.num_nodes-1).id
    Paths = [[gmrca]]
    edges = ts.tables.edges
    for node in reversed(list(ts.nodes())):
        paths_with_node = []
        for i, x in enumerate(Paths):
            if node.id in x:
                paths_with_node.append(i)
        child_nodes = np.unique(edges.child[np.where(edges.parent == node.id)])
        inc = 0
        for path in paths_with_node:
            starting_path = Paths[path+inc][:]
            for i, child in enumerate(child_nodes):
                if i > 0:
                    Paths.insert(path+inc+1, starting_path)
                    Paths[path+inc+i].insert(0, child)
                    inc += 1
                else:
                    Paths[path+inc+i].insert(0, child)
    return Paths

def calc_covariance_matrix(ts):
    gmrca = ts.node(ts.num_nodes-1).id
    recomb_nodes = np.where(ts.tables.nodes.flags == 131072)[0]
    recomb_nodes_to_convert = dict(zip(recomb_nodes[1::2], recomb_nodes[::2]))
    edges = ts.tables.edges
    CovMat = np.matrix([[0.0]])
    Indices = defaultdict(list)
    Indices[gmrca] = [0]
    Paths = [[gmrca]]
    edges = ts.tables.edges
    for node in reversed(list(ts.nodes())):
        """
        paths_with_node = []
        for i, x in enumerate(Paths):
            if node.id in x:
                paths_with_node.append(i)
        child_nodes = np.unique(edges.child[np.where(edges.parent == node.id)])
        for path in paths_with_node:
            starting_path = Paths[path][:]
            for i, child in enumerate(child_nodes):
                if i > 0:
                    Paths.append(starting_path)
                    Paths[-1].insert(0, child)
                else:
                    Paths[path].insert(0, child)
        """
        if node.id in recomb_nodes_to_convert or node.flags == 1:
            continue
        path_ind = Indices[node.id]
        child_nodes = np.unique(edges.child[np.where(edges.parent == node.id)])
        for i, child in enumerate(child_nodes):
            for path in path_ind:
                if i > 0:
                    Paths.append(Paths[path][:])
                    Paths[-1][0] = child
                else:
                    Paths[path].insert(0, child)
        npaths = len(path_ind)
        nchild = len(child_nodes)
        for i, child in enumerate(child_nodes):
            if child in recomb_nodes_to_convert:
                child_nodes[i] = recomb_nodes_to_convert[child]
        if nchild == 1:
            child = child_nodes[0]
            edge_len = node.time - ts.node(child).time
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_len*np.ones((npaths,npaths))
            Indices[child] += path_ind
        else:
            edge_lens = [ node.time - ts.node(child).time for child in child_nodes ]
            existing_paths = CovMat.shape[0]
            CovMat = np.hstack(  (CovMat,) + tuple( ( CovMat[:,path_ind] for j in range(nchild-1) ) ) ) #Duplicate the rows
            CovMat = np.vstack(  (CovMat,) + tuple( ( CovMat[path_ind,:] for j in range(nchild-1) ) ) ) #Duplicate the columns
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_lens[0]*np.ones((npaths,npaths))
            Indices[ child_nodes[0] ] += path_ind
            for child_ind in range(1,nchild):
                mod_ind = range(existing_paths+ npaths*(child_ind-1),existing_paths + npaths*child_ind) #indices of the entries that will be modified
                CovMat[ np.ix_( mod_ind , mod_ind  ) ] += edge_lens[child_ind]*np.ones( (npaths,npaths) )
                Indices[ child_nodes[child_ind] ] += mod_ind
    return np.asarray(CovMat), Paths

def calc_node_covariances(ts, paths, node_paths):
    edges = ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    cov_mat = np.zeros((len(node_paths), len(node_paths)+len(paths)))
    for node in ts.nodes():
        if node.id in child_list:
            node_path_indices = []
            for i, path in enumerate(node_paths):
                if node.id in path:
                    node_path_indices.append(i)
            path_indices = []
            for i, path in enumerate(paths):
                if node.id in path:
                    path_indices.append(i+len(node_paths))
            shared_time = ts.node(parent_list[child_list.index(node.id)]).time - node.time
            for a in node_path_indices:
                for b in node_path_indices+path_indices:
                    cov_mat[a, b] += shared_time
    return cov_mat

def benchmark(ts):
    start = time.time()
    cov_mat, paths = calc_covariance_matrix(ts=ts)
    end = time.time()
    return end-start, cov_mat.sum(), "NA"

def create_sample_locations_array(paths, sample_locs):
    """
    Expands sample locations to pair with the unique paths. Accounts for samples having multiple
    paths.
    
    Inputs:
    - paths: list, unique paths within the ARG. Output of identify_unique_paths(). If not
        provided, this will be calculated
    - sample_locs: numpy array, sample locations
    
    Output:
    - path_locs: numpy array, sample locations expanded to match number of paths
    """
    sample_locs_array = []
    for path in paths:
        sample_locs_array.append([sample_locs[path[0]]])
    path_locs = np.array(sample_locs_array)
    return path_locs

def link_node_with_path(ts, paths):
    """
    Adds paths from internal nodes to the root to the paths list used for calculating the
    covariance matrix. Could potentially only do one of the two recombination nodes, but keeping
    it simple for now

    Inputs:
    - ts: tskit tree sequence
    - paths: list, unique paths within the ARG. Output of identify_unique_paths().

    Output:
    - path_list: list, updated paths list
    """
    path_list = []
    for node in ts.nodes():
        if node.flags == 1 or node.time == ts.max_root_time:
            continue
        for i in range(len(paths)):
            if node.id in paths[i]:
                path_list.append(paths[i][paths[i].index(node.id):])
                break
    return path_list

def locate_mle_gmrca(inv_sigma_22, sample_locs):
    """
    Locates the maximum likelihood estimate of the grand most recent common ancestor based on the covariance
    matrix between paths and sample locations (Equation 5.6 from 
    https://lukejharmon.github.io/pcm/pdf/phylogeneticComparativeMethods.pdf). Currently, requires simga_22
    to be pre-inverted (may be worth adding both options in future).
    
    Inputs:
    - inv_sigma_22: numpy array, inverted covariance matrix between paths at sample time
    - sample_locs: numpy array, sample locations expanded to match number of paths. Output of
        create_sample_locations_array().
    
    Output:
    - u1: float, maximum likelihood estimate of the grand most recent common ancestor (GMRCA). Output of 
        locate_mle_gmrca().
    """
    k = len(inv_sigma_22)
    a1 = np.matmul(np.matmul(np.ones(k), inv_sigma_22), np.ones(k).reshape(-1,1))
    a2 = np.matmul(np.matmul(np.ones(k), inv_sigma_22), sample_locs)
    u1 = a2/a1
    return u1

def estimate_mle_dispersal(Tinv, locs):
    '''
    MLE dispersal estimate
    
    parameters
    ----------
    Tinv: inverse covariance matrix among sample locations
    locs: sample locations
    '''
    k = len(locs) #number of paths
    # find MLE MRCA location (eqn 5.6 Harmon book)
    a1 = np.matmul(np.matmul(np.ones(k), Tinv), np.ones(k).reshape(-1,1))
    a2 = np.matmul(np.matmul(np.ones(k), Tinv), locs)
    ahat = a2/a1
    # find MLE dispersal rate (eqn 5.7 Harmon book)
    x = locs.reshape(-1,1) #make locations a column vector
    R1 = x - ahat * np.ones(k).reshape(-1,1)
    Rhat = np.matmul(np.matmul(np.transpose(R1), Tinv), R1) / (k-1)
    return Rhat[0]

def reconstruct_node_locations(ts, sample_locs):
    sigma_22, paths = calc_covariance_matrix(ts=ts)
    node_paths = link_node_with_path(ts=ts, paths=paths)
    node_covariances = calc_node_covariances(ts=ts, paths=paths, node_paths=node_paths)
    sigma_11 = node_covariances[:,:len(node_paths)]
    sigma_12 = node_covariances[:,len(node_paths):]
    sigma_21 = np.transpose(sigma_12)
    inv_sigma_22 = np.linalg.pinv(sigma_22)
    sample_locs_array = create_sample_locations_array(paths=paths, sample_locs=sample_locs)
    dispersal_rate = estimate_mle_dispersal(inv_sigma_22, sample_locs_array)
    u1 = locate_mle_gmrca(inv_sigma_22=inv_sigma_22, sample_locs=sample_locs_array)
    cmvn_u = u1 + np.matmul(np.matmul(sigma_12, inv_sigma_22),sample_locs_array - u1)
    cmvn_sigma = sigma_11 - np.matmul(np.matmul(sigma_12, inv_sigma_22), sigma_21)
    node_times = ts.tables.nodes.time
    node_locs = np.concatenate((sample_locs, np.transpose(cmvn_u)[0], u1))
    return paths, node_times, node_locs, dispersal_rate


def get_dispersal_and_gmrca(file):
    ts = tskit.load(file)
    sample_locs = np.linspace(0, 1, ts.num_samples)
    paths, times, locations, dispersal_rate = reconstruct_node_locations(ts=ts, sample_locs=sample_locs)
    return dispersal_rate, locations[-1]


if __name__ == "__main__":
    
    rs = random.randint(0,10000)
    print(rs)
    ts = msprime.sim_ancestry(
        samples=2,
        recombination_rate=1e-8,
        sequence_length=2_000,
        population_size=10_000,
        record_full_arg=True,
        random_seed=rs
    )
    print(ts.draw_text())

    sample_locs = np.linspace(0, 1, ts.num_samples)
    paths, times, locations, dispersal_rate = reconstruct_node_locations(ts=ts, sample_locs=sample_locs)