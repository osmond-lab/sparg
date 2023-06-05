import random
import msprime
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import chain
import time
import matplotlib.pyplot as plt



def ts_to_nx(ts, connect_recombination_nodes=False, recomb_nodes=[]):
    """
    Converts tskit tree sequence to networkx graph.
    """
    topology = defaultdict(list)
    for tree in ts.trees():
        for k, v in chain(tree.parent_dict.items()):
            if connect_recombination_nodes:
                if recomb_nodes == []:
                    recomb_nodes = list(np.where(ts.tables.nodes.flags == 131072)[0])
                if v in recomb_nodes and recomb_nodes.index(v)%2 == 1:
                    v -= 1
                if k in recomb_nodes and recomb_nodes.index(k)%2 == 1:
                    k -= 1
                if v not in topology[k]:
                    topology[k].append(v)
            else:
                if v not in topology[k]:
                    topology[k].append(v)
    nx_graph = nx.MultiDiGraph(topology)
    return nx_graph

def identify_unique_paths(ts):
    """
    Finds all of the paths within the incomplete ARG, stored as a tskit tree sequence
    
    Input:
    - ts: tskit tree sequence
    
    Output:
    - all_paths: list, unique paths within the ARG
    """
    G = ts_to_nx(ts=ts)
    # originally had grmca but there are instances of multiple roots,
    # so this should handle that.
    roots = [i.id for i in list(ts.nodes()) if i.id not in list(ts.tables.edges.child)]
    all_paths = []
    for sample in ts.samples():
        for root in roots:
            paths = nx.all_simple_paths(G, source=sample, target=root)
            all_paths.extend(paths)
    return all_paths

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

def calc_covariance_matrix(ts, paths=[]):
    edges = ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    if len(paths) == 0:
        paths = identify_unique_paths(ts=ts)
    cov_mat = np.zeros((len(paths), len(paths)))
    for node in ts.nodes():
        if node.id in child_list:
            path_indices = []
            for i, path in enumerate(paths):
                if node.id in path:
                    path_indices.append(i)
            shared_time = ts.node(parent_list[child_list.index(node.id)]).time - node.time
            for a in path_indices:
                for b in path_indices:
                    cov_mat[a, b] += shared_time
    if len(paths) == 0:
        return cov_mat, paths
    else:
        return cov_mat

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
    """
    Calculates the location of ancestral nodes using conditional multivariate normal distribution.

    Inputs:
    - ts: tskit tree sequence. Needed for node times
    - paths: list, unique paths within the ARG. Output of identify_unique_paths()
    - sample_locs: list of sample locations, one location per sample

    Outputs:
    - node_times: list, time of nodes from present
    - node_locs: list, location of nodes
    """
    paths = identify_unique_paths(ts=ts)
    sample_locs_array = create_sample_locations_array(paths=paths, sample_locs=sample_locs) # expands locs
    node_paths = link_node_with_path(ts=ts, paths=paths)
    all_paths = node_paths + paths
    sigma = calc_covariance_matrix(ts=ts, paths=all_paths)
    np.savetxt("path_CM.csv", sigma, delimiter=",")
    sigma_11 = sigma[0:sigma.shape[0]-len(paths),0:sigma.shape[1]-len(paths)]
    sigma_12 = sigma[0:sigma.shape[0]-len(paths),sigma.shape[1]-len(paths):sigma.shape[1]]
    sigma_21 = sigma[sigma.shape[0]-len(paths):sigma.shape[0],0:sigma.shape[1]-len(paths)]
    sigma_22 = sigma[sigma.shape[0]-len(paths):sigma.shape[0],sigma.shape[1]-len(paths):sigma.shape[1]]
    
    mean_centering_M = np.identity(len(paths)) - [[1/len(paths) for _ in range(len(paths))] for _ in range(len(paths))]; mean_centering_M = mean_centering_M[:-1] #mean centering matrix
    print(mean_centering_M)
    mean_centering_X = np.dot(mean_centering_M, sample_locs_array)
    print(mean_centering_X)
    mean_centering_VGT = np.dot(np.matmul(mean_centering_M, sigma_22), np.transpose(mean_centering_M))
    print(mean_centering_VGT)

    print(locate_mle_gmrca(inv_sigma_22=np.linalg.pinv(sigma_22), sample_locs=sample_locs_array))
    print(sample_locs_array.mean() + locate_mle_gmrca(inv_sigma_22=np.linalg.pinv(mean_centering_VGT), sample_locs=mean_centering_X))


    mean_centering_va = np.matmul(mean_centering_M, sigma_21) - np.matmul( np.matmul(mean_centering_M, sigma_22), np.kron(np.ones(len(paths)).reshape(-1,1), np.identity(1)))/len(paths)
    
    la_hat = sample_locs_array.mean() + np.matmul(np.matmul(mean_centering_va.transpose(), np.linalg.pinv(mean_centering_VGT)), mean_centering_X)

    #print(sample_locs_array)
    #print(sigma_22)
    
    #print(sample_locs_array.mean() + locate_mle_gmrca(inv_sigma_22=np.linalg.pinv(mean_centering_VGT), sample_locs=mean_centering_X))


    """
    inv_sigma_22 = np.linalg.pinv(sigma_22)
    dispersal_rate = estimate_mle_dispersal(inv_sigma_22, sample_locs_array)
    u1 = locate_mle_gmrca(inv_sigma_22=inv_sigma_22, sample_locs=sample_locs_array)
    cmvn_u = u1 + np.dot(np.dot(sigma_12, inv_sigma_22),sample_locs_array - u1)
    cmvn_sigma = sigma_11 - np.dot(np.dot(sigma_12, inv_sigma_22), sigma_21)
    node_times = ts.tables.nodes.time
    node_locs = np.concatenate((sample_locs, np.transpose(cmvn_u)[0], u1))
    return paths, node_times, node_locs, dispersal_rate
    """


rs = random.randint(0,10000)
print(rs)
ts = msprime.sim_ancestry(
    samples=1,#30
    recombination_rate=1e-8,
    sequence_length=2_000,#1_000
    population_size=10_000,
    record_full_arg=True,
    random_seed=7976#9080
)

print(ts.draw_text())


sample_locs = np.linspace(0, 1, ts.num_samples)
reconstruct_node_locations(ts=ts, sample_locs=sample_locs)