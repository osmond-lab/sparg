import msprime
import tskit
import numpy as np
import math
import random
from collections import defaultdict
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt
import time





def ts_to_nx(ts, connect_recombination_nodes=False, recomb_nodes=[]):
    """
    Converts tskit tree sequence to networkx graph.

    Need to add a check to ensure that the list of recombination nodes is valid
    (there should always be an even number of recombination nodes if following
    tskit setup)
    """
    if recomb_nodes == []:
        recomb_nodes = list(np.where(ts.tables.nodes.flags == 131072)[0])
    recomb_nodes_to_remove = recomb_nodes[1::2]
    topology = defaultdict(list)
    for tree in ts.trees():
        for k, v in chain(tree.parent_dict.items()):
            if connect_recombination_nodes:
                if v in recomb_nodes_to_remove:
                    v -= 1
                if k in recomb_nodes_to_remove:
                    k -= 1
                if v not in topology[k]:
                    topology[k].append(v)
            else:
                if v not in topology[k]:
                    topology[k].append(v)
    nx_graph = nx.DiGraph(topology)
    node_times = {v: k for v, k in enumerate(ts.tables.nodes.time)}
    nx.set_node_attributes(nx_graph, node_times, "time")
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
    
    #unique paths up the ARG
    gmrca = ts.node(ts.num_nodes-1).id
    all_paths = []
    for sample in ts.samples():
        paths = nx.all_simple_paths(G, source=sample, target=gmrca)
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
        if node.flags == tskit.NODE_IS_SAMPLE or node.time == ts.max_root_time:
            continue
        for i in range(len(paths)):
            if node.id in paths[i]:
                path_list.append(paths[i][paths[i].index(node.id):])
                break
    return path_list

def calc_covariance_matrix(paths, ts):
    """
    Calculates the covariance matrix between paths in a full ARG, stored as a tskit tree sequence.
    
    Inputs:
    - paths: list, unique paths within the ARG. Output of identify_unique_paths()
    - ts: tskit tree sequence. Needed for node times
    
    Output:
    - times: numpy array, shared times between the paths within the ARG
    """
    edges = ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    gmrca = ts.node(ts.num_nodes-1).id
    tgmrca = ts.node(gmrca).time
    times = np.empty((len(paths),len(paths)))
    tree = ts.first()
    for i, p in enumerate(paths):
        for j in range(i+1):
            intersect = list(set(p).intersection(paths[j]))
            if i == j:
                times[i,j] = tgmrca
            elif intersect == [gmrca]:
                times[i,j] = 0
            else:
                edges = []
                for child in intersect:
                    if child != gmrca:
                        edges.append(ts.node(parent_list[child_list.index(child)]).time - ts.node(child).time)
                times[i,j] = np.sum(edges) # Do I need np.unique()? Ask Matt, because it was previously in his
            times[j,i] = times[i,j]
    return times
    
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

def reconstruct_node_locations(ts, paths, sample_locs):
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
    sample_locs_array = create_sample_locations_array(paths=paths, sample_locs=sample_locs) # expands locs
    #node_paths = link_node_with_path(ts=ts, paths=paths)
    all_paths = paths #node_paths + paths
    sigma = calc_covariance_matrix(paths=all_paths, ts=ts)
    return sigma
    """
    #np.savetxt("path_CM.csv", sigma, delimiter=",")
    sigma_11 = sigma[0:sigma.shape[0]-len(paths),0:sigma.shape[1]-len(paths)]
    sigma_12 = sigma[0:sigma.shape[0]-len(paths),sigma.shape[1]-len(paths):sigma.shape[1]]
    sigma_21 = sigma[sigma.shape[0]-len(paths):sigma.shape[0],0:sigma.shape[1]-len(paths)]
    sigma_22 = sigma[sigma.shape[0]-len(paths):sigma.shape[0],sigma.shape[1]-len(paths):sigma.shape[1]]
    inv_sigma_22 = np.linalg.pinv(sigma_22)
    dispersal_rate = estimate_mle_dispersal(inv_sigma_22, sample_locs_array)
    u1 = locate_mle_gmrca(inv_sigma_22=inv_sigma_22, sample_locs=sample_locs_array)
    cmvn_u = u1 + np.dot(np.dot(sigma_12, inv_sigma_22),sample_locs_array - u1)
    cmvn_sigma = sigma_11 - np.dot(np.dot(sigma_12, inv_sigma_22), sigma_21)
    node_times = ts.tables.nodes.time
    node_locs = np.concatenate((sample_locs, np.transpose(cmvn_u)[0], u1))
    return node_times, node_locs, dispersal_rate
    """










def locate_loop_group_nodes(ARG):
    """
    First calculates the cycle basis of the graph (this utilizes the ARG with connected
    recombination nodes), then groups those loops based on whether they are interconnected.
    Returns a list of lists of nodes that are interconnected by loops in the ARG.

    THE OUTPUT HAS BEEN CHANGED FROM PREVIOUS ITERATIONS. Loops themselves are no longer
    preserved, only loop groups.
    """
    
    loop_list = nx.cycle_basis(ARG.nx_graph_connected_recomb_nodes.to_undirected())
    edges = ARG.ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    if len(loop_list) != len(ARG.recomb_nodes)/2:
        for node in ARG.recomb_nodes[::2]:
            parent = parent_list[child_list.index(node)]
            if parent == parent_list[child_list.index(node+1)]:
                loop_list.append([node, parent])
    num_loops = len(loop_list)
    loop_group_nodes = []
    if num_loops > 1:
        build_instructions = []
        for loop in loop_list:
            for n in range(len(loop)):
                if n == len(loop)-1:
                    a, b = loop[n], loop[0]
                else:
                    a, b = loop[n], loop[n+1]
                build_instructions.append([a, b])
        g = nx.Graph(build_instructions)
        loop_group_nodes = list(nx.connected_components(g))
    elif num_loops == 1:
        loop_group_nodes = [set(loop_list[0])]
    return loop_group_nodes

def generate_skeleton(ARG):
    """
    This matches Puneeth's skeleton graph but stored as a dictionary.
    """

    loop_group_nodes = locate_loop_group_nodes(ARG)
    gmrca = ARG.ts.node(ARG.ts.num_nodes-1).id
    all_loop_nodes = set([node for lg in loop_group_nodes for node in lg])
    skeleton = defaultdict(list)
    previously_found = []
    working_nodes = list(ARG.ts.samples())
    for node in working_nodes:
        no_shallower_connection = True
        for lg in loop_group_nodes:
            input_node = max(lg)
            if node < input_node:
                if nx.has_path(ARG.nx_graph_connected_recomb_nodes, source=node, target=input_node):
                    path = nx.shortest_path(ARG.nx_graph_connected_recomb_nodes, source=node, target=input_node)
                    youngest_connection_node = min(list(set(path[1:]) & all_loop_nodes))
                    #if youngest_connection_node not in lg:
                    #    continue
                    skeleton[youngest_connection_node].append(node)
                    if youngest_connection_node not in previously_found:
                        skeleton[input_node].append(youngest_connection_node)
                        previously_found.append(youngest_connection_node)
                        #if youngest_connection_node in ARG.recomb_nodes:
                        #    skeleton[input_node].append(youngest_connection_node+1)
                    if input_node != gmrca and input_node not in working_nodes:
                        working_nodes.append(input_node)
                    no_shallower_connection = False
                    break
        if no_shallower_connection:
            skeleton[gmrca].append(node)
    for group in skeleton:
        skeleton[group] = sorted(skeleton[group])
    return skeleton

def combine_cov_submatrices(skeleton, sub_cov_mats, row_column_ids, group_node, samples):
    if group_node in samples:
        return np.array([0])
    else:
        num_output_nodes = len(skeleton[group_node])
        output_mat = [list(x) for x in np.zeros((num_output_nodes, num_output_nodes))]
        for i, output_node_1 in enumerate(skeleton[group_node]):
            for j, output_node_2 in enumerate(skeleton[group_node]):
                print(group_node, output_node_1, output_node_2)
                current_cov = combine_cov_submatrices(skeleton=skeleton, sub_cov_mats=sub_cov_mats, row_column_ids=row_column_ids, group_node=output_node_2, samples=samples)
                new_cov = sub_cov_mats[group_node][row_column_ids[output_node_1]["start"]:row_column_ids[output_node_1]["stop"], row_column_ids[output_node_2]["start"]:row_column_ids[output_node_2]["stop"]]
                if i == j:
                    output_mat[i][j] = np.kron( current_cov, np.ones(new_cov.shape)) + np.kron(np.ones(current_cov.shape), new_cov)
                else:
                    output_mat[i][j] = np.kron(np.ones(current_cov.shape), new_cov)
                print(output_mat)
        print(group_node)
        print(output_mat)
        return np.bmat(output_mat)

def non_recursive_combine_cov_submatrices(skeleton, sub_cov_mats, row_column_ids, samples):
    aggregated_cov_mats = {}
    for group in sorted(skeleton.keys()):
        if all(child not in sub_cov_mats for child in skeleton[group]):
            aggregated_cov_mats[group] = sub_cov_mats[group]
        else:
            num_output_nodes = len(skeleton[group])
            output_mat = [list(x) for x in np.zeros((num_output_nodes, num_output_nodes))]
            for i, output_node_1 in enumerate(skeleton[group]):
                if output_node_1 in samples:
                    output_node_1_cov = np.array([[0]])
                else:
                    output_node_1_cov = aggregated_cov_mats[output_node_1]
                for j, output_node_2 in enumerate(skeleton[group]):
                    if output_node_2 in samples:
                        output_node_2_cov = np.array([[0]])
                    else:
                        output_node_2_cov = aggregated_cov_mats[output_node_2]
                    new_cov = sub_cov_mats[group][row_column_ids[output_node_1]["start"]:row_column_ids[output_node_1]["stop"], row_column_ids[output_node_2]["start"]:row_column_ids[output_node_2]["stop"]]
                    if output_node_1 == output_node_2:
                        output_mat[i][j] = np.kron(output_node_1_cov, np.ones(new_cov.shape)) + np.kron(np.ones(output_node_1_cov.shape), new_cov)
                    else:
                        output_mat[i][j] = np.kron(np.ones((output_node_1_cov.shape[0],output_node_2_cov.shape[1])), new_cov)
            aggregated_cov_mats[group] = np.bmat(output_mat)
    return aggregated_cov_mats[max(aggregated_cov_mats.keys())]

def calc_cov_matrix(ARG):
    """
    THIS METHOD DOES NOT MATCH UP PUNEETH'S, I misremembered it so will have to redo.
    
    Strips the ARG to its skeleton graph, where a group is a set of nodes which share the same parent
    in the skeleton graph. Calculates all of the paths between each node and the parent of the group.
    Calculates the covariance matrix between these paths.

    This function should then merge the covariance matrices of each group to build the full covariance
    matrix of the whole ARG, which is when I realized I was going about this process in a different
    way to Puneeth.
    """

    skeleton = generate_skeleton(ARG)
    sub_cov_mats = {}
    number_of_paths_per_node = {}
    for group in skeleton:
        all_paths = []
        path_counter = 0
        for output_node in skeleton[group]:
            node_paths = list(nx.all_simple_paths(ARG.nx_graph, source=output_node, target=group))
            if output_node in ARG.recomb_nodes:
                node_paths.extend(nx.all_simple_paths(ARG.nx_graph, source=output_node+1, target=group))
            number_of_paths_per_node[output_node] = {"start":path_counter,"stop":path_counter+len(node_paths)}
            path_counter += len(node_paths)
            all_paths.extend(node_paths)
        edges = ARG.ts.tables.edges
        parent_list = list(edges.parent)
        child_list = list(edges.child)
        times = np.empty((len(all_paths),len(all_paths)))
        for i, p in enumerate(all_paths):
            for j in range(i+1):
                intersect = list(set(p).intersection(all_paths[j]))
                edges = []
                for child in intersect:
                    if child != group:
                        edges.append(ARG.ts.node(parent_list[child_list.index(child)]).time - ARG.ts.node(child).time)
                times[i,j] = np.sum(edges)
                times[j,i] = times[i,j]
        sub_cov_mats[group] = times
    return non_recursive_combine_cov_submatrices(skeleton=skeleton, sub_cov_mats=sub_cov_mats, row_column_ids=number_of_paths_per_node, samples=ARG.ts.samples())


class ARGObject:
    """
    A ARGObject contains various methods for reference an ARG:
        
        ts - tskit tree sequence
        nx_graph - directed networkx graph built from tree sequence instructions
        nx_graph_connected_recomb_nodes - directed networkx graph where
            recombination nodes have been merged

    There are also an attributes that is often used with this class:
        recomb_nodes - recombination nodes (very commonly referenced)
    """

    def __init__(self, ts):
        self.ts = ts
        self.recomb_nodes = list(np.where(self.ts.tables.nodes.flags == 131072)[0])
        self.nx_graph = ts_to_nx(ts=ts, recomb_nodes=self.recomb_nodes)
        self.nx_graph_connected_recomb_nodes = ts_to_nx(ts=ts, connect_recombination_nodes=True, recomb_nodes=self.recomb_nodes)


rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=100,
    recombination_rate=1e-8,
    sequence_length=2_000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=rs#6334 #7483,8131
)

#print(ts.draw_text())
print("TREES:", ts.num_trees)
print("NODES:", len(ts.nodes()))

start = time.time()
arg = ARGObject(ts=ts)
cov_mat = calc_cov_matrix(ARG=arg)
end = time.time()
print("HYBRID - Total Execution Time:", round((end - start)/60, 2), "minutes")
print(cov_mat.shape)
np.savetxt("hybrid_cm.csv", cov_mat, delimiter=",")

start = time.time()
paths = identify_unique_paths(ts=ts)
sample_locs = np.linspace(0, 1, ts.num_samples) # evenly space the samples, ignore ordering of tree samples
sigma = reconstruct_node_locations(
    ts=ts,
    paths=paths,
    sample_locs=sample_locs
)
end = time.time()
print("PATHS - Total Execution Time:", round((end - start)/60, 2), "minutes")
print(sigma.shape)
np.savetxt("paths_cm.csv", sigma, delimiter=",")