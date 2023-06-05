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

def locate_loops(g, cycle_root=-1):
    """
    Finds loops within the ARG. I thought that it would be easiest to utilize functions from
    networkx package. Identifies recombination events, converts the tree sequence into a networkx
    graph. The paired recombination nodes are merged together in this graph. Converts graph to 
    undirected, then calculates cycle basis. This does not identify 'bubbles', so we need to add
    an extra step to this.
    """
    
    if cycle_root < 0:
        cycle_root = list(g.nodes())[0]
    recomb_nodes = [ x for x in g.nodes() if len(list(g.successors(x))) == 2 ]
    # print(recomb_nodes)
    g_un = g.to_undirected()
    loop_list = nx.cycle_basis(g_un, root=cycle_root)
    if len(loop_list) != len(recomb_nodes):
        for node in recomb_nodes:
            parent = list(g.successors(node))[0]
            loop_list.append([node, parent])  
    return loop_list

def locate_loops_ts(ts, cycle_root=-1):
    """
    Finds loops within the ARG. I thought that it would be easiest to utilize functions from
    networkx package. Identifies recombination events, converts the tree sequence into a networkx
    graph. The paired recombination nodes are merged together in this graph. Converts graph to 
    undirected, then calculates cycle basis. This does not identify 'bubbles', so we need to add
    an extra step to this.
    """
    
    if cycle_root < 0:
        cycle_root = ts.node(ts.num_nodes-1).id
    edges = ts.tables.edges
    parent_list = list(edges.parent)
    child_list = list(edges.child)
    recomb_nodes = list(np.where(ts.tables.nodes.flags == 131072)[0])
    g = ts_to_nx(ts=ts, connect_recombination_nodes=True, recomb_nodes=recomb_nodes)
    g = g.to_undirected()
    loop_list = nx.cycle_basis(g, root=cycle_root)
    if len(loop_list) != len(recomb_nodes)/2:
        for node in recomb_nodes[::2]:
            parent = parent_list[child_list.index(node)]
            if parent == parent_list[child_list.index(node+1)]:
                loop_list.append([node, parent])  
    return loop_list

def group_loops(loops, plot=False):
    """
    Groups intersecting loops in list. Builds networkx graph based on the loop list. Determines
    if the nodes are connected through the graph. Returns a list of lists of loops.
    """
    
    num_loops = len(loops)
    if num_loops == 0:
        return []
    else:
        if num_loops > 1:
            build_instructions = []
            for loop in loops:
                for n in range(len(loop)):
                    if n == len(loop)-1:
                        a, b = loop[n], loop[0]
                    else:
                        a, b = loop[n], loop[n+1]
                    build_instructions.append([a, b])
            g = nx.Graph(build_instructions)
            grouped_nodes = list(nx.connected_components(g))
            if plot:
                nx.draw(g, with_labels=True)
            grouped_loops = [[] for i in range(len(grouped_nodes))]
            for loop in loops:
                for i in range(len(grouped_loops)):
                    if loop[0] in grouped_nodes[i]:
                        grouped_loops[i].append(loop)
            return grouped_loops
        else:
            return [loops]

    
def Cov(edge_path1,edge_path2,G):
    """ 
    edge_path1 : A path in the graph G as a list of edges 
    edge_path2 : A path in the graph G as a list of edges 
    G : A graph G in which the paths exist and the nodes have an attribute time
    returns the Covariance between the two paths = the shared time between the two paths in G.
    """
    # edges_path1 = set(path_to_edges(path1))
    # edges_path2 = set(path_to_edges(path2))
    # print('check',edge_path2)
    common_edges = set(edge_path1).intersection(set(edge_path2))
    cov = 0 
    for edge in common_edges:
        t_edge = G.nodes()[edge[1]]['time'] - G.nodes()[edge[0]]['time']
        cov += t_edge 
    return cov 



rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=10,
    recombination_rate=1e-8,
    sequence_length=2000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=3439
)

print(ts.num_trees)

# Paths Method
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
np.savetxt("paths_cm.csv", sigma, delimiter=",")

print(len(paths), sigma.shape)

# Hybrid Method
start = time.time()
Sample_Nodes = list(ts.samples())
G = ts_to_nx(ts=ts, connect_recombination_nodes=True)
loops = locate_loops_ts(ts=ts) #Identify each loop as a list of nodes 
grouped_loops = group_loops(loops=loops) #Group the loops if they shared edges
vGMRCA = max(list(G.nodes()), key = lambda nd:G.nodes()[nd]['time'] ) #This is the GMRCA of the entire ARG identified as the oldest node in the ARG 
nodes_in_all_groups = [] #This will store all the nodes present in any loop in the Graph 
Grp_details = {} #This will store information about each group of nodes including the index of the group, the nodes in the group and the oldest node in the group
ind = 0 #This will keep track of the number of groups of loops present 
for Lp_grp in grouped_loops: 
    nodes_in_group = [] #To keep track of nodes in a given group of loops 
    for lp in Lp_grp: 
        nodes_in_group += lp
        nodes_in_all_groups += lp
    Grp_details[ind] = {'ind':ind, 'nodes': nodes_in_group, 'maximum': max(nodes_in_group, key = lambda nd: G.nodes()[nd]['time']) }
    ind += 1 
nodes_in_all_groups = set(nodes_in_all_groups)

G_skeleton = nx.DiGraph() #This will record connection nodes after we break down the ARG into simple components (trees and loops)
S = Sample_Nodes
while len(S) > 0 :
    sample_connects = {} #For each s in S, this will record the earliest ancestor of s that is part of any loop
    for s in S: 
        if s == vGMRCA : 
            continue 
        
        group_connections = [vGMRCA] #This list will containt all the nodes in any loop that s is connected to. vGMRCA is included here for algorithmic purposes. 
        for g in nodes_in_all_groups : 
            if nx.has_path(G,s,g) and s != g:  
                group_connections += [g]
        
        earliest_connection = min(group_connections, key = lambda nd: G.nodes()[nd]['time']  )#Choose the earliest in group_connection
        sample_connects[s] = earliest_connection
        G_skeleton.add_node(earliest_connection, time= G.nodes()[earliest_connection]['time'] )
        G_skeleton.add_edge(s,earliest_connection, typ ='tree', paths = list(nx.all_simple_edge_paths(G,s,earliest_connection)) ) #To each edge in the skeleton, we all attribute all the paths between the two nodes in the real Graph. The attribute typ keeps track of whether this edge is part of a loop or a tree. typ attribute was mainly included for verification purposes and can eventually be removed. 
    
    loop_st_nds = list(sample_connects.values()) #This is the list of nodes in the loops which are connected to S 
    loop_st_nds_grpwise = { ind: list( set(Grp_details[ind]['nodes']).intersection(set(loop_st_nds))) for ind in Grp_details if len(list( set(Grp_details[ind]['nodes']).intersection(set(loop_st_nds)))) != 0 } #We divide and label these nodes according the group of loops they belong to.
    
    S_new = [] #The next set of S will be the the MRCA of each loop involved in this step
    for ind in loop_st_nds_grpwise : 
        loop_MRCA = Grp_details[ind]['maximum']
        S_new += [loop_MRCA]
        G_skeleton.add_node( loop_MRCA )
        for v in loop_st_nds_grpwise[ind]: 
            G_skeleton.add_edge(v,loop_MRCA, typ='loop', paths = list(nx.all_simple_edge_paths(G,v,loop_MRCA)))
    S = S_new
    Gske_nodes = sorted(G_skeleton.nodes, key = lambda nd:G.nodes()[nd]['time']) #Ordering the nodes in the skeleton according to their times

#To each node in G_skeleton we will associcate two matrices. CovMatrix of a nodes will be the covariance matrix of all the paths from its immediate predecessor to itself where the CovFullMatrix will be the covariance matrix from the samples to itself. 

for v in Gske_nodes: 
    if v in Sample_Nodes:
        G_skeleton.add_node(v, CovMatrix = np.matrix([0]), CovMatrixFull = np.matrix([0]))
    else:    
        v_pred  = list(sorted(G_skeleton.predecessors(v), key = lambda nd:G.nodes()[nd]['time'])) #The predecessors nodes of v
        n_pred = len(v_pred)
        CovMat = {} #This will record the CovMatrix between all the paths from the predecessors to v. This is stored as a dictionary of dictonary. The [v1][v2] entry is the covariance between paths from v1 to v and v2 to v. 
        CovMatFull = [] #This will record the CovMatrixFull which the covariance matrix from all paths from the samples to v
        for v1 in v_pred:
            Covrow = {}
            CovFullrow = []
            
            v1_paths = G_skeleton.edges()[(v1,v)]['paths']
            n1 = len(v1_paths)
            v1FullMat = G_skeleton.nodes()[v1]['CovMatrixFull'] #The covariance matrix between all paths from the samples to v1
            m1 = len(v1FullMat)
            
            for v2 in v_pred : 
                print('v2',v2)
                v2_paths = G_skeleton.edges()[(v2,v)]['paths']
                n2 = len(v2_paths)
                v2FullMat = G_skeleton.nodes()[v2]['CovMatrixFull'] #The covariance matrix between all paths from the samples to v2
                m2 = len(v2FullMat)
                
                Matv1v2 = np.matrix( [ [ Cov(path1,path2,G) for path2 in v2_paths] for path1 in v1_paths ] )
                MatFullv1v2 = np.kron(np.matrix(np.ones( (m1,m2) )), Matv1v2 )
                print('Matv1v2',Matv1v2)
                print('MatFullv1v2',MatFullv1v2)
                if v1 == v2 :
                    MatFullv1v2 = MatFullv1v2 + np.kron( v1FullMat, np.matrix(np.ones( (n1,n1) )) ) 
                Covrow[v2] = Matv1v2
                CovFullrow += [ MatFullv1v2 ]
            CovMat[v1] = Covrow
            CovMatFull += [CovFullrow ]
        
        G_skeleton.add_node(v, CovMatrix = CovMat, CovMatrixFull = np.bmat(CovMatFull))
CMFull = G_skeleton.nodes()[vGMRCA]['CovMatrixFull'] 
end = time.time()
print("HYBRID - Total Execution Time:", round((end - start)/60, 2), "minutes")
np.savetxt("hybrid_cm.csv", CMFull, delimiter=",")

print(CMFull.shape)