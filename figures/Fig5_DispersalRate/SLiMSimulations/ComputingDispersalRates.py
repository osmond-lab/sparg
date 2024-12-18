# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sparg as sparg
import tskit
import numpy as np
import multiprocessing as mp 

def locate_nodes(nodes, SpARG):
    """Calculate the location of the list of nodes

    Parameters
    ----------
    nodes : list
        List of nodes for which the locations will be calculated
    SpARG : SpatialARG
    	An instance of the SpatialARG class defined below. 

    Returns
    -------
    node_locations : dict
        The keys are the nodes. The values are the geographic location of the nodes in n dimensions
    variance_in_node_locations : dict
        The keys are the nodes. The values are the variance in the geographic location of the nodes in n dimensions
    """

    sigma = SpARG.dispersal_rate_matrix
    paths = SpARG.paths
    inverted_cov_mat = SpARG.inverted_paths_shared_time_matrix
    sample_locs_to_root_locs = SpARG.path_dispersal_distances
    root_locations = SpARG.root_locations
    node_shared_times = SpARG.node_paths_shared_times
    node_paths = SpARG.node_paths
    ts = SpARG.ts
    
    location_of_nodes = {} 
    variance_in_node_locations = {} 
    
    for i,node in enumerate(nodes): 
        node_specific_sharing = node_shared_times[node,:].copy()
        matmul_prod = np.matmul(node_specific_sharing, inverted_cov_mat)
        root_location = root_locations[node_paths[node][-1]]
        node_location = root_location + np.matmul(matmul_prod, sample_locs_to_root_locs)
        location_of_nodes[node] = node_location
        
        base_variance = np.matmul(matmul_prod, np.transpose(node_specific_sharing))
        ones = np.ones(inverted_cov_mat.shape[0])
        correction_denominator = np.matmul(np.matmul(np.transpose(ones),inverted_cov_mat),ones)
        correction_numerator = (1-np.matmul(np.matmul(np.transpose(node_specific_sharing),inverted_cov_mat),ones))**2
        corrected_variance_scaling_factor = (ts.max_root_time-ts.node(node).time)-base_variance+(correction_numerator/correction_denominator)
        total_variance_node = sigma*corrected_variance_scaling_factor
        variance_in_node_locations[node] = total_variance_node
    return location_of_nodes, variance_in_node_locations



def average_dispersal_treewise(ts, locations_of_nodes):
    branch_lengths = ts.tables.nodes.time[ts.tables.edges.parent] - ts.tables.nodes.time[ts.tables.edges.child]
    child_locations = np.array(list( map(locations_of_nodes.get, ts.tables.edges.child) ))
    parent_locations = np.array(list( map(locations_of_nodes.get, ts.tables.edges.parent) ))
    branch_distances = parent_locations - child_locations 
    ts_trees = ts.aslist()
    dispersal_rate = []
    average_dispersal_rate = []
    for ts_tree in ts_trees:     
        edge_ind = ts_tree.edge_array[ts_tree.edge_array>-1]
        tree_branch_lengths = branch_lengths[edge_ind]
        tree_branch_distances = branch_distances[edge_ind]
        tree_dispersal_rate = [ np.matmul( np.transpose([tree_branch_distances[i]]),[tree_branch_distances[i]] )/tree_branch_lengths[i] for i in range(len(tree_branch_distances)) ]
        tree_dispersal_rate = np.sum(np.array(tree_dispersal_rate), axis=0)/ts.num_samples   
        dispersal_rate += [tree_dispersal_rate]
        average_dispersal_rate += [ np.average(np.array(dispersal_rate), axis=0) ]
    return dispersal_rate, average_dispersal_rate 

def disp_figure(fname):
    ts = tskit.load(fname)
    #ts = tskit.load("slim_"+str(disp)+"rep"+str(rep)+"sigma.trees")
        
    sd = 1
    np.random.seed(sd)
    keep_nodes = list(np.random.choice(ts.samples(), 100, replace=False))

    ts_sim, sim_map = ts.simplify(samples=keep_nodes, keep_input_roots=False, keep_unary=True, map_nodes=True)
    ts_filtered, filtered_maps = sparg.simplify_with_recombination(ts=ts_sim)
    ts_chopped = sparg.chop_arg(ts_filtered,1000)
    FullARG = sparg.SpatialARG(ts=ts_chopped, dimensions=2, verbose=False)
    locations_of_nodes, var_nodes = locate_nodes(nodes = range(ts_chopped.num_nodes), SpARG = FullARG)
    dispersal_rate_treewise, average_dispersal_rate_treewise = average_dispersal_treewise(ts=ts_chopped, locations_of_nodes = locations_of_nodes)
    
    outfile = open(fname+"_output"+".txt", "w")
    i = 0
    ts_breaks = ts_chopped.breakpoints(as_array=True)
    
    FI1_trees = []
    FI2_trees = []
    sigma_trees = [] 
    for (bp_i,bp) in enumerate(ts_breaks[:-1]):
        if bp_i > 0:
             if bp_i%10 == 0 or bp_i == 1 :
                if bp_i in range(0,1551,100):
                    print(bp_i)
                ts_short = ts_chopped.keep_intervals(intervals=[(0,bp)], simplify=False).trim()
                ts_short_sim, maps_short_sim = sparg.simplify_with_recombination(ts=ts_short)
                ts_short_attached = sparg.chop_arg(ts=ts_short, time = ts_short.max_time )

                #print(sd, bp_i,ts_breaks[bp_i-1], bp)
                ts_tree = ts_short_attached.keep_intervals(intervals=[(ts_breaks[bp_i-1],bp)], simplify=False).trim() 
                ts_tree = sparg.chop_arg(ts=ts_tree,time= ts_tree.max_time)

                PartialARG = sparg.SpatialARG(ts=ts_short_attached,dimensions=2, verbose=False)
                dispersal_rate = PartialARG.dispersal_rate_matrix
                FI1 = PartialARG.fishers_information_1
                FI2 = PartialARG.fishers_information_2 
                loc_nodes, var_nodes = locate_nodes(nodes = range(ts_short_attached.num_nodes), SpARG = PartialARG)

                disp_ARG, avg_disp_ARG = average_dispersal_treewise(ts=ts_short_attached, locations_of_nodes = loc_nodes)

                Tree = sparg.SpatialARG(ts=ts_tree, dimensions=2, verbose=False)

                dispersal_rate_tree = Tree.dispersal_rate_matrix
                FI1_tree = Tree.fishers_information_1 
                FI2_tree = Tree.fishers_information_2

                FI1_trees += [ dispersal_rate_tree[0][0]**2*FI1 ]
                FI2_trees += [ dispersal_rate_tree[0][0]**3*FI2 ]
                sigma_trees += [ dispersal_rate_tree[0][0] ]

                sigma_avg = np.average(sigma_trees)
                FI1_avg = np.sum(FI1_trees)/(sigma_avg**2)
                FI2_avg = np.sum(FI2_trees)/(sigma_avg**3)

                outfile.write(str(ts_short_attached.num_trees) + " " + str(bp) + " " + str(dispersal_rate[0][0]) + " " + str(dispersal_rate[1][1]) + " " + str(FI1 + FI2)+ " " + str(sigma_avg) + " " + str(FI1_avg + FI2_avg) + " " + str(dispersal_rate_treewise[bp_i][0][0])+ " " + str(dispersal_rate_treewise[bp_i][1][1]) + " " + str(average_dispersal_rate_treewise[bp_i][0][0])+ " " + str(average_dispersal_rate_treewise[bp_i][1][1]) + " " + str(avg_disp_ARG[-1][0][0])+ " " + str(avg_disp_ARG[-1][1][1]) + "\n")

    outfile.close()
    return 0 

mp_cpu = mp.cpu_count()
pool = mp.Pool(mp_cpu-4)

fnames = [] 


for mating in np.arange(0.1,1.1,0.1):
    for rep in range(1,6):
        mating = round(mating,2)
        comp = 1
        fname = "slim_disp0.25" +"_mating"+str(mating)+"_comp"+str(comp)+ "rep"+str(rep)+ "sigma.trees"
        fnames += [(fname,)]

for comp in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
    for rep in range(1,6):
        comp = round(comp,2)
        mating = 1
        fname = "slim_disp0.25" +"_mating"+str(mating)+"_comp"+str(comp)+ "rep"+str(rep)+ "sigma.trees"
        fnames += [(fname,)]        

results = pool.starmap(disp_figure, fnames)
print("end")
pool.close()
pool.join()

