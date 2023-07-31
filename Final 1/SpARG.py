#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:36:48 2023

@author: puneeth
"""

import msprime
import random
import numpy as np
from collections import defaultdict
import sympy as sym
import matplotlib.pyplot as plt 

def calc_covariance_matrix(ts, internal_nodes = 'None' ):
    """
    Parameters
    ----------
    ts : Tree Sequence
        DESCRIPTION.
    internal_nodes : string or list 
        DESCRIPTION. A list of internal nodes for which you want the shared times. 
        It can take values 'None', 'All' or a list. The default is 'None'.

    Returns
    -------
    CovMat : TYPE
        DESCRIPTION.
    Paths : TYPE
        DESCRIPTION.
    shared_time : TYPE
        DESCRIPTION.

    """
    edges = ts.tables.edges 
    
    #Covariance Matrix of Samples
    CovMat = np.zeros(shape=(ts.num_samples, ts.num_samples)) #Initialize the covariance matrix. Initial size = #samples. Will increase to #paths
    Indices = defaultdict(list) #Keeps track of the indices of paths that enter (from bottom) a particular node.
    for i, sample in enumerate(ts.samples()):
        Indices[sample] = [i] #Initialize indices for each path which at this point also corresponds to the sample.    
    Paths = [[sample] for sample in ts.samples()] #Keeps track of different paths. To begin with, as many paths as samples.
    
    #Shared Time of Internal Nodes 
    int_nodes = {}
    if internal_nodes != 'None':
        if internal_nodes =='All': 
            int_nodes = {nd.id:i for i,nd in enumerate(ts.nodes()) }    
            internal_paths = [ [nd.id] for nd in ts.nodes() ]
        else:
            int_nodes = {nd:i for i,nd in enumerate(internal_nodes) }    
            internal_paths = [ [nd] for nd in internal_nodes ]
        shared_time = np.zeros(shape=(len(int_nodes),ts.num_samples)) 
        internal_indices = defaultdict(list) #For each path, identifies internal nodes that are using that path for shared times.
    
    
    for node in ts.nodes():
        path_ind = Indices[node.id]
        parent_nodes = np.unique(edges.parent[np.where(edges.child == node.id)])
        
        if internal_nodes != 'None': 
            if node.id in int_nodes: 
                internal_indices[path_ind[0]] += [int_nodes[node.id]]
                
        for i, parent in enumerate(parent_nodes):
            for path in path_ind:
                if i == 0:
                    Paths[path].append(parent)
                    for internal_path_ind in internal_indices[path]: 
                        internal_paths[internal_path_ind] += [parent]
                else:
                    Paths.append(Paths[path][:])
                    Paths[-1][-1] = parent
                
                    
        
                    
        npaths = len(path_ind)
        nparent = len(parent_nodes)
        
        if nparent == 0:
            continue
        else:
            edge_len = ts.node(parent_nodes[0]).time - node.time
            
            CovMat = np.hstack(  (CovMat,) + tuple( ( CovMat[:,path_ind] for j in range(nparent-1) ) ) ) #Duplicate the columns
            CovMat = np.vstack(  (CovMat,) + tuple( ( CovMat[path_ind,:] for j in range(nparent-1) ) ) ) #Duplicate the rows
            new_ind = path_ind + [len(CovMat) + x for x in range((-(nparent-1)*len(path_ind)),0)]
            CovMat[ np.ix_( new_ind, new_ind ) ] += edge_len
            for i,parent in enumerate(parent_nodes): 
                Indices[parent] += new_ind[i*npaths:(i+1)*npaths]
                
            if internal_nodes == 'None': 
                shared_time = [] 
            else : 
                shared_time = np.hstack( (shared_time, ) + tuple( ( shared_time[:,path_ind] for j in range(nparent-1) ) ) )
                int_nodes_update = []
                for i in path_ind: 
                    int_nodes_update += internal_indices[i]
                shared_time[ np.ix_( int_nodes_update, new_ind) ] += edge_len

    return CovMat, Paths, [list(int_nodes.keys()),shared_time, internal_paths]


def MLE(S_inv, loc, path_roots, n) :  
    """
    
    Parameters
    ----------
    S_inv : TYPE
        DESCRIPTION.
    loc : TYPE
        DESCRIPTION.
    path_roots : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    """
    S_inv = np.array(S_inv)
    lenloc = len(loc)
    mu_vect = np.zeros((lenloc,1)) 
    roots = np.unique(path_roots)
    k = len(roots)
    R  = np.zeros((k,lenloc)) #dij is 1 if the jth path has the ith root
    for i,root in enumerate(roots): 
        R[i][np.where( np.array(path_roots) == root)[0]] += 1.0
    
    A = np.matmul(R,np.matmul(S_inv,np.transpose(R))) #Matrix of coefficients of the system of linear equations 
    b = np.matmul(R,np.matmul(S_inv,loc)) #Vector of constants of the system of linear equations. 
    augmented_matrix = np.column_stack((A, b)) # Construct the augmented matrix [A|b]
    rre_form, pivots = sym.Matrix(augmented_matrix).rref() # Perform row reduction on the augmented matrix
    null_space = sym.Matrix(A).nullspace() # Find the basis of the null space of A i.e. all x such that Ax = 0
    
    if len(pivots) == A.shape[0]:
        if int(A.shape[0]) in pivots: 
            raise TypeError("No Solutions")
        else: 
            # print("Unique Solution")
            mu_list = np.array(rre_form.col(-1))
            mu_vect = np.matmul(np.transpose(R),mu_list)    
            sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
            return mu_list, sigma, roots, R
    else: 
        print("Multiple Solution")
        mu_particular = rre_form.col(-1)
        mu_vect = np.matmul(np.transpose(R),mu_particular)    
        sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
        
        # sigma_list = []
        # for rep in range(20): 
        #     mu_list = mu_particular
        #     mu_vect = np.matmul(np.transpose(R),mu_list)    
        #     sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
        #     sigma_list += [float(sigma[0][0])]
        # print(sigma_list)
        #     for col in null_space:
        #         a = np.random.uniform(-50,50)
        #         mu_list += a*col
        # print( np.average(sigma_list))
        return mu_particular, sigma, roots, R

def ARG_estimate(ts, internal_nodes = 'All', return_roots = False): 
    CM, paths, internal_data = calc_covariance_matrix(ts, internal_nodes = internal_nodes)  
    path_roots = [ row[-1] for row in paths ]
    samples = [ row[0] for row in paths ] 
    loc = np.zeros((CM.shape[0],1))
    for i,nd in enumerate(samples): 
        ind = ts.tables.nodes[nd].individual
        loc[i][0] = ts.tables.individuals[ind].location[0]
    # print(CM,loc)
    CMinv = np.linalg.pinv(CM)
    mu_roots, sigma, roots, R  = MLE(CMinv, loc, path_roots, len(np.unique(samples)))
    
    internal_path_roots = [ internal_path[-1] for internal_path in internal_data[-1]  ]
    # print(mu_roots, [rt for rt in internal_path_roots])
    internal_path_root_locations = [ mu_roots[np.where(roots == rt)[0]][0] for rt in internal_path_roots ]
    internal_path_root_locations = np.array(internal_path_root_locations)
    internal_locations = internal_path_root_locations + np.matmul(np.matmul(internal_data[1], CMinv),loc - np.matmul(np.transpose(R),mu_roots))
                            
    internal_loc_variance = [] 
    for i,nd in enumerate(internal_data[0]) : 
        t_int = ts.tables.nodes.time[nd]
        s_int = np.array([internal_data[1][i]])
        print(s_int.shape)
        var_int = t_int - np.matmul(np.matmul(s_int,CMinv),np.transpose(s_int)) 
        # + (1 - np.sum(np.matmul(s_int,CMinv) ) )**2/(np.sum(CMinv))
        internal_loc_variance += [sigma*var_int]
        
    return mu_roots, sigma, internal_locations, internal_loc_variance

    
    
    
if __name__ == "__main__":
    
    rs = random.randint(0, 10000)
    print("Random Seed:", rs)

    ts = msprime.sim_ancestry(
        samples=2,
        recombination_rate=1e-8,
        sequence_length=2_000,
        population_size=10_000,
        record_full_arg=True,
        random_seed=rs
    )
    
    
    print(ts.draw_text())
    
    cov_mat, paths, internal_data = calc_covariance_matrix(ts=ts, internal_nodes=[5,8])

    tstables = ts.dump_tables()
    loc_list = [] 
    for ind in tstables.individuals : 
        loc_list += [ [ np.random.uniform(-100,100) ] ]
    
    tstables.individuals.packset_location(np.array( loc_list ))
    tstables.sort()
    ts_loc = tstables.tree_sequence()
    
    mu_roots, sigma, internal_locations, internal_var  = ARG_estimate(ts=ts_loc, internal_nodes = [5,8] )
    
    