import msprime
import random
import numpy as np
from collections import defaultdict
import time
import Toy_Examples
import sympy as sym
import matplotlib.pyplot as plt

def calc_covariance_matrix(ts):
    nodes_id = sorted(np.unique(list(ts.tables.edges.parent) + list(ts.tables.edges.child)))
    # print(len(list(ts.nodes())), len(nodes_id))
    nodes = [nd for nd in ts.nodes() if nd.id in nodes_id ]
    
    # gmrca = ts.node(ts.num_nodes-1).id
    # gmrca = max(nodes_id, key = lambda nd : ts.node(nd).time )
    roots = list(set(nodes_id).difference(set( ts.tables.edges.child )) )
    # print(roots)
    n_gmrca = len(roots)
    
    recomb_nodes = np.where(ts.tables.nodes.flags == 131072)[0]
    recomb_nodes_to_convert = dict(zip(recomb_nodes[1::2], recomb_nodes[::2]))
    recomb_nodes_to_convert = { nd: recomb_nodes_to_convert[nd] for nd in recomb_nodes_to_convert if nd in nodes_id and recomb_nodes_to_convert[nd] in nodes_id }
    # print('Recomb Nodes', recomb_nodes, recomb_nodes_to_convert)
    edges = ts.tables.edges
    CovMat = np.matrix(np.zeros((n_gmrca,n_gmrca)))
    Indices = defaultdict(list)
    RootIndices = []
    for i,nd in enumerate(roots): 
        Indices[nd] = [i]
        RootIndices += [nd]
    edges = ts.tables.edges
    nodes = sorted(nodes,key = lambda nd:nd.time,reverse = True)
    for node in nodes:
        # print(node.id, node.time)
        # print(node)
        
        if node.id in recomb_nodes_to_convert or node.flags == 1:
            continue
        path_ind = Indices[node.id]
        npaths = len(path_ind)
        # if npaths == 0: 
        #     print(Indices, roots, recomb_nodes_to_convert )
        child_nodes = np.unique(edges.child[np.where(edges.parent == node.id)])
        for i, child in enumerate(child_nodes):
            if child in recomb_nodes_to_convert:
                child_nodes[i] = recomb_nodes_to_convert[child]
        nchild = len(child_nodes)
        if nchild == 1:
            # print('Enter', path_ind, RootIndices)
            child = child_nodes[0]
            edge_len = node.time - ts.node(child).time
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_len*np.ones((npaths,npaths))
            Indices[child] += path_ind
        else:
            # print('Enter', path_ind, RootIndices)
            edge_lens = [ node.time - ts.node(child).time for child in child_nodes ]
            existing_paths = CovMat.shape[0]
            CovMat = np.hstack(  (CovMat,) + tuple( ( CovMat[:,path_ind] for j in range(nchild-1) ) ) ) #Duplicate the rows
            CovMat = np.vstack(  (CovMat,) + tuple( ( CovMat[path_ind,:] for j in range(nchild-1) ) ) ) #Duplicate the columns
            # print(edge_lens, child_nodes, node.id)
            # print(CovMat)
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_lens[0]*np.ones((npaths,npaths))
            Indices[ child_nodes[0] ] += path_ind
            for child_ind in range(1,nchild):
                mod_ind = range(existing_paths+ npaths*(child_ind-1),existing_paths + npaths*child_ind) #indices of the entries that will be modified
                CovMat[ np.ix_( mod_ind , mod_ind  ) ] += edge_lens[child_ind]*np.ones( (npaths,npaths) )
                Indices[ child_nodes[child_ind] ] += mod_ind
                RootIndices += list( np.array(RootIndices)[ np.array(path_ind) ] )
        # print(child_nodes)
        # print(node.id, '----------------------------------------------------------' )
        # print(CovMat, Indices)
        # print(Indices)
    return CovMat, Indices, RootIndices

def MLE(S_inv, loc, rootind, n) :  
    S_inv = np.array(S_inv)
    lenloc = len(loc)
    # one_vect = np.ones((lenloc,1)) 
    mu_vect = np.zeros((lenloc,1)) 
    roots = np.unique(rootind)
    k = len(roots)
    D  = np.zeros((k,lenloc))
    for i,root in enumerate(roots): 
        D[i][np.where( np.array(rootind) == root)[0]] += 1.0
    
    A = np.matmul(D,np.matmul(S_inv,np.transpose(D))) #Matrix of coefficients of the system of linear equations 
    b = np.matmul(D,np.matmul(S_inv,loc)) #Vector of constants of the system of linear equations. 
    augmented_matrix = np.column_stack((A, b)) # Construct the augmented matrix [A|b]
    rre_form, pivots = sym.Matrix(augmented_matrix).rref() # Perform row reduction on the augmented matrix
    null_space = sym.Matrix(A).nullspace() # Find the basis of the null space of A i.e. all x such that Ax = 0
    
    if len(pivots) == A.shape[0]:
        if int(A.shape[0]) in pivots: 
            raise TypeError("No Solutions")
        else: 
            print("Unique Solution")
            mu_list = np.array(rre_form.col(-1))
            mu_vect = np.matmul(np.transpose(D),mu_list)    
            sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
            return mu_list, sigma[0][0]
    else: 
        print("Multiple Solution", D, roots, null_space)
        mu_particular = rre_form.col(-1)
        sigma_list = []
        
        for rep in range(20): 
            mu_list = mu_particular
            for col in null_space:
                a = np.random.uniform(-50,50)
                mu_list += a*col
            mu_vect = np.matmul(np.transpose(D),mu_list)    
            sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
            sigma_list += [float(sigma[0][0])]
        print(sigma_list)
        print( np.average(sigma_list))
        return mu_list, np.average(sigma_list)
    

# def MLE(S_inv, loc, rootind, n) :  
#     S_inv = np.array(S_inv)
#     lenloc = len(loc)
#     # one_vect = np.ones((lenloc,1)) 
#     mu_vect = np.zeros((lenloc,1)) 
#     roots = np.unique(rootind)
#     k = len(roots)
#     D  = np.zeros((k,lenloc))
#     for i,root in enumerate(roots): 
#         D[i][np.where( np.array(rootind) == root)[0]] += 1.0
    
    
#     mu_list = np.linalg.solve(np.matmul(D,np.matmul(S_inv,np.transpose(D))), np.matmul(D,np.matmul(S_inv,loc)))
#     # print(mu_list)
#     for i,root in enumerate(roots):
#         print(i,roots,len(mu_vect),np.where( np.array(rootind) == root),mu_vect,mu_list)
#         mu_vect[np.where( np.array(rootind) == root)[0]][1] += mu_list[i] 
#     print(mu_vect)
#     # mu = np.matmul(np.matmul(np.transpose(one_vect), S_inv),loc) / np.matmul(np.matmul(np.transpose(one_vect), S_inv),one_vect)
#     # sigma = np.matmul(np.matmul(np.transpose(loc - mu*one_vect), S_inv), (loc - mu*one_vect))/n
    
#     sigma = np.matmul(np.matmul(np.transpose(loc - mu_vect), S_inv), (loc - mu_vect))/n
    
#     return mu_list, sigma


def ARG_estimate(ts): 
    CM, indices, rootind = calc_covariance_matrix(ts)  
    S = list(ts.samples())
    loc = np.zeros((CM.shape[0],1))
    for i in S: 
        for j in indices[i]:
            ind = ts.tables.nodes[i].individual
            loc[j][0] = ts.tables.individuals[ind].location[0]
    
    CMinv = np.linalg.pinv(CM)
    mu, sigma = MLE(CMinv, loc, rootind, len(S))
    return mu, sigma

def benchmark(ts):
    start = time.time()
    cov_mat = calc_covariance_matrix(ts=ts)
    end = time.time()
    return end-start, cov_mat.sum(), "NA"

if __name__ == "__main__":
    sigma_plot = []
    t_rng = np.arange(0,2.5,0.01)
    for t in t_rng:
        ts = Toy_Examples.ts_stacked(n=3)
        # ts = ts.decapitate(time=1.25)
        ts = ts.decapitate(time=t)
        # CM, Ind, RootInd = calc_covariance_matrix(ts)
        mu, sigma = ARG_estimate(ts)
        sigma_plot += [sigma] 
        print(t,sigma)
    plt.plot(t_rng, sigma_plot)
    
    # for i in range(1000):
    # rs = random.randint(0,10000)
    # print(rs)
    # ts = msprime.sim_ancestry(
    #     samples=3,
    #     recombination_rate=1e-8,
    #     sequence_length=2_000,
    #     population_size=10_000,
    #     record_full_arg=True,
    #     random_seed=rs
    # )
    # mu, sigma = ARG_estimate(ts)
    
    
    # print(ts.draw_text())
    
    #print(cov_mat.sum())
    #np.savetxt("topdown.csv", cov_mat, delimiter=",")

    #print(benchmark(ts=ts))

    # cov_mat = calc_covariance_matrix(ts=ts)
    #print(cov_mat.sum(), cov_mat.shape[0])
    #true_cov_mat, paths = paths_modified.calc_covariance_matrix(ts=ts)
    #print(true_cov_mat.sum(), true_cov_mat.shape[0])
    #print(paths)
    #exit()
    #np.savetxt("paths_modified.csv", true_cov_mat, delimiter=",")
    #if round(cov_mat.sum()) != round(true_cov_mat.sum()):
    #    print(rs, "FAIL", cov_mat.sum(), true_cov_mat.sum())
        
