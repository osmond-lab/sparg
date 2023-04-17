import msprime
import random
import numpy as np
from collections import defaultdict
import time
import Toy_Examples


def calc_covariance_matrix(ts):
    nodes_id = sorted(np.unique(list(ts.tables.edges.parent) + list(ts.tables.edges.child)))
    nodes = [nd for nd in ts.nodes() if nd.id in nodes_id ]
    
    # gmrca = ts.node(ts.num_nodes-1).id
    # gmrca = max(nodes_id, key = lambda nd : ts.node(nd).time )
    gmrca = list(set(nodes_id).difference(set( ts.tables.edges.child )) )
    n_gmrca = len(gmrca)
    
    recomb_nodes = np.where(ts.tables.nodes.flags == 131072)[0]
    recomb_nodes_to_convert = dict(zip(recomb_nodes[1::2], recomb_nodes[::2]))
    recomb_nodes_to_convert = { nd: recomb_nodes_to_convert[nd] for nd in recomb_nodes_to_convert if nd in nodes_id and recomb_nodes_to_convert[nd] in nodes_id }
    # print('Recomb Nodes', recomb_nodes, recomb_nodes_to_convert)
    edges = ts.tables.edges
    CovMat = np.matrix(np.zeros((n_gmrca,n_gmrca)))
    Indices = defaultdict(list)
    for i,nd in enumerate(gmrca): 
        Indices[nd] = [i]
    edges = ts.tables.edges
    nodes = sorted(nodes,key = lambda nd:nd.time,reverse = True)
    for node in nodes:
        # print(node.id, node.time, Indices)
        # print(node)
        
        if node.id in recomb_nodes_to_convert or node.flags == 1:
            continue
        path_ind = Indices[node.id]
        npaths = len(path_ind)
        child_nodes = np.unique(edges.child[np.where(edges.parent == node.id)])
        for i, child in enumerate(child_nodes):
            if child in recomb_nodes_to_convert:
                child_nodes[i] = recomb_nodes_to_convert[child]
        nchild = len(child_nodes)
        if nchild == 1:
            # print('Enter', path_ind)
            child = child_nodes[0]
            edge_len = node.time - ts.node(child).time
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_len*np.ones((npaths,npaths))
            Indices[child] += path_ind
        else:
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
        # print(node.id, '----------------------------------------------------------' )
        # print(CovMat, Indices)
        # print(Indices)
    return CovMat, Indices

def MLE(S_inv, loc, n) :  
    S_inv = np.array(S_inv)
    # print(loc)
    lenloc = len(loc)
    one_vect = np.ones((lenloc,1)) 
    # print(np.matmul(np.matmul(np.transpose(one_vect), S_inv),loc), np.matmul(np.matmul(np.transpose(one_vect), S_inv),one_vect) ) 
    mu = np.matmul(np.matmul(np.transpose(one_vect), S_inv),loc) / np.matmul(np.matmul(np.transpose(one_vect), S_inv),one_vect)
    # print(mu)
    sigma = np.matmul(np.matmul(np.transpose(loc - mu*one_vect), S_inv), (loc - mu*one_vect))/n
    return mu, sigma


def ARG_estimate(ts): 
    CM, indices = calc_covariance_matrix(ts)  
    # print(CM)
    # S = [ i for i,row in enumerate(ts.tables.nodes) if row.flags == 1 ] 
    S = list(ts.samples())
    # print(S, indices)
    loc = np.zeros((CM.shape[0],1))
    for i in S: 
        # print(i)
        for j in indices[i]:
            ind = ts.tables.nodes[i].individual
            # print(ind)
            loc[j][0] = ts.tables.individuals[ind].location[0]
    CMinv = np.linalg.pinv(CM)
    # print(CMinv, loc)
    mu, sigma = MLE(CMinv, loc, len(S))
    return mu, sigma

def benchmark(ts):
    start = time.time()
    cov_mat = calc_covariance_matrix(ts=ts)
    end = time.time()
    return end-start, cov_mat.sum(), "NA"

if __name__ == "__main__":
    
    ts = Toy_Examples.ts_singlecompound()
    CM, Ind = calc_covariance_matrix(ts)
    print(CM)
    
    #for i in range(1000):
    rs = random.randint(0,10000)
    print(rs)
    ts = msprime.sim_ancestry(
        samples=2,
        recombination_rate=1e-8,
        sequence_length=2_000,
        population_size=10_000,
        record_full_arg=True,
        random_seed=9203
    )
    
    
    
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
        
