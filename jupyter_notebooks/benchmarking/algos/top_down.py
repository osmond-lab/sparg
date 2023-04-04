import msprime
import random
import numpy as np
from collections import defaultdict
import time


def calc_covariance_matrix(ts):
    gmrca = ts.node(ts.num_nodes-1).id
    recomb_nodes = np.where(ts.tables.nodes.flags == 131072)[0]
    recomb_nodes_to_convert = dict(zip(recomb_nodes[1::2], recomb_nodes[::2]))
    edges = ts.tables.edges
    CovMat = np.matrix([[0.0]])
    Indices = defaultdict(list)
    Indices[gmrca] = [0]
    edges = ts.tables.edges
    for node in reversed(list(ts.nodes())):
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
    return CovMat

def benchmark(ts):
    start = time.time()
    cov_mat = calc_covariance_matrix(ts=ts)
    end = time.time()
    return end-start, cov_mat.sum(), "NA"

if __name__ == "__main__":
    rs = random.randint(0,10000)
    print(7960)
    ts = msprime.sim_ancestry(
        samples=2,
        recombination_rate=1e-8,
        sequence_length=2_000,
        population_size=10_000,
        record_full_arg=True,
        random_seed=7960
    )
    print(ts.draw_text())
    cov_mat = calc_covariance_matrix(ts=ts)
    #print(cov_mat.sum())
    #np.savetxt("topdown.csv", cov_mat, delimiter=",")

    #print(benchmark(ts=ts))

    #true_cov_mat, paths = paths_modified.calc_covariance_matrix(ts=ts)
    #np.savetxt("paths_modified.csv", true_cov_mat, delimiter=",")
    #print(paths)