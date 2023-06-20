import msprime
import random
import numpy as np
from collections import defaultdict
import time


def calc_covariance_matrix(ts):
    edges = ts.tables.edges
    CovMat = np.zeros(shape=(ts.num_samples, ts.num_samples))
    Indices = defaultdict(list)
    for i, sample in enumerate(ts.samples()):
        Indices[sample] = [i]
    Paths = [[sample] for sample in ts.samples()]
    for node in ts.nodes():
        path_ind = Indices[node.id]
        parent_nodes = np.unique(edges.parent[np.where(edges.child == node.id)])
        for i, parent in enumerate(parent_nodes):
            for path in path_ind:
                if i > 0:
                    Paths.append(Paths[path][:])
                    Paths[-1][-1] = parent
                else:
                    Paths[path].append(parent)
        npaths = len(path_ind)
        nparent = len(parent_nodes)
        if nparent == 0:
            continue
        elif nparent == 1:
            parent = parent_nodes[0]
            edge_len = ts.node(parent).time - node.time
            CovMat[ np.ix_( path_ind, path_ind ) ] += edge_len*np.ones((npaths,npaths))
            Indices[parent] += path_ind
        else:
            edge_len = ts.node(parent_nodes[0]).time - node.time
            CovMat = np.hstack(  (CovMat,) + tuple( ( CovMat[:,path_ind] for j in range(nparent-1) ) ) ) #Duplicate the rows
            CovMat = np.vstack(  (CovMat,) + tuple( ( CovMat[path_ind,:] for j in range(nparent-1) ) ) ) #Duplicate the columns
            new_ind = path_ind + list(range((-(nparent-1)*len(path_ind)),0))
            CovMat[ np.ix_( new_ind, new_ind ) ] += edge_len
            for i, parent in enumerate(parent_nodes):
                for x in range(i*npaths,(i+1)*npaths):
                    ind = new_ind[x]
                    if ind < 0:
                        ind = len(CovMat) + ind
                    Indices[parent] += [ind]
    return CovMat, Paths


def benchmark(ts):
    start = time.time()
    cov_mat = calc_covariance_matrix(ts=ts)
    end = time.time()
    return end-start, cov_mat.sum(), "NA"

if __name__ == "__main__":
    
    rs = random.randint(0,10000)
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
    
    cov_mat, paths = calc_covariance_matrix(ts=ts)

    print(paths, cov_mat)

        