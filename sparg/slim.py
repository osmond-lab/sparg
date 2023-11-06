import msprime
import numpy as np
import tskit
import networkx as nx
from . import comparisons


def merge_roots(ts): 
    ts_tables = ts.dump_tables() 
    edge_table = ts_tables.edges 
    parent = edge_table.parent 
    
    roots = np.where(ts_tables.nodes.time == ts.max_time)[0]
    root_children = []
    for root in roots:
        root_children += list(ts.tables.edges.child[np.where(ts.tables.edges.parent == root)[0]])
    for root_child in root_children: 
        pts = np.unique(ts.tables.edges.parent[np.where(ts.tables.edges.child == root_child)[0]])
        if len(pts) > 2 : 
            for i,pt in enumerate(pts): 
                parent[np.where(ts.tables.edges.parent == pt)[0]] = pts[0] 
    edge_table.parent = parent 
    ts_tables.sort() 
    ts_new = ts_tables.tree_sequence() 
    return ts_new

def add_recomb_node_flags_to_ts(ts):
    ts_tables = ts.dump_tables()
    node_table = ts_tables.nodes
    flags = node_table.flags
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has. 
    multiple_parents = nd[count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    flags[recomb_nodes] = msprime.NODE_IS_RE_EVENT
    node_table.flags = flags
    ts_tables.sort() 
    return ts_tables.tree_sequence()

def add_recomb_node_flags_to_ts_check(ts):
    ts_tables = ts.dump_tables()
    node_table = ts_tables.nodes
    flags = node_table.flags
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has. 
    multiple_parents = nd[count > 1] #Find children who have more than 1 parent. 
    #recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    flags[multiple_parents] = msprime.NODE_IS_RE_EVENT
    node_table.flags = flags
    ts_tables.sort() 
    return ts_tables.tree_sequence()


def remove_excess_nodes(ts, keep_unary_roots=False, keep_young_nodes={}):
    """Removes nodes in tree sequence output from SLiM

    
    Parameters
    ----------
    ts : tskit.trees.TreeSequence
    keep_young_nodes (optional) : dictionary
        Used when tracking the lineages in recent time. This dictionary has two keys: "below" and "step". Keeps nodes younger than
        "below" with gaps between nodes set by "step".

    Returns
    -------
    ts_final :
    maps : 
    critical_nodes_mapped : 
    """

    ts_tables = ts.dump_tables()
    node_table = ts_tables.nodes
    flags = node_table.flags
    
    roots = np.where(ts.tables.nodes.time == ts.max_time)[0]

    recomb_nodes = []
    coal_nodes = []
    
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 0], return_counts=True) #For each child, count how many parents it has. 
    multiple_parents = nd[count > 1] #Find children who have more than 1 parent. 
    recomb_nodes = ts.edges_parent[np.in1d(ts.edges_child, multiple_parents)] #Find the parent nodes of the children with multiple parents. 
    flags[recomb_nodes] = msprime.NODE_IS_RE_EVENT
    
    nd, count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has. 
    coal_nodes = nd[count > 1] #Find parent who have more than 1 children. 
    
    node_table.flags = flags
    ts_tables.sort() 
    ts_new = ts_tables.tree_sequence()

    critical_nodes = list(ts_new.samples()) + list(recomb_nodes) + list(coal_nodes)
    if keep_unary_roots:
        critical_nodes += list(roots)
    if len(keep_young_nodes) > 0:
        keep_nodes = list(np.unique(critical_nodes + list(np.where((node_table.time<=keep_young_nodes["below"]) & (node_table.time%keep_young_nodes["step"]==0))[0])))
    else:
        keep_nodes = list(np.unique(critical_nodes))
    ts_final, maps = ts_new.simplify(samples=keep_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
    critical_nodes_mapped = []
    for node in critical_nodes:
        critical_nodes_mapped.append(maps[node])
    return ts_final, maps, critical_nodes_mapped


def identify_gmrca(ts):
    """Returns the ID of the grand most recent common ancestor of the ARG, if one exists.

    Parameters
    ----------
    ts : tskit.trees.TreeSequence
        This must be a "full" ARG, though node flags are not needed. Simplified tree sequences
        will return the deepest root in the tree sequence but not necessarily the GMRCA.

    Returns
    -------
    gmrca : int
        The ID of the grand most recent common ancestor. If none exists, return -1.

    """

    edge_by_time = np.empty((ts.num_edges, 4))
    for i,edge in enumerate(ts.edges()):
        edge_by_time[i,0] = i
        edge_by_time[i,1] = edge.child
        edge_by_time[i,2] = ts.node(edge.child).time
        edge_by_time[i,3] = ts.node(edge.parent).time
    gmrca = -1
    for node in ts.nodes():
        if node.flags != 1:
            edges_start_before_time = edge_by_time[edge_by_time[:,2]<=node.time,:]
            active_edges = edges_start_before_time[edges_start_before_time[:,3]>node.time,:]
            other_edges = active_edges[active_edges[:,1]!=node.id,0]
            if len(other_edges) == 0:
                if len(np.where(ts.tables.nodes.time == node.time)[0]) == 1:
                    gmrca = node.id
                    break
    return gmrca


def cut_ts_at_gmrca(ts):
    gmrca = identify_gmrca(ts=ts)
    if gmrca != -1:
        return ts.subset(nodes=list(range(gmrca+1)))
    else:
        print("Tree sequence does not have a GMRCA - did not cut.")
        return ts
    

def cut_ts_at_gmrca_old(ts, gmrca):
    tables = tskit.TableCollection(sequence_length=ts.sequence_length)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    node_table = tables.nodes
    for i, node in enumerate(ts.tables.nodes):
        if i <= gmrca:
            node_table.add_row(
                flags=node.flags,
                time=node.time,
                metadata=node.metadata
            )
    edge_table = tables.edges
    for i, edge in enumerate(ts.tables.edges):
        if (edge.parent <= gmrca) and (edge.child <= gmrca):
            edge_table.add_row(
                left=edge.left,
                right=edge.right,
                parent=edge.parent,
                child=edge.child
            )
    tables.sort()
    condensed_ts = tables.tree_sequence()
    return condensed_ts