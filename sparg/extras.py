import numpy as np
import math
import networkx as nx
from collections import defaultdict
from itertools import chain


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

def manually_add_node_below_recombination_event(ts):
    tables = ts.dump_tables()
    node_table = tables.nodes
    edge_table = tables.edges
    node_table.add_row(time=ts.node(11).time-1)
    rows_of_interest = np.where(np.isin(edge_table.parent,[11,12]))[0]
    for row in rows_of_interest:
        edge_table[row] = edge_table[row].replace(child=len(node_table)-1)
    edge_table.add_row(left=0, right=2000, parent=len(node_table)-1, child=8)
    tables.sort()
    return tables.tree_sequence()


def add_nodes_along_sample_paths(ts, tracked_samples):
    ts = manually_add_node_below_recombination_event(ts=ts)
    tables = ts.dump_tables()
    node_table = tables.nodes
    edge_table = tables.edges
    recombination_nodes = np.where(node_table.flags==131072)[0]
    nodes_to_do = tracked_samples[:]
    previously_handled_nodes = tracked_samples[:]
    while len(nodes_to_do) > 0:
        rows_of_interest = np.where(edge_table.child==nodes_to_do[0])[0]
        current_node_time = ts.node(nodes_to_do[0]).time
        for row in rows_of_interest:
            edge = edge_table[row]
            if edge.parent not in recombination_nodes:
                parent_time = ts.node(edge.parent).time
                interval = 500
                if parent_time - current_node_time > interval:
                    if current_node_time == 0:
                        lower_bound = interval
                    else:
                        lower_bound = math.ceil(current_node_time/interval)*interval
                    upper_bound = math.ceil(parent_time/interval)*interval
                    child = nodes_to_do[0]
                    for middle_node_time in range(lower_bound, upper_bound, interval):
                        node_table.add_row(time=middle_node_time)
                        if middle_node_time == lower_bound:
                            edge_table[row] = edge_table[row].replace(parent=len(node_table)-1)
                        else:
                            edge_table.add_row(left=edge.left, right=edge.right, parent=len(node_table)-1, child=child)
                        child = len(node_table)-1
                    edge_table.add_row(left=edge.left, right=edge.right, parent=edge.parent, child=child)
            if edge.parent not in previously_handled_nodes:
                nodes_to_do.append(edge.parent)
            previously_handled_nodes.append(edge.parent)
        nodes_to_do.pop(0)
    tables.sort()
    return tables.tree_sequence()


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

def remove_unattached_nodes_old(ts):
    ts_nx = ts_to_nx(ts=ts)
    sub_graphs = nx.connected_components(ts_nx.to_undirected())
    attached_nodes = []
    for sg in sub_graphs:
        for node in sg:
            attached_nodes.append(node)
    critical_nodes = []
    for node in ts.nodes():
        if node.id in attached_nodes:
            critical_nodes.append(node.id) #something in attached nodes that cannot be passed to simplify 
    ts_final, maps = ts.simplify(samples=critical_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False) #Subset might be better
    return ts_final, maps

def remove_unattached_nodes(ts):
    edge_table = ts.tables.edges
    connected_nodes = np.sort(np.unique(np.concatenate((edge_table.parent,edge_table.child))))
    ts_final = ts.subset(nodes=connected_nodes)
    return ts_final
    

def merge_unnecessary_roots(ts):
    ts_tables = ts.dump_tables()
    edge_table = ts_tables.edges 
    parent = edge_table.parent
    roots = np.where(ts_tables.nodes.time == ts.max_time)[0]
    children = defaultdict(list)
    for root in roots:
        root_children = edge_table.child[np.where(edge_table.parent == root)[0]]
        for child in root_children:
            children[child] += [root]
    for child in children:
        pts = children[child]
        if len(pts) > 1:
            for pt in pts:
                if len(np.unique(edge_table.child[np.where(edge_table.parent == pt)[0]])) > 1:
                    print(pt, "has multiple children! Merge roots with caution.")
                parent[np.where(ts.tables.edges.parent == pt)[0]] = pts[0]
    edge_table.parent = parent 
    ts_tables.sort() 
    ts_new = remove_unattached_nodes(ts=ts_tables.tree_sequence())
    return ts_new

def remove_uninformative_nodes(ts, keep_young_nodes={}):
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each parent, count how many children it has. 
    coal_nodes = nd[count > 1] #Find parent who have more than 1 children.
    recomb_nodes_first_id = np.where(ts.tables.nodes.flags==131072)[0][::2]
    important_recomb_nodes = []
    for node in recomb_nodes_first_id:
        if node in uniq_child_parent[:,0]:# or (node in uniq_child_parent[:,1]):
            if (node+1) in uniq_child_parent[:,0]:# or (node+1 in uniq_child_parent[:,1]):
                #print(uniq_child_parent[np.where(uniq_child_parent[:,0]==348)[0],:])
                important_recomb_nodes.append(node)
                important_recomb_nodes.append(node+1)
    max_time = max(ts.tables.nodes[nd_id].time for nd_id in np.union1d(ts.edges_child, ts.edges_parent) )
    roots = np.where(ts.tables.nodes.time == max_time)[0]
    critical_nodes = list(np.unique( list(ts.samples()) + list(np.unique(important_recomb_nodes)) + list(np.unique(coal_nodes)) + list(np.unique(roots))))
    if len(keep_young_nodes) > 0:
        critical_nodes = list(np.unique(critical_nodes + list(np.where((ts.tables.nodes.time<=keep_young_nodes["below"]) & (ts.tables.nodes.time%keep_young_nodes["step"]==0))[0])))
    ts_final, maps = ts.simplify(samples=critical_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
    return ts_final, maps 

def old_remove_uninformative_nodes(ts, keep_young_nodes={}):
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has. 
    coal_nodes = nd[count > 1] #Find parent who have more than 1 children.
    recomb_nodes_first_id = np.where(ts.tables.nodes.flags==131072)[0][::2]
    important_recomb_nodes = []
    for node in recomb_nodes_first_id:
        if node in uniq_child_parent[:,0]:# or (node in uniq_child_parent[:,1]):
            if (node+1) in uniq_child_parent[:,0]:# or (node+1 in uniq_child_parent[:,1]):
                #print(uniq_child_parent[np.where(uniq_child_parent[:,0]==348)[0],:])
                important_recomb_nodes.append(node)
                important_recomb_nodes.append(node+1)
    roots = np.where(ts.tables.nodes.time == ts.max_time)[0]
    critical_nodes = list(np.unique( list(ts.samples()) + list(np.unique(important_recomb_nodes)) + list(np.unique(coal_nodes)) + list(np.unique(roots))))
    if len(keep_young_nodes) > 0:
        critical_nodes = list(np.unique(critical_nodes + list(np.where((ts.tables.nodes.time<=keep_young_nodes["below"]) & (ts.tables.nodes.time%keep_young_nodes["step"]==0))[0])))
    ts_final, maps = ts.simplify(samples=critical_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
    return ts_final, maps


def old_remove_useless_nodes(ts):
    uniq_child_parent = np.unique(np.column_stack((ts.edges_child, ts.edges_parent)), axis=0) #Find unique parent-child pairs. 
    nd, count = np.unique(uniq_child_parent[:, 1], return_counts=True) #For each child, count how many parents it has. 
    coal_nodes = nd[count > 1] #Find parent who have more than 1 children.
    recomb_nodes_first_id = np.where(ts.tables.nodes.flags==131072)[0][::2]
    important_recomb_nodes = []
    for node in recomb_nodes_first_id:
        if node in uniq_child_parent[:,0]:# or (node in uniq_child_parent[:,1]):
            if (node+1) in uniq_child_parent[:,0]:# or (node+1 in uniq_child_parent[:,1]):
                #print(uniq_child_parent[np.where(uniq_child_parent[:,0]==348)[0],:])
                important_recomb_nodes.append(node)
                important_recomb_nodes.append(node+1)
    critical_nodes = list(np.unique( list(ts.samples()) + list(np.unique(important_recomb_nodes)) + list(np.unique(coal_nodes)) ))  
    
    #for node in critical_nodes[:]:
    #    if node not in all_nodes:
    #        critical_nodes.remove(node)
    #node_counts = []
    #for graph in sub_graphs:
    #    node_counts.append(len(graph))
    #critical_nodes = list(sub_graphs[np.argmax(node_counts)])
    ts_final, maps = ts.simplify(samples=critical_nodes, map_nodes = True, keep_input_roots=False, keep_unary=False, update_sample_flags = False)
    return ts_final, maps


def special_merge_roots(ts): 
    ts_tables = ts.dump_tables() 
    edge_table = ts_tables.edges 
    parent = edge_table.parent 
    
    roots = np.where(ts_tables.nodes.time == ts.max_time)[0]
    print(roots)
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



