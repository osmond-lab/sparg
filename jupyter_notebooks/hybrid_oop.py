import msprime
import tskit
import numpy as np
import math
import random
from collections import defaultdict
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt


class ARGObject:
    """
    A ARGObject contains various methods for reference an ARG:
        
        ts - tskit tree sequence
        nx_graph - directed networkx graph built from tree sequence instructions
        nx_graph_connected_recomb_nodes - directed networkx graph where
            recombination nodes have been merged
        skeleton_graph - directed networkx graph after loops have been simplified

    There are also a few attributes that are useful:
        recomb_nodes - recombination nodes (very commonly referenced)
        skeleton_groups - list of working units in the graph
    """

    def __init__(self, ts):
        self.ts = ts
        self.recomb_nodes = list(np.where(self.ts.tables.nodes.flags == 131072)[0])
        self.nx_graph = self.ts_to_nx()
        self.nx_graph_connected_recomb_nodes = self.ts_to_nx(connect_recombination_nodes=True)
        self.cov_matrix = self.calc_cov_matrix()

    def ts_to_nx(self, connect_recombination_nodes=False):
        """
        Converts tskit tree sequence to networkx graph.

        Need to add a check to ensure that the list of recombination nodes is valid
        (there should always be an even number of recombination nodes if following
        tskit setup)
        """
        recomb_nodes_to_remove = self.recomb_nodes[1::2]
        topology = defaultdict(list)
        for tree in self.ts.trees():
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
        node_times = {v: k for v, k in enumerate(self.ts.tables.nodes.time)}
        nx.set_node_attributes(nx_graph, node_times, "time")
        return nx_graph
    
    def group_loops(self, loops):
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
                return grouped_nodes
            else:
                return [set(loops[0])]
        
    def locate_loop_groups(self):
        """
        Finds loops within the ARG. I thought that it would be easiest to utilize functions from
        networkx package. Identifies recombination events, converts the tree sequence into a networkx
        graph. The paired recombination nodes are merged together in this graph. Converts graph to 
        undirected, then calculates cycle basis. This does not identify 'bubbles', so we need to add
        an extra step to this.
        """
        
        loop_list = nx.cycle_basis(self.nx_graph_connected_recomb_nodes.to_undirected())
        edges = self.ts.tables.edges
        parent_list = list(edges.parent)
        child_list = list(edges.child)
        if len(loop_list) != len(self.recomb_nodes)/2:
            for node in self.recomb_nodes[::2]:
                parent = parent_list[child_list.index(node)]
                if parent == parent_list[child_list.index(node+1)]:
                    loop_list.append([node, parent])
        grouped_loops = self.group_loops(loops=loop_list)
        return grouped_loops
    
    def generate_skeleton(self):
        loop_group_nodes = self.locate_loop_groups()
        print(loop_group_nodes)
        gmrca = self.ts.node(self.ts.num_nodes-1).id
        all_loop_nodes = set([node for lg in loop_group_nodes for node in lg])
        skeleton = defaultdict(list)
        previously_found = []
        working_nodes = list(self.ts.samples())
        for node in working_nodes:
            no_shallower_connection = True
            for lg in loop_group_nodes:
                input_node = max(lg)
                if node < input_node:
                    if nx.has_path(self.nx_graph_connected_recomb_nodes, source=node, target=input_node):
                        path = nx.shortest_path(self.nx_graph_connected_recomb_nodes, source=node, target=input_node)
                        youngest_connection_node = min(list(set(path[1:]) & all_loop_nodes))
                        if youngest_connection_node not in lg:
                            continue
                        skeleton[youngest_connection_node].append(node)
                        if youngest_connection_node not in previously_found:
                            skeleton[input_node].append(youngest_connection_node)
                            previously_found.append(youngest_connection_node)
                            if youngest_connection_node in self.recomb_nodes:
                                skeleton[input_node].append(youngest_connection_node+1)
                        if input_node != gmrca and input_node not in working_nodes:
                            working_nodes.append(input_node)
                        no_shallower_connection = False
                        break
            if no_shallower_connection:
                skeleton[gmrca].append(node)
        return skeleton

    def calc_cov_matrix(self):
        skeleton = self.generate_skeleton()
        sub_cov_mats = {}
        for group in skeleton:
            paths = []
            for output_node in skeleton[group]:
                paths.extend(nx.all_simple_paths(self.nx_graph, source=output_node, target=group))
                if group in self.recomb_nodes:
                    paths.extend(nx.all_simple_paths(self.nx_graph, source=output_node, target=group+1))
            edges = self.ts.tables.edges
            parent_list = list(edges.parent)
            child_list = list(edges.child)
            times = np.empty((len(paths),len(paths)))
            for i, p in enumerate(paths):
                for j in range(i+1):
                    intersect = list(set(p).intersection(paths[j]))
                    if i == j:
                        times[i,j] = self.nx_graph.nodes[group]["time"]
                    elif intersect == [group]:
                        times[i,j] = 0
                    else:
                        edges = []
                        for child in intersect:
                            if child != group:
                                edges.append(self.ts.node(parent_list[child_list.index(child)]).time - self.ts.node(child).time)
                        times[i,j] = np.sum(edges) # Do I need np.unique()? Ask Matt, because it was previously in his
                    times[j,i] = times[i,j]
            sub_cov_mats[group] = times
            print(group, paths)
            print(times)
            


class SkeletonGroup:
    def __init__(self, input_nodes, output_nodes, nx_graph):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.nx_graph = nx_graph
    
    def __str__(self):
        return f'Change'
    
        
    def calc_paths(self):
        all_paths = []
        for node in self.output_nodes:
            all_paths.extend(nx.all_simple_paths(self.nx_graph, source=node, target=self.input_node))
        self.paths = all_paths
        return all_paths
    
    def calc_cov_matrix(self, ts):
        """
        I'll eventually want to get rid of the ts parameter. Should have enough information in the nx_graph
        """

        paths = self.calc_paths()
        edges = ts.tables.edges
        parent_list = list(edges.parent)
        child_list = list(edges.child)
        times = np.empty((len(paths),len(paths)))
        for i, p in enumerate(paths):
            for j in range(i+1):
                intersect = list(set(p).intersection(paths[j]))
                if i == j:
                    times[i,j] = self.nx_graph.nodes[self.input_node]["time"]
                elif intersect == [self.input_node]:
                    times[i,j] = 0
                else:
                    edges = []
                    for child in intersect:
                        if child != self.input_node:
                            edges.append(ts.node(parent_list[child_list.index(child)]).time - ts.node(child).time)
                    times[i,j] = np.sum(edges) # Do I need np.unique()? Ask Matt, because it was previously in his
                times[j,i] = times[i,j]
        self.cov_matrix = times
        return times
            
            
            





rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=2,
    recombination_rate=1e-8,
    sequence_length=2000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=5841#7483
)

print(ts.draw_text())

ARG = ARGObject(ts=ts)



def group_loops(self, loops):
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
                    if a > b:
                        build_instructions.append([b, a])
                    else:
                        build_instructions.append([a, b])
            g = nx.DiGraph(build_instructions)
            node_times = {v: k for v, k in enumerate(self.ts.tables.nodes.time)}
            nx.set_node_attributes(g, node_times, "time")
            grouped_nodes = list(nx.connected_components(g.to_undirected()))
            grouped_loops = []
            for grouping in grouped_nodes:
                lg_graph = self.nx_graph.copy()
                multi_recomb_id_grouping = []
                for element in grouping:
                    if element in self.recomb_nodes:
                        multi_recomb_id_grouping.append(element + 1)
                    multi_recomb_id_grouping.append(element)
                lg_graph.remove_nodes_from(lg_graph.nodes() - set(multi_recomb_id_grouping))
                lg_graph_connected = self.nx_graph_connected_recomb_nodes.copy()
                lg_graph_connected.remove_nodes_from(lg_graph_connected.nodes() - set(grouping))
                grouped_loops.append(SkeletonGroup(nodes=grouping, loop_list=loops, nx_graph=lg_graph, nx_graph_connected_recomb_nodes=lg_graph_connected))
            grouped_loops.sort(key=lambda x: x.youngest_node)
            return grouped_loops
        else:
            return [SkeletonGroup(nodes=loops[0], loop_list=loops)]

def strip_graph(self):
    """
    Creates a skeleton graph from the full ARG
    """
    loop_groups = self.locate_loop_groups()
    gmrca = self.ts.node(self.ts.num_nodes-1).id
    all_loop_nodes = set([node for loop in loop_groups for node in loop.nodes])
    skeleton_topology = defaultdict(list)
    skeleton_groupings = defaultdict(list)
    previously_found = []
    working_nodes = list(self.ts.samples())
    for node in working_nodes:
        no_shallower_connection = True
        for lg in loop_groups:
            if node < lg.input_node:
                if nx.has_path(self.nx_graph_connected_recomb_nodes, source=node, target=lg.input_node):
                    path = nx.shortest_path(self.nx_graph_connected_recomb_nodes, source=node, target=lg.input_node)
                    youngest_connection_node = min(list(set(path[1:]) & all_loop_nodes))
                    if youngest_connection_node not in lg.nodes:
                        continue
                    skeleton_topology[node].append(youngest_connection_node)
                    skeleton_groupings[youngest_connection_node].append(node)
                    if youngest_connection_node not in previously_found:
                        skeleton_topology[youngest_connection_node].append(lg.input_node)
                        skeleton_groupings[lg.input_node].append(youngest_connection_node)
                        previously_found.append(youngest_connection_node)
                        lg.add_output_node(youngest_connection_node)
                        if youngest_connection_node in self.recomb_nodes:
                            skeleton_groupings[lg.input_node].append(youngest_connection_node+1)
                            lg.add_output_node(youngest_connection_node+1)
                    if lg.input_node != gmrca and lg.input_node not in working_nodes:
                        working_nodes.append(lg.input_node)
                    no_shallower_connection = False
                    break
        if no_shallower_connection:
            skeleton_topology[node].append(gmrca)
    print(skeleton_groupings)
    return nx.DiGraph(skeleton_topology), {}