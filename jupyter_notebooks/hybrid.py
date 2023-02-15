import msprime
import tskit
import numpy as np
import math
import random
from collections import defaultdict
from itertools import chain
import networkx as nx
import matplotlib.pyplot as plt


class LoopGroup:
    def __init__(self, nodes, loop_list):
        self.nodes = set(nodes)
        self.loops = []
        for loop in loop_list:
            if loop[0] in self.nodes:
                self.loops.append(loop)
        self.youngest_node = min(nodes)
        self.input_node = max(nodes) #should this be changed to self.input_node
        self.output_nodes = None
        """
        I don't know what best practice is here. I don't want the attribute
        unless I've added to it. Should it be a list or set?
        """
        
        """
        We may want to also have a networkx graph representation here.
        This would allow us to calculate all of the paths through the
        LoopGroup, needed for the covariance matrix.
        """
    
    def __str__(self):
        return f'LoopGroup contains {len(self.loops)} loop(s): {self.loops}'
    
    def add_output_node(self, node):
        if self.output_nodes == None:
            self.output_nodes = [node]
        else:
            self.output_nodes.append(node)
            self.output_nodes.sort()


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
    return nx_graph

def locate_loops_combo(nx_graph, ts=None):
    """
    Finds loops within the ARG. I thought that it would be easiest to utilize functions from
    networkx package. Identifies recombination events, converts the tree sequence into a networkx
    graph. The paired recombination nodes are merged together in this graph. Converts graph to 
    undirected, then calculates cycle basis. This does not identify 'bubbles', so we need to add
    an extra step to this.
    """
    
    loop_list = nx.cycle_basis(nx_graph.to_undirected())
    if ts:
        edges = ts.tables.edges
        parent_list = list(edges.parent)
        child_list = list(edges.child)
        recomb_nodes = list(np.where(ts.tables.nodes.flags == 131072)[0])
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
            if plot:
                nx.draw(g, with_labels=True)
            grouped_nodes = list(nx.connected_components(g))
            grouped_loops = []
            for grouping in grouped_nodes:
                grouped_loops.append(LoopGroup(nodes=grouping, loop_list=loops))
            grouped_loops.sort(key=lambda x: x.youngest_node)
            return grouped_loops
        else:
            return [LoopGroup(nodes=loops[0], loop_list=loops)]

def strip_graph(nx_graph, ts, grouped_loops):
    """
    Creates a skeleton graph from the full graph
    """
    
    gmrca = ts.node(ts.num_nodes-1).id
    skeleton_topology = defaultdict(list)
    previously_found = []
    working_nodes = list(ts.samples())
    for node in working_nodes:
        no_shallower_connection = True
        for lg in grouped_loops:
            if node < lg.input_node:
                if nx.has_path(nx_graph, source=node, target=lg.input_node):
                    path = nx.shortest_path(nx_graph, source=node, target=lg.input_node)
                    youngest_connection_node = min(list(set(path) & lg.nodes))
                    skeleton_topology[node].append(youngest_connection_node)
                    if youngest_connection_node not in previously_found:
                        skeleton_topology[youngest_connection_node].append(lg.input_node)
                        previously_found.append(youngest_connection_node)
                        lg.add_output_node(youngest_connection_node)
                    if lg.input_node != gmrca and lg.input_node not in working_nodes:
                        working_nodes.append(lg.input_node)
                    no_shallower_connection = False
                    break
        if no_shallower_connection:
            skeleton_topology[node].append(gmrca)
    return nx.DiGraph(skeleton_topology)


rs = random.randint(0,10000)
print(rs)

ts = msprime.sim_ancestry(
    samples=2,
    recombination_rate=1e-8,
    sequence_length=2000,
    population_size=10_000,
    record_full_arg=True,
    random_seed=7483
)

print(ts.draw_text())

G = ts_to_nx(ts=ts, connect_recombination_nodes=True)
loops = locate_loops_combo(nx_graph=G, ts=ts) #Identify each loop as a list of nodes 
grouped_loops = group_loops(loops=loops) #Group the loops if they shared edges
skeleton_G = strip_graph(nx_graph=G, ts=ts, grouped_loops=grouped_loops)

#print(grouped_loops)

for lg in grouped_loops:
    print(lg, lg.output_nodes)