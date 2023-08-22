import numpy as np
import math


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