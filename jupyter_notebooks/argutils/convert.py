import collections

import pandas as pd
import networkx as nx
import numpy as np
import tskit

import argutils


def convert_argweaver(infile):
    """
    Convert an ARGweaver .arg file to a tree sequence. An example .arg file is at

    https://github.com/CshlSiepelLab/argweaver/blob/master/test/data/test_trans/0.arg
    """
    start, end = next(infile).strip().split()
    assert start.startswith("start=")
    start = int(start[len("start=") :])
    assert end.startswith("end=")
    end = int(end[len("end=") :])
    # the "name" field can be a string. Force it to be so, in case it is just numbers
    df = pd.read_csv(infile, header=0, sep="\t", dtype={"name": str, "parents": str})

    name_to_record = {}
    for _, row in df.iterrows():
        row = dict(row)
        name_to_record[row["name"]] = row
    # We could use nx to do this, but we want to be sure the order is correct.
    parent_map = collections.defaultdict(list)

    # Make an nx DiGraph so we can do a topological sort.
    G = nx.DiGraph()
    time_map = {} # argweaver times to allocated time
    for row in name_to_record.values():
        child = row["name"]
        parents = row["parents"]
        time_map[row["age"]] = row["age"]
        G.add_node(child)
        if isinstance(parents, str):
            for parent in row["parents"].split(","):
                G.add_edge(child, parent)
                parent_map[child].append(parent)

    tables = tskit.TableCollection(sequence_length=end)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    breakpoints = np.full(len(G), tables.sequence_length)
    aw_to_tsk_id = {}
    min_time_diff = min(np.diff(sorted(time_map.keys())))
    epsilon = min_time_diff / 1e6
    for node in nx.lexicographical_topological_sort(G):
        record = name_to_record[node]
        flags = 0
        # Sample nodes are marked as "gene" events
        if record["event"] == "gene":
            flags = tskit.NODE_IS_SAMPLE
            assert record["age"] == 0
            time = record["age"]
        else:
            time = time_map[record["age"]]
            # Argweaver allows age of parent and child to be the same, so we
            # need to add epsilons to enforce parent_age > child_age
            time_map[record["age"]] += epsilon
        tsk_id = tables.nodes.add_row(flags=flags, time=time, metadata=record)
        aw_to_tsk_id[node] = tsk_id
        if record["event"] == "recomb":
            breakpoints[tsk_id] = record["pos"]

    L = tables.sequence_length
    for aw_node in G:
        child = aw_to_tsk_id[aw_node]
        parents = [aw_to_tsk_id[aw_parent] for aw_parent in parent_map[aw_node]]
        if len(parents) == 1:
            tables.edges.add_row(0, L, parents[0], child)
        elif len(parents) == 2:
            # Recombination node.
            # If we wanted a GARG here we'd add an extra node
            x = breakpoints[child]
            tables.edges.add_row(0, x, parents[0], child)
            tables.edges.add_row(x, L, parents[1], child)
        else:
            assert len(parents) == 0
    # print(tables)
    tables.sort()
    # print(tables)
    ts = tables.tree_sequence()
    # The plan here originally was to use the earg_to_garg method to
    # convert the recombination events to two parents (making a
    # standard GARG). However, there are some complexities here so
    # returning the ARG topology as defined for now. There is an
    # argument that we should do this anyway, since that's the structure
    # that was returned and makes very little difference.

    # garg = argutils.earg_to_garg(ts)

    return ts.simplify(keep_unary=True)

def convert_kwarg(infile, num_samples, sequence_length, sample_names=None):
    """
    Convert a KwARG output file to a tree sequence.
    """

    tables = tskit.TableCollection(sequence_length=sequence_length)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    time = 0.0

    node_ids = {}
    for n in range(num_samples):
        node_ids[n] = n
        if sample_names is not None and n in sample_names:
            name = sample_names[n]
        else:
            name = n
        tsk_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE, time=0.0, metadata={"name": name})

    for x in infile:
        line = x.split()
        if (line[0] == "Mutation"):
            site_id = tables.sites.add_row(position=int(line[3]) - 1, ancestral_state='0')
            tables.mutations.add_row(site=site_id, node=node_ids[int(line[6]) - 1], derived_state='1')
        elif (line[0] == "Coalescing"):
            c1 = node_ids[int(line[2]) - 1]
            c2 = node_ids[int(line[4]) - 1]
            time += 1
            tsk_id = tables.nodes.add_row(flags=0, time=time, metadata={"coal": line[2]})
            node_ids[int(line[2]) - 1] = tsk_id
            tables.edges.add_row(0, sequence_length, tsk_id, c1)
            tables.edges.add_row(0, sequence_length, tsk_id, c2)
        elif (line[0] == "---->Recombination"):
            breakpoint = int(line[6][:-1]) - 1
            n1 = node_ids[int(line[3]) - 1]
            time += 1
            tsk_id_1 = tables.nodes.add_row(flags=0, time=time, metadata={"rec": line[3]})
            node_ids[int(line[3]) - 1] = tsk_id_1
            tsk_id_2 = tables.nodes.add_row(flags=0, time=time, metadata={"rec": line[11]})
            node_ids[int(line[11]) - 1] = tsk_id_2
            tables.edges.add_row(0, breakpoint, tsk_id_1, n1)
            tables.edges.add_row(breakpoint, sequence_length, tsk_id_2, n1)

    tables.sort()
    ts = tables.tree_sequence()

    return argutils.simplify_keeping_all_nodes(ts)


def relate_ts_JBOT_to_ts(ts, additional_equivalents=None):
    """
    Convert a tree sequence from Relate (converted via relate_lib/bin/Convert, which
    provides equivalenet via metadata, into a tree sequence in which the nodes
    have been merged, and new times allocated that are consistent with the ARG
    
    the additional_equivalents parameter is a hack because relate_lib doesn't
    calculate equivalent edges if the edges are above sample nodes. This is therefore
    a supplementary dictionary of cases where *nodes* (not edges) in the input
    tree sequence can be thought as equivalent.
    """
    # make a map of relate node_id -> canonical node id (the first one used in relate)
    node_map = additional_equivalents.copy() or {}
    ns = set()
    G = nx.DiGraph()
    assert ts.num_nodes == ts.num_samples + (ts.num_samples - 1) * ts.num_trees
    for ediff in ts.edge_diffs():
        # each tree should have new edges
        if len(ediff.edges_in) != 0 and len(ediff.edges_out) != 0:
            assert len(ediff.edges_in) == len(ediff.edges_out) == (2 * (ts.num_samples - 1))
        # look for equivalent edges based on node metadata
        out_edges = {e.child: e for e in ediff.edges_out}
        for e_in in ediff.edges_in:
            # equivalents stored in the child node
            md = np.fromstring(ts.node(e_in.child).metadata, dtype=int, sep=' ')
            if len(md) != 0 and md[0] != -1:
                equivalent_to = out_edges[md[0]]
                # both parent and child nodes of this edge have an equivalent
                equivalent_child = node_map.get(equivalent_to.child, equivalent_to.child)
                equivalent_parent = node_map.get(equivalent_to.parent, equivalent_to.parent)
                if e_in.child in node_map:
                    # could happen because another edge could have same parent
                    assert node_map[e_in.child] == equivalent_child
                else:
                    node_map[e_in.child] = equivalent_child
                if e_in.parent in node_map:
                    assert node_map[e_in.parent] == equivalent_parent
                else:
                    node_map[e_in.parent] = equivalent_parent
    for node in ts.nodes():
        if node.id not in node_map:
            node_map[node.id] = node.id
    for u in set(node_map.values()):
        G.add_node(u, flags=ts.node(u).flags, time=0, relate_times={})
    for node in ts.nodes():
        assert G.nodes[node_map[node.id]]["flags"] == node.flags
        G.nodes[node_map[node.id]]["relate_times"][node.id] = node.time

    # collect all the intervals for each edge (will be squashed later)
    edges = collections.defaultdict(list)
    for e in ts.edges():
        edges[(node_map[e.child], node_map[e.parent])].append((e.left, e.right))

    # Turn this into a graph so we can do a topological sort
    for k, v in edges.items():
        G.add_edge(*k, intervals=v)
    # Set arbitrary times by topological_sort
    for i, n in enumerate(nx.lexicographical_topological_sort(G)):
        if G.nodes[n]["flags"] & tskit.NODE_IS_SAMPLE:
            assert G.nodes[n]["time"] == 0
            G.nodes[n]["time"] = 0
        else:
            G.nodes[n]["time"] = i

    # Create a new tree sequence using these times
    tables = tskit.TableCollection(ts.sequence_length)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    for n in range(max(G.nodes())+1):
        try:
            node = G.nodes[n]
            tables.nodes.add_row(flags=node["flags"], time=node["time"], metadata={"Relate_times": node["relate_times"]})
        except:
            tables.nodes.add_row(time=0) # dummy, unused

    for child, parent, attribute in G.edges(data=True):
        for i in attribute["intervals"]:
            tables.edges.add_row(child = child, parent=parent, left=i[0], right=i[1])
    tables.sort()
    tables.simplify()
    tables.edges.squash()  # probably don't need this: I think simplify() squashes edges
    return tables.tree_sequence()        
            
        