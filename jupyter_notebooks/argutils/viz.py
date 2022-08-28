"""
Viz routines for args.
"""
import collections
import colorsys
import itertools
import string

import colorcet
import matplotlib as mpl
import networkx as nx
import numpy as np
import pydot
import scipy.stats


def arity_colors(n_parents):
    assert n_parents >= 0
    if n_parents == 0:
        return colorcet.cm.CET_I1(255)  # Red
    if n_parents == 1:
        return colorcet.cm.CET_I1(200)  # Yellow
    # Max out at full blue if 6 or more parents; use log scale
    multiple_parents_val = (min(np.log2(n_parents), np.log2(6)) - 1) / (np.log2(6) - 1)
    return colorcet.cm.CET_I1(100 - int(multiple_parents_val * 100))


def make_color(rgb, lighten=0):
    """
    Make a hex colour from this rgb colour and potentially lighten it (to white if l=0)
    By default, make a tiny bit darker anyway
    """
    amount = 1.2 * (1 - lighten)
    c = colorsys.rgb_to_hls(*rgb[:3])
    c = colorsys.hls_to_rgb(c[0], 1 - amount * (1-c[1]), c[2])
    return mpl.colors.to_hex(c)


def draw(
    ts,
    ax,
    *,
    use_ranked_times=None,
    tweak_x=None,
    tweak_y=None,
    arrows=False,
    pos=None,
    draw_edge_widths=False,
    max_edge_width=5,
    draw_edge_alpha=False,
    node_arity_colors=False,
    edge_colors=None,
    nonsample_node_shrink=None,
    rotated_sample_labels=None,
    node_size=None,
    font_size=None,
    node_color=None,
    font_color=None,
    reverse_x_axis=None,
):
    """
    Draw a graphical representation of a tree sequence, returning the node
    positions and a networkx graph object. If a metadata key
    called "name" exists for the node, it is taken as
    a node label, otherwise the node ID will be used as a label instead.

    If use_ranked_times is True, the y axis uses the time ranks, with the
    same times sharing a rank. If False, it uses the true (tree sequence)
    times. If None, times from the tree sequence are not used and the
    standard dot layout is used.

    tweak_x is a dict of {u: x_adjustment_percent} which allows
    the x position of node u to be hand-adjusted by adding or
    subtracting a percentage of the total x width of the plot

    tweak_y is a dict of {u: y_adjustment_percent} which allows
    the y position of node u to be hand-adjusted by adding or
    subtracting a unit of time ranking (only if use_ranked_times
    is true).

    If pos is passed in, it should be a dictionary mapping nodes to
    positions.

    If draw_edge_widths is True, draw the widths of edges in the graphical
    representation of the tree sequence in proportion to their genomic span.
    max_edge_width specifies the maximum edge width to draw (default is 5).
    If draw_edge_widths is False, all edges are drawn with a width of
    max_edge_width.

    If draw_edge_alpha is True, draw edges with an alpha value equal to their
    genomic span / the total sequence length of the tree sequence.

    If `node_arity_colours` is True, the colour of the nodes is determined
    by the number of children and number of parents. In particular, the brightness
    of the fill colour is determined by the proportion of genome over which the node
    has 2 or more children (i.e. is "coalescent") compared to the amount of genome over
    which it has any children (if never coalescent, it it white). The hue of the
    colour (both for the stroke and the fill) indicated the number of parents of a
    node (red is 0, yellow/brown is 1, green to blue are 2 -> N parents. If
    `node_arity_colours` is True, node_colour is ignored and font_color is
    set to white for sample nodes.

    If `edge_colors` is not None, it should be a dictionary where keys are a tuple
    of (edge child node, edge parent node) and values are a hex color. If
    `edge_colors` is None, then

    If "nonsample_node_shrink" is not None, it should be an integer giving the
    amount by which nonsample node symbols are reduced; in this case labels on
    the nodes are omitted

    If "rotated_sample_labels" is True, sample lables are rotated and placed
    below the nodes

    node_size, font_size, font_color, and node_color are all passed to nx.draw
    directly. In particular this means that node_color can either be a single
    colour for all nodes (e.g. `mpl.colors.to_hex(mpl.pyplot.cm.tab20(1))`)
    or a list of colours (which is ignored if node_arity_colors is True)

    if reverse_x_axis is True, the graph is reflected horizontally

    """
    if node_size is None:
        node_size = 200
    if nonsample_node_shrink is not None:
        # Shrink the nonsample label size
        node_size_array = np.full(ts.num_nodes, node_size / nonsample_node_shrink)
        node_size_array[ts.samples()] = node_size
        node_size = node_size_array
    if font_size is None:
        font_size = 9
    if font_color is None:
        font_color = "k"

    G = convert_nx(ts)
    labels_by_colour = {font_color: {}}  # so we can change font colour
    is_sample = {}
    for nd in ts.nodes():
        is_sample[nd.id] = nd.is_sample()
        if nonsample_node_shrink is not None and not nd.is_sample():
            labels_by_colour[font_color][nd.id] = ""
        else:
            try:
                labels_by_colour[font_color][nd.id] = str(nd.metadata["name"])
            except (TypeError, KeyError):
                labels_by_colour[font_color][nd.id] = str(nd.id)

    if pos is None:
        pos = nx_get_dot_pos(G, reverse_x_axis)

    if use_ranked_times is not None:
        if use_ranked_times:
            ranked_times = scipy.stats.rankdata(ts.tables.nodes.time, method="dense")
            for i, p in pos.items():
                p[1] = ranked_times[i]
        else:
            times = ts.tables.nodes.time
            for i, p in pos.items():
                p[1] = times[i]
        if tweak_y:
            for node, tweak_val in tweak_y.items():
                pos[node] = pos[node] + np.array([0, tweak_val])
    if tweak_x:
        plot_width = np.ptp([x for x, _ in pos.values()])
        for node, tweak_val in tweak_x.items():
            pos[node] = pos[node] + np.array([(tweak_val/100*plot_width), 0])

    if draw_edge_widths:
        edge_widths = get_edge_spans(G)
        sequence_length = ts.get_sequence_length()
        edge_widths = (edge_widths / sequence_length) * max_edge_width
    else:
        edge_widths = max_edge_width
    if draw_edge_alpha:
        edge_alpha = get_edge_alpha(G, ts)
        if not draw_edge_widths:
            edge_widths = [max_edge_width for edge in G.edges()]
    else:
        edge_alpha = None

    # Draw just the nodes
    arity_edge_color = None
    if node_arity_colors:
        # Isoluminant colours
        spans = collections.defaultdict(float)
        c_spans = collections.defaultdict(float)
        for tree in ts.trees():
            for node in tree.nodes():
                if tree.num_children(node) > 0:
                    spans[node] += tree.span
                if tree.num_children(node) > 1:
                    c_spans[node] += tree.span
        node_color = []
        arity_edge_color = []
        for n, deg in G.out_degree():
            col = arity_colors(deg)
            # get span over which this is not unary
            if spans[n] == 0:
                assert ts.node(n).is_sample()
                node_color.append("#000000")
                arity_edge_color.append(make_color(col))
            else:
                arity_edge_color.append(make_color(col))
                node_color.append(make_color(col, lighten= 1 - (c_spans[n] / spans[n])))
        if not rotated_sample_labels:
            # Sample nodes are filled black, so we need to change the label colour
            labs = labels_by_colour[font_color]
            labels_by_colour["w"] = {k: v for k, v in labs.items() if ts.node(k).is_sample()}
            labs = {k: v for k, v in labs.items() if not ts.node(k).is_sample()}

    nx.draw(
        G,
        pos,
        node_color=node_color,
        edgecolors=arity_edge_color,
        with_labels=False,
        node_shape="o",
        node_size=node_size,
        font_size=font_size,
        ax=ax,
        edgelist=[],
    )
    for font_color, labels in labels_by_colour.items():
        text = nx.draw_networkx_labels(
            G,
            pos,
            font_color=font_color,
            font_size=font_size,
            labels=labels,
            ax=ax,
        )
        y_below_extra = 0.018 * np.diff(ax.get_ylim())  # 2% of the y range
        if rotated_sample_labels:
            for nd, t in text.items():
                if is_sample[nd]:
                    x, y = t.get_position()
                    t.set_rotation(-90)
                    t.set_position((x, y - y_below_extra))
                    t.set_va('top')
    # Assign edge colors using the ordering of edges in the networkx representation
    if edge_colors is not None:
        ordered_edge_color = []
        for edge_id, edge in enumerate(G.edges()):
            ordered_edge_color.append(edge_colors[(edge[0], edge[1])])
    else:
        ordered_edge_color = None

    # Now add the edges
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=list(G.edges()),
        arrowstyle="-|>" if arrows else "-",
        ax=ax,
        node_size=node_size,
        width=edge_widths,
        edge_color=ordered_edge_color
    )
    if edge_alpha is not None:
        for i, edge in enumerate(edges):
            edge.set_alpha(edge_alpha[i])

    return pos, G


def label_nodes(ts, labels=None):
    """
    Adds a metadata item called "name" to each node, whose value
    is given by the labels dictionary passed in. If labels is None (default),
    use the dictionary {0: 'a', 1: 'b', 2: 'c', ... 26: 'z'}.
    Any nodes without a corresponding key in the labels dictionary will
    simply have their metadata value set to their node id.

    Note that this means that if no labels are given, nodes 26 onwards
    will be labelled with numbers rather than ascii uppercase letters.
    """
    if labels is None:
        labels = {i: lab for i, lab in enumerate(string.ascii_lowercase)}
    tables = ts.dump_tables()

    tables.nodes.clear()
    for i, nd in enumerate(ts.tables.nodes):
        m = nd.metadata or {}
        # assume we can set metadata to a dict: e.g. that node.metadata_schema is json
        tables.nodes.append(nd.replace(metadata={**m, "name": labels.get(i, i)}))
    return tables.tree_sequence()


def convert_nx(ts):
    """
    Returns the specified tree sequence as an networkx directed graph.
    """
    G = nx.DiGraph()
    edges = collections.defaultdict(list)
    for edge in ts.edges():
        edges[(edge.child, edge.parent)].append((edge.left, edge.right))
    for node in ts.nodes():
        G.add_node(node.id, time=node.time, flags=node.flags)
    for edge, intervals in edges.items():
        G.add_edge(*edge, intervals=intervals)
    return G


def nx_get_dot_pos(G, reverse_x_axis=None):
    """
    Layout using graphviz's "dot" algorithm and return a dict of positions in the
    format required by networkx. We assume that the nodes have a "time" attribute
    """
    if reverse_x_axis is None:
        reverse_x_axis = False
    P=nx.drawing.nx_pydot.to_pydot(G) # switch to a Pydot representation
    nodes_at_time = collections.defaultdict(list)
    for nd in P.get_nodes():
        nodes_at_time[float(nd.get("time"))].append(nd)

    # First cluster all nodes at the same times (probably mostly samples)
    for t, nodes in nodes_at_time.items():
        if len(nodes) > 1:
            S = pydot.Cluster(f"cluster_t{t}")
            for nd in nodes:
                S.add_node(nd)
            P.add_subgraph(S)
    graphviz_bytes = P.create_dot()
    graphs = pydot.graph_from_dot_data(graphviz_bytes.decode())
    assert len(graphs) == 1
    graph = graphs[0]
    # negate at least the y axis coords, to get the graph going in the direction we want
    if reverse_x_axis:
        coord_direction = np.array([-1, -1])
    else:
        coord_direction = np.array([1, -1])
        
    return {
        # [1:-1] snips off enclosing quotes
        int(n.get_name()): np.fromstring(n.get_pos()[1:-1], sep=",") * coord_direction
        # need to iterate over this graph and also all the subgraphs to get all the nodes
        for g in itertools.chain([graph], graph.get_subgraphs()) for n in g.get_nodes()
        if n.get_pos() and n.get_name().isdigit()
    }


def get_edge_spans(G):
    """
    Returns a list of length "num_edges" in G, the graphical representation of the
    tree sequence, where each entry is the genomic span of the corresponding edge in the
    tree sequence divided by the total sequence length.
    """
    edge_spans = list()
    for edge in G.edges():
        edge_data = G.get_edge_data(edge[0], edge[1])
        total_span = sum(right - left for (left, right) in edge_data["intervals"])
        edge_spans.append(total_span)
    edge_spans = np.array(edge_spans)
    return edge_spans


def get_edge_alpha(G, ts):
    sequence_length = ts.get_sequence_length()
    edge_spans = get_edge_spans(G)
    edge_alpha = (edge_spans / sequence_length)
    return edge_alpha
