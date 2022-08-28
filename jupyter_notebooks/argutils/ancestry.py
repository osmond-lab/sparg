"""
Utilities for generating and converting ARGs in various formats.
"""
from __future__ import annotations

import collections
import random
import math
import dataclasses
from typing import List
from typing import Any

import numpy as np
import tskit
import networkx as nx

NODE_IS_RECOMB = 1 << 1


# AncestryInterval is the equivalent of msprime's Segment class. The
# important different here is that we don't associated nodes with
# individual intervals here: because this is an ARG, nodes that
# we pass through are recorded.
#
# (The ancestral_to field is also different here, but that's because
# I realised that the way we're tracking extant ancestral material
# in msprime is unnecessarily complicated, and we can actually
# track it locally. There is potentially quite a large performance
# increase available in msprime from this.)


@dataclasses.dataclass
class AncestryInterval:
    """
    Records that the specified interval contains genetic material ancestral
    to the specified number of samples.
    """

    left: int
    right: int
    ancestral_to: int

    @property
    def span(self):
        return self.right - self.left


@dataclasses.dataclass
class Lineage:
    """
    A single lineage that is present during the simulation of the coalescent
    with recombination. The node field represents the last (as we go backwards
    in time) genome in which an ARG event occured. That is, we can imagine
    a lineage representing the passage of the ancestral material through
    a sequence of ancestral genomes in which it is not modified.
    """

    node: int
    ancestry: List[AncestryInterval]

    def __str__(self):
        s = f"{self.node}:["
        for interval in self.ancestry:
            s += str((interval.left, interval.right, interval.ancestral_to))
            s += ", "
        if len(self.ancestry) > 0:
            s = s[:-2]
        return s + "]"

    @property
    def num_recombination_links(self):
        """
        The number of positions along this lineage's genome at which a recombination
        event can occur.
        """
        return self.right - self.left - 1

    @property
    def left(self):
        """
        Returns the leftmost position of ancestral material.
        """
        return self.ancestry[0].left

    @property
    def right(self):
        """
        Returns the rightmost position of ancestral material.
        """
        return self.ancestry[-1].right

    def split(self, breakpoint):
        """
        Splits the ancestral material for this lineage at the specified
        breakpoint, and returns a second lineage with the ancestral
        material to the right.
        """
        left_ancestry = []
        right_ancestry = []
        for interval in self.ancestry:
            if interval.right <= breakpoint:
                left_ancestry.append(interval)
            elif interval.left >= breakpoint:
                right_ancestry.append(interval)
            else:
                assert interval.left < breakpoint < interval.right
                left_ancestry.append(dataclasses.replace(interval, right=breakpoint))
                right_ancestry.append(dataclasses.replace(interval, left=breakpoint))
        self.ancestry = left_ancestry
        return Lineage(self.node, right_ancestry)


# The details of the machinery in the next two functions aren't important.
# It could be done more cleanly and efficiently. The basic idea is that
# we're providing a simple way to find the overlaps in the ancestral
# material of two or more lineages, abstracting the complex interval
# logic out of the main simulation.
@dataclasses.dataclass
class MappingSegment:
    left: int
    right: int
    value: Any = None


def overlapping_segments(segments):
    """
    Returns an iterator over the (left, right, X) tuples describing the
    distinct overlapping segments in the specified set.
    """
    S = sorted(segments, key=lambda x: x.left)
    n = len(S)
    # Insert a sentinel at the end for convenience.
    S.append(MappingSegment(math.inf, 0))
    right = S[0].left
    X = []
    j = 0
    while j < n:
        # Remove any elements of X with right <= left
        left = right
        X = [x for x in X if x.right > left]
        if len(X) == 0:
            left = S[j].left
        while j < n and S[j].left == left:
            X.append(S[j])
            j += 1
        j -= 1
        right = min(x.right for x in X)
        right = min(right, S[j + 1].left)
        yield left, right, X
        j += 1

    while len(X) > 0:
        left = right
        X = [x for x in X if x.right > left]
        if len(X) > 0:
            right = min(x.right for x in X)
            yield left, right, X


def merge_ancestry(lineages):
    """
    Return an iterator over the ancestral material for the specified lineages.
    For each distinct interval at which ancestral material exists, we return
    the AncestryInterval and the corresponding list of lineages.
    """
    # See note above on the implementation - this could be done more cleanly.
    segments = []
    for lineage in lineages:
        for interval in lineage.ancestry:
            segments.append(
                MappingSegment(interval.left, interval.right, (lineage, interval))
            )

    for left, right, U in overlapping_segments(segments):
        ancestral_to = sum(u.value[1].ancestral_to for u in U)
        interval = AncestryInterval(left, right, ancestral_to)
        yield interval, [u.value[0] for u in U]


@dataclasses.dataclass
class Node:
    time: float
    flags: int = 0
    metadata: dict = dataclasses.field(default_factory=dict)


def fully_coalesced(lineages, n):
    """
    Returns True if all segments are ancestral to n samples in all
    lineages.
    """
    for lineage in lineages:
        for segment in lineage.ancestry:
            if segment.ancestral_to < n:
                return False
    return True


def sim_coalescent(n, rho, L, seed=None, resolved=True):
    """
    Simulate an ancestry-resolved ARG under the coalescent with recombination
    and return the tskit TreeSequence object.

    NOTE! This hasn't been statistically tested and is probably not correct.
    """
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    lineages = []
    nodes = []
    for _ in range(n):
        lineages.append(Lineage(len(nodes), [AncestryInterval(0, L, 1)]))
        nodes.append(Node(time=0, flags=tskit.NODE_IS_SAMPLE))

    t = 0

    # NOTE: the stopping condition here is more complicated because
    # we have to still keep track of segments after the MRCA has been
    # reached. We don't simulate back to the GMRCA as this may take a
    # very long time.

    # while len(lineages) > 0:
    while not fully_coalesced(lineages, n):
        # print(f"t = {t:.2f} k = {len(lineages)}")
        # for lineage in lineages:
        #     print(f"\t{lineage}")
        lineage_links = [lineage.num_recombination_links for lineage in lineages]
        total_links = sum(lineage_links)
        re_rate = total_links * rho
        t_re = math.inf if re_rate == 0 else rng.expovariate(re_rate)
        k = len(lineages)
        ca_rate = k * (k - 1) / 2
        t_ca = rng.expovariate(ca_rate)
        t_inc = min(t_re, t_ca)
        t += t_inc

        if t_inc == t_re:
            left_lineage = rng.choices(lineages, weights=lineage_links)[0]
            breakpoint = rng.randrange(left_lineage.left + 1, left_lineage.right)
            assert left_lineage.left < breakpoint < left_lineage.right
            right_lineage = left_lineage.split(breakpoint)
            lineages.append(right_lineage)
            child = left_lineage.node
            assert nodes[child].flags & NODE_IS_RECOMB == 0
            nodes[child].flags |= NODE_IS_RECOMB
            assert "breakpoint" not in nodes[child].metadata
            nodes[child].metadata["breakpoint"] = breakpoint
            for lineage in left_lineage, right_lineage:
                lineage.node = len(nodes)
                nodes.append(Node(time=t))
                if resolved:
                    for interval in lineage.ancestry:
                        tables.edges.add_row(
                            interval.left, interval.right, lineage.node, child
                        )
            if not resolved:
                tables.edges.add_row(0, breakpoint, left_lineage.node, child)
                tables.edges.add_row(breakpoint, L, right_lineage.node, child)
        else:
            a = lineages.pop(rng.randrange(len(lineages)))
            b = lineages.pop(rng.randrange(len(lineages)))
            c = Lineage(len(nodes), [])
            for interval, intersecting_lineages in merge_ancestry([a, b]):
                # if interval.ancestral_to < n:
                c.ancestry.append(interval)
                if resolved:
                    for lineage in intersecting_lineages:
                        tables.edges.add_row(
                            interval.left, interval.right, c.node, lineage.node
                        )
            if not resolved:
                tables.edges.add_row(0, L, c.node, a.node)
                tables.edges.add_row(0, L, c.node, b.node)

            nodes.append(Node(time=t))
            # if len(c.ancestry) > 0:
            lineages.append(c)

    for node in nodes:
        tables.nodes.add_row(flags=node.flags, time=node.time, metadata=node.metadata)
    tables.sort()
    # TODO not sure if this is the right thing to do, but it makes it easier
    # to compare with examples.
    tables.edges.squash()
    return tables.tree_sequence()


@dataclasses.dataclass
class Individual:
    id: int = -1
    lineages: List[Lineage] = dataclasses.field(default_factory=list)
    collected_lineages: List[List[Lineage]] = dataclasses.field(
        default_factory=lambda: [[], []]
    )


def iter_lineages(individuals):
    for individual in individuals:
        for lineage in individual.lineages:
            yield lineage


def remove_unused_nodes(ts):
    """
    Remove any nodes not referenced in the edge table, but keep the edges
    """
    tables = ts.tables
    used_nodes = np.unique(np.concatenate((tables.edges.parent, tables.edges.child)))
    return ts.subset(used_nodes)


def simplify_keeping_all_nodes(ts):
    """
    Run the Hudson algorithm to convert from an implicit to an explicit edge encoding,
    but keep all the nodes (like simplify(filter_nodes=False) if that existed)
    """
    # get the edges to keep
    ts2, node_map = ts.simplify(keep_unary=True, map_nodes=True)
    val, inverted_map = np.unique(node_map, return_index=True)
    inverted_map = inverted_map[val != tskit.NULL]
    # only use the edges in the simplified one, but keep the nodes from the original
    tables = ts.dump_tables()
    tables.edges.clear()
    for edge in ts2.tables.edges:
        tables.edges.append(
            edge.replace(
                child=inverted_map[edge.child], parent=inverted_map[edge.parent]
            )
        )
    tables.sort()
    return tables.tree_sequence()


def simplify_remove_pass_through(ts, repeat=False, map_nodes=False):
    """
    Remove nonsample nodes that have same single child and parent everywhere in the ts.
    Removing some pass though nodes can turn previously non-pass-through nodes
    into pass-through nodes (for example, at the top of a diamond). If repeat==True,
    we carry on doing removing these nodes repeatedly until
    there are no pass-though nodes remaining
    """
    tables = ts.dump_tables()
    # remove existing individuals. We will reinstate them later
    node_map = np.arange(ts.num_nodes)

    while True:
        tables.individuals.clear()
        tables.nodes.individual = np.full_like(tables.nodes.individual, tskit.NULL)
        parent_children_dict = collections.defaultdict(set)
        for t in ts.trees():
            # TODO: there must be a more efficient way to do this using edge dicts
            for u in t.nodes():
                parent_children_dict[u].add((t.parent(u), tuple(sorted(t.children(u)))))
        to_keep = np.ones(tables.nodes.num_rows, dtype=bool)
        for u, parent_children in parent_children_dict.items():
            if ts.node(u).is_sample():
                continue
            if len(parent_children) != 1:
                continue
            parent, children = next(iter(parent_children))
            if len(children) == 1:
                to_keep[u] = False
        if np.all(to_keep):
            break
        # Add an individual for each kept node, so we can run
        # simplify(keep_unary_in_individuals=True) to leave the unary portions in.
        for u in np.where(to_keep)[0]:
            i = tables.individuals.add_row()
            tables.nodes[u] = tables.nodes[u].replace(individual=i)
        tmp_node_map = tables.simplify(
            keep_unary_in_individuals=True,
            filter_sites=False,
            filter_populations=False,
        )
        # Keep track of the repeated node mappings
        node_map[node_map != tskit.NULL] = tmp_node_map[
            node_map[node_map != tskit.NULL]
        ]
        if repeat:
            ts = tables.tree_sequence()
        else:
            break

    # Remove all traces of the added individuals
    tables.individuals.clear()
    tables.nodes.individual = np.full_like(tables.nodes.individual, tskit.NULL)
    if map_nodes:
        return tables.tree_sequence(), node_map
    else:
        return tables.tree_sequence()


def simplify_keeping_unary_in_coal(ts, map_nodes=False):
    """
    Keep the unary regions of nodes that are coalescent at least someone in the tree seq
    Temporary hack until https://github.com/tskit-dev/tskit/issues/2127 is addressed
    """
    tables = ts.dump_tables()
    # remove existing individuals. We will reinstate them later
    tables.individuals.clear()
    tables.nodes.individual = np.full_like(tables.nodes.individual, tskit.NULL)

    _, node_map = ts.simplify(map_nodes=True)
    keep_nodes = np.where(node_map != tskit.NULL)[0]
    # Add an individual for each coalescent node, so we can run
    # simplify(keep_unary_in_individuals=True) to leave the unary portions in.
    for u in keep_nodes:
        i = tables.individuals.add_row()
        tables.nodes[u] = tables.nodes[u].replace(individual=i)
    node_map = tables.simplify(keep_unary_in_individuals=True)

    # Reinstate individuals
    tables.individuals.clear()
    for i in ts.individuals():
        tables.individuals.append(i)
    val, inverted_map = np.unique(node_map, return_index=True)
    inverted_map = inverted_map[val != tskit.NULL]
    tables.nodes.individual = ts.tables.nodes.individual[inverted_map]
    if map_nodes:
        return tables.tree_sequence(), node_map
    else:
        return tables.tree_sequence()


def is_recombinant(flags):
    return (flags & NODE_IS_RECOMB) != 0


def sim_wright_fisher(n, N, L, recomb_proba=1, seed=None):
    """
    NOTE! This hasn't been statistically tested and is probably not correct.

    The ``recomb_proba`` is the probability of recombination occuring
    when a genome chooses its parents. With probability ``recomb_proba``
    a breakpoint will be chosen and the ancestry split among two parents
    (if present); otherwise, one parent genome will be chosen randomly.
    (This almost certainly not a good way to set things up model-wise,
    and just a quick hack to give us a lever to control the amount of
    recombination happening.)

    We don't keep track of the pedigree because this is inconsistent
    with the practise of dropping "pass through" nodes in which
    nothing happens.
    """
    # We don't offer a "resolved" option here, but it should be
    # easy enough to implement the resolved=False version. It should be
    # a case of skipping the second loop in which we go through the
    # collected ancestry, and instead writing out the edges directly when
    # parents are chosen
    rng = random.Random(seed)
    tables = tskit.TableCollection(L)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()

    ancestors = []
    node_flags = []
    for _ in range(n):
        ind = Individual(tables.individuals.add_row(), [])
        for _ in range(2):
            node = tables.nodes.add_row(time=0, individual=ind.id)
            ind.lineages.append(Lineage(node, [AncestryInterval(0, L, 1)]))
            node_flags.append(tskit.NODE_IS_SAMPLE)
        ancestors.append(ind)

    t = 0
    while not fully_coalesced(iter_lineages(ancestors), 2 * n):
        t += 1
        # print("===")
        # print("T = ", t, "|A|=", len(ancestors))
        # for ancestor in ancestors:
        #     print(ancestor)
        # Group the ancestral lineages among the parent individuals chosen
        # among the previous generation. We map the individual's index
        # in the population to the lineages inherited from is maternal
        # and paternal genomes
        parents = {}
        for ancestor in ancestors:
            for lineage in ancestor.lineages:
                parent_index = rng.randrange(N)
                if parent_index not in parents:
                    parents[parent_index] = Individual()
                parent = parents[parent_index]
                # Randomise the order we add the lineages
                rng.shuffle(parent.collected_lineages)
                j = 0
                if rng.random() < recomb_proba:
                    breakpoint = rng.randrange(1, L)
                    # print("breakpoint", breakpoint)
                    if lineage.left < breakpoint < lineage.right:
                        # print("effective recombination!", lineage.node)
                        right_lineage = lineage.split(breakpoint)
                        parent.collected_lineages[j].append(right_lineage)
                        j += 1
                        node_flags[lineage.node] |= NODE_IS_RECOMB
                parent.collected_lineages[j].append(lineage)

        # All the ancestral material has been distributed to the parental
        # lineages.
        ancestors.clear()

        for parent in parents.values():
            for lineages in parent.collected_lineages:
                # These are all the lineages that have been collected
                # together for this lineage on this individual. If there
                # is at least one piece of ancestral material going through,
                # we create a node for it.
                # print("\tlineages = ")
                # for lin in lineages:
                #     print("\t\t", lin)

                merged_lineage = None
                # is_recombinant = True
                if len(lineages) == 1 and not is_recombinant(
                    node_flags[lineages[0].node]
                ):
                    merged_lineage = lineages[0]
                elif len(lineages) > 0:
                    if parent.id == -1:
                        parent.id = tables.individuals.add_row()
                    node = len(tables.nodes)
                    merged_lineage = Lineage(node, [])
                    node_flags.append(0)
                    for interval, intersecting_lineages in merge_ancestry(lineages):
                        # if interval.ancestral_to < 2 * n:  # n is *diploid* sample size
                        merged_lineage.ancestry.append(interval)
                        for child_lineage in intersecting_lineages:
                            tables.edges.add_row(
                                interval.left, interval.right, node, child_lineage.node
                            )
                    tables.nodes.add_row(time=t, individual=parent.id, metadata={})
                if merged_lineage is not None:
                    parent.lineages.append(merged_lineage)
            if len(parent.lineages) > 0:
                ancestors.append(parent)

        assert len(ancestors) <= N

    tables.nodes.flags = node_flags

    tables.sort()
    return tables.tree_sequence()


def wh99_example():
    """
    The example ARG from figure 1 of Wiuf and Hein 99, Recombination as a Point Process
    along Sequences. Each event is given a time increment of 1.
    """
    L = 7
    tables = tskit.TableCollection(L)
    nodes = tables.nodes
    edges = tables.edges
    nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    t = 0
    for _ in range(3):
        nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=t)

    def re(child, x):
        nonlocal t
        t += 1
        left_parent = nodes.add_row(time=t)
        right_parent = nodes.add_row(time=t)
        edges.add_row(0, x, left_parent, child)
        edges.add_row(x, L, right_parent, child)
        nodes[child] = nodes[child].replace(flags=nodes[child].flags | NODE_IS_RECOMB)

    def ca(*args):
        nonlocal t
        t += 1
        parent = nodes.add_row(time=t)
        for child in args:
            edges.add_row(0, L, parent, child)

    re(1, x=4)
    re(2, x=2)
    ca(0, 3)
    re(5, x=5)
    ca(4, 8)
    re(10, x=3)
    ca(12, 9)
    ca(7, 11)
    re(13, x=6)
    ca(16, 6)
    ca(14, 15)
    re(18, x=1)
    ca(20, 17)
    ca(19, 21)

    tables.sort()
    return tables.tree_sequence()


class IntervalSet:
    """
    Naive and simple implementation of discrete intervals.
    """

    def __init__(self, L, tuples=None):
        assert int(L) == L
        self.I = np.zeros(int(L), dtype=int)
        if tuples is not None:
            for left, right in tuples:
                self.insert(left, right)

    def __str__(self):
        return str(self.I)

    def __repr__(self):
        return repr(list(self.I))

    def __eq__(self, other):
        return np.array_equal(self.I == 0, other.I == 0)

    def insert(self, left, right):
        assert int(left) == left
        assert int(right) == right
        self.I[int(left) : int(right)] = 1

    def contains(self, x):
        assert int(x) == x
        return self.I[int(x)] != 0

    def union(self, other):
        """
        Returns a new IntervalSet with the union of intervals in this and
        other.
        """
        new = IntervalSet(self.I.shape[0])
        assert other.I.shape == self.I.shape
        new.I[:] = np.logical_or(self.I, other.I)
        return new

    def intersection(self, other):
        """
        Returns a new IntervalSet with the intersection of intervals in this and
        other.
        """
        new = IntervalSet(self.I.shape[0])
        assert other.I.shape == self.I.shape
        new.I[:] = np.logical_and(self.I, other.I)
        return new

    def is_subset(self, other):
        """
        Return True if this set is a subset of other.
        """
        a = np.all(other.I[self.I == 1] == 1)
        b = np.all(self.I[other.I == 0] == 0)
        return a and b


def as_garg(ts):
    """
    Returns the specified unresolved ARG as an GARG E.
    """
    E = collections.defaultdict(lambda: IntervalSet(ts.sequence_length))
    for edge in ts.edges():
        E[(edge.child, edge.parent)].insert(edge.left, edge.right)
    return [(u, v, I) for (u, v), I in E.items()]


def as_resolved_garg(ts):
    """
    Returns the specified unresolved ARG as an GARG E with respect to
    the tree sequences samples.
    """
    E = as_garg(ts)
    G = nx.DiGraph()
    G.add_edges_from([e[:2] for e in E])
    # We have to get the topological sorting of the nodes because we don't
    # have time ordering.
    topological = list(nx.topological_sort(G))

    I = collections.defaultdict(lambda: IntervalSet(ts.sequence_length))
    for u in ts.samples():
        I[u] = IntervalSet(ts.sequence_length, [(0, ts.sequence_length)])

    Ep = []
    for c in topological:
        # print("c = ", c)
        for e in [e for e in E if e[0] == c]:
            p, Ie = e[1:]
            inter = I[c].intersection(Ie)
            Ep.append((c, p, inter))
            I[p] = I[p].union(inter)
    return Ep


# TODO Need to be clearer here - what are the assumptions about how things
# are encoded in the tree sequence? We should probably call this
# as_minimial_arg or something.
def as_earg(ts):
    """
    Returns the specified unresolved ARG as an EARG (E, sigma).
    """
    sigma = np.full(ts.num_nodes, int(ts.sequence_length), dtype=int)
    edges = iter(ts.edges())
    edge = next(edges, None)
    E = []
    while edge is not None:
        if edge.left == 0 and edge.right == ts.sequence_length:
            E.append((edge.child, edge.parent))
        else:
            assert edge.left == 0
            breakpoint = edge.right
            child = edge.child
            sigma[child] = breakpoint
            E.append((edge.child, edge.parent))
            edge = next(edges, None)
            assert edge.left == breakpoint
            assert edge.child == child
            E.append((edge.child, edge.parent))
        edge = next(edges, None)
    # We could wrap sigma as a function to be literal about the definition
    # but this is simpler for debugging.
    return E, sigma


def earg_get_tree(E, sigma, S, x):
    """
    Given a minimal EARG definition (E, sigma), return the tree for a the given
    set of samples at the specified position as a dictionary parent->child.
    """
    N = set(S)
    parent = {}
    while len(N) > 0:
        c = N.pop()
        P = [e[1] for e in E if e[0] == c]
        if len(P) > 0:
            if x < sigma[c]:
                p = P[0]
            else:
                p = P[1]
            parent[c] = p
            N.add(p)
    return parent


def garg_get_tree(E, S, x):
    """
    Given a minimal GARG definition E, return the tree for a the given
    set of samples at the specified position as a dictionary parent->child.
    """
    N = set(S)
    parent = {}
    while len(N) > 0:
        c = N.pop()
        P = [e[1] for e in E if e[0] == c and e[2].contains(x)]
        # print(c, P)
        if len(P) > 0:
            assert len(P) == 1
            parent[c] = P[0]
            N.add(P[0])
    return parent


@dataclasses.dataclass
class RecombinationEvent:
    parent_edges: List[tskit.Edge] = dataclasses.field(default_factory=list)
    child_edge: tskit.Edge | None = None


def earg_to_garg(ts):
    """
    Given an EARG encoded in the specified tree sequence, return the
    corresponding GARG in which we create two parental nodes
    instead of a single recombination event.

    NOTE: it's important to realise here that we can't tell if an
    input ARG is a GARG or an EARG just by looking at the edges.
    There's no difference topologically between an EARG in which
    the focal recombination node is the child in two edges,
    or a GARG in which the *recombinant* node is the child in
    two edges. We could check some properties of the nodes (like,
    the two parents should be at different times in an EARG), but
    they make assumptions also. So, this transformation may be useless in
    practise.

    However, the general approach in which we graft on some
    extra topology in a minimally destructive way is worth
    keeping for now.
    """
    # TODO need to resolve what we mean by the terminology here, the
    # ts EARG is different to what we're working with above.

    L = ts.sequence_length
    # Map from recombination nodes to the RecombinationEvent objects.
    recombination_events = collections.defaultdict(RecombinationEvent)

    for edge in ts.edges():
        if edge.left != 0 and edge.right != ts.sequence_length:
            raise ValueError("only edges like (0, x], (x, L] or (0, L]) are allowed")
        if edge.left != 0 or edge.right != ts.sequence_length:
            recombination_events[edge.child].parent_edges.append(edge)
    for edge in ts.edges():
        if edge.parent in recombination_events:
            event = recombination_events[edge.parent]
            assert event.child_edge == None
            event.child_edge = edge

    # print(recombination_events)
    tables = ts.dump_tables()
    tables.edges.clear()
    for edge in ts.edges():
        if (
            edge.child not in recombination_events
            and edge.parent not in recombination_events
        ):
            tables.edges.add_row(edge.left, edge.right, edge.parent, edge.child)

    for u, event in recombination_events.items():
        assert len(event.parent_edges) == 2
        pe1, pe2 = event.parent_edges
        # Rewrite the edges for the first parent, reusing the existing node.
        tables.edges.add_row(pe1.left, pe1.right, u, event.child_edge.child)
        tables.edges.add_row(0, L, pe1.parent, u)
        # Add a new node for the second path
        # TODO there must be a better way of doing this:
        u_obj = ts.node(u)
        v = tables.nodes.add_row(
            flags=u_obj.flags,
            time=u_obj.time,
            individual=u_obj.individual,
            population=u_obj.population,
            metadata=u_obj.metadata,
        )
        # Add the edges from the other path.
        tables.edges.add_row(pe2.left, pe2.right, v, event.child_edge.child)
        tables.edges.add_row(0, L, pe2.parent, v)

    tables.sort()
    return tables.tree_sequence()
