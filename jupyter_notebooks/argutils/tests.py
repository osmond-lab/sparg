# Run using python3 -m pytest argutils/tests.py
import collections
import string

import tskit
import pytest
import numpy as np

import argutils


def assert_arg_properties(ts):
    # Each node should have at most two parents along the genome.
    parents = collections.defaultdict(set)
    for tree in ts.trees():
        for u in tree.nodes():
            parent = tree.parent(u)
            if parent != tskit.NULL:
                parents[u].add(parent)
    for nodes in parents.values():
        assert len(nodes) <= 2
        if len(nodes) == 2:
            assert len(set(ts.node(u).time for u in nodes)) == 1


class TestSimulate:
    def test_basic_coalescent(self):
        ts = argutils.sim_coalescent(4, L=5, rho=0.1, seed=1)
        assert ts.num_samples == 4
        assert ts.sequence_length == 5
        assert ts.num_trees > 1
        assert all(tree.num_roots == 1 for tree in ts.trees())

    def test_basic_wright_fisher(self):
        ts = argutils.sim_wright_fisher(4, N=10, L=5, seed=1)
        assert ts.num_samples == 8
        assert ts.sequence_length == 5
        assert ts.num_trees > 1
        assert all(tree.num_roots == 1 for tree in ts.trees())

    def test_wright_fisher_no_recomb(self):
        ts = argutils.sim_wright_fisher(4, N=10, L=5, recomb_proba=0, seed=1)
        assert ts.num_samples == 8
        assert ts.sequence_length == 5
        assert ts.num_trees == 1
        assert ts.num_nodes == 15
        assert all(tree.num_roots == 1 for tree in ts.trees())

    @pytest.mark.parametrize("seed", range(1, 10))
    def test_wright_fisher_high_recomb(self, seed):
        ts = argutils.sim_wright_fisher(4, N=10, L=5, recomb_proba=0.9, seed=seed)
        assert ts.num_samples == 8
        assert ts.sequence_length == 5
        assert ts.num_trees > 1
        assert all(tree.num_roots == 1 for tree in ts.trees())
        assert_arg_properties(ts)

    def test_basic_coalescent_unresolved(self):
        ts = argutils.sim_coalescent(4, L=5, rho=0.1, seed=1, resolved=False)
        # Very basic - real test is the resolution process.
        assert ts.num_nodes > 4
        assert ts.num_edges > 4

    @pytest.mark.parametrize("seed", range(1, 10))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [100, 4],
            [10, 100],
        ],
    )
    def test_coalescent_resolved_equal(self, n, L, seed):
        rho = 0.1
        ts1 = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        ts2 = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=True)
        ts3 = ts1.simplify(keep_unary=True)
        ts2.tables.assert_equals(ts3.tables, ignore_provenance=True)


def parent_dict(tree):
    """
    Returns the parent dict reachable from the samples for the specified
    tskit tree. We can't use tree.parent_dict as it includes non-sample
    leaves.
    """
    d = {}
    # Could make this more efficient avoiding already written paths,
    # but can't be bothered.
    for u in tree.tree_sequence.samples():
        v = tree.parent(u)
        while v != tskit.NULL:
            d[u] = v
            u = v
            v = tree.parent(u)
    return d


class TestEARG:
    @pytest.mark.parametrize("seed", range(1, 3))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_trees_equal_left(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E, sigma = argutils.as_earg(ts)
        for tree in ts.trees():
            t = argutils.earg_get_tree(E, sigma, ts.samples(), tree.interval[0])
            assert t == parent_dict(tree)

    @pytest.mark.parametrize("seed", range(10, 12))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_trees_equal_mid(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E, sigma = argutils.as_earg(ts)
        for tree in ts.trees():
            x = tree.interval.left + tree.interval.span / 2
            t = argutils.earg_get_tree(E, sigma, ts.samples(), int(x))
            assert t == parent_dict(tree)


class TestGARG:
    @pytest.mark.parametrize("seed", range(1, 3))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_trees_equal_left(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E = argutils.as_garg(ts)
        for tree in ts.trees():
            t = argutils.garg_get_tree(E, ts.samples(), tree.interval[0])
            assert t == parent_dict(tree)

    @pytest.mark.parametrize("seed", range(5, 8))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_trees_equal_mid(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E = argutils.as_garg(ts)
        for tree in ts.trees():
            x = tree.interval.left + tree.interval.span / 2
            t = argutils.garg_get_tree(E, ts.samples(), int(x))
            assert t == parent_dict(tree)


def assert_gargs_equal(E1, E2):
    assert len(E1) == len(E2)
    d1 = {(u, v): I for (u, v, I) in E1}
    d2 = {(u, v): I for (u, v, I) in E2}
    assert len(d1) == len(d2)
    for key, I1 in d1.items():
        I2 = d2[key]
        # print(key, I1, I2)
        assert I1 == I2


class TestResolvedGARG:
    @pytest.mark.parametrize("seed", range(1, 3))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_trees_equal_left(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E = argutils.as_resolved_garg(ts)
        for tree in ts.trees():
            t = argutils.garg_get_tree(E, ts.samples(), tree.interval[0])
            assert t == parent_dict(tree)

    @pytest.mark.parametrize("seed", range(5, 8))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_trees_equal_mid(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E = argutils.as_resolved_garg(ts)
        for tree in ts.trees():
            x = tree.interval.left + tree.interval.span / 2
            t = argutils.garg_get_tree(E, ts.samples(), int(x))
            assert t == parent_dict(tree)

    @pytest.mark.parametrize("seed", range(1, 3))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_garg_subset(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        d1 = {(u, v): I for (u, v, I) in argutils.as_garg(ts)}
        d2 = {(u, v): I for (u, v, I) in argutils.as_resolved_garg(ts)}
        assert len(d1) == len(d2)
        for key, I1 in d1.items():
            I2 = d2[key]
            # The intervals for the resolved edges should be a subset of
            # the unresolved.
            assert I2.is_subset(I1)

    @pytest.mark.parametrize("seed", range(1, 3))
    @pytest.mark.parametrize(
        ["n", "L"],
        [
            [2, 2],
            [8, 2],
            [16, 2],
            [2, 10],
            [8, 10],
            [16, 10],
            [10, 50],
        ],
    )
    def test_equal_to_resolved(self, n, L, seed):
        rho = 0.1
        ts = argutils.sim_coalescent(n, L=L, rho=rho, seed=seed, resolved=False)
        E1 = argutils.as_resolved_garg(ts)
        tsr = ts.simplify(keep_unary=True)
        E2 = argutils.as_garg(tsr)
        assert_gargs_equal(E1, E2)

    def test_equal_to_resolved_example(self):
        E1 = argutils.as_resolved_garg(gmrca_example_unresolved())
        E2 = argutils.as_garg(gmrca_example_resolved())
        assert_gargs_equal(E1, E2)


def gmrca_example_resolved():
    # 3.00┊  5  ┊  5  ┊
    #     ┊  ┃  ┊ ┏┻┓ ┊
    # 2.00┊  4  ┊ 4 ┃ ┊
    #     ┊ ┏┻┓ ┊ ┃ ┃ ┊
    # 1.00┊ ┃ 2 ┊ ┃ 3 ┊
    #     ┊ ┃ ┃ ┊ ┃ ┃ ┊
    # 0.00┊ 0 1 ┊ 0 1 ┊
    #     0     1     2
    tables = tskit.TableCollection(2)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=1)
    tables.nodes.add_row(time=1)
    tables.nodes.add_row(time=2)
    tables.nodes.add_row(time=3)

    tables.edges.add_row(0, 2, 4, 0)
    tables.edges.add_row(0, 2, 5, 4)
    tables.edges.add_row(0, 1, 2, 1)
    tables.edges.add_row(0, 1, 4, 2)
    tables.edges.add_row(1, 2, 3, 1)
    tables.edges.add_row(1, 2, 5, 3)
    tables.sort()
    return tables.tree_sequence()


def gmrca_example_unresolved():
    # 3.00┊     5   ┊
    #     ┊  ┏━━┻━┓ ┊
    # 2.00┊  4    ┃ ┊
    #     ┊ ┏┻┓   ┃ ┊
    # 1.00┊ ┃ 2━┳━3 ┊
    #     ┊ ┃   ┃   ┊
    # 0.00┊ 0   1   ┊
    #
    tables = tskit.TableCollection(2)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=1)
    tables.nodes.add_row(time=1)
    tables.nodes.add_row(time=2)
    tables.nodes.add_row(time=3)

    tables.edges.add_row(0, 1, 2, 1)
    tables.edges.add_row(1, 2, 3, 1)
    tables.edges.add_row(0, 2, 4, 0)
    tables.edges.add_row(0, 2, 5, 4)
    tables.edges.add_row(0, 2, 4, 2)
    tables.edges.add_row(0, 2, 5, 3)
    tables.sort()
    return tables.tree_sequence()


class TestEARGtoGARG:
    def test_example(self):
        # 3.00┊     4   ┊
        #     ┊  ┏━━┻━┓ ┊
        # 2.00┊  3    ┃ ┊
        #     ┊ ┏┻┓   ┃ ┊
        # 1.00┊ ┃ ┗━2━┛ ┊
        #     ┊ ┃   ┃   ┊
        # 0.00┊ 0   1   ┊
        #
        tables = tskit.TableCollection(2)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(time=2)
        tables.nodes.add_row(time=3)

        tables.edges.add_row(0, 2, 2, 1)
        tables.edges.add_row(0, 2, 3, 0)
        tables.edges.add_row(0, 2, 4, 3)
        tables.edges.add_row(0, 1, 3, 2)
        tables.edges.add_row(1, 2, 4, 2)

        tables.sort()
        ts = tables.tree_sequence()
        ts2 = argutils.earg_to_garg(ts)
        # Can't easily compare the unresolved version because the node IDs
        # differ.
        ts3 = ts2.simplify(keep_unary=True)
        ts3.tables.assert_equals(
            gmrca_example_resolved().tables, ignore_provenance=True
        )


class TestResolve:
    @pytest.mark.parametrize(
        ["unresolved", "resolved"],
        [(gmrca_example_unresolved(), gmrca_example_resolved())],
    )
    def test_examples(self, unresolved, resolved):
        resolved2 = unresolved.simplify(keep_unary=True)
        assert resolved.equals(resolved2, ignore_provenance=True)


class TestLabels:
    def test_viz_label_nodes_many(self):
        ts = argutils.sim_coalescent(20, 0.1, 10, seed=123)  # over 26 nodes
        ts = argutils.viz.label_nodes(ts)
        assert ts.num_nodes > len(string.ascii_uppercase)
        i = 0
        for c in string.ascii_uppercase:
            assert ts.node(i).metadata["name"] == c
            i += 1
        while i < ts.num_nodes:
            assert ts.node(i).metadata["name"] == i
            i += 1

    def test_viz_label_nodes_bespoke(self):
        ts = argutils.sim_coalescent(20, 0.1, 10, seed=123)  # over 26 nodes
        ts = argutils.viz.label_nodes(
            ts, labels={n: str(i) for n, i in enumerate(range(ts.num_nodes, 0, -1))}
        )
        for nd in ts.nodes():
            assert nd.metadata["name"] == f"{ts.num_nodes - nd.id}"


class TestSimplifyFunctions:
    def test_simplify_keeping_all_nodes(self):
        ts = argutils.viz.label_nodes(argutils.wh99_example())
        ts2 = argutils.simplify_keeping_all_nodes(ts)
        assert ts.num_edges > ts2.num_edges
        assert ts.num_nodes == ts2.num_nodes
        for n1, n2 in zip(ts.nodes(), ts2.nodes()):
            assert "name" in n1.metadata and n1.metadata["name"]
            assert n1.metadata["name"] == n2.metadata["name"]

    def test_simplify_remove_pass_through(self):
        ts = argutils.sim_coalescent(10, 0.1, 10, seed=3)
        node_edges = np.zeros((ts.num_nodes, 2), dtype=int)
        has_pass_through = False
        for e in ts.edges():
            node_edges[e.child][0] += 1
            node_edges[e.parent][1] += 1
        # at least one node with 1 parent and 1 child only
        for row in node_edges:
            if list(row) == [1, 1]:
                has_pass_through = True
        assert has_pass_through

        # This example has a diamond, so a single pass of the remove_pass_through
        # algorithm won't remove the top of the diamon
        ts2 = argutils.simplify_remove_pass_through(ts, repeat=False)
        node_edges = np.zeros((ts.num_nodes, 2), dtype=int)
        has_pass_through = False
        for e in ts.edges():
            node_edges[e.child][0] += 1
            node_edges[e.parent][1] += 1
        # still at least one node with 1 parent and 1 child only
        for row in node_edges:
            if list(row) == [1, 1]:
                has_pass_through = True
        assert has_pass_through

        ts3 = argutils.simplify_remove_pass_through(ts, repeat=True)
        node_edges = np.zeros((ts.num_nodes, 2), dtype=int)
        for e in ts3.edges():
            node_edges[e.child][0] += 1
            node_edges[e.parent][1] += 1
        for i, row in enumerate(node_edges):
            # No more pass thoguh nodes left
            assert list(row) != [1, 1]

    @pytest.mark.parametrize(
        "ts",
        [
            argutils.wh99_example(),
            argutils.sim_wright_fisher(4, N=10, L=5, seed=1),
            argutils.sim_coalescent(10, 0.1, 10, seed=3),
        ],
    )
    def test_simplify_keeping_unary_in_coal(self, ts):
        ts2 = argutils.simplify_keeping_unary_in_coal(ts)
        ts3 = ts.simplify()
        assert ts2.num_nodes == ts3.num_nodes
        has_unary = False
        for tree in ts2.trees():
            for n in tree.nodes():
                if tree.num_children(n) == 1:
                    has_unary = True
        assert has_unary
        assert ts2.num_individuals == ts.num_individuals

        has_unary = False
        for tree in ts3.trees():
            for n in tree.nodes():
                if tree.num_children(n) == 1:
                    has_unary = True
        assert not has_unary


class TestIntervalSet:
    def test_single_interval(self):
        s = argutils.IntervalSet(4, [(1, 4)])
        assert list(s.I) == [0, 1, 1, 1]

    def test_two_intervals(self):
        s = argutils.IntervalSet(6, [(0, 1), (5, 6)])
        assert list(s.I) == [1, 0, 0, 0, 0, 1]

    def test_intersection(self):
        s1 = argutils.IntervalSet(6, [(0, 4)])
        s2 = argutils.IntervalSet(6, [(3, 6)])
        s3 = s2.intersection(s1)
        assert list(s1.I) == [1, 1, 1, 1, 0, 0]
        assert list(s2.I) == [0, 0, 0, 1, 1, 1]
        assert list(s3.I) == [0, 0, 0, 1, 0, 0]

    def test_union(self):
        s1 = argutils.IntervalSet(6, [(0, 4)])
        s2 = argutils.IntervalSet(6, [(3, 6)])
        s3 = s2.union(s1)
        assert list(s1.I) == [1, 1, 1, 1, 0, 0]
        assert list(s2.I) == [0, 0, 0, 1, 1, 1]
        assert list(s3.I) == [1, 1, 1, 1, 1, 1]

    def test_is_subset_overlapping(self):
        s1 = argutils.IntervalSet(6, [(0, 4)])
        s2 = argutils.IntervalSet(6, [(3, 6)])
        assert not s1.is_subset(s2)
        assert not s2.is_subset(s1)

    def test_is_subset_contained(self):
        s1 = argutils.IntervalSet(6, [(0, 4)])
        s2 = argutils.IntervalSet(6, [(2, 4)])
        assert s2.is_subset(s1)
        assert not s1.is_subset(s2)


class TestConvertArgweaver:
    def test_example(self):
        with open("examples/argweaver.arg") as f:
            ts = argutils.convert_argweaver(f)
        # print(ts.draw_text())
        assert ts.sequence_length == 1000
        assert ts.num_samples == 8
        # TODO more tests

class TestConvertKwARG:
    def test_example(self):
        with open("examples/kreitman.kwarg") as f:
            ts = argutils.convert_kwarg(f, 11, 43)
        assert ts.sequence_length == 43
        assert ts.num_samples == 11