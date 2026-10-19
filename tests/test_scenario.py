"""Generator und Liniennetz: eingefrorene Instanzen, Spannbaum gegen Kruskal, kürzeste Wege gegen Floyd-Warshall (unabhängige Gegenproben)."""

import itertools
import math

import numpy as np
import pytest

import tar_scenario as S

FROZEN = [  # (Zufalls-Seed, Haltepunkte, Konzentration) -> Summe der Koordinaten, Summe der Nachfrage, Kanten, Kantenkennzahl, innere, Einzonenpaare, Zentren
    ((1, 20, 50), (1915, 3336, 28, 15478, 10, 45, [11, 19])),
    ((7, 25, 0), (2467, 3236, 36, 28090, 12, 66, [16, 15])),
    ((12, 30, 100), (3318, 9804, 40, 41069, 15, 105, [10, 1])),
]


@pytest.mark.parametrize("args,expected", FROZEN)
def test_frozen_instances(args, expected):
    """Ganzzahlen und Entscheidungen (Spannbaum, Nachbarn, Zonen) sind über Plattformen und Bibliotheksversionen gleich."""
    i = S.make_instance(*args)
    got = (int(i.grid.sum()), int(i.demand.sum()), i.E, int((i.eu * 100 + i.ev).sum()), int(i.inner.sum()), int((i.z == 1).sum()), i.hubs)
    assert got == expected


def test_same_input_same_instance():
    a, b = S.make_instance(3, 22, 50), S.make_instance(3, 22, 50)
    assert np.array_equal(a.grid, b.grid) and np.array_equal(a.inc, b.inc) and np.array_equal(a.t_pt, b.t_pt)


def test_demand_is_symmetric_with_zero_diagonal_and_positive_off_diagonal():
    i = S.make_instance(5, 25, 50)
    assert np.array_equal(i.demand, i.demand.T) and not np.any(np.diag(i.demand))
    assert np.all(i.demand[np.triu_indices(25, 1)] > 0)


def test_hub_concentration_raises_hub_demand():
    low, high = S.make_instance(5, 25, 0), S.make_instance(5, 25, 100)
    h = low.hubs[0]
    assert high.demand[h].sum() / high.demand.sum() > low.demand[h].sum() / low.demand.sum()


def _kruskal_weight(sq):
    n = len(sq)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    total = 0.0
    for d, u, v in sorted((math.sqrt(sq[u][v]), u, v) for u, v in itertools.combinations(range(n), 2)):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            total += d
    return total


@pytest.mark.parametrize("seed", range(1, 9))
def test_spanning_tree_has_the_kruskal_weight_and_connects(seed):
    i = S.make_instance(seed, 20 + seed, 50)
    sq = S.squared_distances(i.grid)
    tree = S.spanning_tree_edges(sq)
    assert len(tree) == i.n - 1
    assert sum(math.sqrt(sq[u][v]) for u, v in tree) == pytest.approx(_kruskal_weight(sq), rel=1e-12)
    adj = {k: set() for k in range(i.n)}
    for u, v in tree:
        adj[u].add(v)
        adj[v].add(u)
    seen, stack = {0}, [0]
    while stack:
        for w in adj[stack.pop()]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    assert len(seen) == i.n


def test_line_network_contains_the_tree_and_two_nearest_neighbours():
    i = S.make_instance(4, 25, 50)
    sq = S.squared_distances(i.grid)
    edges = set(S.line_network_edges(sq))
    assert set(S.spanning_tree_edges(sq)) <= edges
    for a in range(i.n):
        nearest = sorted((j for j in range(i.n) if j != a), key=lambda j: (sq[a][j], j))[:2]
        assert all((min(a, j), max(a, j)) in edges for j in nearest)


@pytest.mark.parametrize("seed", range(1, 6))
def test_shortest_paths_match_floyd_warshall_and_path_incidence_sums_to_the_length(seed):
    i = S.make_instance(seed, 22, 50)
    n = i.n
    w = np.full((n, n), np.inf)
    np.fill_diagonal(w, 0.0)
    for u, v, length in zip(i.eu, i.ev, i.elen):
        w[u, v] = w[v, u] = 1.0 + 3.0 * length
    for k in range(n):
        w = np.minimum(w, w[:, k:k + 1] + w[k:k + 1, :])
    assert np.allclose(i.t_pt, w[i.I, i.J], rtol=1e-12)
    # Die Weg-Inzidenz bildet genau diese Zeit nach: Summe (1 + 3 · Länge) über die Kanten des Weges
    assert np.allclose(i.inc @ (1.0 + 3.0 * i.elen), i.t_pt, rtol=1e-12)


def test_inner_zone_is_the_half_closest_to_the_demand_weighted_centre():
    i = S.make_instance(6, 24, 50)
    assert int(i.inner.sum()) == 12
    wt = i.demand.sum(axis=1)
    cx, cy = (wt * i.grid[:, 0]).sum() / wt.sum(), (wt * i.grid[:, 1]).sum() / wt.sum()
    d = np.hypot(i.grid[:, 0] - cx, i.grid[:, 1] - cy)
    assert d[i.inner].max() <= d[~i.inner].min() + 1e-9


def test_trip_distribution_is_a_probability_vector_and_stable():
    pk = S.trip_distribution(10)
    assert pk.sum() == pytest.approx(1.0) and np.all(pk > 0)
    assert np.array_equal(pk, S.trip_distribution(10)) and not np.array_equal(pk, S.trip_distribution(11))


def test_pair_distances_are_the_grid_distances_in_kilometres():
    i = S.make_instance(2, 20, 50)
    q = 17
    a, b = int(i.I[q]), int(i.J[q])
    assert i.l_km[q] == pytest.approx(0.2 * math.hypot(*(i.grid[a] - i.grid[b])), rel=1e-12)
    assert i.persons[q] == 10.0 * i.demand[a, b]


def test_ties_between_equally_near_neighbours_go_to_the_smaller_index():
    """Punkt 0 hat die Punkte 1, 2, 3 im selben Abstand; der Spannbaum verbindet ihn nur mit 1. Die zwei nächsten Nachbarn sind 1 und 2."""
    sq = [[0, 100, 100, 100], [100, 0, 1, 1], [100, 1, 0, 1], [100, 1, 1, 0]]
    assert S.spanning_tree_edges(sq) == [(0, 1), (1, 2), (1, 3)]
    assert S.line_network_edges(sq) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
