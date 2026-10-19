"""Szenario: ganzzahliger Netz- und Nachfragegenerator, Liniennetz, kürzeste Wege, Zonen und Fahrtenverteilung einer Instanz.

Alles, was Entscheidungen beeinflusst, kommt aus Ganzzahlen (SplitMix64, Raster 5..95, ganzzahlige Nachfragegewichte, Ganzzahl-Abstände mit
Index-Tie-Break) oder aus eigenem Code (Spannbaum nach Prim, Dijkstra mit Heap): keine Bibliothek, deren Gleichstandsentscheidungen oder
Zufallsströme sich zwischen Versionen ändern.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np

import tar_constants as C
from tar_rng import SplitMix64


@dataclass
class Instance:
    seed: int
    n: int
    hub_pct: int
    grid: np.ndarray        # (n, 2) ganzzahlige Koordinaten
    demand: np.ndarray      # (n, n) ganzzahlig, symmetrisch
    hubs: list
    I: np.ndarray           # OD-Paare i < j
    J: np.ndarray
    persons: np.ndarray     # (P,) Personen je Paar
    l_km: np.ndarray        # (P,) Luftlinie in km
    t_pt: np.ndarray        # (P,) Fahrzeit im ÖV in Minuten
    t_car: np.ndarray       # (P,) Fahrzeit mit dem Auto in Minuten
    eu: np.ndarray          # (E,) Kanten des Liniennetzes
    ev: np.ndarray
    elen: np.ndarray        # (E,) Kantenlänge in km
    inc: np.ndarray         # (P, E) Wegekanten-Inzidenz
    inner: np.ndarray       # (n,) Haltepunkt in der inneren Zone
    z: np.ndarray           # (P,) Zonenzahl 1 oder 2
    pk: np.ndarray          # (7,) Verteilung der Fahrtenzahl
    share_unit: np.ndarray  # (P,) tanh des z-Werts der Entfernung

    @property
    def P(self):
        return len(self.I)

    @property
    def E(self):
        return len(self.eu)


def generate_network(n_stops, seed, hub_pct=50, n_hubs=2):
    """Ganzzahlige Haltepunkte (Raster 5..95) und symmetrische ganzzahlige Nachfrage. hub_pct = Nachfragekonzentration in Prozent:
    Zeilen und Spalten der Zentren bekommen den Faktor 1 + 6 · hub_pct / 100 (50 entspricht dem Faktor 4 der Liniennetz-Demo)."""
    rng = SplitMix64(seed)
    coords = np.array([[C.GRID_LO + rng.below(C.GRID_SPAN), C.GRID_LO + rng.below(C.GRID_SPAN)] for _ in range(n_stops)], dtype=np.int64)
    base = [[0] * n_stops for _ in range(n_stops)]
    for i in range(n_stops):
        for j in range(i + 1, n_stops):
            base[i][j] = base[j][i] = 1 + rng.below(10)
    pool = list(range(n_stops))
    hubs = [pool.pop(rng.below(len(pool))) for _ in range(max(1, min(n_hubs, n_stops)))]
    boost = 100 + 6 * hub_pct
    b = [boost if i in hubs else 100 for i in range(n_stops)]
    demand = np.zeros((n_stops, n_stops), dtype=np.int64)
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                demand[i, j] = (base[i][j] * b[i] * b[j] + 5000) // 10000
    return coords, demand, hubs


def squared_distances(grid):
    g = [(int(x), int(y)) for x, y in grid]
    return [[(a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 for b in g] for a in g]


def spanning_tree_edges(sq):
    """Minimaler Spannbaum nach Prim auf den ganzzahligen Quadratabständen; Gleichstand: kleinerer Schlüssel (Abstand, Knoten, Vorgänger)."""
    n = len(sq)
    in_tree = [False] * n
    in_tree[0] = True
    key = [(sq[0][j], j, 0) for j in range(n)]
    edges = []
    for _ in range(n - 1):
        j = min((key[v] for v in range(n) if not in_tree[v]))
        d, v, u = j
        in_tree[v] = True
        edges.append((min(u, v), max(u, v)))
        for w in range(n):
            if not in_tree[w] and (sq[v][w], w, v) < key[w]:
                key[w] = (sq[v][w], w, v)
    return edges


def line_network_edges(sq, extra=C.NEAREST_EXTRA):
    """Liniennetz: Spannbaum plus die `extra` nächsten Nachbarn je Haltepunkt (Gleichstand: kleinerer Index)."""
    n = len(sq)
    edges = set(spanning_tree_edges(sq))
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: (sq[i][j], j))
        for j in order[:extra]:
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def shortest_paths(n, edges, weights):
    """Dijkstra von jedem Haltepunkt aus: Entfernungsmatrix und Vorgängermatrix (Gleichstand: Knoten mit kleinerem Index zuerst)."""
    adj = [[] for _ in range(n)]
    for (u, v), w in zip(edges, weights):
        adj[u].append((v, w))
        adj[v].append((u, w))
    for a in adj:
        a.sort()
    dist = [[math.inf] * n for _ in range(n)]
    pred = [[-1] * n for _ in range(n)]
    for s in range(n):
        d, p = dist[s], pred[s]
        d[s] = 0.0
        heap = [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for v, w in adj[u]:
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    p[v] = u
                    heapq.heappush(heap, (nd, v))
    return dist, pred


def inner_zone(grid, demand, frac=C.INNER_FRAC):
    """Die inneren Haltepunkte: die n // 2 nächsten am nachfragegewichteten Schwerpunkt (ganzzahlig gerechnet, Gleichstand: kleinerer Index)."""
    n = len(grid)
    wt = [int(demand[i].sum()) for i in range(n)]
    total = sum(wt)
    cx = sum(wt[i] * int(grid[i][0]) for i in range(n))
    cy = sum(wt[i] * int(grid[i][1]) for i in range(n))
    key = [((total * int(grid[i][0]) - cx) ** 2 + (total * int(grid[i][1]) - cy) ** 2, i) for i in range(n)]
    order = sorted(range(n), key=lambda i: key[i])
    inner = np.zeros(n, dtype=bool)
    inner[order[: int(frac * n)]] = True
    return inner


def trip_distribution(seed):
    """Gewichte der sieben Fahrtenzahlen: K_BASE · exp(0,3 · N(0,1)) mit N aus der Summe von 12 Gleichverteilten minus 6, normiert."""
    rng = SplitMix64(int(seed) + (1 << 40))
    noise = [sum(rng.u32() / 4294967296.0 for _ in range(12)) - 6.0 for _ in range(len(C.K_VALS))]
    pk = C.K_BASE * np.exp(C.K_NOISE * np.array(noise))
    return pk / pk.sum()


def make_instance(seed, n_stops=C.DEFAULT_STOPS, hub_pct=C.DEFAULT_HUB):
    """Deterministisch aus (seed, n_stops, hub_pct): gleiche Eingabe, gleiche Instanz auf jeder Plattform (bis auf Rundung der letzten Stelle)."""
    seed, n = int(seed), int(n_stops)
    grid, demand, hubs = generate_network(n, seed, int(hub_pct))
    sq = squared_distances(grid)
    I, J = np.triu_indices(n, 1)
    km = [[math.sqrt(s) * C.KM_PER_UNIT for s in row] for row in sq]
    l_km = np.array([km[i][j] for i, j in zip(I, J)])
    edges = line_network_edges(sq)
    eu = np.array([e[0] for e in edges])
    ev = np.array([e[1] for e in edges])
    elen = np.array([km[u][v] for u, v in edges])
    weights = [C.DWELL + C.PT_MIN_PER_KM * float(x) for x in elen]
    dist, pred = shortest_paths(n, edges, weights)
    eidx = {}
    for e, (a, b) in enumerate(edges):
        eidx[(a, b)] = eidx[(b, a)] = e
    inc = np.zeros((len(I), len(edges)))
    for q, (i, j) in enumerate(zip(I, J)):
        cur = int(j)
        while cur != int(i):
            prv = pred[int(i)][cur]
            inc[q, eidx[(prv, cur)]] = 1.0
            cur = prv
    inner = inner_zone(grid, demand)
    z_unit = (l_km - l_km.mean()) / l_km.std()
    return Instance(
        seed=seed, n=n, hub_pct=int(hub_pct), grid=grid, demand=demand, hubs=hubs, I=I, J=J,
        persons=demand[I, J] * C.PERSONS_SCALE, l_km=l_km,
        t_pt=np.array([dist[int(i)][int(j)] for i, j in zip(I, J)]),
        t_car=C.CAR_BASE_MIN + C.CAR_MIN_PER_KM * C.CAR_DETOUR * l_km,
        eu=eu, ev=ev, elen=elen, inc=inc, inner=inner,
        z=np.where(inner[I] & inner[J], 1.0, 2.0), pk=trip_distribution(seed), share_unit=np.tanh(z_unit),
    )
