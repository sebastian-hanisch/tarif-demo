"""Orakel-Tests (ORACLE-PLAYBOOK §2): unabhängige Referenzen für die Rechnung des Modells und der Löser.

1. Logit-Erwartungen gegen Monte-Carlo (Gumbel-argmax mit vektorisiertem SplitMix64, nicht aus dem Modell).
2. Analytischer Gradient gegen zentrale finite Differenzen.
3. SLSQP-Mehrfachstart gegen ein feines Gitter (nie schlechter als das Gitter) und gegen 20 Starts (dasselbe Optimum).
4. Struktur: Verhältnis zweier Alternativen hängt nicht von der dritten ab (IIA); Nulltarif ist das Optimum von Max-D ohne Erlösbedingung."""

import numpy as np
import pytest

import tar_algorithm as A
import tar_constants as C
import tar_evaluation as ev
from tar_model import Model
from tar_scenario import make_instance

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix_uniform(seed, n):
    """n Gleichverteilte in (0, 1) aus SplitMix64, vektorisiert (uint64-Arithmetik läuft modulo 2^64 über)."""
    with np.errstate(over="ignore"):
        state = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15) * np.arange(1, n + 1, dtype=np.uint64)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def monte_carlo(m, x, n_draws, seed):
    """Je Zelle (OD-Paar, Fahrtenzahl) n_draws Wahlen aus V + Gumbel; Erlös, Fahrten und Auto-Anteil mit Standardfehler."""
    pr = (m.Cflat @ x).reshape(m.P, m.K, 2)
    V = np.empty((m.P, m.K, 3))
    V[:, :, :2] = -pr / m.kk[..., None] + m.base_pt[:, :, None]
    V[:, :, 2] = m.base_car
    V = m.mu * V
    w = m.w
    wk = w * m.kk
    price = np.concatenate([pr, np.zeros((m.P, m.K, 1))], axis=2)
    R = T = car = 0.0
    var_R = var_T = var_car = 0.0
    for p in range(m.P):
        for k in range(m.K):
            if w[p, k] <= 0:
                continue
            u = splitmix_uniform(seed + 1000003 * (p * m.K + k), 3 * n_draws).reshape(3, n_draws)
            choice = np.argmax(V[p, k][:, None] - np.log(-np.log(u)), axis=0)
            rev = price[p, k][choice]
            ride = (choice < 2).astype(float)
            R += w[p, k] * rev.mean()
            var_R += w[p, k] ** 2 * rev.var() / n_draws
            T += wk[p, k] * ride.mean()
            var_T += wk[p, k] ** 2 * ride.var() / n_draws
            car += wk[p, k] * (1 - ride.mean())
            var_car += wk[p, k] ** 2 * ride.var() / n_draws
    return (R, np.sqrt(var_R)), (T, np.sqrt(var_T)), (car / wk.sum(), np.sqrt(var_car) / wk.sum())


@pytest.mark.parametrize("seed,factor", [(1, 1.0), (2, 1.7), (3, 0.4)])
def test_logit_expectations_match_monte_carlo(seed, factor):
    m = Model(make_instance(seed, 6, 50))
    x = m.x0 * factor
    r = m.core(x)
    (R, sR), (T, sT), (car, scar) = monte_carlo(m, x, 20000, 17 * seed)
    assert abs(r["R"] - R) < 3.5 * sR and abs(r["T"] - T) < 3.5 * sT
    assert abs(m.shares(x)[2] - car) < 3.5 * scar


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gradient_matches_central_differences(seed):
    m = Model(make_instance(seed, 20, 50))
    for x in (m.x0, m.x0 * np.array([1.4, 1.8]), m.x0 * np.array([0.6, 2.2])):
        g = m.core(x, grad=True)
        for name, grad in (("R", g["dR"]), ("T", g["dT"]), ("C", g["dC"])):
            for j in range(2):
                h = 1e-5 * x[j]
                xp, xm = x.copy(), x.copy()
                xp[j] += h
                xm[j] -= h
                fd = (m.core(xp)[name] - m.core(xm)[name]) / (2 * h)
                assert grad[j] == pytest.approx(fd, rel=1e-5, abs=1e-8 * abs(g[name])), (name, j)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_local_solution_is_never_worse_than_a_fine_grid(seed):
    m = Model(make_instance(seed, 20 + seed, 50))
    sub = m.S_today
    for kind in ("R", "P", "D"):
        sol = A.multistart(m, kind, subsidy=sub, seed=seed)
        grid = A.grid_search(m, kind, 41, subsidy=sub)
        assert sol is not None and sol["feas"]
        assert sol["val"] >= grid["val"] - 1e-9 * abs(grid["val"]), kind
        # Das dichte Gitter bleibt nah am lokalen Optimum (R und P; bei Max-D liegt das Optimum auf dem Rand der Bilanzbedingung, das Gitter
        # trifft diesen Rand nicht): der Löser fand wirklich das Maximum und nicht irgendeinen Punkt
        if kind != "D":
            assert sol["val"] <= grid["val"] + 2e-3 * m.R0, kind


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_more_starts_find_the_same_optimum(seed):
    m = Model(make_instance(seed, 20 + seed, 50))
    for kind in ("R", "P", "D"):
        a = A.multistart(m, kind, 5, subsidy=m.S_today, seed=seed)
        b = A.multistart(m, kind, 20, subsidy=m.S_today, seed=seed + 100)
        assert a["val"] == pytest.approx(b["val"], rel=1e-6)


def test_ratio_of_two_alternatives_does_not_depend_on_the_third():
    """IIA des Logit: Ändert man nur den Preis der Monatskarte, bleibt das Verhältnis Einzelkarte : Auto je Zelle gleich."""
    m = Model(make_instance(4, 20, 50))
    a = m.core(m.x0)["Pr"]
    b = m.core(m.x0 * np.array([1.0, 0.7]))["Pr"]
    assert np.allclose(a[:, :, 0] / a[:, :, 2], b[:, :, 0] / b[:, :, 2], rtol=1e-9)


def test_max_d_without_revenue_condition_is_the_free_tariff():
    """Ab der Subvention S_null hält der Nulltarif die Bilanz: Max-D wählt dann den Nulltarif (Fahrten gleich denen bei x = 0)."""
    m = Model(make_instance(3, 24, 50))
    s_null = A.null_tariff_subsidy(m)
    sol = A.multistart(m, "D", subsidy=1.001 * s_null, seed=3)
    assert sol["val"] == pytest.approx(m.core(np.zeros(2))["T"], rel=1e-6)
    below = A.multistart(m, "D", subsidy=0.9 * s_null, seed=3)
    assert below["val"] < m.core(np.zeros(2))["T"]


def test_max_d_respects_the_balance_constraint_and_max_p_ignores_the_subsidy():
    m = Model(make_instance(2, 22, 50))
    g1 = A.solve_goals(m, m.S_today, seed=2)
    g2 = A.solve_goals(m, 0.8 * m.C0, seed=2)
    assert g1["D"]["kpi"]["profit"] >= -1e-6 * m.R0
    assert np.allclose(g1["P"]["x"], g2["P"]["x"], rtol=1e-5)          # die Subvention verschiebt den Gewinn, nicht den optimalen Preis
    assert np.allclose(g1["R"]["x"], g2["R"]["x"], rtol=1e-5)


def test_max_d_feasibility_flag_at_the_minimum_subsidy():
    """Unter der kleinsten zulässigen Subvention S_min gibt es keinen Tarif mit Bilanz ≥ 0; knapp darüber gibt es einen."""
    m = Model(make_instance(3, 24, 50))
    s_min = A.minimum_subsidy(m)
    assert A.multistart(m, "D", subsidy=s_min - 0.05 * m.R0, seed=3) is None
    assert A.multistart(m, "D", subsidy=s_min + 0.05 * m.R0, seed=3) is not None


def test_uniform_rule_for_max_d_is_nan_where_the_balance_is_negative():
    m = Model(make_instance(3, 24, 50))
    vals = A.uniform_rule(m, "D", [0.0, 1.0], m.S_today + 0.01 * m.R0)
    assert np.isnan(vals[0]) and vals[1] == pytest.approx(m.T0)


def test_best_uniform_factor_reproduces_its_value():
    """Der gemeldete Prozentsatz X gehört zum gemeldeten Zielwert: Preise x0 · (1 + X) liefern genau diesen Erlös."""
    m = Model(make_instance(3, 24, 50))
    u = A.best_uniform(m, "R")
    assert m.core(m.x0 * (1 + u["X"]))["R"] == pytest.approx(u["val"], rel=1e-12)
    assert u["X"] > 0
