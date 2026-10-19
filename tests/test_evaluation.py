"""Live-Auswertung: Zusammenhänge, die für jedes Szenario gelten müssen (Reihenfolgen, Monotonie, Randwerte)."""

import numpy as np
import pytest

import tar_constants as C
import tar_evaluation as ev


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings(seed=3, stops=22))


def test_goal_ordering_on_the_standard_scenario(a):
    """Max-R hat den höchsten Erlös, Max-P den höchsten Gewinn, Max-D die meisten Fahrten; die Preise steigen von Max-D über Max-R zu Max-P."""
    k = {g: a.goals[g]["kpi"] for g in "RPD"}
    assert k["R"]["R"] >= k["P"]["R"] >= k["D"]["R"] - 1e-6 * a.model.R0
    assert k["P"]["profit"] >= k["R"]["profit"] >= k["D"]["profit"] - 1e-6 * a.model.R0
    assert k["D"]["T"] >= k["R"]["T"] >= k["P"]["T"]
    lv = a.price_level
    assert lv["D"] < lv["R"] < lv["P"]


def test_today_and_zero_tariff_reference_points(a):
    m = a.model
    assert a.today["R"] == pytest.approx(m.R0) and a.today["T"] == pytest.approx(m.T0)
    assert a.zero["R"] == 0.0 and a.zero["T"] > a.today["T"]
    assert a.s_null == pytest.approx(a.zero["C"])


def test_max_d_balance_is_non_negative_and_default_subsidy_is_cost_minus_revenue(a):
    assert a.subsidy == pytest.approx(a.model.C0 - a.model.R0)
    assert a.goals["D"]["kpi"]["profit"] >= -1e-6 * a.model.R0


def test_pareto_curve_is_monotone(a):
    pts = ev.pareto_curve(a)
    assert [p["f"] for p in pts] == list(ev.FRACTIONS)
    r = [p["R"] for p in pts]
    t = [p["T"] for p in pts]
    assert all(x <= y + 1e-6 for x, y in zip(r, r[1:])) and all(x >= y - 1e-6 for x, y in zip(t, t[1:]))
    assert r[-1] == pytest.approx(100.0 * a.goals["R"]["kpi"]["R"] / a.model.R0, rel=1e-6)   # f = 1 ist das Erlösmaximum


def test_subsidy_curve_is_monotone_and_reaches_the_free_tariff(a):
    pts = ev.subsidy_curve(a)
    assert all(p["feasible"] for p in pts)
    t = [p["T"] for p in pts]
    assert all(x <= y + 1e-6 for x, y in zip(t, t[1:]))
    assert t[-1] == pytest.approx(100.0 * a.zero["T"] / a.model.T0, rel=1e-4)       # 1,4 × C0 reicht für den Nulltarif


def test_assumption_table_orders_price_level_by_sensitivity_and_has_zero_regret_at_nominal(a):
    rows = ev.assumption_table(a.settings)
    lv = [r["level"]["R"] for r in rows]
    assert lv[0] > lv[1] > lv[2]                                                   # unempfindlich: höheres Erlösoptimum
    assert rows[1]["regret"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["regret"] > 0 and rows[2]["regret"] > 0
    assert [r["elasticity"] for r in rows][0] > rows[1]["elasticity"] > rows[2]["elasticity"]


def test_method_table_local_is_best_and_practice_rule_is_close(a):
    rows = ev.method_table(a, "R")
    assert rows[0]["gap"] == pytest.approx(0.0, abs=1e-9)
    assert all(r["gap"] >= -1e-9 for r in rows)
    practice = next(r for r in rows if r["name"].startswith("Praxisregel"))
    assert 0 <= practice["gap"] < 3.0


def test_cannibalization_follows_the_cut(a):
    rows = ev.cannibalization(a)
    assert [r["cut"] for r in rows] == [0.1, 0.2, 0.3]
    assert all(r["d_month"] > 0 and r["d_single"] < 0 and r["d_car"] < 0 for r in rows)
    assert rows[0]["R"] > rows[1]["R"] > rows[2]["R"]
    assert all(r["d_single"] + r["d_month"] + r["d_car"] == pytest.approx(0.0, abs=1e-9) for r in rows)


def test_price_plane_has_the_revenue_maximum_near_the_solution(a):
    xs, ys, R, B = ev.price_plane(a, n=41)
    i, j = np.unravel_index(np.argmax(R), R.shape)
    xr = a.goals["R"]["x"]
    assert abs(xs[j] - xr[0]) < 0.35 * xr[0] and abs(ys[i] - xr[1]) < 0.35 * xr[1]
    assert R.max() <= 100.0 * a.goals["R"]["kpi"]["R"] / a.model.R0 + 1e-6


def test_trip_mean_setting_changes_the_scenario():
    base = ev.analyse(ev.Settings(seed=3, stops=22))
    low = ev.analyse(ev.Settings(seed=3, stops=22, trips=15.0))
    assert low.model.T0 != base.model.T0 and low.model.pk @ C.K_VALS == pytest.approx(15.0, abs=1e-6)


def test_given_subsidy_is_a_multiple_of_the_operating_cost():
    a2 = ev.analyse(ev.Settings(seed=3, stops=22, subsidy=0.8))
    assert a2.subsidy == pytest.approx(0.8 * a2.model.C0)


def test_pareto_points_keep_the_required_share_of_the_revenue_maximum(a):
    r_max = a.goals["R"]["kpi"]["R"] / a.model.R0 * 100.0
    for p in ev.pareto_curve(a):
        assert p["R"] >= p["f"] * r_max - 1e-6
