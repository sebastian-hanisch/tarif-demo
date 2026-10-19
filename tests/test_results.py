"""Vorgerechnete Messreihe (data/tar_results.json): Aufbau, innere Konsistenz und Übereinstimmung der Live-Rechnung mit den Sweeps."""

import math

import pytest

import tar_constants as C
import tar_evaluation as ev
import tar_results as Rs
from tar_scenario import make_instance

SWEEP_N = 40


def test_structure_and_network_count():
    d = Rs.load()
    assert set(d) >= {"meta", "modell", "kreuz", "preise", "verlust", "pareto", "subvention", "verfahren", "annahmen", "kannibal", "luecke", "rechenzeit"}
    assert Rs.n_networks() == SWEEP_N == len(C.SWEEP_SEEDS)
    for cost in Rs.VARIANTS:
        assert set(d["kreuz"][cost]) == set(Rs.KREUZ_ROWS)


def test_the_cross_table_reference_rows_are_exact():
    for cost in Rs.VARIANTS:
        base, _r, _p, d_row, zero = Rs.cross_table(cost)
        assert base["R"]["mean"] == pytest.approx(100.0) and base["T"]["mean"] == pytest.approx(100.0) and base["P"]["mean"] == pytest.approx(0.0, abs=1e-9)
        assert zero["R"]["mean"] == 0.0 and zero["zul"]["k"] == 0                         # Nulltarif ist bei heutiger Subvention nie zulässig
        assert d_row["zul"]["k"] == d_row["zul"]["n"] == SWEEP_N                          # Max-D hält die Bilanz in jedem Netz


def test_ordering_of_the_goals_in_the_means():
    for cost in Rs.VARIANTS:
        _b, r, p, d_row, zero = Rs.cross_table(cost)
        assert r["R"]["mean"] >= max(p["R"]["mean"], d_row["R"]["mean"]) - 1e-9           # Max-R hat den höchsten Erlös
        assert p["P"]["mean"] >= max(r["P"]["mean"], d_row["P"]["mean"]) - 1e-9           # Max-P den höchsten Gewinn (bei hohen Grenzkosten kann Max-P unter dem Erlös von Max-D liegen)
        assert d_row["T"]["mean"] > r["T"]["mean"] > p["T"]["mean"]
        assert zero["T"]["mean"] > d_row["T"]["mean"]


def test_cost_structure_moves_max_p_towards_max_r():
    ratios = [Rs.prices(c)["ratio_P_R"]["mean"] for c in ("fix", "mittel", "var")]
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[0] < 1.15 < ratios[1]                         # hohe Fixkosten: Preisniveau-Verhältnis höchstens 1,15


def test_pareto_is_monotone_and_the_last_step_is_the_largest():
    for cost in Rs.VARIANTS:
        par = Rs.pareto(cost)
        r = [s["R"]["mean"] for s in par["stufen"]]
        t = [s["T"]["mean"] for s in par["stufen"]]
        assert all(b >= a - 1e-9 for a, b in zip(r, r[1:])) and all(b <= a + 1e-9 for a, b in zip(t, t[1:]))
        drops = [s["drop"]["mean"] for s in par["schritte"]]
        assert drops[-1] == max(drops) and par["monoton"] and par["knick_letzter_groesster"]


def test_subsidy_ordering_and_the_free_tariff_threshold():
    for cost in Rs.VARIANTS:
        sb = Rs.subsidy(cost)
        assert sb["Smin"]["mean"] < sb["Stoday"]["mean"] < sb["Snull"]["mean"]
        t = [r["T"]["mean"] for r in sb["raster"] if r["T"]["n"]]
        assert all(b >= a - 1e-6 for a, b in zip(t, t[1:]))


def test_time_value_and_car_km_cost_are_without_effect_after_calibration():
    entries = Rs.assumptions()["eintraege"]
    for name in ("lam", "car_km"):
        recal = [e for e in entries if e["name"] == name and e["mode"] == "recal"]
        assert recal and all(e["wirkungslos"] and e["max_rel_aenderung"] < 1e-6 for e in recal)
    assert all(not e["wirkungslos"] for e in entries if e["name"] in ("mu", "car_share", "k_mean", "car_fix", "cdep", "dmin") and e["mode"] == "recal")


def test_live_computation_reproduces_the_series_means():
    """Dieselben 40 Netze (n = 20 + (Seed − 1) mod 11, Konzentration mittel), Kostenstruktur mittel, Subvention heute: die Mittel stimmen mit der Datei überein.
    Der Live-Löser nimmt 5, die Messreihe 20 Starts; Max-P hat einen flachen Grat (bis 1e-4 des heutigen Erlöses), daher das Band von 0,02 Punkten."""
    sums = {"R": [0.0, 0.0], "P": [0.0, 0.0], "D": [0.0, 0.0]}
    for seed in C.SWEEP_SEEDS:
        a = ev.analyse(ev.Settings(stops=20 + (seed - 1) % 11, hub=50, seed=seed), make_instance(seed, 20 + (seed - 1) % 11, 50))
        for g in "RPD":
            sums[g][0] += ev._pct(a, g, "R") / SWEEP_N
            sums[g][1] += ev._pct(a, g, "T") / SWEEP_N
    _b, r, p, d_row, _z = Rs.cross_table("mittel")
    for g, row in (("R", r), ("P", p), ("D", d_row)):
        assert sums[g][0] == pytest.approx(row["R"]["mean"], abs=0.02), g
        assert sums[g][1] == pytest.approx(row["T"]["mean"], abs=0.02), g


def test_stat_text_uses_the_decimal_comma_and_standard_error():
    assert Rs.stat_text({"mean": 108.819, "se": 0.232253}) == "108,8 ± 0,2"
    assert Rs.stat_text({"mean": -0.46, "se": 0.0036}, 2) == "-0,46 ± 0,00"
    assert Rs.share_text({"k": 7, "n": 40}) == "7 von 40"


def test_inf_strings_are_decoded():
    assert Rs._decode({"a": ["inf", "-inf", 1.0]}) == {"a": [math.inf, -math.inf, 1.0]}
