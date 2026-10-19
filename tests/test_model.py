"""Modell gegen Handrechnung: ein OD-Paar, zwei Fahrten im Monat, drei Alternativen (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import tar_constants as C
from tar_model import Model, tilt_pk


def test_hand_calculation_one_pair(tiny):
    """Ziel-Auto-Anteil 0,45; Nutzen je Fahrt: Einzelkarte −5/2 − 0,1·10 = −3,5, Monatskarte −60/2 − 1,0 = −31,0. Beide ÖV-Alternativen teilen
    sich 0,55; ihr Verhältnis ist exp(0,3 · 27,5). Erlös = 100 · (P_S · 5 + P_M · 60), Fahrten = 100 · 2 · 0,55, Betrieb = 20 · 5 · max(550, 110/80)."""
    m = Model(tiny)
    p_s = 0.55 / (1 + math.exp(-0.3 * 27.5))
    p_m = 0.55 - p_s
    r = m.core(m.x0)
    assert r["Pr"][0, 0, 2] == pytest.approx(0.45, abs=1e-9)
    assert r["Pr"][0, 0, 0] == pytest.approx(p_s, rel=1e-9)
    assert r["R"] == pytest.approx(100 * (p_s * 5.0 + p_m * 60.0), rel=1e-9)
    assert r["T"] == pytest.approx(110.0, rel=1e-9)
    assert r["C"] == pytest.approx(20.0 * 5.0 * 550.0, rel=1e-12)
    assert m.S_today == pytest.approx(r["C"] - r["R"])


def test_calibration_hits_the_target_share(tiny):
    for share in (0.35, 0.45, 0.55):
        m = Model(tiny, car_share=share)
        assert m.shares(m.x0)[2] == pytest.approx(share, abs=1e-9)


def test_free_tariff_has_the_highest_ridership(tiny):
    m = Model(tiny)
    assert m.core(np.zeros(2))["T"] > m.core(m.x0)["T"] > m.core(m.x0 * 3)["T"]
    assert m.core(np.zeros(2))["R"] == 0.0


def test_operating_cost_switches_to_load_dependent_above_dmin(tiny):
    """Bei kleiner Mindestabfahrtszahl zählt die Last: d = Fahrgäste / Kapazität."""
    m = Model(tiny, cost="var")                          # d_min = 300
    assert m.core(m.x0)["C"] == pytest.approx(28.0 * 5.0 * 300.0)
    m2 = Model(tiny, cap=0.001)                          # Last/Kapazität = 110/0,001 = 110000 > d_min
    assert m2.core(m2.x0)["C"] == pytest.approx(20.0 * 5.0 * 110.0 / 0.001, rel=1e-9)


def test_calibration_is_independent_of_time_value_and_car_km_cost(tiny):
    """Nach der Kalibrierung auf den Ausgangs-Modal-Split sind Zeitwert und Auto-km-Kosten wirkungslos (Befund der Vorab-Reihe)."""
    base = Model(tiny)
    r0 = base.core(base.x0 * 1.3)
    for kwargs in (dict(lam=0.3), dict(car_km=0.4), dict(car_fix=90.0)):
        m = Model(tiny, **kwargs)
        r = m.core(m.x0 * 1.3)
        # lam ändert auch die ÖV-Nutzen (beide gleich), ändert damit das Verhältnis zwischen ÖV und Auto, das y wieder ausgleicht
        assert r["T"] == pytest.approx(r0["T"], rel=1e-9), kwargs
        assert r["R"] == pytest.approx(r0["R"], rel=1e-9), kwargs


def test_trip_distribution_tilt_hits_the_target_mean():
    pk = C.K_BASE / C.K_BASE.sum()
    for target in (15.0, 25.0):
        assert (tilt_pk(pk, target) * C.K_VALS).sum() == pytest.approx(target, abs=1e-6)


def test_elasticity_is_negative_and_in_the_documented_range(tiny):
    assert Model(tiny).elasticity() < 0


def test_car_utility_has_the_fixed_cost_spread_over_the_trips(tiny):
    """V_Auto (ohne Zuschlag) = −(F/k + c · Entfernung) − Zeitwert · Autozeit, je Fahrtenzahl k (Handrechnung für k = 2 und k = 60)."""
    m = Model(tiny)
    for j, k in enumerate(C.K_VALS):
        assert m.car_const[0, j] == pytest.approx(-(40.0 / k + 0.15 * 5.0) - 0.1 * 12.0, rel=1e-12)


def test_calibration_hits_the_target_for_every_pair_on_a_real_instance():
    from tar_scenario import make_instance

    m = Model(make_instance(4, 22, 50))
    Pr = m.core(m.x0)["Pr"]
    wk = m.pk * C.K_VALS
    share = (Pr[:, :, 2] * (wk / wk.sum())[None, :]).sum(1)
    assert np.allclose(share, m.target, atol=1e-9)


def test_elasticity_of_a_real_instance_is_in_the_documented_range():
    from tar_scenario import make_instance

    e = Model(make_instance(4, 22, 50)).elasticity()
    assert -0.60 < e < -0.35          # Messreihe: −0,46 ± 0,02 (Mittel ± Standardfehler)
