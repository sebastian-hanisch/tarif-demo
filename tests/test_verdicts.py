"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import tar_evaluation as ev
from tar_model import Model


class _Fake:
    """Analyse mit festen Zahlen: R0 = T0 = 100, C0 = 100; je Ziel eine Zeile Preise und Kennzahlen."""

    def __init__(self, subsidy=30.0, s_null=120.0, xr=(3.0, 90.0), xp=(3.5, 120.0)):
        self.model = type("M", (), {"R0": 100.0, "T0": 100.0, "C0": 100.0, "x0": np.array([2.5, 60.0])})()
        self.subsidy, self.s_null = subsidy, s_null
        self.today = dict(R=100.0, T=100.0, profit=0.0)
        self.zero = dict(R=0.0, T=140.0, profit=-120.0)
        self.goals = {
            "R": dict(x=xr, kpi=dict(R=109.0, T=77.0, profit=15.0)),
            "P": dict(x=xp, kpi=dict(R=105.0, T=63.0, profit=16.0)),
            "D": dict(x=(2.5, 58.0), kpi=dict(R=100.0, T=100.0, profit=0.0)),
        }

    verdict = ev.Analysis.verdict


def test_three_states():
    assert _Fake().verdict == "drei"
    assert _Fake(xp=(3.0, 90.0 * 1.0199)).verdict == "zusammen"          # knapp unter 2 % Abstand gilt als zusammen
    assert _Fake(xp=(3.0, 90.0 * 1.0201)).verdict == "drei"
    assert _Fake(subsidy=120.0).verdict == "nulltarif"                  # S = S_null reicht
    assert _Fake(subsidy=119.99).verdict == "drei"


def test_verdict_three_prices_names_prices_and_loss():
    kind, text = ev.verdict(_Fake(xp=(3.5, 120.0)))
    assert kind == "info" and "Drei Ziele, drei Preise" in text
    assert "Einzelkarte 3,00 und Monatskarte 90" in text and "Einzelkarte 3,50 und Monatskarte 120" in text
    assert "109 % Erlös" in text and "77 % Fahrten" in text and "63 % Fahrten" in text
    assert "**23 Prozentpunkte**" in text                                 # 100 − 77


def test_verdict_together_and_zero_tariff():
    kind, text = ev.verdict(_Fake(xp=(3.0, 90.0)))
    assert kind == "info" and "fallen **Gewinn- und Erlösmaximum zusammen**" in text and "Einzelkarte 3,00, Monatskarte 90" in text
    kind, text = ev.verdict(_Fake(subsidy=130.0))
    assert kind == "success" and "1,30 × Betriebskosten" in text and "nötig: 1,20" in text and "**140 %**" in text


def test_trip_loss_is_measured_against_max_d():
    assert ev.trip_loss(_Fake(), "R") == 23.0 and ev.trip_loss(_Fake(), "P") == 37.0
