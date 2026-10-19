"""Preset-Geschichten: Jede stimmt auf dem gezeigten Szenario, und jede Schwelle kippt einzeln an künstlichen Werten (DEMO-PLAYBOOK §12)."""

import pytest

import tar_constants as C
import tar_evaluation as ev
import tar_stories as ST


@pytest.fixture(scope="module")
def analyses():
    out = {}
    for name, preset in C.PRESETS.items():
        s = ST.settings_of(preset)
        out[name] = (ev.analyse(s), ev.analyse(ST.reference_settings(s)) if name in ST.NEEDS_REFERENCE else None)
    return out


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_story_holds_on_its_scenario(name, analyses):
    a, ref = analyses[name]
    assert ST.check(name, a, ref), name


def test_every_preset_has_a_criterion_and_a_seed_outside_the_measurement_series():
    assert set(ST.CRITERIA) == set(C.PRESETS)
    assert all(p["seed"] > max(C.SWEEP_SEEDS) for p in C.PRESETS.values())


class _Fake:
    """Analyse mit festen Zahlen: R0 = T0 = 100."""

    def __init__(self, verdict="drei", level=1.5, r=107.0, t=79.0, zero_t=143.0):
        self.verdict, self.price_level = verdict, {"R": level}
        self.model = type("M", (), {"R0": 100.0, "T0": 100.0})()
        self.goals = {"R": dict(kpi=dict(R=r, T=t, profit=0.0))}
        self.zero = dict(R=0.0, T=zero_t, profit=0.0)
        self.today = dict(R=100.0, T=100.0, profit=0.0)


def test_standard_thresholds_are_inclusive_and_need_three_prices():
    assert ST.check_standard(_Fake(r=ST.STANDARD_R_MIN, t=ST.STANDARD_T_MAX))
    assert not ST.check_standard(_Fake(r=ST.STANDARD_R_MIN - 0.01))
    assert not ST.check_standard(_Fake(t=ST.STANDARD_T_MAX + 0.01))
    assert not ST.check_standard(_Fake(verdict="zusammen"))


def test_zero_tariff_story():
    assert ST.check_nulltarif(_Fake(verdict="nulltarif", zero_t=ST.ZERO_T_MIN))
    assert not ST.check_nulltarif(_Fake(verdict="nulltarif", zero_t=ST.ZERO_T_MIN - 0.01))
    assert not ST.check_nulltarif(_Fake(verdict="drei"))


def test_price_level_stories_flip_at_their_thresholds():
    ref = _Fake(level=1.0)
    assert ST.check_hohe_empfindlichkeit(_Fake(level=0.949), ref) and not ST.check_hohe_empfindlichkeit(_Fake(level=0.951), ref)
    assert ST.check_viel_auto(_Fake(level=0.949), ref) and not ST.check_viel_auto(_Fake(level=0.951), ref)
    assert ST.check_niedrige_empfindlichkeit(_Fake(level=ST.LEVEL_UP_MIN), ref) and not ST.check_niedrige_empfindlichkeit(_Fake(level=ST.LEVEL_UP_MIN - 0.01), ref)
    assert not ST.check_niedrige_empfindlichkeit(_Fake(level=1.3, r=ST.R_MIN_LOW_SENS - 0.1), ref)


def test_fixed_cost_story_needs_the_together_state():
    assert ST.check_hohe_fixkosten(_Fake(verdict="zusammen")) and not ST.check_hohe_fixkosten(_Fake(verdict="drei"))
