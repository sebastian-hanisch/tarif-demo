"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Bereinigung."""

import pytest

import tar_constants as C
import tar_evaluation as ev
import tar_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP.values())


def test_preset_values_are_valid_steps():
    for name, preset in C.PRESETS.items():
        for param, value in preset.items():
            state_key = P.PRESET_KEYS[param]
            spec = P.SETTING_SPECS[state_key]
            assert P.clean_value(spec, value) == value, (name, param)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert ev.settings_of(C.PRESETS[first]) == ev.Settings()


def test_setting_specs_cover_every_url_parameter():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and set(C.DEFAULTS) == set(P.PRESET_KEYS)


def test_presets_differ_from_each_other_in_at_least_one_setting():
    settings = [ev.settings_of(p) for p in C.PRESETS.values()]
    assert len(set(settings)) == len(settings)


@pytest.mark.parametrize("state_key,bad,expected", [
    ("mu_select", "nan", 1.0), ("mu_select", "inf", 1.0), ("mu_select", "0.9", 1.0), ("stops_select", "99", 25), ("stops_select", "abc", 25),
    ("seed_input", "-5", 0), ("seed_input", "99999999", C.SEED_MAX), ("subsidy_select", "7", "heute"), ("cost_select", "x", "mittel"),
])
def test_clean_value_falls_back_or_clamps(state_key, bad, expected):
    assert P.clean_value(P.SETTING_SPECS[state_key], bad) == expected


def test_settings_of_maps_every_key():
    s = ev.settings_of({"stops": 30, "hub": 100, "mu": 1.25, "car": 0.55, "trips": "15", "cost": "fix", "sub": "0.8", "seed": 7})
    assert s == ev.Settings(stops=30, hub=100, mu_factor=1.25, car_share=0.55, trips=15.0, cost="fix", subsidy=0.8, seed=7)
    assert ev.settings_of({"seed": 1}, seed=9).seed == 9
