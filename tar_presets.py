"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import tar_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # Auswahl-Regler: nur diese Werte sind gültig (die Reihe wurde nur auf diesen Stufen gemessen)


SETTING_SPECS = {
    "stops_select": SettingSpec("stops", int, C.DEFAULTS["stops"], options=C.STOPS_CHOICES),
    "hub_select": SettingSpec("hub", int, C.DEFAULTS["hub"], options=C.HUB_CHOICES),
    "mu_select": SettingSpec("mu", float, C.DEFAULTS["mu"], options=C.MU_FACTORS),
    "car_select": SettingSpec("car", float, C.DEFAULTS["car"], options=C.CAR_SHARE_CHOICES),
    "trips_select": SettingSpec("trips", str, C.DEFAULTS["trips"], options=C.TRIPS_CHOICES),
    "cost_select": SettingSpec("cost", str, C.DEFAULTS["cost"], options=tuple(C.COST_VARIANTS)),
    "subsidy_select": SettingSpec("sub", str, C.DEFAULTS["sub"], options=C.SUBSIDY_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULTS["seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {spec.url_param: state_key for state_key, spec in SETTING_SPECS.items()}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean_value(spec, value):
    """Einen eingelesenen Wert auf den gültigen Bereich bringen: Auswahl-Regler nur auf Stufen (sonst Standard), Zahlen-Regler begrenzt;
    NaN, Infinity und Unlesbares fallen auf den Standard."""
    try:
        value = spec.caster(value)
    except (ValueError, TypeError, OverflowError):
        return spec.default
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return spec.default
    if spec.options is not None:
        return value if value in spec.options else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die gemessenen Stufen und Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            st.session_state[state_key] = clean_value(spec, qp[spec.url_param])
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt JEDEN Regler: was es nicht nennt, bekommt den Standardwert (sonst blieben frühere Änderungen stehen)."""
    for param, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(param, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert)
    mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das)."""
    preset_key_of = {state_key: param for param, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
