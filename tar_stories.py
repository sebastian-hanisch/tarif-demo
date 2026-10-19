"""Abnahmekriterien der Presets: Jedes Preset erzählt eine Geschichte, und die muss auf dem gezeigten Szenario stimmen.

`check(name, a, a_ref)` bekommt die Analyse des Presets (`a`) und, wo ein Vergleich nötig ist, dieselbe Analyse mit Standard-Annahmen (`a_ref`: dasselbe Netz,
Preisempfindlichkeit mittel, Auto-Anteil 45 %, Kostenstruktur mittel, Subvention heute). Die Schwellen stehen hier einzeln, damit `tests/test_stories.py` jede mit
künstlichen Werten an ihrer Grenze prüfen kann."""

import tar_constants as C
import tar_evaluation as ev

# Schwellen
STANDARD_R_MIN = 105.0       # Erlös von Max-R mindestens so viele % von heute
STANDARD_T_MAX = 85.0        # Fahrten von Max-R höchstens so viele % von heute
ZERO_T_MIN = 135.0           # Nulltarif: Fahrten mindestens so viele % von heute
LEVEL_SHIFT = 0.05           # Preisniveau von Max-R weicht um mehr als diesen Anteil vom Referenzniveau ab
LEVEL_UP_MIN = 1.15          # niedrige Preisempfindlichkeit: Preisniveau mindestens so viel mal das Referenzniveau
R_MIN_LOW_SENS = 100.0       # ... bei weiterhin höherem Erlös als heute


def _pct(a, key, which):
    return ev._pct(a, key, which)


def _level(a, goal="R"):
    return a.price_level[goal]


def check_standard(a, a_ref=None):
    return a.verdict == "drei" and _pct(a, "R", "R") >= STANDARD_R_MIN and _pct(a, "R", "T") <= STANDARD_T_MAX


def check_nulltarif(a, a_ref=None):
    return a.verdict == "nulltarif" and _pct(a, "zero", "T") >= ZERO_T_MIN


def check_hohe_empfindlichkeit(a, a_ref):
    """Empfindlichere Fahrgäste: das Preisniveau des Erlösmaximums liegt unter dem der Referenz (um mehr als LEVEL_SHIFT)."""
    return _level(a) < _level(a_ref) * (1.0 - LEVEL_SHIFT)


def check_niedrige_empfindlichkeit(a, a_ref):
    """Unempfindlichere Fahrgäste: das Preisniveau steigt stark (mindestens LEVEL_UP_MIN mal die Referenz), der Erlös bleibt über dem heutigen."""
    return _level(a) >= _level(a_ref) * LEVEL_UP_MIN and _pct(a, "R", "R") >= R_MIN_LOW_SENS


def check_hohe_fixkosten(a, a_ref=None):
    """Gewinn- und Erlösmaximum fallen zusammen (Zustand „zusammen“ der Meldung)."""
    return a.verdict == "zusammen"


def check_viel_auto(a, a_ref):
    """Mehr Autofahrer: das Preisniveau des Erlösmaximums liegt unter dem der Referenz (um mehr als LEVEL_SHIFT)."""
    return _level(a) < _level(a_ref) * (1.0 - LEVEL_SHIFT)


CRITERIA = {
    "Standardfall": check_standard,
    "Nulltarif": check_nulltarif,
    "Hohe Preisempfindlichkeit": check_hohe_empfindlichkeit,
    "Niedrige Preisempfindlichkeit": check_niedrige_empfindlichkeit,
    "Hohe Fixkosten": check_hohe_fixkosten,
    "Viel Autoverkehr": check_viel_auto,
}
NEEDS_REFERENCE = {"Hohe Preisempfindlichkeit", "Niedrige Preisempfindlichkeit", "Viel Autoverkehr"}


def settings_of(preset, seed=None):
    """Einstellungen eines Presets (Schlüssel wie in tar_constants.PRESETS); `seed` überschreibt den Zufalls-Seed des Presets."""
    return ev.settings_of(preset, seed=seed)


def reference_settings(settings: ev.Settings) -> ev.Settings:
    """Dasselbe Netz mit den Standard-Annahmen, für die Vergleiche der Presets zur Preisempfindlichkeit und zum Auto-Anteil."""
    return ev.Settings(stops=settings.stops, hub=settings.hub, seed=settings.seed)


def check(name, a, a_ref=None):
    return bool(CRITERIA[name](a, a_ref))
