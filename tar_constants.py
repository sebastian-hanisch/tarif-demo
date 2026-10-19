"""Konstanten: feste Modellannahmen, Reglerstufen, Kostenvarianten, Farben und Presets.

Alle Modellannahmen sind frei gesetzt (kein Messwert): Sie stehen hier an einer Stelle und im README in der Annahmen-Tabelle."""

import numpy as np

# --- Netz (Raster 5..95, 1 Einheit = 0,2 km) -------------------------------------------------------------------------
GRID_LO, GRID_SPAN = 5, 91
KM_PER_UNIT = 0.2
PT_MIN_PER_KM = 3.0          # Fahrzeit im ÖV je km Kante
DWELL = 1.0                  # Halt je Kante in Minuten
CAR_BASE_MIN = 3.0           # Auto: Zugang/Parken in Minuten
CAR_MIN_PER_KM = 1.8         # Auto: Minuten je km (Wegelänge = 1,2 × Luftlinie)
CAR_DETOUR = 1.2
NEAREST_EXTRA = 2            # Liniennetz = minimaler Spannbaum + so viele nächste Nachbarn je Haltepunkt
INNER_FRAC = 0.5             # Anteil der Haltepunkte in der inneren Zone
PERSONS_SCALE = 10.0         # Personen je Nachfrageeinheit

# --- Fahrtenverteilung: Fahrten je Person und Monat ---------------------------------------------------------------------
K_VALS = np.array([2, 6, 12, 20, 30, 44, 60], dtype=float)
K_BASE = np.array([0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10])
K_NOISE = 0.3                # Streuung des Logarithmus der Gewichte je Instanz

# --- Verhalten und Kalibrierung -------------------------------------------------------------------------------------
MU = 0.30                    # Logit-Skala je Geldeinheit und Fahrt (Fahrpreiselastizität ≈ −0,46)
LAM = 0.10                   # Zeitwert in Geldeinheiten je Minute (nach der Kalibrierung wirkungslos)
CAR_FIX = 40.0               # Auto: Fixkosten je Monat (durch k geteilt)
CAR_KM = 0.15                # Auto: Kosten je km (nach der Kalibrierung wirkungslos)
CAR_SHARE = 0.45             # Ausgangs-Auto-Anteil (Wege), Mittel über OD-Paare
CAR_SHARE_SLOPE = 0.10       # Auto-Anteil steigt mit der Entfernung (tanh des z-Werts)
CAR_SHARE_CLIP = (0.03, 0.92)

# --- Tarif (Zonentarif) und Betrieb ----------------------------------------------------------------------------------
X0 = (2.5, 60.0)             # heutiger Tarif: Einzelkarte je Fahrt und Zone, Monatskarte je Zone
CAP = 80.0                   # Fahrgäste je Abfahrt
COST_VARIANTS = {
    "fix": dict(cdep=14.0, dmin=900.0),
    "mittel": dict(cdep=20.0, dmin=550.0),
    "var": dict(cdep=28.0, dmin=300.0),
}
COST_LABELS = {"fix": "Fixkosten hoch", "mittel": "Fixkosten mittel", "var": "Fixkosten niedrig"}
DEFAULT_COST = "mittel"

# --- Löser ---------------------------------------------------------------------------------------------------------------
UB = 2.5                     # obere Grenze in u = x / (2 · x0), also x ≤ 5 · x0
N_STARTS = 5
START_LO, START_HI = 0.05, 2.0

# --- Regler ------------------------------------------------------------------------------------------------------------
STOPS_CHOICES = (20, 25, 30)
HUB_CHOICES = (0, 50, 100)                 # Prozent: Nachfragekonzentration auf zwei Zentren
HUB_LABELS = {0: "gleichverteilt", 50: "mittel", 100: "stark"}
MU_FACTORS = (0.75, 1.0, 1.25)
MU_LABELS = {0.75: "niedrig", 1.0: "mittel", 1.25: "hoch"}
CAR_SHARE_CHOICES = (0.35, 0.45, 0.55)
TRIPS_CHOICES = ("auto", "15", "25")       # "auto": Verteilung wie erzeugt (Mittel ≈ 19,8 Fahrten), sonst Zielmittel
TRIPS_LABELS = {"auto": "wie erzeugt (Mittel ≈ 20)", "15": "15 Fahrten", "25": "25 Fahrten"}
SUBSIDY_CHOICES = ("heute", "0.5", "0.8", "1.0", "1.2", "1.4")   # "heute" oder Vielfache der heutigen Betriebskosten C0
SEED_MAX, DEFAULT_SEED = 9999, 503
DEFAULT_STOPS, DEFAULT_HUB = 25, 50

SWEEP_SEEDS = tuple(range(1, 41))          # feste Netze für alle Messreihen: nie ein einzelner Seed als Befund

COLOR_R = "#2F6B65"
COLOR_P = "#c8872b"
COLOR_D = "#4c78a8"
COLOR_TODAY = "#7f7f7f"
COLOR_ZERO = "#b04a4a"

DEFAULTS = {"stops": DEFAULT_STOPS, "hub": DEFAULT_HUB, "mu": 1.0, "car": CAR_SHARE, "trips": "auto", "cost": DEFAULT_COST, "sub": "heute", "seed": DEFAULT_SEED}

PRESETS = {
    "Standardfall": {"seed": DEFAULT_SEED},
    "Nulltarif": {"sub": "1.4", "seed": DEFAULT_SEED},
    "Hohe Preisempfindlichkeit": {"mu": 1.25, "seed": DEFAULT_SEED},
    "Niedrige Preisempfindlichkeit": {"mu": 0.75, "seed": DEFAULT_SEED},
    "Hohe Fixkosten": {"cost": "fix", "seed": DEFAULT_SEED},
    "Viel Autoverkehr": {"car": 0.55, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall": "Drei Ziele, drei Preise: Der Erlös steigt um mehr als 5 %, die Fahrten sinken um mehr als 15 %.",
    "Nulltarif": "Bei 1,4-facher Subvention hält der Nulltarif die Bilanz: Max-D ist dann der Nulltarif und fährt mehr als 35 % mehr Fahrten als heute.",
    "Hohe Preisempfindlichkeit": "Empfindlichere Fahrgäste: Das Preisniveau des Erlösmaximums sinkt um mehr als 5 % gegenüber mittlerer Empfindlichkeit.",
    "Niedrige Preisempfindlichkeit": "Unempfindlichere Fahrgäste: Das Preisniveau des Erlösmaximums steigt um mehr als 15 % gegenüber mittlerer Empfindlichkeit.",
    "Hohe Fixkosten": "Der Betrieb kostet kaum mehr bei mehr Fahrgästen: Gewinn- und Erlösmaximum fallen zusammen.",
    "Viel Autoverkehr": "Mehr Fahrgäste fahren heute Auto: Das Preisniveau des Erlösmaximums liegt mehr als 5 % unter dem bei 45 % Auto-Anteil.",
}
