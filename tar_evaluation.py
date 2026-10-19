"""Live-Szenario: Einstellungen, die drei Tarifziele, Nulltarif, Preis-Ebene, Pareto- und Subventions-Kurve, Annahmen-Tabelle und Verdikt in drei Zuständen."""

import time
from dataclasses import dataclass, field

import numpy as np

import tar_algorithm as A
import tar_constants as C
from tar_model import Model
from tar_scenario import Instance, make_instance

GOAL_LABEL = {"R": "Max-R (Erlös)", "P": "Max-P (Gewinn)", "D": "Max-D (Fahrgäste, Bilanz wie heute)"}
GOAL_SHORT = {"R": "Max-R", "P": "Max-P", "D": "Max-D"}
MU_LABEL = {0.75: "niedrig", 1.0: "mittel", 1.25: "hoch"}
TOGETHER_BAND = 0.02     # Max-P und Max-R gelten als zusammenfallend, wenn beide Preise um höchstens 2 % abweichen


@dataclass(frozen=True)
class Settings:
    stops: int = C.DEFAULT_STOPS
    hub: int = C.DEFAULT_HUB
    mu_factor: float = 1.0
    car_share: float = C.CAR_SHARE
    trips: float | None = None          # None: Fahrtenverteilung wie erzeugt (Mittel ≈ 19,8); sonst Zielmittel
    cost: str = C.DEFAULT_COST
    subsidy: float | None = None        # None: heutige Subvention; sonst Vielfaches der heutigen Betriebskosten C0
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    inst: Instance
    model: Model
    subsidy: float                      # Geldeinheiten
    today: dict
    zero: dict
    goals: dict                         # "R" / "P" / "D" -> Lösung samt "kpi"
    s_null: float                       # Subvention, ab der der Nulltarif die Bilanz hält
    elasticity: float
    time_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def relative_prices(self):
        x0 = self.model.x0
        return {k: tuple(float(v / x) for v, x in zip(g["x"], x0)) for k, g in self.goals.items()}

    @property
    def price_level(self):
        """Mittleres Preisniveau (x / x0) je Ziel."""
        return {k: float(np.mean(v)) for k, v in self.relative_prices.items()}

    @property
    def verdict(self):
        """Drei Zustände: 'nulltarif' (die Subvention reicht für den Nulltarif: Max-D ist der Nulltarif), 'zusammen' (Gewinn- und Erlösmaximum fallen
        zusammen: Preise um höchstens 2 % verschieden) und 'drei' (drei verschiedene Preise)."""
        if self.subsidy >= self.s_null:
            return "nulltarif"
        xr, xp = np.array(self.goals["R"]["x"]), np.array(self.goals["P"]["x"])
        if np.all(np.abs(xp / xr - 1.0) <= TOGETHER_BAND):
            return "zusammen"
        return "drei"


def model_of(s: Settings, inst: Instance | None = None) -> Model:
    inst = inst or make_instance(s.seed, s.stops, s.hub)
    return Model(inst, mu=C.MU * s.mu_factor, car_share=s.car_share, cost=s.cost, trips_mean=s.trips)


def subsidy_of(s: Settings, m: Model) -> float:
    return m.S_today if s.subsidy is None else s.subsidy * m.C0


def analyse(s: Settings, inst: Instance | None = None) -> Analysis:
    t0 = time.perf_counter()
    m = model_of(s, inst)
    sub = subsidy_of(s, m)
    goals = A.solve_goals(m, sub, seed=s.seed)
    for k, g in goals.items():
        if g is None:
            raise RuntimeError(f"Ziel {k} nicht lösbar")
    return Analysis(s, m.inst, m, sub, m.kpi(m.x0, sub), m.kpi(np.zeros(2), sub), goals, A.null_tariff_subsidy(m), m.elasticity(), time.perf_counter() - t0)


def price_plane(a: Analysis, n=31, span=3.2):
    """Erlös (in % des heutigen Erlöses) und Bilanz (Erlös − Kosten + Subvention, in % des heutigen Erlöses) auf einem Gitter der Preis-Ebene;
    Achsen in Geldeinheiten von 0 bis span · x0."""
    m = a.model
    xs = np.linspace(0.0, span * m.x0[0], n)
    ys = np.linspace(0.0, span * m.x0[1], n)
    R = np.zeros((n, n))
    B = np.zeros((n, n))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            r = m.core(np.array([x, y]))
            R[i, j] = 100.0 * r["R"] / m.R0
            B[i, j] = 100.0 * (r["R"] - r["C"] + a.subsidy) / m.R0
    return xs, ys, R, B


FRACTIONS = tuple(round(0.1 * i, 1) for i in range(11))


def pareto_curve(a: Analysis, fractions=FRACTIONS):
    """Max-D unter Erlös ≥ f · Erlösmaximum: Erlös und Fahrten in % von heute."""
    m = a.model
    pts = A.pareto(m, fractions, seed=a.settings.seed)
    return [dict(f=p["f"], R=100.0 * p["R"] / m.R0, T=100.0 * p["T"] / m.T0, x=p["x"]) for p in pts]


SUBSIDY_STEPS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4)


def subsidy_curve(a: Analysis, steps=SUBSIDY_STEPS):
    """Max-D mit Kostenbilanz für wachsende Subvention (in Vielfachen von C0): Fahrten in % von heute."""
    m = a.model
    pts = A.subsidy_curve(m, [s * m.C0 for s in steps], seed=a.settings.seed)
    out = []
    for s, p in zip(steps, pts):
        out.append(dict(s=s, feasible=p["feasible"], T=100.0 * p["T"] / m.T0 if p["feasible"] else None,
                        level=float(np.mean(np.array(p["x"]) / m.x0)) if p["feasible"] else None))
    return out


def assumption_table(s: Settings, factors=C.MU_FACTORS):
    """Dasselbe Szenario bei niedriger, mittlerer und hoher Preisempfindlichkeit: Preisniveau je Ziel und Verlust des Preises der mittleren
    Annahme („Regret“: Erlöspunkte in % des heutigen Erlöses, die der nominale Max-R-Preis in der anderen Welt verliert)."""
    inst = make_instance(s.seed, s.stops, s.hub)
    nominal = None
    rows = []
    for f in factors:
        sf = Settings(**{**s.__dict__, "mu_factor": f})
        a = analyse(sf, inst)
        row = dict(mu_factor=f, level=a.price_level, elasticity=a.elasticity, R_opt=a.goals["R"]["kpi"]["R"] / a.model.R0 * 100.0)
        if f == 1.0:
            nominal = a.goals["R"]["x"]
        rows.append((a, row))
    out = []
    for a, row in rows:
        if nominal is not None:
            row["regret"] = row["R_opt"] - 100.0 * a.model.core(np.array(nominal))["R"] / a.model.R0
        out.append(row)
    return out


def settings_of(values, seed=None):
    """Einstellungen aus einem Preset-Wörterbuch (Schlüssel wie `tar_constants.DEFAULTS`: stops, hub, mu, car, trips, cost, sub, seed);
    fehlende Schlüssel bekommen den Standard, `seed` überschreibt den Zufalls-Seed."""
    v = {**C.DEFAULTS, **values}
    return Settings(
        stops=int(v["stops"]), hub=int(v["hub"]), mu_factor=float(v["mu"]), car_share=float(v["car"]),
        trips=None if v["trips"] == "auto" else float(v["trips"]), cost=v["cost"],
        subsidy=None if v["sub"] == "heute" else float(v["sub"]), seed=int(v["seed"] if seed is None else seed),
    )


def de(x, digits=1):
    """Zahl mit Dezimalkomma; nur die Zahl wird umformatiert, nie ein ganzer Satz (Tausenderpunkt-Falle)."""
    out = f"{x:.{digits}f}"
    if float(out) == 0.0:
        out = out.lstrip("-")          # kein „-0“ und „-0,0“ bei kleinen negativen Werten
    return out.replace(".", ",")


def _pct(a, key, which):
    m = a.model
    k = {"today": a.today, "zero": a.zero}.get(key) or a.goals[key]["kpi"]
    return 100.0 * k[which] / (m.R0 if which in ("R", "profit") else m.T0)


def trip_loss(a, key="R"):
    """Verlust an Fahrten gegenüber Max-D in Prozentpunkten von heute, wenn die Preise des Ziels `key` gelten."""
    return _pct(a, "D", "T") - _pct(a, key, "T")


def verdict(a: Analysis):
    """Meldung in drei Zuständen (Art, Text): 'nulltarif' (Max-D ist der Nulltarif), 'zusammen' (Max-P und Max-R fallen zusammen),
    'drei' (drei verschiedene Preise). Alle Zahlen stammen aus der Analyse des gezeigten Szenarios."""
    xr, xp, xd = (a.goals[k]["x"] for k in "RPD")
    state = a.verdict
    if state == "nulltarif":
        return "success", (
            f"Die Subvention von **{de(a.subsidy / a.model.C0, 2)} × Betriebskosten** reicht für den **Nulltarif** (nötig: {de(a.s_null / a.model.C0, 2)} ×): "
            f"Max-D verschenkt die Fahrten und fährt **{de(_pct(a, 'zero', 'T'), 0)} %** der heutigen Fahrten bei Bilanz ±0. Der Erlös-optimale Tarif (Max-R) kommt auf "
            f"{de(_pct(a, 'R', 'R'), 0)} % Erlös, aber nur {de(_pct(a, 'R', 'T'), 0)} % der Fahrten: Zwischen beiden liegt der Zielkonflikt."
        )
    if state == "zusammen":
        return "info", (
            f"Bei dieser Kostenstruktur fallen **Gewinn- und Erlösmaximum zusammen** (die Preise unterscheiden sich um höchstens {int(TOGETHER_BAND * 100)} %): "
            f"Einzelkarte {de(xr[0], 2)}, Monatskarte {de(xr[1], 0)}. Erlös {de(_pct(a, 'R', 'R'), 0)} %, Fahrten {de(_pct(a, 'R', 'T'), 0)} % von heute. "
            f"Max-D hält die Bilanz und fährt {de(_pct(a, 'D', 'T'), 0)} % der Fahrten: Der Konflikt bleibt zwischen Erlös und Fahrgästen."
        )
    return "info", (
        f"**Drei Ziele, drei Preise.** Max-R verlangt Einzelkarte {de(xr[0], 2)} und Monatskarte {de(xr[1], 0)} ({de(_pct(a, 'R', 'R'), 0)} % Erlös, "
        f"{de(_pct(a, 'R', 'T'), 0)} % Fahrten), Max-P Einzelkarte {de(xp[0], 2)} und Monatskarte {de(xp[1], 0)} ({de(_pct(a, 'P', 'R'), 0)} % Erlös, "
        f"{de(_pct(a, 'P', 'T'), 0)} % Fahrten), Max-D Einzelkarte {de(xd[0], 2)} und Monatskarte {de(xd[1], 0)} ({de(_pct(a, 'D', 'T'), 0)} % Fahrten bei Bilanz wie heute). "
        f"Mit den Preisen von Max-R fehlen gegenüber Max-D **{de(trip_loss(a, 'R'), 0)} Prozentpunkte** der Fahrten."
    )


def method_table(a: Analysis, goal="R"):
    """Verfahren-Vergleich für ein Ziel (R oder P) auf diesem Szenario: Zielwert in % des heutigen Erlöses und Abstand zum besten Fund (in Punkten)."""
    m = a.model
    sub = a.subsidy
    rows = []
    local = A.multistart(m, goal, subsidy=sub, seed=a.settings.seed)
    rows.append(("lokal (SLSQP, 5 Startpunkte)", local["val"], local["x"]))
    for npts in (21, 41):
        g = A.grid_search(m, goal, npts, subsidy=sub)
        rows.append((f"Gitter {npts} × {npts}", g["val"], g["x"]))
    u = A.best_uniform(m, goal, sub)
    rows.append((f"Praxisregel: alle Preise um {de(100 * u['X'], 0)} % ändern", u["val"], m.x0 * (1 + u["X"])))
    rows.append(("heutiger Tarif", value_at(m, goal, m.x0, sub), m.x0))
    best = max(v for _, v, _ in rows)
    return [dict(name=n, value=100.0 * v / m.R0, gap=100.0 * (best - v) / m.R0, x=tuple(float(t) for t in x)) for n, v, x in rows]


def value_at(m, goal, x, sub):
    r = m.core(x)
    return {"R": r["R"], "P": r["R"] - r["C"] + sub}[goal]


def cannibalization(a: Analysis, cuts=(0.1, 0.2, 0.3)):
    """Monatskarte −10 / −20 / −30 % bei sonst heutigem Tarif: Erlös, Fahrten und Verschiebung des Modal Split (Prozentpunkte der Fahrten)."""
    m = a.model
    base = m.shares(m.x0)
    r0 = m.core(m.x0)
    rows = []
    for cut in cuts:
        x = np.array([m.x0[0], m.x0[1] * (1 - cut)])
        r = m.core(x)
        sh = m.shares(x)
        rows.append(dict(cut=cut, R=100.0 * r["R"] / m.R0, T=100.0 * r["T"] / m.T0, d_single=100.0 * (sh[0] - base[0]), d_month=100.0 * (sh[1] - base[1]),
                         d_car=100.0 * (sh[2] - base[2]), R_loss_points=100.0 * (r0["R"] - r["R"]) / m.R0))
    return rows
