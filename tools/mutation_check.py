"""Mutationsprüfung: jede Änderung an einer Rechenstelle muss mindestens einen Test fehlschlagen lassen.

Aufruf (aus dem Repo-Ordner, mit dem Test-venv): python tools/mutation_check.py [--jobs 4] [--only Teil]
Jeder Mutant läuft in einer Kopie des Repos (Temp-Ordner), die Tests des Kerns (ohne App-Tests) laufen mit -x. Ergebnis: pro Mutant
„gefunden“ (ein Test scheitert) oder „ÜBERLEBT“; Rückgabewert 1, wenn ein Mutant überlebt."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ["tests/test_model.py", "tests/test_scenario.py", "tests/test_oracle_tarif.py", "tests/test_evaluation.py", "tests/test_verdicts.py"]

# (Datei, alter Text, neuer Text, Beschreibung[, n-te Stelle]); jeder alte Text kommt in der Datei genau einmal vor, sonst mit Stellenangabe.
# Bewusst NICHT in der Liste (gleichwertige Änderungen, im README begründet):
#  - Fahrzeit des Autos (t_car) und Auto-km-Kosten: nach der Kalibrierung auf den Ausgangs-Modal-Split durch den Zuschlag y je Paar aufgefangen.
#  - Maßstab der Auto-Anteil-Steigung (car_share / 0,45): ändert nur die Streuung der Zielanteile um den Mittelwert.
#  - Rundung `f >= 1.0` im Pareto-Aufruf: bei f = 1 liefert der Löser mit Erlösbedingung dasselbe Maximum.
MUTANTS = [
    ("tar_model.py", "Cm[:, :, 0, 0] = self.kk * z", "Cm[:, :, 0, 0] = z", "Einzelkarte nicht mit der Fahrtenzahl multiplizieren"),
    ("tar_model.py", "Cm[:, :, 1, 1] = z", "Cm[:, :, 1, 1] = 1.0", "Monatskarte ohne Zonenzahl"),
    ("tar_model.py", "V[:, :, :2] = -pr / self.kk[..., None] + self.base_pt[:, :, None]", "V[:, :, :2] = -pr + self.base_pt[:, :, None]", "Kalibrierung: Preis nicht durch k teilen", 0),
    ("tar_model.py", "V[:, :, :2] = -pr / self.kk[..., None] + self.base_pt[:, :, None]", "V[:, :, :2] = -pr + self.base_pt[:, :, None]", "Auswertung: Preis nicht durch k teilen", 1),
    ("tar_model.py", "self.car_const = -(car_fix / self.kk + car_km * l)", "self.car_const = -(car_fix * self.kk + car_km * l)", "Auto-Fixkosten mal statt durch k"),
    ("tar_model.py", "self.S_today = self.C0 - self.R0", "self.S_today = self.R0 - self.C0", "heutige Subvention mit falschem Vorzeichen"),
    ("tar_model.py", "d = np.maximum(self.dmin, load / self.cap)", "d = np.minimum(self.dmin, load / self.cap)", "Abfahrten: Minimum statt Maximum"),
    ("tar_model.py", "Cost = self.cdep * (self.inst.elen * d).sum()", "Cost = self.cdep * d.sum()", "Betriebskosten ohne Kantenlänge"),
    ("tar_model.py", "pair = (wk * (1 - Pc)).sum(1)", "pair = (w * (1 - Pc)).sum(1)", "Fahrten ohne Fahrtenzahl"),
    ("tar_model.py", "R = (w * (Pr[:, :, :2] * pr).sum(2)).sum()", "R = (w * (Pr[:, :, :2] * pr).sum(2)).sum() * 0.99", "Erlös um 1 % verfälschen"),
    ("tar_model.py", "dR = (w[..., None] * (first + second)).sum((0, 1))", "dR = (w[..., None] * (first - second)).sum((0, 1))", "Gradient des Erlöses: Vorzeichen des zweiten Terms"),
    ("tar_model.py", "dPc = -self.mu * Pc[..., None] * gbar", "dPc = self.mu * Pc[..., None] * gbar", "Gradient der Auto-Wahrscheinlichkeit: Vorzeichen"),
    ("tar_model.py", "(self.inst.elen * act)[:, None] * dload).sum(0) / self.cap", "(self.inst.elen * act)[:, None] * dload).sum(0)", "Gradient der Kosten ohne Kapazität"),
    ("tar_model.py", "act = (load / self.cap > self.dmin).astype(float)", "act = (load / self.cap < self.dmin).astype(float)", "Gradient der Kosten: aktive Kanten umkehren"),
    ("tar_model.py", "up = share < self.target", "up = share > self.target", "Kalibrierung: Bisektion umkehren"),
    ("tar_model.py", "wk = self.pk * K_VALS\n        wk = wk / wk.sum()", "wk = self.pk\n        wk = wk / wk.sum()", "Kalibrierung mit Personen statt Fahrten gewichten"),
    ("tar_model.py", "if (q * K_VALS).sum() < target_mean:", "if (q * K_VALS).sum() > target_mean:", "Kippung der Fahrtenverteilung umkehren"),
    ("tar_model.py", "profit=float(r[\"R\"] - r[\"C\"] + subsidy)", "profit=float(r[\"R\"] - r[\"C\"] - subsidy)", "Bilanz: Subvention abziehen"),
    ("tar_model.py", "return ((a - b) / self.T0) / (2 * rel)", "return ((a - b) / self.T0) / rel", "Elastizität: falscher Nenner"),
    ("tar_algorithm.py", "return -r[\"T\"] / T0, -r[\"dT\"] * xs / T0", "return -r[\"T\"] / T0, r[\"dT\"] * xs / T0", "Gradient der Fahrten: Vorzeichen"),
    ("tar_algorithm.py", "return -r[\"R\"] / R0, -r[\"dR\"] * xs / R0", "return -r[\"R\"] / R0, r[\"dR\"] * xs / R0", "Gradient des Erlöses im Löser: Vorzeichen"),
    ("tar_algorithm.py", "feas = con >= -1e-6 * R0", "feas = con >= -0.5 * R0", "Zulässigkeit von Max-D aufweichen"),
    ("tar_algorithm.py", "if con < 0:\n                continue", "if con < -0.1 * m.R0:\n                continue", "Gitter: unzulässige Punkte zulassen"),
    ("tar_algorithm.py", "vals = uniform_rule(m, kind, 1 + xs, subsidy)", "vals = uniform_rule(m, kind, 1 - xs, subsidy)", "Praxisregel: Faktor umkehren"),
    ("tar_algorithm.py", "return r[\"T\"] if r[\"R\"] - r[\"C\"] + subsidy >= 0 else np.nan", "return r[\"T\"] if r[\"R\"] - r[\"C\"] + subsidy <= 0 else np.nan", "Praxisregel Max-D: Zulässigkeit umkehren"),
    ("tar_algorithm.py", "return float(m.core(np.zeros(2))[\"C\"])", "return float(m.core(np.zeros(2))[\"R\"])", "Nulltarif-Subvention aus dem Erlös statt den Kosten"),
    ("tar_algorithm.py", "rev_rhs=f * r_max[\"val\"], seed=seed)", "rev_rhs=0.9 * f * r_max[\"val\"], seed=seed)", "Pareto: Erlösschranke lockern"),
    ("tar_scenario.py", "boost = 100 + 6 * hub_pct", "boost = 100 + 5 * hub_pct", "Zentren-Zuschlag"),
    ("tar_scenario.py", "km = [[math.sqrt(s) * C.KM_PER_UNIT for s in row] for row in sq]", "km = [[math.sqrt(s) * C.KM_PER_UNIT * 1.1 for s in row] for row in sq]", "Entfernungsumrechnung"),
    ("tar_scenario.py", "inner[order[: int(frac * n)]] = True", "inner[order[: int(frac * n) + 1]] = True", "innere Zone um einen Haltepunkt vergrößern"),
    ("tar_scenario.py", "z=np.where(inner[I] & inner[J], 1.0, 2.0)", "z=np.where(inner[I] | inner[J], 1.0, 2.0)", "Zonenzahl: oder statt und"),
    ("tar_scenario.py", "key = [(sq[0][j], j, 0) for j in range(n)]", "key = [(sq[0][j], -j, 0) for j in range(n)]", "Spannbaum: Gleichstand umkehren"),
    ("tar_scenario.py", "order = sorted((j for j in range(n) if j != i), key=lambda j: (sq[i][j], j))", "order = sorted((j for j in range(n) if j != i), key=lambda j: (sq[i][j], -j))", "nächste Nachbarn: Gleichstand umkehren"),
    ("tar_scenario.py", "weights = [C.DWELL + C.PT_MIN_PER_KM * float(x) for x in elen]", "weights = [C.PT_MIN_PER_KM * float(x) for x in elen]", "Halt je Kante streichen"),
    ("tar_constants.py", "NEAREST_EXTRA = 2 ", "NEAREST_EXTRA = 1 ", "nur den nächsten Nachbarn zusätzlich verbinden"),
    ("tar_evaluation.py", "TOGETHER_BAND = 0.02", "TOGETHER_BAND = 0.05", "Schwelle „zusammenfallend“"),
    ("tar_evaluation.py", "if self.subsidy >= self.s_null:", "if self.subsidy > self.s_null:", "Nulltarif-Grenze nicht inklusive"),
    ("tar_evaluation.py", "return _pct(a, \"D\", \"T\") - _pct(a, key, \"T\")", "return _pct(a, key, \"T\") - _pct(a, \"D\", \"T\")", "Fahrtenverlust mit falschem Vorzeichen"),
    ("tar_evaluation.py", "return m.S_today if s.subsidy is None else s.subsidy * m.C0", "return m.S_today if s.subsidy is None else s.subsidy", "Subvention nicht mit den Betriebskosten malnehmen"),
    ("tar_evaluation.py", "return Model(inst, mu=C.MU * s.mu_factor,", "return Model(inst, mu=C.MU,", "Preisempfindlichkeit nicht anwenden"),
    ("tar_evaluation.py", "x = np.array([m.x0[0], m.x0[1] * (1 - cut)])", "x = np.array([m.x0[0] * (1 - cut), m.x0[1]])", "Kannibalisierung: falsche Karte senken"),
]


def run_mutant(idx, mutant, workdir):
    fname, old, new, desc = mutant[:4]
    nth = mutant[4] if len(mutant) > 4 else None
    work = Path(workdir) / f"m{idx}"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns("__pycache__", ".git", "data", "venv", ".pytest_cache", "tools"))
    p = work / fname
    text = p.read_text(encoding="utf-8")
    if nth is None and text.count(old) != 1:
        return idx, desc, f"FEHLER: Stelle kommt {text.count(old)}-mal vor"
    if nth is None:
        text = text.replace(old, new)
    else:
        parts = text.split(old)
        if nth >= len(parts) - 1:
            return idx, desc, "FEHLER: Stelle zu selten"
        text = old.join(parts[:nth + 1]) + new + old.join(parts[nth + 1:])
    p.write_text(text, encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS], cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    shutil.rmtree(work, ignore_errors=True)
    return idx, desc, "gefunden" if r.returncode != 0 else "ÜBERLEBT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    todo = [(i, m) for i, m in enumerate(MUTANTS) if a.only in m[0] or a.only in m[3]]
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(lambda im: run_mutant(im[0], im[1], tmp), todo))
    bad = 0
    for idx, desc, res in sorted(results):
        print(f"{idx:2d} {res:10s} {MUTANTS[idx][0]}: {desc}")
        bad += res != "gefunden"
    print(f"\n{len(results) - bad}/{len(results)} Mutanten gefunden")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
