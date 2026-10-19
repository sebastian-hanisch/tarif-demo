"""Vorgerechnete Messreihe (data/tar_results.json, erzeugt von tools/sweep_tarif.py, tools/analyse.py, tools/dump_results.py): Laden und Abfragen.

Nichts hier rechnet im Browser; alle Zahlen stammen aus 40 Netzen (siehe `meta`). Statistikfelder haben die Form {n, mean, se, med, p10, p90},
Anteile die Form {k, n, p}."""
import json
import os
import pathlib
from functools import lru_cache

PATH = pathlib.Path(os.environ.get("TAR_RESULTS_PATH") or pathlib.Path(__file__).parent / "data" / "tar_results.json")
INF = float("inf")
VARIANTS = ("fix", "mittel", "var")
KREUZ_ROWS = ("base", "R", "P", "D", "null")


def _decode(o):
    """±Unendlich steht in der Datei als "inf" / "-inf" (JSON kennt kein Infinity); hier wieder als Zahl."""
    if isinstance(o, dict):
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(v) for v in o]
    if o == "inf":
        return INF
    if o == "-inf":
        return -INF
    return o


@lru_cache(maxsize=1)
def load():
    with open(PATH, encoding="utf-8") as fh:
        return _decode(json.load(fh))


def n_networks():
    return load()["modell"]["mittel"]["n"]["n"]


def model_stats(cost="mittel"):
    return load()["modell"][cost]


def cross_table(cost):
    """Zielkonflikt-Kreuztabelle: Zeilen base (heutiger Tarif), R, P, D, null; Spalten R, P, T, oev (Statistik) und zul (Anteil)."""
    return [load()["kreuz"][cost][k] for k in KREUZ_ROWS]


def prices(cost):
    return load()["preise"][cost]


def loss_matrix(cost):
    return load()["verlust"][cost]


def pareto(cost):
    return load()["pareto"][cost]


def subsidy(cost):
    return load()["subvention"][cost]


def methods(goal):
    return load()["verfahren"][goal]


def assumptions():
    return load()["annahmen"]


def cannibalization():
    return load()["kannibal"]


def practice_gap():
    return load()["luecke"]


def stat_text(s, digits=1):
    """„Mittel ± Standardfehler“ mit Dezimalkomma."""
    return f"{s['mean']:.{digits}f} ± {s['se']:.{digits}f}".replace(".", ",")


def share_text(p):
    return f"{p['k']} von {p['n']}"
