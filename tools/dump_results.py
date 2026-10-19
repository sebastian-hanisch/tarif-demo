"""Schreibt aus auswertung.json (tools/analyse.py) die kompakte Ergebnisdatei data/tar_results.json für die App.

Aufruf:  python -B tools/dump_results.py [Pfad zu auswertung.json] [Ausgabedatei]
Standard: TAR_RAW_DIR/auswertung.json (sonst ../oepnv-planung/messreihe_tarif_ap0) -> data/tar_results.json

Regeln: UTF-8, +inf als String "inf", -inf als "-inf", NaN als null. Alle Gleitkommazahlen auf 6 gültige Stellen gerundet, keine Rohdaten, kompakte
Trennzeichen. Statistiken sind {n, mean, se, med, p10, p90}, Anteile {k, n, p}. Aufbau oberste Ebene:
  meta, modell, kreuz, preise, verlust, pareto, subvention, verfahren, annahmen, kannibal, luecke, rechenzeit, pruefungen
je Kostenvariante (fix/mittel/var) bei modell, kreuz, preise, verlust, pareto, subvention; verfahren, annahmen, kannibal, luecke nur für "mittel".
`pruefungen` enthält nur die Kurzurteile der Pflichtprüfungen (keine Zeilen), `rechenzeit` die Zeiten der Live-Pfade.
"""
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
RAW = os.environ.get("TAR_RAW_DIR") or os.path.join(DEMO, "..", "oepnv-planung", "messreihe_tarif_ap0")
SRC = os.path.join(RAW, "auswertung.json")
DST = os.path.join(DEMO, "data", "tar_results.json")
KEEP = ("meta", "modell", "kreuz", "preise", "verlust", "pareto", "subvention", "verfahren", "annahmen", "kannibal", "luecke", "rechenzeit")


def rnd(o):
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, float):
        if math.isnan(o):
            return None
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return float("%.6g" % o)
    return o


def pruef_kurz(R):
    out = {}
    if "pr_mc" in R:
        m = R["pr_mc"]
        out["monte_carlo"] = {k: m[k] for k in ("draws", "max_abs_z", "max_abs_z_cells", "max_rel_R", "max_relC", "bestanden")}
    if "pr_gradient" in R:
        g = R["pr_gradient"]
        out["gradient"] = {k: g[k] for k in ("punkte", "max_rel_fehler", "bestanden")}
    if "pr_loeser" in R:
        l = R["pr_loeser"]
        out["loeser"] = {k: l[k] for k in ("netze", "min_diff", "loeser_nie_schlechter", "max_gap_51_starts", "alle_starts_gleich")}
    if "pr_determinismus" in R:
        d = R["pr_determinismus"]
        out["determinismus"] = {k: d[k] for k in ("zahlen", "bitgleich", "max_rel_abweichung", "bestanden")}
    return out


def main(src=SRC, dst=DST):
    with open(src, encoding="utf-8") as fh:
        R = json.load(fh)
    R = dict(R)
    if "rechenzeit" not in R and "pr_timing" in R:
        R["rechenzeit"] = R["pr_timing"]
    out = {k: R[k] for k in KEEP if k in R}
    out["pruefungen"] = pruef_kurz(R)
    out = rnd(out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("geschrieben:", dst, "%.0f kB" % (os.path.getsize(dst) / 1024.0), "Schlüssel:", list(out))


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
