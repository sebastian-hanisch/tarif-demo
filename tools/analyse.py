"""Auswertung der Tarif-Rohdaten (tools/sweep_tarif.py) -> auswertung.json und auswertung.txt.

Aufruf:  python -B tools/analyse.py [Rohdatenordner]
Standard: Umgebungsvariable TAR_RAW_DIR, sonst ../oepnv-planung/messreihe_tarif_ap0 (relativ zum Projekte-Ordner).

`aggregate(...)` ist von den Rohdateien getrennt, damit dieselben Definitionen auch auf die Vorab-Messreihe angewandt werden können (vergleich.py).
Jede Kennzahl heißt `stat`: {n, mean, se, med, p10, p90}; Anteile heißen `share`: {k, n, p}. Beschriftungen und Aufbau folgen der alten auswertung.txt.
"""
import json
import math
import os
import sys
from collections import Counter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
RAW = os.environ.get("TAR_RAW_DIR") or os.path.join(DEMO, "..", "oepnv-planung", "messreihe_tarif_ap0")

import numpy as np  # noqa: E402

COSTS = ("fix", "mittel", "var")
S_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4)
NULL_MULT = (0.9, 0.97, 1.001)
PARETO_F = tuple(round(0.1 * i, 1) for i in range(11))
XS = np.round(np.arange(-1.0, 3.0001, 0.01), 4)
GOALS = ("R", "P", "D")
GOAL_LABEL = {"R": "Max-R", "P": "Max-P", "D": "Max-D (Bilanz wie heute)"}
ROWS_KREUZ = (("base", "heutiger Tarif"), ("R", "Preise Max-R"), ("P", "Preise Max-P"), ("D", "Preise Max-D"), ("null", "Nulltarif"))


# ------------------------------------------------------------------------------------------------ Bausteine
def clean(values):
    return np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], float)


def stat(values):
    a = clean(values)
    n = len(a)
    if n == 0:
        return dict(n=0, mean=None, se=None, med=None, p10=None, p90=None)
    se = float(a.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return dict(n=int(n), mean=float(a.mean()), se=se, med=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))


def share(flags):
    flags = [bool(f) for f in flags]
    k, n = int(sum(flags)), len(flags)
    return dict(k=k, n=n, p=(k / n if n else None))


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pick(r, src):
    if src == "base":
        return r["base"]
    if src == "null":
        return r["opt_0"]
    return r["cross"][src]


# ------------------------------------------------------------------------------------------------ Abschnitte
def agg_modell(rows):
    keys = ("n", "E", "tpt", "l", "elasticity", "kdg", "sS", "sM", "sC", "kmean", "outer", "fixshare", "bound")
    out = {k: stat([r["facts"][k] for r in rows]) for k in keys}
    out["kdg"] = stat([r["R0"] / r["C0"] for r in rows])
    out["S_today_over_C0"] = stat([r["S_today"] / r["C0"] for r in rows])
    out["auto_heute_pct"] = stat([r["share0"][2] * 100 for r in rows])
    return out


def agg_kreuz(rows):
    out = {}
    for src, lab in ROWS_KREUZ:
        d = {"label": lab}
        d["R"] = stat([pick(r, src)["R"] / r["R0"] * 100 for r in rows])
        d["P"] = stat([pick(r, src)["P"] / r["R0"] * 100 for r in rows])
        d["T"] = stat([pick(r, src)["T"] / r["T0"] * 100 for r in rows])
        d["oev"] = stat([(1 - pick(r, src)["share"][2]) * 100 for r in rows])
        d["zul"] = share([pick(r, src)["P"] >= -1e-6 * r["R0"] for r in rows])
        out[src] = d
    return out


def agg_preise(rows):
    out = {}
    lev = {k: np.array([np.mean(np.array(r["opt_" + k]["x"]) / np.array(r["x0"])) for r in rows]) for k in GOALS}
    for k in GOALS:
        out[k] = dict(
            x_s_rel=stat([r["opt_" + k]["x"][0] / r["x0"][0] for r in rows]),
            x_m_rel=stat([r["opt_" + k]["x"][1] / r["x0"][1] for r in rows]),
            x_s=stat([r["opt_" + k]["x"][0] for r in rows]),
            x_m=stat([r["opt_" + k]["x"][1] for r in rows]),
            niveau=stat(lev[k]),
        )
    out["ratio_P_R"] = stat(lev["P"] / lev["R"])
    out["ratio_D_R"] = stat(lev["D"] / lev["R"])
    out["ratio_D_P"] = stat(lev["D"] / lev["P"])
    out["P_gleich_R"] = share(np.abs(lev["P"] / lev["R"] - 1) < 1e-3)
    out["D_nahe_heute"] = share(np.abs(lev["D"] - 1.0) < 0.02)
    order = Counter("".join(GOALS[i] for i in np.argsort([lev["R"][j], lev["P"][j], lev["D"][j]])) for j in range(len(rows)))
    out["rangfolge_aufsteigend"] = dict(order)
    return out


def agg_verlust(rows):
    out = {}
    for src in GOALS:
        out[src] = dict(
            R=stat([(r["opt_R"]["R"] - r["cross"][src]["R"]) / r["R0"] * 100 for r in rows]),
            P=stat([(r["opt_P"]["P"] - r["cross"][src]["P"]) / r["R0"] * 100 for r in rows]),
            T=stat([(r["opt_D"]["T"] - r["cross"][src]["T"]) / r["T0"] * 100 for r in rows]),
        )
    return out


def agg_pareto(rows):
    per_f = {f: [] for f in PARETO_F}
    for r in rows:
        for p in r["pareto"]:
            if "T" in p:
                for f in PARETO_F:
                    if abs(p["f"] - f) < 1e-9:
                        per_f[f].append((r, p))
    steps = []
    for f, items in per_f.items():
        steps.append(dict(
            f=f,
            R=stat([p["R"] / r["R0"] * 100 for r, p in items]),
            T=stat([p["T"] / r["T0"] * 100 for r, p in items]),
            oev=stat([(1 - p["share"][2]) * 100 for r, p in items]),
            niveau=stat([np.mean(np.array(p["x"]) / np.array(r["x0"])) for r, p in items]),
        ))
    drops = []
    byseed = []
    for r in rows:
        d = {round(p["f"], 1): p for p in r["pareto"] if "T" in p}
        if all(f in d for f in PARETO_F):
            byseed.append([d[f]["T"] / r["T0"] * 100 for f in PARETO_F])
    byseed = np.array(byseed)
    for i in range(len(PARETO_F) - 1):
        drops.append(dict(von=PARETO_F[i], bis=PARETO_F[i + 1], drop=stat(byseed[:, i] - byseed[:, i + 1]) if len(byseed) else stat([])))
    means = [d["drop"]["mean"] for d in drops]
    return dict(stufen=steps, schritte=drops, n_vollstaendig=int(len(byseed)),
                monoton=bool(all(m is not None and m >= -1e-9 for m in means)),
                knick_letzter_groesster=bool(means[-1] > max(means[:-1])) if all(m is not None for m in means) else None)


def agg_subvention(rows):
    out = dict(
        Smin=stat([r["Smin_over_C0"] for r in rows]),
        Snull=stat([r["Snull_over_C0"] for r in rows]),
        Stoday=stat([r["S_today"] / r["C0"] for r in rows]),
        Snull_over_Stoday=stat([r["Snull_over_C0"] / (r["S_today"] / r["C0"]) for r in rows]),
    )
    out["Smin_unter_Stoday_unter_Snull"] = bool(out["Smin"]["mean"] < out["Stoday"]["mean"] < out["Snull"]["mean"])

    def row(sel, ref_fn):
        pts = [(r, p) for r in rows for p in r["sweep"] if sel(r, p)]
        feas = [(r, p) for r, p in pts if p.get("feasible", not p.get("infeasible"))]
        return dict(
            zul=share([bool(p.get("feasible", not p.get("infeasible"))) for r, p in pts]),
            T=stat([p["T"] / ref_fn(r) * 100 for r, p in feas]),
            oev=stat([(1 - p["share"][2]) * 100 for r, p in feas]),
            niveau=stat([np.mean(np.array(p["x"]) / np.array(r["x0"])) for r, p in feas]),
            nulltarif=share([p["x"][0] / r["x0"][0] < 1e-3 for r, p in feas]),
            nulltarif_beide=share([max(p["x"][0] / r["x0"][0], p["x"][1] / r["x0"][1]) < 1e-3 for r, p in feas]),
        )

    out["heute"] = row(lambda r, p: p.get("tag") == "heute", lambda r: r["T0"])
    out["raster"] = []
    for s in S_GRID:
        d = row(lambda r, p, s=s: p.get("tag") == "grid" and abs(p["s"] - s) < 1e-9, lambda r: r["T0"])
        d["s"] = s
        out["raster"].append(d)
    out["null_vielfache"] = []
    for fac in NULL_MULT:
        d = row(lambda r, p, fac=fac: p.get("tag") == "null" and abs(p["s"] - fac) < 1e-9, lambda r: r["opt_0"]["T"])
        d["faktor"] = fac
        out["null_vielfache"].append(d)
    return out


def cluster_count(us, tol=0.03):
    centers = []
    for u in us:
        u = np.array(u)
        if not any(np.max(np.abs(u - c)) < tol for c in centers):
            centers.append(u)
    return len(centers)


def agg_verfahren(rows):
    out = {}
    for kind in GOALS:
        R = rows
        noise, allb, refs = [], [], []
        for r in R:
            rec = r[kind]
            cands = [s["val"] for s in rec["ms50"] if s["feas"]]
            for key in ("runA", "runB"):
                if rec[key] is not None:
                    cands.append(rec[key])
            for key in ("grid_coarse", "grid_med"):
                if rec[key] is not None:
                    cands.append(rec[key]["val"])
            rv = np.array([np.nan if v is None else v for v in rec["rule_vals"]])
            if not np.all(np.isnan(rv)):
                cands.append(float(np.nanmax(rv)))
            allb.append(max(cands))
            refs.append(rec["ref"])
            noise.append(abs(rec["runA"] - rec["runB"]) / rec["ref"] if rec["runA"] is not None and rec["runB"] is not None else np.nan)
        noise = np.array(noise)
        theta = float(np.nanpercentile(noise, 95))
        thr = max(theta, 1e-5)
        normv = np.array([(best - np.array([np.nan if v is None else v for v in r[kind]["rule_vals"]])) / ref for r, best, ref in zip(R, allb, refs)])
        pen = np.where(np.isnan(normv), 1.0, normv)
        xg = int(np.argmin(pen.mean(0)))
        j10 = int(np.argmin(np.abs(XS - 0.10)))
        gaps = {k: [] for k in ("x0", "rule_p10", "rule_global", "rule_oracle", "grid_coarse", "grid_med", "ms10", "ms50")}
        nclu, nclu_val, hit, gain_share, same50, nonfeas, nonok, ntot = [], [], [], [], [], 0, 0, 0
        local_minus_grid = []
        for i, (r, best, ref) in enumerate(zip(R, allb, refs)):
            rec = r[kind]
            ms = [s for s in rec["ms50"] if s["feas"]]
            nonfeas += sum(1 for s in rec["ms50"] if not s["feas"])
            nonok += sum(1 for s in rec["ms50"] if not s["ok"])
            ntot += len(rec["ms50"])
            gaps["ms50"].append((best - max(s["val"] for s in ms)) / ref)
            gaps["ms10"].append((best - rec["runA"]) / ref if rec["runA"] is not None else np.nan)
            gaps["grid_coarse"].append((best - rec["grid_coarse"]["val"]) / ref if rec["grid_coarse"] else np.nan)
            gaps["grid_med"].append((best - rec["grid_med"]["val"]) / ref if rec["grid_med"] else np.nan)
            gaps["rule_oracle"].append(float(np.nanmin(normv[i])) if not np.all(np.isnan(normv[i])) else np.nan)
            gaps["rule_global"].append(normv[i, xg])
            gaps["rule_p10"].append(normv[i, j10])
            gaps["x0"].append((best - rec["val_x0"]) / ref)
            if rec["grid_med"] is not None:
                local_minus_grid.append((max(s["val"] for s in ms) - rec["grid_med"]["val"]) / ref)
            us = [s["u"] for s in ms]
            vs = [s["val"] for s in ms]
            bv = max(vs)
            nclu.append(cluster_count([us[j] for j in np.argsort(-np.array(vs))]))
            hit.append(float(np.mean([(bv - v) / ref <= thr for v in vs])))
            same50.append(all((bv - v) / ref <= thr for v in vs))
            order = np.argsort(-np.array(vs))
            cl = []
            for k_ in order:
                u = np.array(us[k_])
                if not any(np.max(np.abs(u - c[0])) < 0.03 for c in cl):
                    cl.append((u, vs[k_]))
            nclu_val.append(sum(1 for c in cl if (bv - c[1]) / ref > thr))
            if kind in ("R", "P"):
                gain = best - rec["val_x0"]
                rv = np.array([np.nan if v is None else v for v in rec["rule_vals"]])
                if gain > 1e-6 * ref:
                    gain_share.append((np.nanmax(rv) - rec["val_x0"]) / gain * 100)
        meth = {}
        for k, lab in (("x0", "heutiger Tarif"), ("rule_p10", "Praxisregel +10 % (fest)"), ("rule_global", "Praxisregel global bester Wert"),
                       ("rule_oracle", "Praxisregel mit je Instanz bestem X (fairster Wert)"), ("grid_coarse", "Gittersuche grob (21x21)"),
                       ("grid_med", "Gittersuche mittel (41x41)"), ("ms10", "lokal, 10 Starts"), ("ms50", "lokal, 50 Starts")):
            a = np.array(gaps[k], float)
            ok = a[~np.isnan(a)]
            meth[k] = dict(label=lab, gap_pct=stat(ok * 100), le_thr=share(ok <= thr), fehlt=int(np.isnan(a).sum()))
        xo = []
        for r in R:
            rv = np.array([np.nan if v is None else v for v in r[kind]["rule_vals"]])
            if not np.all(np.isnan(rv)):
                xo.append(XS[int(np.nanargmax(rv))] * 100)
        out[kind] = dict(
            rauschen=dict(median=float(np.nanmedian(noise)), p95=theta, max=float(np.nanmax(noise))), schwelle=thr, methoden=meth,
            gewinnanteil_regel=stat(gain_share) if gain_share else None, bestes_X_pct=stat(xo), X_global_pct=float(XS[xg] * 100),
            optima=dict(endpunkte=stat(nclu), schlechtere=stat(nclu_val), anteil_mehrere=share(np.array(nclu) > 1), anteil_wertverschieden=share(np.array(nclu_val) > 0),
                        treffer50=stat(np.array(hit) * 100), alle50_gleich=share(same50), starts_unzulaessig=int(nonfeas), starts_nicht_konvergiert=int(nonok), starts_gesamt=int(ntot)),
            lokal_minus_gitter_min=float(min(local_minus_grid)) if local_minus_grid else None,
            zeit=dict(ms10_med=float(np.median([r[kind]["t_ms10"] for r in R])), grid_grob_med=float(np.median([r[kind]["t_grid_coarse"] for r in R])),
                      grid_mittel_med=float(np.median([r[kind]["t_grid_med"] for r in R]))),
            auswertungen_gitter_mittel=int(R[0][kind]["grid_med"]["nev"]) if R[0][kind]["grid_med"] and "nev" in R[0][kind]["grid_med"] else None,
        )
    return out


def agg_annahmen(rows):
    out = {}
    nom = {k: stat([r["nominal"][k]["level"] if r["nominal"][k] else np.nan for r in rows]) for k in GOALS}

    def order_of(levs):
        return tuple(int(i) for i in np.argsort(levs))

    nomord = Counter(order_of([r["nominal"][k]["level"] for k in GOALS]) for r in rows if all(r["nominal"][k] for k in GOALS))
    out["nominal_niveau"] = nom
    out["nominal_rangfolge"] = {"".join(GOALS[i] for i in k): v for k, v in nomord.items()}
    names = []
    for p in rows[0]["pert"]:
        key = (p["name"], p["mode"])
        if key not in names:
            names.append(key)
    entries = []
    for name, mode in names:
        for fac in (0.75, 1.25):
            items = [(r, [p for p in r["pert"] if p["name"] == name and p["mode"] == mode and abs(p["fac"] - fac) < 1e-9][0]) for r in rows]
            cell = {}
            for k in GOALS:
                sh = []
                for r, p in items:
                    if p["opt"][k] is None or r["nominal"][k] is None:
                        continue
                    xp, xn = np.array(p["opt"][k]["x"]), np.array(r["nominal"][k]["x"])
                    sh.append(float(np.mean(xp / xn - 1) * 100))
                cell[k] = stat(sh)
            regR = [(p["opt"]["R"]["R"] - p["nom_in_pert"]["R"]["R"]) / p["R0p"] * 100 for r, p in items if p["opt"]["R"] and "R" in p["nom_in_pert"]]
            regP = [(p["opt"]["P"]["P"] - p["nom_in_pert"]["P"]["P"]) / p["R0p"] * 100 for r, p in items if p["opt"]["P"] and "P" in p["nom_in_pert"]]
            feas = [p["nom_in_pert"]["D"]["feasD"] for r, p in items if "D" in p["nom_in_pert"]]
            shortD = [(p["opt"]["D"]["T"] - p["nom_in_pert"]["D"]["T"]) / p["T0p"] * 100 for r, p in items if p["opt"]["D"] and "D" in p["nom_in_pert"]]
            nD = sum(1 for r, p in items if p["opt"]["D"] is None)
            arp = {}
            for k in ("R", "P"):
                arp[k] = stat([(p["opt"][k]["R"] / p["opt"][k]["T"]) / (r["nominal"][k]["R"] / r["nominal"][k]["T"]) * 100 - 100
                               for r, p in items if p["opt"][k] and r["nominal"][k]])
            same = [order_of([p["opt"][k]["level"] for k in GOALS]) == order_of([r["nominal"][k]["level"] for k in GOALS])
                    for r, p in items if all(p["opt"][k] for k in GOALS) and all(r["nominal"][k] for k in GOALS)]
            # größte relative Änderung irgendeiner Preis-/Kennzahl gegenüber nominal (Prüfung auf Wirkungslosigkeit)
            mx = 0.0
            for r, p in items:
                for k in GOALS:
                    if p["opt"][k] and r["nominal"][k]:
                        a, b = np.array(p["opt"][k]["x"]), np.array(r["nominal"][k]["x"])
                        mx = max(mx, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-3))))
                mx = max(mx, abs(p["R0p"] - r["R0"]) / r["R0"], abs(p["T0p"] - r["T0"]) / r["T0"], abs(p["C0p"] - r["C0"]) / r["C0"])
            entries.append(dict(name=name, mode=mode, fac=fac, shift=cell, erloes_je_fahrt=arp, regret=dict(R=stat(regR), P=stat(regP)), regret_P90=dict(R=float(np.percentile(regR, 90)), P=float(np.percentile(regP, 90))),
                                zul=share(feas), D_verlust=stat(shortD), D_unzulaessig=int(nD), gleiche_rangfolge=share(same), max_rel_aenderung=mx,
                                wirkungslos=bool(mx < 1e-6)))
    out["eintraege"] = entries
    return out


def agg_kannibal(rows):
    d0 = [r["q5"] for r in rows]
    out = dict(einzel_erloesanteil_pct=stat([d["Rs"] / (d["Rs"] + d["Rm"]) * 100 for d in d0]),
               personenanteile_pct=[float(np.mean([d["pshare"][i] for d in d0]) * 100) for i in range(3)], schnitte=[])
    for ci, c in enumerate((0.9, 0.8, 0.7)):
        cs = [d["cuts"][ci] for d in d0]
        dPS = np.array([c_["pshare"][0] - d["pshare"][0] for c_, d in zip(cs, d0)])
        dPM = np.array([c_["pshare"][1] - d["pshare"][1] for c_, d in zip(cs, d0)])
        dPC = np.array([c_["pshare"][2] - d["pshare"][2] for c_, d in zip(cs, d0)])
        out["schnitte"].append(dict(
            c=c,
            dR=stat([(c_["R"] - d["R"]) / d["R"] * 100 for c_, d in zip(cs, d0)]),
            dRs=stat([(c_["Rs"] - d["Rs"]) / d["R"] * 100 for c_, d in zip(cs, d0)]),
            dRm=stat([(c_["Rm"] - d["Rm"]) / d["R"] * 100 for c_, d in zip(cs, d0)]),
            dT=stat([(c_["T"] - d["T"]) / d["T"] * 100 for c_, d in zip(cs, d0)]),
            dPM_pkt=stat(dPM * 100), dPS_pkt=stat(dPS * 100), dPC_pkt=stat(dPC * 100),
            aus_einzelkarte_pct=stat(-dPS / dPM * 100),
        ))
    out["iia_max_abw"] = float(max(r["iia_max_dev"] for r in rows)) if rows and "iia_max_dev" in rows[0] else None
    return out


def agg_luecke(rows):
    out = {}
    for kind in ("R", "P"):
        g = [a["gap"] * 100 for r in rows for a in r["A"] if a["kind"] == kind]
        g0 = [a["gap_at_x0"] * 100 for r in rows for a in r["A"] if a["kind"] == kind]
        xb = [a["Xbest"] * 100 for r in rows for a in r["A"] if a["kind"] == kind]
        out[kind] = dict(luecke_beste_anhebung_pct=stat(g), luecke_ohne_anhebung_pct=stat(g0), bestes_X_pct=stat(xb))
    return out


def aggregate(q1, q2, q3, q5, extra):
    res = dict(meta=dict(n_q1=len(q1), n_q2=len(q2), n_q3=len(q3), n_q5=len(q5), n_extra=len(extra)))
    for key in ("modell", "kreuz", "preise", "verlust", "pareto", "subvention"):
        res[key] = {}
    for cost in COSTS:
        rows = sorted((r for r in q1 if r["cost"] == cost), key=lambda r: r["seed"])
        if not rows:
            continue
        res["modell"][cost] = agg_modell(rows)
        res["kreuz"][cost] = agg_kreuz(rows)
        res["preise"][cost] = agg_preise(rows)
        res["verlust"][cost] = agg_verlust(rows)
        res["pareto"][cost] = agg_pareto(rows)
        res["subvention"][cost] = agg_subvention(rows)
    if q2:
        res["verfahren"] = agg_verfahren(sorted(q2, key=lambda r: r["seed"]))
    if q3:
        res["annahmen"] = agg_annahmen(sorted(q3, key=lambda r: r["seed"]))
    if q5:
        res["kannibal"] = agg_kannibal(sorted(q5, key=lambda r: r["seed"]))
    if extra:
        res["luecke"] = agg_luecke(sorted(extra, key=lambda r: r["seed"]))
    return res


# ------------------------------------------------------------------------------------------------ Text
def fs(s, f="{:.3f}", scale=1.0):
    if not s or s.get("n", 0) == 0:
        return "n=0"
    g = lambda v: f.format(v * scale)
    return f"{g(s['mean'])} ± {g(s['se'])} (Med {g(s['med'])}; P10 {g(s['p10'])}; P90 {g(s['p90'])})"


def fshare(s):
    return f"{s['k']}/{s['n']}"


def text_report(res, extra_sections=None):
    out = []
    pr = out.append
    if "modell" in res and "mittel" in res["modell"]:
        m = res["modell"]["mittel"]
        pr("=" * 100)
        pr("MODELL-KENNZAHLEN der 40 Instanzen (Seeds 1..40, n = 20 + (seed-1) mod 11 Haltepunkte, Kostenvariante 'mittel')")
        pr("=" * 100)
        names = dict(n="Haltepunkte", E="Kanten", tpt="Fahrzeit ÖV (min, nachfragegewichtet)", l="Distanz (km, nachfragegewichtet)",
                     elasticity="Fahrpreiselastizität der ÖV-Fahrten (Zonentarif, heute)", kdg="Kostendeckungsgrad heute (Zonentarif)",
                     sS="Wege-Anteil Einzelkarte heute", sM="Wege-Anteil Monatskarte heute", sC="Wege-Anteil Auto heute", kmean="mittlere Fahrtenzahl je Person und Monat",
                     outer="Nachfrageanteil 2-Zonen-Fahrten", fixshare="Fixkostenanteil heute", bound="Anteil lastgebundener Kanten heute")
        for k, lab in names.items():
            pr(f"  {lab}: {fs(m[k])}")
        for cost in COSTS:
            if cost in res["modell"]:
                pr(f"  Kostendeckungsgrad heute, Kostenvariante '{cost}': {fs(res['modell'][cost]['kdg'])}")
    pr("=" * 100)
    pr("FRAGE 1: Zielkonflikt (q1.jsonl)")
    pr("=" * 100)
    for cost in COSTS:
        if cost not in res["kreuz"]:
            continue
        pr(f"\n--- Kosten '{cost}', System 'zone', {res['modell'][cost]['kdg']['n']} Instanzen ---")
        pr("Kostendeckungsgrad heute R0/C0: " + fs(res["modell"][cost]["kdg"]))
        pr("Heutige Subvention S_today / C0: " + fs(res["modell"][cost]["S_today_over_C0"]))
        pr("Auto-Anteil heute (Wege): " + fs(res["modell"][cost]["auto_heute_pct"]) + " %")
        pp = res["preise"][cost]
        for k in GOALS:
            pr(f"  Preis {GOAL_LABEL[k]} Komponente 0 / heute: {fs(pp[k]['x_s_rel'], '{:.2f}')}")
            pr(f"  Preis {GOAL_LABEL[k]} Komponente 1 / heute: {fs(pp[k]['x_m_rel'], '{:.2f}')}")
        pr("  Preisniveau (Mittel x/x0): " + "; ".join(f"{k}: {fs(pp[k]['niveau'], '{:.2f}')}" for k in GOALS))
        pr(f"  Level P/R: {fs(pp['ratio_P_R'])}; D/R: {fs(pp['ratio_D_R'])}; D/P: {fs(pp['ratio_D_P'])}")
        pr(f"  Anteil Instanzen mit x_P = x_R (Niveau auf 0,1 %): {pp['P_gleich_R']['p'] * 100:.0f} %  (Prüfung toter Zweig: Max-P weicht auf {pp['P_gleich_R']['n'] - pp['P_gleich_R']['k']} von {pp['P_gleich_R']['n']} Instanzen ab)")
        pr(f"  Max-D (Bilanz wie heute) liegt auf 2 % am heutigen Tarif: {pp['D_nahe_heute']['p'] * 100:.0f} % der Instanzen")
        pr(f"  Rangfolge der Preisniveaus (aufsteigend, Anzahl Instanzen): {pp['rangfolge_aufsteigend']}")
        pr("  Kreuztabelle: Zeile = verwendete Preise; Spalte = Zielwert. R, P in % von R0 (P = Bilanzgewinn gegenüber heute), T in % von T0; ÖV-Anteil %; zulässig für Max-D (Bilanz >= heute) %")
        for src, lab in ROWS_KREUZ:
            d = res["kreuz"][cost][src]
            pr(f"    {lab:16s} R {d['R']['mean']:7.1f} ± {d['R']['se']:4.1f} | P {d['P']['mean']:7.1f} ± {d['P']['se']:4.1f} | T {d['T']['mean']:6.1f} ± {d['T']['se']:4.1f} | ÖV {d['oev']['mean']:5.1f} | zulässig {d['zul']['p'] * 100:4.0f} %")
        pr("  Verlust gegenüber dem eigenen Optimum (Zeile = fremde Preise): R-Verlust in %R0, P-Verlust in %R0, T-Verlust in %T0")
        for src in GOALS:
            d = res["verlust"][cost][src]
            pr(f"    Preise {src}: R-Verlust {fs(d['R'], '{:.2f}')} | P-Verlust {fs(d['P'], '{:.2f}')} | T-Verlust {fs(d['T'], '{:.2f}')}")
        pa = res["pareto"][cost]
        pr("  Pareto (max Fahrten bei Erlös >= f * R_max), Mittel über Instanzen: f | T/T0 % | ÖV-Anteil % | Preisniveau x/x0")
        for st_ in pa["stufen"]:
            if st_["T"]["n"]:
                pr(f"    f={st_['f']:.1f}: Erlös {st_['R']['mean']:6.1f} % R0 | T {st_['T']['mean']:6.1f} ± {st_['T']['se']:4.1f} | ÖV {st_['oev']['mean']:5.1f} | Niveau {st_['niveau']['mean']:.2f}")
        pr("  Fahrtenverlust je Schritt (Prozentpunkte T/T0): " + ", ".join(f"{d['von']:.1f}->{d['bis']:.1f}: {d['drop']['mean']:.2f}" for d in pa["schritte"])
           + f" | monoton: {pa['monoton']} | Knick (letzter Schritt größter): {pa['knick_letzter_groesster']}")
        sv = res["subvention"][cost]
        pr("  Subventionssweep (Max-D mit Kosten; S in Vielfachen der heutigen Betriebskosten C0):")
        pr("    kleinste zulässige Subvention S_min/C0: " + fs(sv["Smin"]))
        pr("    Nulltarif zulässig ab S_null/C0:        " + fs(sv["Snull"]))
        pr("    Verhältnis S_null / S_today:            " + fs(sv["Snull_over_Stoday"], "{:.2f}"))
        h = sv["heute"]
        pr(f"    s=heute: zulässig {fshare(h['zul'])} | T {h['T']['mean']:6.1f} % T0 | ÖV-Anteil {h['oev']['mean']:5.1f} % | Niveau {h['niveau']['mean']:.2f} | Nulltarif in {h['nulltarif']['p'] * 100:3.0f} %")
        for d in sv["raster"]:
            if d["T"]["n"] == 0:
                pr(f"    s={d['s']:.1f}: zulässig in 0 von {d['zul']['n']}")
                continue
            pr(f"    s={d['s']:.1f}: zulässig {fshare(d['zul'])} | T {d['T']['mean']:6.1f} % T0 | ÖV-Anteil {d['oev']['mean']:5.1f} % | Niveau {d['niveau']['mean']:.2f} | Nulltarif in {d['nulltarif']['p'] * 100:3.0f} % (beide Preise 0: {d['nulltarif_beide']['p'] * 100:3.0f} %)")
        for d in sv["null_vielfache"]:
            if d["T"]["n"]:
                pr(f"    S = {str(d['faktor']).replace('.', ',')} * S_null: T / T(Nulltarif) {d['T']['mean']:6.2f} % | Niveau {d['niveau']['mean']:.3f} | Nulltarif-Lösung in {d['nulltarif']['p'] * 100:3.0f} % (beide Preise 0: {d['nulltarif_beide']['p'] * 100:3.0f} %) (n={d['T']['n']})")
    if "verfahren" in res:
        pr("=" * 100)
        pr("FRAGE 2: Verfahrensvergleich (q2.jsonl), Kostenvariante 'mittel'")
        pr("=" * 100)
        for kind in GOALS:
            v = res["verfahren"][kind]
            n = v["methoden"]["x0"]["gap_pct"]["n"]
            pr(f"\n--- System zone, Ziel {GOAL_LABEL[kind]}, {n} Instanzen ---")
            r_ = v["rauschen"]
            pr(f"Rauschen: |Lauf A - Lauf B| / Referenz (zwei unabhängige 10er-Mehrfachstarts): Median {r_['median'] * 100:.4f} %, P95 {r_['p95'] * 100:.4f} %, Max {r_['max'] * 100:.4f} %")
            pr(f"Meldeschwelle (P95, mindestens 1e-5): {v['schwelle'] * 100:.4f} % der Referenz (R0 bzw. T0)")
            pr("Abstand zum besten bekannten Wert, in % der Referenz (Mittel ± SE; Median; P90) und Anteil <= Meldeschwelle:")
            for k, d in v["methoden"].items():
                lab = d["label"] + (f" X*={v['X_global_pct']:+.0f} %" if k == "rule_global" else "")
                pr(f"  {lab:55s} {fs(d['gap_pct'], '{:.4f}')} | <= Schwelle: {d['le_thr']['p'] * 100:.0f} %" + (f" | unzulässig/fehlt: {d['fehlt']}" if d["fehlt"] else ""))
            if v["gewinnanteil_regel"]:
                pr(f"Anteil des erreichbaren Gewinns (Optimum gegenüber heute), den die Praxisregel mit je Instanz bestem X erreicht: {fs(v['gewinnanteil_regel'], '{:.1f}')} %")
            pr(f"Bestes X je Instanz (Praxisregel): {fs(v['bestes_X_pct'], '{:.0f}')} %; global X* = {v['X_global_pct']:+.0f} %")
            o = v["optima"]
            pr(f"Lokale Optima bei 50 Starts: Zahl räumlich verschiedener Endpunkte (Max-Norm 0,03 in u): {fs(o['endpunkte'], '{:.2f}')}; davon mit schlechterem Wert (> Schwelle): {fs(o['schlechtere'], '{:.2f}')}; Anteil Instanzen mit >1 Endpunkt: {o['anteil_mehrere']['p'] * 100:.0f} %, mit wertverschiedenem schlechteren Optimum: {o['anteil_wertverschieden']['p'] * 100:.0f} %")
            pr(f"Anteil der 50 Starts, die das beste Optimum treffen: {fs(o['treffer50'], '{:.1f}')} %; Instanzen, in denen alle 50 Starts dasselbe Optimum finden: {fshare(o['alle50_gleich'])}; unzulässige Starts {o['starts_unzulaessig']} von {o['starts_gesamt']}, nicht konvergiert {o['starts_nicht_konvergiert']}")
            z = v["zeit"]
            pr("Rechenzeit: 10 Starts lokal %.2f s (Med), Gitter grob %.2f s, Gitter mittel %.2f s (Med)" % (z["ms10_med"], z["grid_grob_med"], z["grid_mittel_med"]))
            pr(f"Auswertungen Gitter mittel: {v['auswertungen_gitter_mittel']}")
    if "annahmen" in res:
        an = res["annahmen"]
        pr("=" * 100)
        pr("FRAGE 3: Annahmen-Empfindlichkeit (q3.jsonl), Kostenvariante 'mittel'; Annahme +-25 %")
        pr("=" * 100)
        pr("\n--- System zone, 40 Instanzen ---")
        pr("Nominale Preisniveaus x/x0: " + "; ".join(f"{k}: {fs(an['nominal_niveau'][k], '{:.2f}')}" for k in GOALS))
        pr("Spalten: mittlere Preisverschiebung der optimalen Preise gegenüber nominal (%; Mittel über Komponenten, dann über Instanzen, ± SE) für R / P / D;")
        pr("          Regret = Verlust beim Einsatz des nominalen Optimums in der gestörten Welt, in % von R0' (R, P) bzw. T0' (D); Zul = Anteil Instanzen, in denen die nominalen D-Preise die Bilanz noch halten.")
        for e in an["eintraege"]:
            sh = e["shift"]
            tag = f"{e['name']:9s}{'(neu kal.)' if e['mode'] == 'recal' else '(y fest)  '}{e['fac']:5.2f}"
            f1 = lambda s: "n/a" if s["n"] == 0 else f"{s['mean']:+7.1f} ± {s['se']:4.1f}"
            pr(f"  {tag} | Preis R {f1(sh['R'])} | P {f1(sh['P'])} | D {f1(sh['D'])} (n={sh['D']['n']})"
               f" | Erlös/Fahrt R {e['erloes_je_fahrt']['R']['mean']:+6.1f} P {e['erloes_je_fahrt']['P']['mean']:+6.1f}"
               f" | Regret R {e['regret']['R']['mean']:5.2f} (P90 {e['regret_P90']['R']:5.2f}) P {e['regret']['P']['mean']:5.2f} (P90 {e['regret_P90']['P']:5.2f})"
               f" | Zul {e['zul']['p'] * 100:3.0f} %, D-Verlust {e['D_verlust']['mean'] if e['D_verlust']['n'] else float('nan'):5.2f} | D unzulässig in Welt: {e['D_unzulaessig']}"
               f" | größte Änderung {e['max_rel_aenderung']:.1e}{' (wirkungslos)' if e['wirkungslos'] else ''}")
        pr("Rangfolge der Preisniveaus (aufsteigend) der drei Zielfunktionen: nominal vs. gestört; Anteil Instanzen mit gleicher Reihenfolge wie nominal")
        pr(f"  nominal: {an['nominal_rangfolge']} (R, P, D: Buchstaben in aufsteigender Reihenfolge)")
        for e in an["eintraege"]:
            pr(f"  {e['name']:9s}{e['mode']:6s}{e['fac']:5.2f}: gleiche Reihenfolge in {e['gleiche_rangfolge']['p'] * 100:5.1f} % (n={e['gleiche_rangfolge']['n']})")
    if "kannibal" in res:
        k = res["kannibal"]
        pr("=" * 100)
        pr("FRAGE 5: Kannibalisierung (q5.jsonl)")
        pr("=" * 100)
        pr("\n--- Zonentarif: Monatskarte x_m billiger ---")
        pr(f"Heute: Anteil Einzelkarte-Erlös am ÖV-Erlös: {fs(k['einzel_erloesanteil_pct'], '{:.1f}')} %; Personenanteile Einzel/Monat/Auto: " + " / ".join(f"{v:.1f}" for v in k["personenanteile_pct"]) + " %")
        for s in k["schnitte"]:
            pr(f"  Karte auf {s['c'] * 100:.0f} % des Preises: Erlös gesamt {fs(s['dR'], '{:.2f}')} % | Einzelkarten-Erlös {fs(s['dRs'], '{:.2f}')} %R0 | Monatskarten-Erlös {fs(s['dRm'], '{:.2f}')} %R0 | Fahrten {fs(s['dT'], '{:.2f}')} %")
            pr(f"      Zuwachs Karteninhaber (Personen) {s['dPM_pkt']['mean']:+.2f} Pkt; davon aus Einzelkarte {s['aus_einzelkarte_pct']['mean']:.1f} ± {s['aus_einzelkarte_pct']['se']:.1f} % (Rest vom Auto); Verlust Einzelkarte {s['dPS_pkt']['mean']:+.2f} Pkt, Auto {s['dPC_pkt']['mean']:+.2f} Pkt")
        pr(f"IIA-Identität (Logit): dP_S/dP_M = -P_S/(1-P_M) je Zelle; maximale Abweichung bei 0,1 % Preisänderung über alle Zellen/Instanzen: {k['iia_max_abw']:.2e}")
    if "luecke" in res:
        pr("=" * 100)
        pr("ZUSATZ A: Praxisregel-Lücke bei zufälliger Ausgangs-Tarifstruktur (extra.jsonl), 40 Instanzen x 5 Starts")
        pr("=" * 100)
        for kind in ("R", "P"):
            d = res["luecke"][kind]
            pr(f"  zone Max-{kind}: Lücke der besten Anhebung zum Optimum (% von R0): {fs(d['luecke_beste_anhebung_pct'], '{:.2f}')}; Lücke ohne Anhebung: {fs(d['luecke_ohne_anhebung_pct'], '{:.2f}')}; bestes X: {fs(d['bestes_X_pct'], '{:.0f}')} %  (n={d['luecke_beste_anhebung_pct']['n']})")
    for sec in (extra_sections or []):
        out.append(sec)
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ Hauptprogramm
def main(raw=RAW):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    q1 = load(os.path.join(raw, "q1.jsonl"))
    q2 = load(os.path.join(raw, "q2.jsonl"))
    q3 = load(os.path.join(raw, "q3.jsonl"))
    q5 = load(os.path.join(raw, "q5.jsonl"))
    ex = load(os.path.join(raw, "extra.jsonl"))
    res = aggregate(q1, q2, q3, q5, ex)
    for name in ("pruefungen_timing", "pruefungen_mc", "pruefungen_gradient", "pruefungen_loeser", "pruefungen_determinismus", "pruefungen_weitere"):
        fn = os.path.join(raw, name + ".json")
        if os.path.exists(fn):
            with open(fn, encoding="utf-8") as fh:
                res[name.replace("pruefungen_", "pr_")] = json.load(fh)
    if "pr_timing" in res:
        res["rechenzeit"] = res["pr_timing"]
    with open(os.path.join(raw, "auswertung.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    txt = text_report(res)
    with open(os.path.join(raw, "auswertung.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RAW)
