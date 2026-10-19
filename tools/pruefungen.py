"""Pflichtprüfungen AP 0 für den Tarif-Kern. Aufruf:  python -B tools/pruefungen.py mc|gradient|loeser|nullspalten|determinismus <zweiter Ordner>|timing|rng
Schreibt pruefungen_<teil>.json in den Rohdatenordner (Umgebungsvariable TAR_RAW_DIR, sonst ../oepnv-planung/messreihe_tarif_ap0).
Zufall nur aus SplitMix64 (`tar_rng`, hier auch als Feld-Version mit numpy.uint64 für die Monte-Carlo-Kontrolle; `rng` prüft beide Fassungen auf Gleichheit).

 mc            (a) Monte-Carlo-Kontrolle von Erlös, Fahrten, Auto-Anteil (Gumbel-argmax) an 4 Netzen x 3 Tarifpunkten, >= 200 000 Ziehungen
 gradient      (b) analytischer Gradient gegen zentrale Differenzen
 loeser        (c) Löser gegen Gitter 41x41 an 20 Netzen; (d) 50 Starts je Netz und Ziel
 nullspalten   (e) Sweep-Spalten mit lauter Nullen; (f) Wirkungslosigkeit der Annahmen
 determinismus (g) zwei Sweep-Läufe (Ordner 1 = Rohdatenordner, Ordner 2 = Argument) bis auf 1e-9 relativ gleich
 timing        Rechenzeiten der Live-Pfade (Modellaufbau, solve_goals, Pareto, Subventionskurve, Preis-Ebene), 20/25/30 Haltepunkte
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
sys.path.insert(0, DEMO)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import tar_algorithm as A  # noqa: E402
import tar_constants as C  # noqa: E402
import tar_evaluation as E  # noqa: E402
from sweep_tarif import build, n_of, retune_cost  # noqa: E402
from tar_model import Model  # noqa: E402
from tar_rng import SplitMix64  # noqa: E402
from tar_scenario import make_instance  # noqa: E402

RAW = os.environ.get("TAR_RAW_DIR") or os.path.join(DEMO, "..", "oepnv-planung", "messreihe_tarif_ap0")
MASK = (1 << 64) - 1


def dump(name, obj):
    fn = os.path.join(RAW, "pruefungen_%s.json" % name)
    with open(fn, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("geschrieben:", fn)


# ------------------------------------------------------------------------------------------------ SplitMix64 als Feld
def sm64_stream(seed, n):
    """Die ersten n Ausgaben von SplitMix64(seed) als uint64-Feld (gleiche Zahlen wie `tar_rng.SplitMix64`, aber ohne Python-Schleife)."""
    with np.errstate(over="ignore"):
        k = np.arange(1, n + 1, dtype=np.uint64)
        z = np.uint64(int(seed) & MASK) + k * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def uniform_stream(seed, n):
    """Gleichverteilte Zahlen im offenen Intervall (0,1) aus den oberen 32 Bit."""
    return ((sm64_stream(seed, n) >> np.uint64(32)).astype(np.float64) + 0.5) / 4294967296.0


def part_rng():
    ok = True
    for seed in (0, 1, 12345, (1 << 63) + 17, MASK):
        ref = SplitMix64(seed)
        want = np.array([ref.next() for _ in range(2000)], dtype=np.uint64)
        ok &= bool(np.array_equal(want, sm64_stream(seed, 2000)))
    out = dict(gleich=bool(ok))
    dump("rng", out)
    print(out)


# ------------------------------------------------------------------------------------------------ (a) Monte Carlo
N_DRAWS = 400_000
MC_NETS = ((101, 20, 0, "mittel"), (102, 25, 50, "fix"), (103, 30, 100, "var"), (104, 28, 50, "mittel"))


def mc_run(m, x, N, master):
    """Eine Monte-Carlo-Rechnung. Nutzen aus den Rohgrößen (Preisformeln, Zeiten, Distanzen) gebildet, ohne Cmat und ohne die Softmax-Routine des Modells;
    nur das kalibrierte y (die Kalibrierung prüft ein eigener Test) wird übernommen."""
    inst = m.inst
    seeds = [master.next() for _ in range(5)]
    persons = inst.persons
    q = np.searchsorted(np.cumsum(persons / persons.sum()), uniform_stream(seeds[0], N))
    q = np.minimum(q, inst.P - 1)
    kc = np.minimum(np.searchsorted(np.cumsum(m.pk), uniform_stream(seeds[1], N)), len(C.K_VALS) - 1)
    k = C.K_VALS[kc]
    z = inst.z[q]
    l = inst.l_km[q]
    price_s = z * x[0] * k
    price_m = z * x[1]
    V = np.empty((N, 3))
    V[:, 0] = -price_s / k - m.lam_ * inst.t_pt[q]
    V[:, 1] = -price_m / k - m.lam_ * inst.t_pt[q]
    V[:, 2] = -(m.car_fix_ / k + m.car_km_ * l) - m.lam_ * inst.t_car[q] + m.y[q]
    G = np.stack([-np.log(-np.log(uniform_stream(seeds[2 + j], N))) for j in range(3)], axis=1)
    ch = (V + G / m.mu).argmax(1)
    rev = np.where(ch == 0, price_s, np.where(ch == 1, price_m, 0.0))
    return dict(q=q, k=k, ch=ch, rev=rev, pt=(ch < 2))


def part_mc():
    master = SplitMix64(20261018)
    runs = []
    for seed, n, hub, cost in MC_NETS:
        inst = make_instance(seed, n, hub)
        m = Model(inst, cost=cost)
        m.lam_, m.car_fix_, m.car_km_ = C.LAM, C.CAR_FIX, C.CAR_KM
        tot = inst.persons.sum()
        for pt in range(3):
            if pt == 0:
                x = m.x0.copy()
            else:
                x = m.x0 * np.array([0.3 + 2.2 * master.u32() / 4294967296.0 for _ in range(2)])
            an = m.core(x)
            s = mc_run(m, x, N_DRAWS, master)
            r_an, r_mc, r_se = an["R"] / tot, s["rev"].mean(), s["rev"].std(ddof=1) / math.sqrt(N_DRAWS)
            tv = s["k"] * s["pt"]
            t_an, t_mc, t_se = an["T"] / tot, tv.mean(), tv.std(ddof=1) / math.sqrt(N_DRAWS)
            car_an = m.shares(x)[2]
            kcar = s["k"] * (s["ch"] == 2)
            car_mc = kcar.sum() / s["k"].sum()
            car_se = (kcar - car_mc * s["k"]).std(ddof=1) / math.sqrt(N_DRAWS) / s["k"].mean()
            load = ((s["k"] * s["pt"])[:, None] * inst.inc[s["q"]]).sum(0) * tot / N_DRAWS
            C_mc = m.cdep * (inst.elen * np.maximum(m.dmin, load / m.cap)).sum()
            runs.append(dict(seed=seed, n=n, hub=hub, cost=cost, point=pt, x=[float(v) for v in x], N=N_DRAWS,
                             R_an=float(r_an), R_mc=float(r_mc), R_se=float(r_se), T_an=float(t_an), T_mc=float(t_mc), T_se=float(t_se),
                             car_an=float(car_an), car_mc=float(car_mc), car_se=float(car_se), C_an=float(an["C"]), C_mc=float(C_mc),
                             zR=float((r_mc - r_an) / r_se), zT=float((t_mc - t_an) / t_se), zCar=float((car_mc - car_an) / car_se), relC=float((C_mc - an["C"]) / an["C"])))
            print(seed, pt, "zR %.2f zT %.2f zCar %.2f relC %.4f" % (runs[-1]["zR"], runs[-1]["zT"], runs[-1]["zCar"], runs[-1]["relC"]), flush=True)
    # Zellen-Wahrscheinlichkeiten: ein Paar, alle sieben Fahrtenzahlen, 1e5 Ziehungen je Zelle
    cells = []
    for seed, n, hub, cost in MC_NETS:
        inst = make_instance(seed, n, hub)
        m = Model(inst, cost=cost)
        x = m.x0 * np.array([0.5 + 1.5 * master.u32() / 4294967296.0 for _ in range(2)])
        an = m.core(x)
        q = master.below(inst.P)
        for kc, kv in enumerate(C.K_VALS):
            z, l = inst.z[q], inst.l_km[q]
            V = np.array([-z * x[0] - C.LAM * inst.t_pt[q], -z * x[1] / kv - C.LAM * inst.t_pt[q], -(C.CAR_FIX / kv + C.CAR_KM * l) - C.LAM * inst.t_car[q] + m.y[q]])
            sd = master.next()
            G = np.stack([-np.log(-np.log(uniform_stream(sd + j * 0x1234567, 100_000))) for j in range(3)], axis=1)
            f = np.bincount((V[None, :] + G / m.mu).argmax(1), minlength=3) / 100_000
            pa = an["Pr"][q, kc]
            zs = (f - pa) / np.sqrt(pa * (1 - pa) / 100_000)
            cells.append(dict(seed=seed, pair=int(q), k=float(kv), p_an=[float(v) for v in pa], f_mc=[float(v) for v in f], z=[float(v) for v in zs]))
    # Kalibrierung: Auto-Anteil je OD-Paar beim heutigen Tarif gegen das Ziel
    cal = []
    for seed, n, hub, cost in MC_NETS:
        inst = make_instance(seed, n, hub)
        m = Model(inst, cost=cost)
        wk = m.pk * C.K_VALS
        pair = (m.core(m.x0)["Pr"][:, :, 2] * (wk / wk.sum())[None, :]).sum(1)
        cal.append(dict(seed=seed, max_abs_err=float(np.abs(pair - m.target).max())))
    zs_all = np.array([[r["zR"], r["zT"], r["zCar"]] for r in runs])
    zc = np.array([c["z"] for c in cells]).ravel()
    out = dict(draws=N_DRAWS, runs=runs, cells=cells, cal=cal,
               max_abs_z=float(np.abs(zs_all).max()), max_abs_z_cells=float(np.abs(zc).max()), max_relC=float(max(abs(r["relC"]) for r in runs)),
               max_rel_R=float(max(abs(r["R_mc"] - r["R_an"]) / r["R_an"] for r in runs)),
               std_z=[float(zs_all[:, j].std(ddof=1)) for j in range(3)], mean_z=[float(zs_all[:, j].mean()) for j in range(3)],
               bestanden=bool(np.abs(zs_all).max() <= 3.5))
    dump("mc", out)
    print({k: v for k, v in out.items() if k not in ("runs", "cells", "cal")})


# ------------------------------------------------------------------------------------------------ (b) Gradient
def part_gradient():
    master = SplitMix64(777)
    worst = {"R": 0.0, "T": 0.0, "C": 0.0}
    npts, nskip = 0, 0
    variants = [dict(), dict(mu=C.MU * 1.25), dict(mu=C.MU * 0.75), dict(trips_mean=25.0), dict(car_share=0.55)]
    detail = []
    for seed in (1, 2, 3, 8, 12, 21, 30):
        for cost, var in zip(("fix", "mittel", "var", "mittel", "fix"), variants):
            m = build(seed, cost, **var)
            for _ in range(3):
                x = m.x0 * np.array([0.2 + 2.8 * master.u32() / 4294967296.0 for _ in range(2)])
                g = m.core(x, grad=True)
                for key, gk in (("R", "dR"), ("T", "dT"), ("C", "dC")):
                    fd = np.zeros(2)
                    ok = np.ones(2, bool)
                    for j in range(2):
                        h = 1e-5 * max(abs(x[j]), 1e-3)
                        xp, xm = x.copy(), x.copy()
                        xp[j] += h
                        xm[j] -= h
                        rp, rm = m.core(xp), m.core(xm)
                        if key == "C" and not np.array_equal(rp["load"] / m.cap > m.dmin, rm["load"] / m.cap > m.dmin):
                            ok[j] = False
                            nskip += 1
                            continue
                        fd[j] = (rp[key] - rm[key]) / (2 * h)
                    if ok.any():
                        err = float(np.abs(g[gk][ok] - fd[ok]).max() / (np.abs(g[gk][ok]).max() + 1e-9))
                        worst[key] = max(worst[key], err)
                npts += 1
    out = dict(punkte=npts, kinkpunkte_uebersprungen=nskip, max_rel_fehler=worst, bestanden=bool(max(worst.values()) < 1e-5))
    dump("gradient", out)
    print(out)


# ------------------------------------------------------------------------------------------------ (c), (d) Löser
def part_loeser():
    rng = SplitMix64(31415926)
    costs = ("fix", "mittel", "var")
    rows = []
    for i in range(20):
        seed = 301 + i
        n = 20 + rng.below(11)
        hub = (0, 50, 100)[rng.below(3)]
        cost = costs[i % 3]
        mu_f = (0.75, 1.0, 1.25)[rng.below(3)]
        inst = make_instance(seed, n, hub)
        m = Model(inst, mu=C.MU * mu_f, cost=cost)
        S0 = m.S_today
        for kind in "RPD":
            ref = m.T0 if kind == "D" else m.R0
            loc = A.multistart(m, kind, C.N_STARTS, subsidy=S0, seed=seed)
            g = A.grid_search(m, kind, 41, S0)
            sols = [A.solve_local(m, kind, u0, S0) for u0 in A.start_points(2, 51, seed + 1000)]
            feas = [s for s in sols if s["feas"]]
            best = max(s["val"] for s in feas)
            gap_all = max((best - s["val"]) / ref for s in feas)
            rows.append(dict(seed=seed, n=n, hub=hub, cost=cost, mu_factor=mu_f, kind=kind, local=float(loc["val"]), grid=None if g is None else float(g["val"]),
                             diff=None if g is None else float((loc["val"] - g["val"]) / ref), start51_best=float(best), start51_max_gap=float(gap_all),
                             start51_infeasible=int(sum(1 for s in sols if not s["feas"])), start51_not_converged=int(sum(1 for s in sols if not s["ok"])),
                             local_vs_best51=float((loc["val"] - best) / ref)))
    diffs = [r["diff"] for r in rows if r["diff"] is not None]
    out = dict(netze=20, zeilen=rows, min_diff=float(min(diffs)), fehlende_gitter=int(sum(1 for r in rows if r["diff"] is None)),
               loeser_nie_schlechter=bool(min(diffs) >= -1e-6), max_gap_51_starts=float(max(r["start51_max_gap"] for r in rows)),
               max_abweichung_5_starts_von_bestem51=float(max(-r["local_vs_best51"] for r in rows)),
               starts_unzulaessig=int(sum(r["start51_infeasible"] for r in rows)), starts_nicht_konvergiert=int(sum(r["start51_not_converged"] for r in rows)),
               alle_starts_gleich=bool(max(r["start51_max_gap"] for r in rows) <= 1e-5))
    dump("loeser", out)
    print({k: v for k, v in out.items() if k != "zeilen"})


# ------------------------------------------------------------------------------------------------ (e), (f) Nullspalten, Wirkungslosigkeit
def flatten(obj, prefix=""):
    """Zahlen-Blätter eines Roh-Datensatzes als (Pfad, Wert). Listen von Dicts werden nach ihrem Schlüssel beschriftet; Vektoren nach Index."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, prefix + "." + k if prefix else k)
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            for i, e in enumerate(obj):
                if "f" in e and "kind" not in e:
                    lab = "f=%.1f" % e["f"]
                elif "tag" in e:
                    lab = "%s=%g" % (e["tag"], e["s"])
                elif "name" in e:
                    lab = "%s/%s/%g" % (e["name"], e["mode"], e["fac"])
                elif "kind" in e:
                    lab = "%s%d" % (e["kind"], e.get("rep", 0))
                else:
                    lab = "[]"
                yield from flatten(e, prefix + "." + lab)
        else:
            for i, e in enumerate(obj):
                yield from flatten(e, prefix + "[%d]" % i)


# Spalten, die per Definition null sind (Muster auf dem Spaltenpfad, Begründung)
NULL_RULES = (
    (r"^opt_0\.(x\[\d\]|R)$", "Nulltarif: Preise und Erlös sind 0"),
    (r"^base\.P$", "Bilanzgewinn des heutigen Tarifs ist mit S = Kosten - Erlös heute 0"),
    (r"^pareto\.f=0\.0\.(x\[\d\]|f)$", "Pareto f = 0: kein Erlösziel, Max-D = Nulltarif (und das Feld f selbst)"),
    (r"^sweep\.grid=1\.4\.x\[\d\]$", "Subvention 1,4 C0 liegt über S_null der Instanzen: Max-D ist der Nulltarif"),
    (r"^sweep\.null=1\.001\.x\[\d\]$", "S = 1,001 S_null: Nulltarif zulässig, Max-D ist der Nulltarif"),
    (r"^sweep\.null=0\.97\.x\[0\]$", "BEFUND, kein Fehler: knapp unter S_null erreicht zuerst der Einzelkartenpreis die Grenze 0, die Monatskarte bleibt positiv"),
    (r"^(R|P)\.rule_vals\[(0|100)\]$", "Praxisregel X = -1 (Nulltarif, Erlös 0) bzw. X = 0 (heutiger Tarif, Bilanzgewinn 0)"),
    (r"^P\.val_x0$", "Bilanzgewinn beim heutigen Tarif ist per Definition 0"),
    (r"^A\.[RP]\d\.rep$", "Feld rep des ersten Wiederholungslaufs (Zähler 0), keine Messspalte"),
)


def part_nullspalten():
    cols = {}
    for fn in ("q1", "q2", "q3", "q5", "extra"):
        for line in open(os.path.join(RAW, fn + ".jsonl"), encoding="utf-8"):
            r = json.loads(line)
            cost = r.get("cost", "")
            for path, v in flatten({k: v for k, v in r.items() if k not in ("secs", "seed", "cost")}):
                cols.setdefault((fn, cost, path), []).append(v)
    n_cols = len(cols)
    zero = []
    for (fn, cost, path), vals in cols.items():
        if len(vals) and max(abs(v) for v in vals) <= 1e-9:
            zero.append(dict(datei=fn, kosten=cost, spalte=path, n=len(vals)))
    # Klassifikation
    for z in zero:
        p = z["spalte"]
        base = p.split(".", 1)[0]
        z["grund"] = None
        for pat, why in NULL_RULES:
            if re.search(pat, p):
                z["grund"] = why
                break
    out = dict(spalten_geprueft=n_cols, null_spalten=zero, unerwartet=[z for z in zero if z["grund"] is None])
    dump("nullspalten_roh", out)
    print("Spalten:", n_cols, "Nullspalten:", len(zero))
    for z in zero:
        print(" ", z["datei"], z["kosten"], z["spalte"], z["n"], z["grund"])


def part_wirkungslos():
    rows = [json.loads(l) for l in open(os.path.join(RAW, "q3.jsonl"), encoding="utf-8")]
    import analyse as AN
    an = AN.agg_annahmen(sorted(rows, key=lambda r: r["seed"]))
    out = [dict(name=e["name"], mode=e["mode"], fac=e["fac"], max_rel_aenderung=e["max_rel_aenderung"], wirkungslos=e["wirkungslos"]) for e in an["eintraege"]]
    dump("wirkungslos", out)
    for e in out:
        print(e)


# ------------------------------------------------------------------------------------------------ (g) Determinismus
def part_determinismus(other):
    res = {}
    worst_all, exact_all, total_all = 0.0, 0, 0
    for fn in ("q1", "q2", "q3", "q5", "extra"):
        a = [json.loads(l) for l in open(os.path.join(RAW, fn + ".jsonl"), encoding="utf-8")]
        b = [json.loads(l) for l in open(os.path.join(other, fn + ".jsonl"), encoding="utf-8")]
        key = lambda r: (r["seed"], r.get("cost", ""))
        a.sort(key=key)
        b.sort(key=key)
        assert [key(r) for r in a] == [key(r) for r in b], fn
        worst, exact, total = 0.0, 0, 0
        for ra, rb in zip(a, b):
            fa = dict(flatten({k: v for k, v in ra.items() if k != "secs"}))
            fb = dict(flatten({k: v for k, v in rb.items() if k != "secs"}))
            assert fa.keys() == fb.keys(), fn
            for p, va in fa.items():
                if p.split(".")[-1].startswith("t_"):
                    continue          # Zeitmessungen
                vb = fb[p]
                total += 1
                if va == vb:
                    exact += 1
                    continue
                worst = max(worst, abs(va - vb) / max(abs(va), abs(vb), 1e-9))
        res[fn] = dict(zahlen=total, bitgleich=exact, max_rel_abweichung=worst)
        worst_all, exact_all, total_all = max(worst_all, worst), exact_all + exact, total_all + total
        print(fn, res[fn])
    out = dict(dateien=res, zahlen=total_all, bitgleich=exact_all, max_rel_abweichung=worst_all, bestanden=bool(worst_all <= 1e-9))
    dump("determinismus", out)
    print(out)


# ------------------------------------------------------------------------------------------------ Rechenzeiten
def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def pstat(a):
    a = np.array(a, float)
    return dict(med=float(np.median(a)), p90=float(np.percentile(a, 90)), max=float(a.max()), min=float(a.min()), n=int(len(a)))


def part_timing(reps=10):
    # Aufwärmlauf (Importe, erste BLAS-Aufrufe)
    E.analyse(E.Settings(stops=20, seed=1))
    out = dict(rechner=platform.processor(), python=platform.python_version(), numpy=np.__version__, wiederholungen=reps,
               hinweis="Median/P90/Max über 10 Zufallsinstanzen (Zufalls-Seeds 1..10, Nachfragekonzentration mittel); ein Prozess; andere Sitzungen laufen ggf. parallel; Streamlit Cloud nicht gemessen", n={})
    for n in (20, 25, 30):
        rec = {k: [] for k in ("instanz", "modell", "solve_goals", "analyse", "pareto", "subventionskurve", "preisebene31", "annahmen_tabelle", "alles")}
        for seed in range(1, reps + 1):
            s = E.Settings(stops=n, seed=seed)
            inst, t_inst = timed(lambda: make_instance(seed, n, 50))
            m, t_mod = timed(lambda: Model(inst, cost="mittel"))
            _, t_goals = timed(lambda: A.solve_goals(m, m.S_today, seed=seed))
            a, t_an = timed(lambda: E.analyse(s, inst))
            _, t_par = timed(lambda: E.pareto_curve(a))
            _, t_sub = timed(lambda: E.subsidy_curve(a))
            _, t_pl = timed(lambda: E.price_plane(a, 31))
            _, t_as = timed(lambda: E.assumption_table(s))
            for k, v in (("instanz", t_inst), ("modell", t_mod), ("solve_goals", t_goals), ("analyse", t_an), ("pareto", t_par), ("subventionskurve", t_sub),
                         ("preisebene31", t_pl), ("annahmen_tabelle", t_as), ("alles", t_an + t_par + t_sub + t_pl)):
                rec[k].append(v)
        out["n"][str(n)] = {k: pstat(v) for k, v in rec.items()}
        print(n, {k: round(out["n"][str(n)][k]["med"], 3) for k in rec}, flush=True)
    dump("timing", out)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part == "rng":
        part_rng()
    elif part == "mc":
        part_mc()
    elif part == "gradient":
        part_gradient()
    elif part == "loeser":
        part_loeser()
    elif part == "nullspalten":
        part_nullspalten()
        part_wirkungslos()
    elif part == "determinismus":
        part_determinismus(sys.argv[2])
    elif part == "timing":
        part_timing()
    else:
        print(__doc__)
