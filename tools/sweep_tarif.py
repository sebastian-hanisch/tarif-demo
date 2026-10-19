"""Tarif-Sweeps (Port der Vorab-Messreihe `oepnv-planung/messreihe_tarif/run_q*.py` auf den ganzzahligen Generator und den eigenen Kern der Demo).

Nur der Zonentarif (der Entfernungstarif entfällt bewusst). Netze wie in der Vorab-Reihe: Zufalls-Seeds 1 bis 40, n_stops = 20 + (Zufalls-Seed - 1) % 11,
Nachfragekonzentration hub_pct = 50. Zufall nur aus SplitMix64 (`tar_rng`); Startpunkte der Löser aus `tar_algorithm.start_points`.

Rohdaten (eine JSON-Zeile je Auftrag):
    q1.jsonl     Zielkonflikt: Max-R/Max-P/Max-D (20 Starts), Kreuztabelle, Nulltarif, Pareto (Kern: `pareto`), Subventionskurve (Kern: `subsidy_curve`),
                 Modell-Kennzahlen; je Zufalls-Seed und Kostenvariante fix/mittel/var
    q2.jsonl     Verfahrensvergleich (Kostenvariante mittel): lokal 10 Starts (zwei unabhängige Läufe), lokal 50 Starts, Gitter 21x21 und 41x41, Praxisregel
    q3.jsonl     Annahmen +-25 % (mu, lam, car_fix, car_km, car_share, trips_mean, cdep, dmin; Neukalibrierung) und y fest (lam, car_km, car_fix)
    q5.jsonl     Kannibalisierung: Monatskarte auf 90/80/70 % des Preises, IIA-Identität
    extra.jsonl  Praxisregel-Lücke bei zufälliger Ausgangs-Tarifstruktur

Aufruf:  python -B tools/sweep_tarif.py [Ausgabeordner] [Prozesse] [probe]
Standard: Umgebungsvariable TAR_RAW_DIR, sonst ../oepnv-planung/messreihe_tarif_ap0 (relativ zum Projekte-Ordner), 6 Prozesse.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.dirname(HERE)
sys.path.insert(0, DEMO)

import numpy as np  # noqa: E402

import tar_algorithm as A  # noqa: E402
import tar_constants as C  # noqa: E402
from tar_model import Model  # noqa: E402
from tar_rng import SplitMix64  # noqa: E402
from tar_scenario import make_instance  # noqa: E402

OUT_DIR = os.environ.get("TAR_RAW_DIR") or os.path.join(DEMO, "..", "oepnv-planung", "messreihe_tarif_ap0")
SEEDS = list(range(1, 41))
COSTS = ("fix", "mittel", "var")
HUB = 50
S_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4)       # Subvention in Vielfachen der heutigen Betriebskosten C0
NULL_MULT = (0.9, 0.97, 1.001)                                   # Vielfache von S_null
PARETO_F = tuple(round(0.1 * i, 1) for i in range(11))
XS_RULE = np.round(np.arange(-1.0, 3.0001, 0.01), 4)             # Praxisregel: alle Preise mit 1 + X (X = -1 ist Nulltarif)
XS_EXTRA = np.round(np.arange(-0.9, 3.0001, 0.01), 4)
CUTS = (0.9, 0.8, 0.7)
FILES = {"q1": "q1.jsonl", "q2": "q2.jsonl", "q3": "q3.jsonl", "q5": "q5.jsonl", "extra": "extra.jsonl"}


def n_of(seed):
    return 20 + (seed - 1) % 11


def build(seed, cost="mittel", **kw):
    inst = make_instance(seed, n_of(seed), HUB)
    return Model(inst, cost=cost, **kw)


def retune_cost(m, cdep, dmin):
    """Kostenparameter nachträglich setzen (der Kern kennt nur die drei Varianten); heutige Kennzahlen neu rechnen. y hängt nicht von den Kosten ab."""
    m.cdep, m.dmin = float(cdep), float(dmin)
    r = m.core(m.x0)
    m.R0, m.T0, m.C0 = r["R"], r["T"], r["C"]
    m.S_today = m.C0 - m.R0
    return m


def kp(m, x, S):
    """Kennzahlen eines Tarifs x bei Subvention S: Preise, Erlös, Kosten, Fahrten, Bilanz, Modal Split."""
    x = np.asarray(x, float)
    r = m.core(x)
    return dict(x=[float(v) for v in x], R=float(r["R"]), C=float(r["C"]), T=float(r["T"]), P=float(r["R"] - r["C"] + S),
                share=[float(v) for v in m.shares(x)])


def level(m, x):
    return float(np.mean(np.asarray(x, float) / m.x0))


def start_list(m, n_random, seed):
    """n_random Zufallsstarts (ohne den Mittelpunkt) aus SplitMix64."""
    return A.start_points(m.J, n_random + 1, seed)[1:]


def multistart_random(m, kind, n, seed, subsidy=0.0, rev_rhs=None):
    """n unabhängige Zufallsstarts (ohne Mittelpunkt); gibt alle Lösungen zurück."""
    return [A.solve_local(m, kind, u0, subsidy, rev_rhs) for u0 in start_list(m, n, seed)]


def best_of(sols):
    feas = [s for s in sols if s["feas"]]
    return max(feas, key=lambda s: s["val"]) if feas else None


def model_facts(m, inst):
    """Modell-Kennzahlen eines Netzes (Zonentarif, heute)."""
    sh = m.shares(m.x0)
    r = m.core(m.x0)
    bound = r["load"] / m.cap > m.dmin
    return dict(n=int(inst.n), E=int(inst.E), tpt=float(np.average(inst.t_pt, weights=inst.persons)), l=float(np.average(inst.l_km, weights=inst.persons)),
                elasticity=float(m.elasticity()), kdg=float(m.R0 / m.C0), sS=float(sh[0]), sM=float(sh[1]), sC=float(sh[2]),
                kmean=float((m.pk * C.K_VALS).sum()), outer=float(np.average(inst.z == 2, weights=inst.persons)),
                bound=float(bound.mean()), fixshare=float((inst.elen * np.where(bound, 0, m.dmin)).sum() * m.cdep / m.C0))


# ================================================================================================ Frage 1
def q1_task(seed, cost):
    t0 = time.time()
    m = build(seed, cost)
    inst = m.inst
    S0 = m.S_today
    out = dict(seed=seed, cost=cost, R0=float(m.R0), C0=float(m.C0), T0=float(m.T0), S_today=float(S0), x0=[float(v) for v in m.x0],
               share0=[float(v) for v in m.shares(m.x0)], facts=model_facts(m, inst))
    sol = {}
    for kind in "RPD":
        b = A.multistart(m, kind, 20, subsidy=S0, seed=seed)
        sol[kind] = b
        out["opt_" + kind] = dict(kp(m, b["x"], S0), u=[float(v) for v in b["u"]], val=float(b["val"]))
    g5 = A.solve_goals(m, S0, seed=seed)
    out["goals5"] = {k: dict(x=[float(v) for v in g["x"]], val=float(g["val"])) for k, g in g5.items()}
    out["opt_0"] = kp(m, np.zeros(m.J), S0)
    out["base"] = kp(m, m.x0, S0)
    cross = {}
    for pk in "RPD":
        d = kp(m, sol[pk]["x"], S0)
        d["feasible_D"] = bool(d["P"] >= -1e-6 * m.R0)
        cross[pk] = d
    out["cross"] = cross
    # Pareto: Fahrten maximal bei Erlös >= f * R_max (Kern)
    par = []
    for p in A.pareto(m, PARETO_F, seed=seed):
        d = kp(m, p["x"], S0)
        d["f"] = p["f"]
        par.append(d)
    out["pareto"] = par
    out["pareto_n"] = len(par)
    # Subventionskurve (Max-D mit Kostenbilanz): heute, 0,2 ... 1,4 mal C0, dazu 0,9 / 0,97 / 1,001 mal S_null
    s_null = A.null_tariff_subsidy(m)
    out["Snull_over_C0"] = float(s_null / m.C0)
    pmax = sol["P"]["val"] - S0                    # max (Erlös - Kosten), 20 Starts
    out["Smin_over_C0"] = float(-pmax / m.C0)
    out["Smin_core_over_C0"] = float(A.minimum_subsidy(m) / m.C0)
    req = [("heute", S0 / m.C0, S0)] + [("grid", s, s * m.C0) for s in S_GRID] + [("null", f, f * s_null) for f in NULL_MULT]
    pts = A.subsidy_curve(m, [r[2] for r in req], seed=seed)
    sw = []
    for (tag, s, S), p in zip(req, pts):
        row = dict(tag=tag, s=float(s), S_over_C0=float(S / m.C0), feasible=bool(p["feasible"]))
        if p["feasible"]:
            row.update(kp(m, p["x"], S))
        sw.append(row)
    out["sweep"] = sw
    out["secs"] = time.time() - t0
    return out


# ================================================================================================ Frage 2
def q2_task(seed):
    t0 = time.time()
    m = build(seed, "mittel")
    S0 = m.S_today
    out = dict(seed=seed, cost="mittel", R0=float(m.R0), T0=float(m.T0), C0=float(m.C0), S0=float(S0))
    for kind in "RPD":
        ref = m.T0 if kind == "D" else m.R0
        rec = {}
        t = time.perf_counter()
        a = best_of(multistart_random(m, kind, 10, seed * 1000 + 1, S0))
        rec["t_ms10"] = time.perf_counter() - t
        b = best_of(multistart_random(m, kind, 10, seed * 1000 + 2, S0))
        rec["runA"] = None if a is None else float(a["val"])
        rec["runB"] = None if b is None else float(b["val"])
        sols = multistart_random(m, kind, 50, seed * 1000 + 3, S0)
        rec["ms50"] = [dict(u=[float(v) for v in s["u"]], val=float(s["val"]), feas=bool(s["feas"]), ok=bool(s["ok"])) for s in sols]
        t = time.perf_counter()
        g1 = A.grid_search(m, kind, 21, S0)
        rec["t_grid_coarse"] = time.perf_counter() - t
        rec["grid_coarse"] = None if g1 is None else dict(val=float(g1["val"]), u=[float(v) for v in g1["u"]])
        t = time.perf_counter()
        g2 = A.grid_search(m, kind, 41, S0)
        rec["t_grid_med"] = time.perf_counter() - t
        rec["grid_med"] = None if g2 is None else dict(val=float(g2["val"]), u=[float(v) for v in g2["u"]], nev=41 * 41)
        v = A.uniform_rule(m, kind, 1.0 + XS_RULE, S0)
        rec["rule_vals"] = [None if np.isnan(z) else float(z) for z in v]
        rec["ref"] = float(ref)
        rec["val_x0"] = float(A.value(m, kind, m.x0, S0))
        out[kind] = rec
    out["secs"] = time.time() - t0
    return out


# ================================================================================================ Frage 3
def perturbations():
    """(Name, Modus, Faktor): Neukalibrierung (`recal`, der Kern kalibriert bei mu/lam/car_*/trips_mean selbst neu) und y fest (`fest`)."""
    out = []
    for fac in (0.75, 1.25):
        for name in ("mu", "lam", "car_fix", "car_km", "car_share", "k_mean", "cdep", "dmin"):
            out.append((name, "recal", fac))
        for name in ("lam", "car_km", "car_fix"):
            out.append((name, "fest", fac))
    return out


def perturbed_model(inst, name, mode, fac, y0, kmean0):
    kw = dict(cost="mittel")
    if mode == "fest":
        kw["y"] = y0
    if name == "mu":
        kw["mu"] = C.MU * fac
    elif name == "lam":
        kw["lam"] = C.LAM * fac
    elif name == "car_fix":
        kw["car_fix"] = C.CAR_FIX * fac
    elif name == "car_km":
        kw["car_km"] = C.CAR_KM * fac
    elif name == "car_share":
        kw["car_share"] = C.CAR_SHARE * fac
    elif name == "k_mean":
        kw["trips_mean"] = kmean0 * fac
    mp = Model(inst, **kw)
    if name in ("cdep", "dmin"):
        base = C.COST_VARIANTS["mittel"]
        retune_cost(mp, base["cdep"] * (fac if name == "cdep" else 1.0), base["dmin"] * (fac if name == "dmin" else 1.0))
    return mp


def summarize_goals(m, goals, S):
    out = {}
    for k in "RPD":
        g = goals[k]
        if g is None:
            out[k] = None
            continue
        d = kp(m, g["x"], S)
        d.update(level=level(m, g["x"]), u=[float(v) for v in g["u"]])
        out[k] = d
    return out


def q3_task(seed):
    t0 = time.time()
    m0 = build(seed, "mittel")
    inst = m0.inst
    S0 = m0.S_today
    nom = A.solve_goals(m0, S0, seed=seed)
    out = dict(seed=seed, S0=float(S0), R0=float(m0.R0), T0=float(m0.T0), C0=float(m0.C0), nominal=summarize_goals(m0, nom, S0), pert=[])
    kmean0 = float((m0.pk * C.K_VALS).sum())
    for name, mode, fac in perturbations():
        mp = perturbed_model(inst, name, mode, fac, m0.y, kmean0)
        Sp = (mp.C0 - mp.R0) if name in ("cdep", "dmin") else S0     # Kostenstörung: heutige Subvention der gestörten Welt
        sol = A.solve_goals(mp, Sp, seed=seed)
        rec = dict(name=name, mode=mode, fac=fac, Sp=float(Sp), R0p=float(mp.R0), T0p=float(mp.T0), C0p=float(mp.C0), opt=summarize_goals(mp, sol, Sp))
        ev = {}
        for k in "RPD":
            if nom[k] is None:
                continue
            r = mp.core(nom[k]["x"])
            ev[k] = dict(R=float(r["R"]), C=float(r["C"]), T=float(r["T"]), P=float(r["R"] - r["C"] + Sp), feasD=bool(r["R"] - r["C"] + Sp >= -1e-6 * m0.R0))
        rec["nom_in_pert"] = ev
        out["pert"].append(rec)
    out["secs"] = time.time() - t0
    return out


# ================================================================================================ Frage 5
def split_rev(m, x):
    r = m.core(x)
    rv = m.w[..., None] * r["Pr"][:, :, :2] * r["pr"]
    return float(rv[..., 0].sum()), float(rv[..., 1].sum())


def person_shares(m, x):
    Pr = m.core(x)["Pr"]
    return (m.w[..., None] * Pr).sum((0, 1)) / m.w.sum()


def q5_task(seed):
    t0 = time.time()
    m = build(seed, "mittel")
    r0 = m.core(m.x0)
    Rs0, Rm0 = split_rev(m, m.x0)
    d = dict(R=float(r0["R"]), T=float(r0["T"]), Rs=Rs0, Rm=Rm0, pshare=[float(v) for v in person_shares(m, m.x0)], share=[float(v) for v in m.shares(m.x0)], cuts=[])
    for c in CUTS:
        x = m.x0.copy()
        x[1] *= c
        r1 = m.core(x)
        Rs1, Rm1 = split_rev(m, x)
        d["cuts"].append(dict(c=c, R=float(r1["R"]), T=float(r1["T"]), Rs=Rs1, Rm=Rm1, pshare=[float(v) for v in person_shares(m, x)],
                              share=[float(v) for v in m.shares(x)]))
    # IIA-Identität: dP_S / dP_M = -P_S / (1 - P_M) je Zelle bei kleiner Preisänderung der Monatskarte
    x = m.x0.copy()
    P0 = m.core(x)["Pr"]
    x[1] *= 0.999
    P1 = m.core(x)["Pr"]
    dPM = P1[:, :, 1] - P0[:, :, 1]
    dPS = P1[:, :, 0] - P0[:, :, 0]
    pred = -P0[:, :, 0] / (1 - P0[:, :, 1])
    return dict(seed=seed, q5=d, iia_max_dev=float(np.abs(dPS / dPM - pred).max()), secs=time.time() - t0)


# ================================================================================================ Zusatz: Praxisregel-Lücke
def extra_task(seed):
    t0 = time.time()
    m = build(seed, "mittel")
    rng = SplitMix64(977 * seed + (3 << 40))
    best = {k: float(A.multistart(m, k, 6, subsidy=0.0, seed=seed)["val"]) for k in "RP"}
    rows = []
    for rep in range(5):
        base = m.x0 * np.exp(np.array([-0.35 + 0.7 * rng.u32() / 4294967296.0 for _ in range(m.J)]))
        R = np.empty(len(XS_EXTRA))
        P = np.empty(len(XS_EXTRA))
        for i, X in enumerate(XS_EXTRA):
            r = m.core(base * (1.0 + X))
            R[i], P[i] = r["R"], r["R"] - r["C"]
        r0 = m.core(base)
        at0 = {"R": r0["R"], "P": r0["R"] - r0["C"]}
        for kind, v in (("R", R), ("P", P)):
            rows.append(dict(kind=kind, rep=rep, gap=float((best[kind] - v.max()) / m.R0), Xbest=float(XS_EXTRA[int(v.argmax())]),
                             gap_at_x0=float((best[kind] - at0[kind]) / m.R0)))
    return dict(seed=seed, A=rows, secs=time.time() - t0)


# ================================================================================================ Steuerung
def task(job):
    kind = job[0]
    if kind == "q1":
        return kind, q1_task(job[1], job[2])
    if kind == "q2":
        return kind, q2_task(job[1])
    if kind == "q3":
        return kind, q3_task(job[1])
    if kind == "q5":
        return kind, q5_task(job[1])
    return kind, extra_task(job[1])


def jobs_all(probe=False):
    seeds = SEEDS[:2] if probe else SEEDS
    jobs = [("q1", s, c) for s in seeds for c in COSTS]
    jobs += [("q3", s) for s in seeds] + [("q2", s) for s in seeds] + [("extra", s) for s in seeds] + [("q5", s) for s in seeds]
    weight = {"q3": 0, "q1": 1, "q2": 2, "extra": 3, "q5": 4}
    jobs.sort(key=lambda j: (weight[j[0]], -j[1]))              # teure Aufträge zuerst
    return jobs


def main(out_dir=OUT_DIR, procs=6, probe=False):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    jobs = jobs_all(probe)
    print("Aufträge", len(jobs), "Prozesse", procs, "->", os.path.abspath(out_dir), flush=True)
    t0 = time.time()
    fhs = {k: open(os.path.join(out_dir, ("probe_" if probe else "") + fn), "w", encoding="utf-8") for k, fn in FILES.items()}
    done = 0
    try:
        with Pool(procs) as pool:
            for kind, row in pool.imap_unordered(task, jobs, chunksize=1):
                fhs[kind].write(json.dumps(row, ensure_ascii=False) + "\n")
                fhs[kind].flush()
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(done, "/", len(jobs), "%.0fs" % (time.time() - t0), flush=True)
    finally:
        for fh in fhs.values():
            fh.close()
    wall = time.time() - t0
    print("fertig %.0fs" % wall, flush=True)
    if not probe:
        with open(os.path.join(out_dir, "sweep_tarif_wallclock.txt"), "w", encoding="utf-8") as fh:
            fh.write("%.1f\n" % wall)


if __name__ == "__main__":
    od = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("probe",) else OUT_DIR
    npr = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 6
    main(od, npr, probe=("probe" in sys.argv[1:]))
