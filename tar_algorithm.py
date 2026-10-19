"""Löser: die drei Tarifziele (Max-R, Max-P, Max-D), Mehrfachstart mit SLSQP, Gitter, Praxisregel, Pareto- und Subventions-Kurve.

Variablen u = x / (2 · x0) mit Grenzen 0 bis 2,5 (also Preise bis 5 · x0), damit SLSQP mit Größenordnung 0,1 bis 1 rechnet. Zielfunktionen werden durch
R0 bzw. T0 geteilt. Zufall nur aus SplitMix64 (feste Startpunkte)."""

import numpy as np
from scipy.optimize import minimize

import tar_constants as C
from tar_rng import SplitMix64


def scales(m):
    return 2.0 * m.x0


def value(m, kind, x, subsidy=0.0):
    r = m.core(x)
    return {"R": r["R"], "P": r["R"] - r["C"] + subsidy, "D": r["T"]}[kind]


def solve_local(m, kind, u0, subsidy=0.0, rev_rhs=None, maxiter=200):
    """SLSQP mit analytischem Gradienten. kind R: Erlös; P: Erlös − Kosten + Subvention; D: Fahrten unter Erlös − Kosten + Subvention ≥ 0
    (rev_rhs None) bzw. unter Erlös ≥ rev_rhs. Gibt u, x, Zielwert, Zulässigkeit zurück."""
    xs = scales(m)
    R0, T0 = m.R0, m.T0

    def fun(u):
        r = m.core(u * xs, grad=True)
        if kind == "R":
            return -r["R"] / R0, -r["dR"] * xs / R0
        if kind == "P":
            return -(r["R"] - r["C"] + subsidy) / R0, -(r["dR"] - r["dC"]) * xs / R0
        return -r["T"] / T0, -r["dT"] * xs / T0

    cons = []
    if kind == "D":
        def cf(u):
            r = m.core(u * xs, grad=True)
            if rev_rhs is None:
                return (r["R"] - r["C"] + subsidy) / R0, (r["dR"] - r["dC"]) * xs / R0
            return (r["R"] - rev_rhs) / R0, r["dR"] * xs / R0
        cons = [dict(type="ineq", fun=lambda u: cf(u)[0], jac=lambda u: cf(u)[1])]
    res = minimize(fun, u0, jac=True, bounds=[(0.0, C.UB)] * m.J, constraints=cons, method="SLSQP", options=dict(maxiter=maxiter, ftol=1e-10))
    x = res.x * xs
    r = m.core(x)
    if kind == "D":
        con = (r["R"] - r["C"] + subsidy) if rev_rhs is None else (r["R"] - rev_rhs)
        feas = con >= -1e-6 * R0
    else:
        feas = True
    return dict(u=res.x, x=x, val=float({"R": r["R"], "P": r["R"] - r["C"] + subsidy, "D": r["T"]}[kind]), feas=bool(feas), ok=bool(res.success))


def start_points(J, n, seed):
    """Der Mittelpunkt u = 0,5 und n − 1 weitere Startpunkte aus SplitMix64 (gleichverteilt in 0,05 bis 2,0)."""
    rng = SplitMix64(int(seed) + (7 << 40))
    pts = [np.full(J, 0.5)]
    for _ in range(n - 1):
        pts.append(np.array([C.START_LO + (C.START_HI - C.START_LO) * rng.u32() / 4294967296.0 for _ in range(J)]))
    return pts


def multistart(m, kind, n_starts=C.N_STARTS, subsidy=0.0, rev_rhs=None, seed=0):
    sols = [solve_local(m, kind, u0, subsidy, rev_rhs) for u0 in start_points(m.J, n_starts, seed)]
    feas = [s for s in sols if s["feas"]]
    return max(feas, key=lambda s: s["val"]) if feas else None


def grid_u(npts):
    ax = np.linspace(0.0, C.UB, npts)
    a, b = np.meshgrid(ax, ax, indexing="ij")
    return np.stack([a.ravel(), b.ravel()], axis=1)


def grid_search(m, kind, npts, subsidy=0.0, rev_rhs=None):
    """Gitter über u in [0, 2,5]²; Max-D nur über zulässige Gitterpunkte."""
    xs = scales(m)
    best, bu = -np.inf, None
    for u in grid_u(npts):
        r = m.core(u * xs)
        if kind == "R":
            v = r["R"]
        elif kind == "P":
            v = r["R"] - r["C"] + subsidy
        else:
            con = (r["R"] - r["C"] + subsidy) if rev_rhs is None else (r["R"] - rev_rhs)
            if con < 0:
                continue
            v = r["T"]
        if v > best:
            best, bu = v, u
    return None if bu is None else dict(u=bu, x=bu * xs, val=float(best))


def uniform_rule(m, kind, factors, subsidy=0.0):
    """Praxisregel: alle Preise mit demselben Faktor (1 + X) ändern. Gibt je Faktor den Zielwert (NaN: unzulässig)."""
    vals = np.full(len(factors), np.nan)
    for i, f in enumerate(factors):
        vals[i] = _val_or_nan(m, kind, m.x0 * f, subsidy)
    return vals


def _val_or_nan(m, kind, x, subsidy):
    r = m.core(x)
    if kind == "R":
        return r["R"]
    if kind == "P":
        return r["R"] - r["C"] + subsidy
    return r["T"] if r["R"] - r["C"] + subsidy >= 0 else np.nan


def best_uniform(m, kind, subsidy=0.0, lo=-1.0, hi=3.0, step=0.05):
    """Fairster Faktor der Praxisregel: Raster von X = lo bis hi in Schritten von `step` (X = −1 ist Nulltarif)."""
    xs = np.round(np.arange(lo, hi + 1e-9, step), 10)
    vals = uniform_rule(m, kind, 1 + xs, subsidy)
    if np.all(np.isnan(vals)):
        return None
    i = int(np.nanargmax(vals))
    return dict(X=float(xs[i]), val=float(vals[i]))


def solve_goals(m, subsidy, n_starts=C.N_STARTS, seed=0):
    """Die drei Ziele für eine Subvention: Max-R, Max-P und Max-D (Bilanz Erlös − Kosten + Subvention ≥ 0), jeweils mit Kennzahlen."""
    out = {}
    for kind in ("R", "P", "D"):
        sol = multistart(m, kind, n_starts, subsidy, seed=seed)
        out[kind] = None if sol is None else dict(sol, kpi=m.kpi(sol["x"], subsidy))
    return out


def minimum_subsidy(m, hi=3.0):
    """Kleinste Subvention, bei der Max-D zulässig ist (Bisektion über die Subvention in Einheiten der Betriebskosten C0).

    Zulässigkeit heißt: Es gibt einen Tarif mit Erlös − Kosten + S ≥ 0, das ist bei S ≥ Kosten − Erlösmaximum der Fall."""
    r_max = multistart(m, "P", C.N_STARTS)
    return float(-r_max["val"])   # P-Maximum ohne Subvention: Erlös − Kosten; S_min = max(0, −dieses Maximum)


def null_tariff_subsidy(m):
    """Subvention, ab der der Nulltarif die Bilanz hält: S_null = Kosten bei Nulltarif."""
    return float(m.core(np.zeros(2))["C"])


def pareto(m, fractions, subsidy=0.0, seed=0):
    """Max-D unter Erlös ≥ f · Erlösmaximum für wachsendes f: der Zielkonflikt zwischen Erlös und Fahrten."""
    r_max = multistart(m, "R", C.N_STARTS, seed=seed)
    pts = []
    for f in fractions:
        if f >= 1.0:
            x = r_max["x"]
        else:
            sol = multistart(m, "D", C.N_STARTS, rev_rhs=f * r_max["val"], seed=seed)
            x = sol["x"] if sol is not None else None
        if x is None:
            continue
        r = m.core(x)
        pts.append(dict(f=float(f), x=tuple(float(v) for v in x), R=float(r["R"]), T=float(r["T"]), C=float(r["C"])))
    return pts


def subsidy_curve(m, subsidies, seed=0):
    """Max-D mit Kostenbilanz für wachsende Subvention (in Geldeinheiten): Fahrten, Erlös und Preise."""
    pts = []
    for s in subsidies:
        sol = multistart(m, "D", C.N_STARTS, subsidy=s, seed=seed)
        if sol is None:
            pts.append(dict(S=float(s), feasible=False))
            continue
        r = m.core(sol["x"])
        pts.append(dict(S=float(s), feasible=True, x=tuple(float(v) for v in sol["x"]), R=float(r["R"]), T=float(r["T"]), C=float(r["C"])))
    return pts
