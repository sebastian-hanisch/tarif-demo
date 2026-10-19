"""Modell: Logit-Wahl zwischen Einzelkarte, Monatskarte und Auto, Kalibrierung auf den Ausgangs-Modal-Split, Erlös, Fahrten, Betriebskosten.

Zonentarif mit zwei Preisen x = (x_s, x_m): Preis der Einzelkarte je Fahrt und Zone, Preis der Monatskarte je Zone. Eine Person mit k Fahrten im
Monat wählt eine Alternative für alle k Fahrten. Nutzen je Fahrt: V = −Preis/k − Zeitwert · Fahrzeit (ÖV) bzw. −(Fixkosten/k + km-Kosten · Entfernung)
− Zeitwert · Fahrzeit + y (Auto); P(a) = exp(μ V_a) / Σ_b exp(μ V_b). Der Auto-Zuschlag y je OD-Paar wird so gewählt, dass beim heutigen Tarif der
Auto-Anteil der Fahrten dem Ausgangs-Modal-Split entspricht. Betrieb: je Kante d = max(d_min, Last / Kapazität) Abfahrten, Kosten = c_dep · Σ Länge · d.
"""

import numpy as np

import tar_constants as C
from tar_scenario import Instance

K_VALS = C.K_VALS


def tilt_pk(pk, target_mean):
    """Exponentielle Kippung der Fahrtenverteilung auf ein Zielmittel der Fahrtenzahl (Bisektion)."""
    sd = np.sqrt((pk * K_VALS ** 2).sum() - (pk * K_VALS).sum() ** 2)
    lo, hi = -20.0, 20.0
    q = pk
    for _ in range(80):
        mid = (lo + hi) / 2
        q = pk * np.exp(mid * (K_VALS - (pk * K_VALS).sum()) / sd)
        q = q / q.sum()
        if (q * K_VALS).sum() < target_mean:
            lo = mid
        else:
            hi = mid
    return q


class Model:
    def __init__(self, inst: Instance, *, mu=C.MU, lam=C.LAM, car_fix=C.CAR_FIX, car_km=C.CAR_KM, car_share=C.CAR_SHARE,
                 cost="mittel", cap=C.CAP, trips_mean=None, y=None):
        self.inst = inst
        self.J = 2
        self.x0 = np.array(C.X0, dtype=float)
        self.mu, self.cap = mu, cap
        self.cdep, self.dmin = C.COST_VARIANTS[cost]["cdep"], C.COST_VARIANTS[cost]["dmin"]
        P = inst.P
        self.P, self.K = P, len(K_VALS)
        pk = tilt_pk(inst.pk, trips_mean) if trips_mean is not None else inst.pk
        self.pk = pk
        self.kk = K_VALS[None, :]
        self.w = inst.persons[:, None] * pk[None, :]
        z = inst.z[:, None]
        Cm = np.zeros((P, self.K, 2, 2))
        Cm[:, :, 0, 0] = self.kk * z
        Cm[:, :, 1, 1] = z
        self.Cmat = Cm
        self.Cflat = Cm.reshape(P * self.K * 2, 2)
        self.base_pt = -lam * inst.t_pt[:, None]
        l = inst.l_km[:, None]
        self.car_const = -(car_fix / self.kk + car_km * l) - lam * inst.t_car[:, None]
        tgt = car_share + C.CAR_SHARE_SLOPE * inst.share_unit * (car_share / 0.45)
        self.target = np.clip(tgt, *C.CAR_SHARE_CLIP)
        self.y = self.calibrate(self.x0) if y is None else y
        self.base_car = self.car_const + self.y[:, None]
        r = self.core(self.x0)
        self.R0, self.T0, self.C0 = r["R"], r["T"], r["C"]
        self.S_today = self.C0 - self.R0     # heutige Subvention: Betriebskosten minus Erlös

    # ----- Kalibrierung ------------------------------------------------------------------------------------------------
    def _probs_given_y(self, x, y):
        pr = (self.Cflat @ x).reshape(self.P, self.K, 2)
        V = np.empty((self.P, self.K, 3))
        V[:, :, :2] = -pr / self.kk[..., None] + self.base_pt[:, :, None]
        V[:, :, 2] = self.car_const + y[:, None]
        Z = self.mu * V
        Z -= Z.max(2, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(2, keepdims=True)

    def calibrate(self, x):
        lo, hi = np.full(self.P, -400.0), np.full(self.P, 400.0)
        wk = self.pk * K_VALS
        wk = wk / wk.sum()
        for _ in range(80):
            mid = (lo + hi) / 2
            share = (self._probs_given_y(x, mid)[:, :, 2] * wk[None, :]).sum(1)
            up = share < self.target
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        return (lo + hi) / 2

    # ----- Auswertung ----------------------------------------------------------------------------------------------------
    def core(self, x, grad=False):
        P, K = self.P, self.K
        pr = (self.Cflat @ x).reshape(P, K, 2)
        V = np.empty((P, K, 3))
        V[:, :, :2] = -pr / self.kk[..., None] + self.base_pt[:, :, None]
        V[:, :, 2] = self.base_car
        Z = self.mu * V
        Z -= Z.max(2, keepdims=True)
        E = np.exp(Z)
        Pr = E / E.sum(2, keepdims=True)
        w = self.w
        R = (w * (Pr[:, :, :2] * pr).sum(2)).sum()
        Pc = Pr[:, :, 2]
        wk = w * self.kk
        pair = (wk * (1 - Pc)).sum(1)
        T = pair.sum()
        load = pair @ self.inst.inc
        d = np.maximum(self.dmin, load / self.cap)
        Cost = self.cdep * (self.inst.elen * d).sum()
        out = dict(R=R, T=T, C=Cost, load=load, Pr=Pr, pr=pr)
        if grad:
            g = -self.Cmat / self.kk[..., None, None]
            gbar = (Pr[:, :, :2, None] * g).sum(2)
            dPpt = self.mu * Pr[:, :, :2, None] * (g - gbar[:, :, None, :])
            dPc = -self.mu * Pc[..., None] * gbar
            first = (Pr[:, :, :2, None] * self.Cmat).sum(2)
            second = (dPpt * pr[..., None]).sum(2)
            dR = (w[..., None] * (first + second)).sum((0, 1))
            dpair = -(wk[..., None] * dPc).sum(1)
            dT = dpair.sum(0)
            dload = self.inst.inc.T @ dpair
            act = (load / self.cap > self.dmin).astype(float)
            dC = self.cdep * ((self.inst.elen * act)[:, None] * dload).sum(0) / self.cap
            out.update(dR=dR, dT=dT, dC=dC)
        return out

    def shares(self, x):
        """Modal Split der Fahrten: Einzelkarte, Monatskarte, Auto."""
        Pr = self.core(x)["Pr"]
        wk = self.w * self.kk
        return (wk[..., None] * Pr).sum((0, 1)) / wk.sum()

    def kpi(self, x, subsidy):
        """Kennzahlen eines Tarifs: Erlös, Fahrten, Kosten, Bilanz (Erlös − Kosten + Subvention) und Modal Split."""
        r = self.core(x)
        s = self.shares(x)
        return dict(x=tuple(float(v) for v in x), R=float(r["R"]), T=float(r["T"]), C=float(r["C"]), profit=float(r["R"] - r["C"] + subsidy),
                    share=tuple(float(v) for v in s))

    def elasticity(self, rel=0.01):
        """Fahrpreiselastizität der ÖV-Fahrten bezüglich des gesamten Preisniveaus (alle Preise gemeinsam)."""
        a = self.core(self.x0 * (1 + rel))["T"]
        b = self.core(self.x0 * (1 - rel))["T"]
        return ((a - b) / self.T0) / (2 * rel)
