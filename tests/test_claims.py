"""Jede Zahl der README (Abschnitte „Warum dieses Problem“ und „Befunde“) ist hier aus data/tar_results.json nachgerechnet (DEMO-PLAYBOOK §4)."""

from pathlib import Path

import tar_results as Rs

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
D = Rs.load()
st = Rs.stat_text


def de(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def has(text):
    assert text in README, text


def test_three_goals_row():
    _b, r, p, d_row, zero = Rs.cross_table("mittel")
    has(f"**Max-R:** {st(r['R'])} % Erlös bei {st(r['T'])} % der Fahrten")
    has(f"**Max-P:** {st(p['R'])} % Erlös bei {st(p['T'])} % der Fahrten, Bilanzgewinn {st(p['P'])} % des heutigen Erlöses")
    has(f"**Max-D:** {st(d_row['T'])} % der Fahrten bei Bilanz wie heute")
    has(f"**Nulltarif:** {st(zero['T'])} % der Fahrten, Bilanz {st(zero['P'])} % des heutigen Erlöses".replace("-", "−"))
    has(f"verliert aber {de(100 - r['T']['mean'])} Prozentpunkte der Fahrten")           # Einleitung: 22,5


def test_price_levels_and_rank_order():
    pr = Rs.prices("mittel")
    has(f"Max-R {st(pr['R']['niveau'], 2)}, Max-P {st(pr['P']['niveau'], 2)}, Max-D {st(pr['D']['niveau'], 2)}")
    assert pr["rangfolge_aufsteigend"] == {"DRP": 40}
    has("in 40 von 40 Netzen")
    has(f"Max-D liegt in {Rs.share_text(pr['D_nahe_heute'])} Netzen auf 2 % am heutigen Tarif")


def test_cost_of_a_wrong_goal():
    lm = Rs.loss_matrix("mittel")
    has(f"fehlen gegenüber Max-D {st(lm['R']['T'])} Prozentpunkte der Fahrten")
    has(f"gegenüber Max-R {st(lm['D']['R'])} Prozentpunkte Erlös")
    has(f"{st(lm['R']['P'])} Punkte Gewinn bzw. {st(lm['P']['R'])} Punkte Erlös")


def test_cost_structure_row():
    ratios = [Rs.prices(c)["ratio_P_R"] for c in ("fix", "mittel", "var")]
    has(f"{st(ratios[0], 2)} bei hohen Fixkosten, {st(ratios[1], 2)} bei mittleren und {st(ratios[2], 2)} bei niedrigen")
    has(f"in {Rs.share_text(Rs.prices('fix')['P_gleich_R'])} Netzen zusammen, bei mittleren in {Rs.share_text(Rs.prices('mittel')['P_gleich_R'])}")
    fixshare = [Rs.model_stats(c)["fixshare"]["mean"] for c in ("fix", "mittel", "var")]
    has(f"Fixkostenanteil heute {de(fixshare[0], 2)}, {de(fixshare[1], 2)} und {de(fixshare[2], 2)}")


def test_pareto_row():
    par = Rs.pareto("mittel")
    s90 = next(s for s in par["stufen"] if s["f"] == 0.9)
    has(f"Bei f = 0,9 sind es {st(s90['T'])} % der Fahrten")
    has(f"kostet {st(par['schritte'][-1]['drop'])} Prozentpunkte Fahrten, der davor (0,8 auf 0,9) nur {st(par['schritte'][-2]['drop'])}")
    assert par["knick_letzter_groesster"] and par["n_vollstaendig"] == 40


def test_subsidy_row():
    sb = Rs.subsidy("mittel")
    has(f"bei {st(sb['Smin'], 2)} Betriebskosten")
    has(f"heute sind es {st(sb['Stoday'], 2)}")
    has(f"ab {st(sb['Snull'], 2)} zulässig")
    has(f"das {de(sb['Snull_over_Stoday']['mean'])}-Fache der heutigen Subvention")
    has(f"ab dem {de(sb['Snull']['mean'], 2)}-Fachen")


def test_assumption_rows():
    en = {(e["name"], e["mode"], e["fac"]): e for e in Rs.assumptions()["eintraege"]}
    mu_lo, mu_hi = en[("mu", "recal", 0.75)], en[("mu", "recal", 1.25)]
    has(f"Preisempfindlichkeit ×0,75: Preisniveau von Max-R **+{de(mu_lo['shift']['R']['mean'])} %**; ×1,25: **{de(mu_hi['shift']['R']['mean']).replace('-', '−')} %**")
    has(f"liegt dabei nur bei {de(mu_lo['regret']['R']['mean'])} bzw. {de(mu_hi['regret']['R']['mean'])} Punkten")
    lo, hi = en[("car_share", "recal", 0.75)], en[("car_share", "recal", 1.25)]
    has(f"Auto-Anteil ×0,75 / ×1,25: +{de(lo['shift']['R']['mean'])} % / {de(hi['shift']['R']['mean']).replace('-', '−')} %")
    for name in ("lam", "car_km"):
        assert all(en[(name, "recal", f)]["wirkungslos"] and en[(name, "recal", f)]["max_rel_aenderung"] < 1e-6 for f in (0.75, 1.25))
    fest = [en[(n, "fest", f)]["max_rel_aenderung"] for n in ("lam", "car_km") for f in (0.75, 1.25)]
    assert round(100 * min(fest)) == 9 and round(100 * max(fest)) == 17
    has("mit 9 bis 17 %")


def test_practice_rule_row():
    mm = Rs.methods("R")
    gp = Rs.practice_gap()
    has(f"eine Lücke von {st(mm['methoden']['rule_oracle']['gap_pct'])} % des heutigen Erlöses (global bestes X: {mm['X_global_pct']:.0f} %)")
    has(f"mit dem Anheben um 10 % noch {st(mm['methoden']['rule_p10']['gap_pct'])} %")
    has(f"{st(gp['R']['luecke_beste_anhebung_pct'])} % beim Erlös und {st(gp['P']['luecke_beste_anhebung_pct'])} % beim Gewinn")


def test_solver_row():
    mm = Rs.methods("R")
    mp = Rs.methods("P")
    has(f"({de(mm['optima']['endpunkte']['mean'])} Endpunkte im Mittel, {mm['optima']['anteil_mehrere']['k']} von {mm['optima']['anteil_mehrere']['n']} Netzen mit mehreren)")
    has(f"treffen im Mittel {de(mp['optima']['treffer50']['mean'])} % der Starts den besten Fund")
    has(f"Gitter 41 × 41 liegt {st(mm['methoden']['grid_med']['gap_pct'], 3)} % unter dem Optimum, das Gitter 21 × 21 {st(mm['methoden']['grid_coarse']['gap_pct'], 3)} %")


def test_cannibalization_row():
    s = {round(x["c"], 1): x for x in Rs.cannibalization()["schnitte"]}
    has(f"Monatskarte −10 %: Erlös {st(s[0.9]['dR'])} Punkte, Fahrten +{st(s[0.9]['dT'])} Punkte; {st(s[0.9]['aus_einzelkarte_pct'])} % der neuen Monatskarten-Nutzer".replace("Erlös -", "Erlös −"))
    has(f"bei −30 %: {de(s[0.7]['dR']['mean']).replace('-', '−')} Punkte Erlös")


def test_timing_row():
    n25 = D["rechenzeit"]["n"]["25"]
    has(f"Die drei Ziele {de(n25['solve_goals']['med'], 2)} s, Preis-Ebene {de(n25['preisebene31']['med'], 2)} s, Pareto-Kurve {de(n25['pareto']['med'], 2)} s, Subventions-Kurve {de(n25['subventionskurve']['med'], 2)} s")
    has(f"alles zusammen {de(n25['alles']['med'], 2)} s")


def test_verification_numbers():
    pf = D["pruefungen"]
    has(f"größtes |z| = {de(pf['monte_carlo']['max_abs_z'], 2)}")
    assert pf["determinismus"]["zahlen"] == pf["determinismus"]["bitgleich"] == 137235
    has("137 235 von 137 235 Zahlen")
    assert pf["gradient"]["punkte"] == 105 and abs(pf["gradient"]["max_rel_fehler"]["R"] - 4.7e-9) < 0.05e-9
    has("105 Punkte, größter relativer Fehler 4,7e-9")


def test_series_size_and_file_size():
    assert Rs.n_networks() == 40
    has("40 Netze (Zufalls-Seeds 1 bis 40")
    assert (ROOT / "data" / "tar_results.json").stat().st_size < 100_000
