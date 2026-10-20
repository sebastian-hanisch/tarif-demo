"""Tarifplanung: ein Preis, drei Ziele - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Zonentarif hat zwei Preise (Einzelkarte, Monatskarte). Wer den Erlös maximiert, wer den Gewinn maximiert und wer möglichst viele Fahrgäste
bei Bilanz wie heute fahren will, verlangt drei verschiedene Preise. Live ein Szenario, daneben die vorgerechnete Messreihe über 40 Netze.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import tar_constants as C
import tar_evaluation as ev
import tar_results as Rs
from tar_evaluation import Settings, de
from tar_layout import dunkles_design, kopf_css
from tar_pdf_export import build_report
from tar_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from tar_visualization import (
    ASSUMPTION_LABEL,
    build_assumption_shift,
    build_pareto,
    build_pareto_mean,
    build_price_plane,
    build_subsidy_curve,
    build_subsidy_mean,
    build_tariff_bars,
)

st.set_page_config(page_title="Tarifplanung – Sebastian Hanisch", layout="wide")


@st.cache_resource(show_spinner=False, max_entries=12)
def _analysis(s: Settings):
    return ev.analyse(s)


@st.cache_data(show_spinner=False, max_entries=24)
def _plane(s: Settings):
    return ev.price_plane(_analysis(s))


@st.cache_data(show_spinner=False, max_entries=24)
def _pareto(s: Settings):
    return ev.pareto_curve(_analysis(s))


@st.cache_data(show_spinner=False, max_entries=24)
def _subsidy(s: Settings):
    return ev.subsidy_curve(_analysis(s))


@st.cache_data(show_spinner=False, max_entries=12)
def _assumptions(s: Settings):
    return ev.assumption_table(s)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in tar_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎫 Tarifplanung: ein Preis, drei Ziele")
st.markdown(
    """
Ein **Zonentarif** hat zwei Preise: die **Einzelkarte** (je Fahrt und Zone) und die **Monatskarte**. Welche Preise „richtig“ sind, hängt am Ziel: Wer den **Erlös** maximiert,
wer den **Gewinn** (Erlös minus Betriebskosten) maximiert und wer **möglichst viele Fahrgäste** bei einer Bilanz wie heute fahren will, verlangt drei verschiedene Preise.
Die Demo rechnet live ein Szenario (Netz, Preisempfindlichkeit, Kostenstruktur, Subvention) und zeigt daneben die **vorgerechnete Messreihe** über 40 Netze.

Weiter unten: **📐 Zielkonflikt**, **📊 Was die Messreihe über 40 Netze zeigt** und der Expander **🔧 Wie wir das erreichen**.
Nachbarn: das feste Liniennetz der [ÖPNV-Liniennetz-Demo](https://sebastianhanisch-transit-demo.streamlit.app/), der [Taktfahrplan](https://sebastianhanisch-taktfahrplan-demo.streamlit.app/) und der [Rufbus](https://sebastianhanisch-rufbus-demo.streamlit.app/).
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/tarif-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein **Zonentarif** hat zwei Preise: die Einzelkarte und die Monatskarte. Die Demo fragt, welche Preise richtig sind, und die Antwort hängt davon ab, was man erreichen will. Wer die Einnahmen maximiert, wer den Gewinn maximiert und wer möglichst viele Fahrgäste befördern will, verlangt drei verschiedene Preise. Jede Person im Modell entscheidet sich für Einzelkarte, Monatskarte oder Auto, und zwar nur mit Wahrscheinlichkeiten: Je günstiger eine Alternative gegenüber den anderen ist, desto öfter wird sie gewählt. Die Demo zeigt, was ein falsches Ziel kostet und wie stark der beste Preis an einer Annahme hängt, die sich nicht messen lässt: der Preisempfindlichkeit. Es ist ein stark vereinfachtes Modell mit erfundenen Parametern auf künstlichen Netzen.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/tarif-erklaert.html)
        """
    )
# erklaerseite:einfach:end

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Nachfrage**")
    stops = st.selectbox("Haltepunkte", C.STOPS_CHOICES, key="stops_select", help="Größe des synthetischen Netzes. Das Netz ist Kulisse: die optimalen relativen Preise streuen über Netze kaum, die Betriebskosten schon.")
    hub = st.selectbox("Nachfragekonzentration", C.HUB_CHOICES, key="hub_select", format_func=lambda v: C.HUB_LABELS[v],
                       help="Wie stark die Nachfrage auf zwei Zentren liegt.")
    st.markdown("**Fahrgäste**")
    mu = st.selectbox("Preisempfindlichkeit", C.MU_FACTORS, key="mu_select", format_func=lambda v: C.MU_LABELS[v],
                      help="Die wirksamste Verhaltensannahme: Sie verschiebt den Erlös-optimalen Preis deutlich, den Erlös kaum. Unten steht die daraus folgende Fahrpreiselastizität.")
    car = st.selectbox("Ausgangs-Auto-Anteil", C.CAR_SHARE_CHOICES, key="car_select", format_func=lambda v: f"{int(round(100 * v))} %",
                       help="Anteil der Fahrten, die beim heutigen Tarif mit dem Auto gefahren werden (Mittel über die OD-Paare). Das Modell wird darauf kalibriert.")
    trips = st.selectbox("Fahrten je Person und Monat (Mittel)", C.TRIPS_CHOICES, key="trips_select", format_func=lambda v: C.TRIPS_LABELS[v],
                         help="Verschiebt den Preis, ab dem sich die Monatskarte lohnt.")
    st.markdown("**Betrieb und Geld**")
    cost = st.selectbox("Kostenstruktur", tuple(C.COST_VARIANTS), key="cost_select", format_func=lambda v: C.COST_LABELS[v],
                        help="Wie stark die Betriebskosten mit der Fahrgastzahl wachsen. Bei hohen Fixkosten rücken Gewinn- und Erlösmaximum zusammen.")
    subsidy = st.selectbox("Subvention", C.SUBSIDY_CHOICES, key="subsidy_select", format_func=lambda v: "wie heute" if v == "heute" else f"{de(float(v), 1)} × Betriebskosten",
                           help="Geld, das der Betreiber zusätzlich zum Erlös bekommt (in Vielfachen der heutigen Betriebskosten). Max-D darf die Bilanz nicht unter null drücken.")
    seed = st.number_input("Zufalls-Seed des Szenarios", *(SETTING_SPECS["seed_input"].lo, SETTING_SPECS["seed_input"].hi), key="seed_input", step=1)
    st.button("🎲 Neues Szenario generieren", width="stretch", on_click=randomize_seed)

sync_query_params({k: st.session_state[k] for k in SETTING_SPECS})

settings = Settings(stops=int(stops), hub=int(hub), mu_factor=float(mu), car_share=float(car), trips=None if trips == "auto" else float(trips), cost=cost,
                    subsidy=None if subsidy == "heute" else float(subsidy), seed=int(seed))

with st.spinner("Rechne die drei Tarifziele für dieses Szenario …"):
    a = _analysis(settings)
m = a.model

with st.expander("📍 Netz und Nachfrage (nicht editierbar – hängt an der Erzeugung des Szenarios)"):
    inst = a.inst
    wt = inst.demand.sum(axis=1)
    st.dataframe(pd.DataFrame({"Haltepunkt": [f"H{i + 1}" for i in range(inst.n)], "x": [int(c[0]) for c in inst.grid], "y": [int(c[1]) for c in inst.grid],
                               "Zone": ["innen" if v else "außen" for v in inst.inner], "Nachfragegewicht": [int(v) for v in wt]}), width="stretch", hide_index=True)
    st.caption(f"{inst.n} Haltepunkte, {inst.E} Kanten im Liniennetz (Spannbaum plus die zwei nächsten Nachbarn je Haltepunkt), {inst.P} Paare; eine Koordinateneinheit entspricht 0,2 km.")

# --- Ihr Szenario --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ihr Szenario")
st.caption(
    f"Heute: Einzelkarte {de(m.x0[0], 2)} und Monatskarte {de(m.x0[1], 0)} (Geldeinheiten, GE). Betriebskosten {de(m.C0 / 1000, 0)} Tsd. GE im Monat, Erlös {de(m.R0 / 1000, 0)} Tsd. GE "
    f"(Kostendeckung {de(100 * m.R0 / m.C0, 0)} %), Fahrten {de(m.T0 / 1000, 1)} Tsd. im Monat, ÖV-Anteil {de(100 * (1 - m.shares(m.x0)[2]), 0)} %. "
    f"Fahrpreiselastizität der Fahrten: {de(a.elasticity, 2)}. Subvention in diesem Szenario: {de(a.subsidy / m.C0, 2)} × Betriebskosten."
)

c1, c2, c3 = st.columns(3)
for col, key, label in ((c1, "R", "Erlös maximieren (Max-R)"), (c2, "P", "Gewinn maximieren (Max-P)"), (c3, "D", "Fahrgäste maximieren (Max-D)")):
    g = a.goals[key]["kpi"]
    col.metric(label, f"Erlös {de(100 * g['R'] / m.R0, 0)} %", delta=f"Fahrten {de(100 * g['T'] / m.T0, 0)} %", delta_color="off",
               help="Erlös und Fahrten in Prozent von heute (Zonentarif, Einzelkarte und Monatskarte). Max-D: Bilanz Erlös − Kosten + Subvention darf nicht unter null liegen.")

rows = []
for key, kpi, name in (("today", a.today, "heutiger Tarif"), ("R", a.goals["R"]["kpi"], "Max-R"), ("P", a.goals["P"]["kpi"], "Max-P"), ("D", a.goals["D"]["kpi"], "Max-D"),
                       ("zero", a.zero, "Nulltarif")):
    rows.append({"Tarif": name, "Einzelkarte (GE)": de(kpi["x"][0], 2), "Monatskarte (GE)": de(kpi["x"][1], 0), "Erlös (% von heute)": de(100 * kpi["R"] / m.R0, 1),
                 "Fahrten (% von heute)": de(100 * kpi["T"] / m.T0, 1), "ÖV-Anteil (%)": de(100 * (1 - kpi["share"][2]), 1),
                 "Bilanz (% des heutigen Erlöses)": de(100 * kpi["profit"] / m.R0, 1)})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

kind, text = ev.verdict(a)
getattr(st, kind)(text)

left, right = st.columns([3, 2])
with left:
    st.markdown("**Preis-Ebene.** Höhenlinien des Erlöses über Einzelkarte und Monatskarte; die gestrichelte Linie ist die Grenze „Bilanz = 0“: Max-D liegt auf ihr, auf der Seite von Max-R und Max-P ist die Bilanz positiv.")
    with st.spinner("Rechne die Preis-Ebene …"):
        st.plotly_chart(build_price_plane(a, _plane(settings)), width="stretch", key="plane")
with right:
    st.markdown("**Vier Tarife im Vergleich.**")
    st.plotly_chart(build_tariff_bars(a), width="stretch", key="bars")

# --- Zielkonflikt ------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Zielkonflikt")
st.markdown("Nahe dem Erlösmaximum ist die Erlöskurve flach, die Fahrgastzahl nicht: Die letzten Erlöspunkte kosten viele Fahrten. Und wer Fahrgäste will, braucht Subvention.")
pc, sc = st.columns(2)
with pc:
    st.markdown("**Pareto-Kurve.** Fahrten, die ein Tarif höchstens fährt, wenn er mindestens den Anteil f des Erlösmaximums einnehmen soll.")
    with st.spinner("Rechne die Pareto-Kurve (11 Stufen) …"):
        st.plotly_chart(build_pareto(_pareto(settings), a), width="stretch", key="pareto")
with sc:
    st.markdown("**Subventions-Kurve.** Fahrten bei Bilanz null für wachsende Subvention (Max-D); ab der rechten Marke genügt der Nulltarif.")
    with st.spinner("Rechne die Subventions-Kurve (9 Stützstellen) …"):
        st.plotly_chart(build_subsidy_curve(_subsidy(settings), a), width="stretch", key="subsidy")

st.markdown("**Dasselbe Szenario bei anderer Preisempfindlichkeit.** Der Preis des nominalen Erlösmaximums („Regret“) verliert in der anderen Welt so viele Erlöspunkte (in % des heutigen Erlöses):")
with st.spinner("Rechne drei Welten …"):
    at = _assumptions(settings)
st.dataframe(pd.DataFrame([{"Preisempfindlichkeit": C.MU_LABELS[r["mu_factor"]], "Fahrpreiselastizität": de(r["elasticity"], 2),
                            "Preisniveau Max-R": de(r["level"]["R"], 2), "Preisniveau Max-P": de(r["level"]["P"], 2), "Preisniveau Max-D": de(r["level"]["D"], 2),
                            "Regret des nominalen Max-R-Preises (Punkte)": de(r.get("regret", 0.0), 2)} for r in at]), width="stretch", hide_index=True)
st.caption("Preisniveau = Mittel der beiden Preise relativ zum heutigen Tarif. Die Nutzenfunktion ist eine Annahme, keine Messung: Der optimale Preis hängt an ihr, der Erlös kaum.")

st.markdown("## 📊 Was die Messreihe über 40 Netze zeigt")
_n = Rs.n_networks()
st.caption(f"Vorgerechnet, kein Rechnen im Browser: {_n} synthetische Netze (20 bis 30 Haltepunkte, Zufalls-Seeds 1 bis 40, Nachfragekonzentration mittel), Zonentarif, Preisempfindlichkeit mittel, Ausgangs-Auto-Anteil 45 %. "
           "Zahlen sind Mittel ± Standardfehler über die Netze. Die Auswahl gilt nur für diesen Abschnitt, nicht für die Seitenleiste.")
sc = st.selectbox("Kostenstruktur der Messreihe", Rs.VARIANTS, index=1, key="series_cost", format_func=lambda v: C.COST_LABELS[v],
                  help="Die Reihe wurde für jede der drei Kostenstrukturen auf der Basiseinstellung gemessen (Subvention heute, mittlere Konzentration).")

ms = Rs.model_stats(sc)
st.markdown(f"**Das Modell dieser Reihe.** Fahrpreiselastizität {Rs.stat_text(ms['elasticity'], 2)}, Kostendeckung heute {Rs.stat_text(ms['kdg'], 2)}, Fixkostenanteil {Rs.stat_text(ms['fixshare'], 2)}, "
            f"heutige Subvention {Rs.stat_text(ms['S_today_over_C0'], 2)} × Betriebskosten.")

st.markdown("### Drei Ziele, drei Preise")
ct = Rs.cross_table(sc)
names = ("heutiger Tarif", "Preise Max-R", "Preise Max-P", "Preise Max-D", "Nulltarif")
st.dataframe(pd.DataFrame([{"Tarif": n, "Erlös (% von heute)": Rs.stat_text(r["R"]), "Fahrten (% von heute)": Rs.stat_text(r["T"]), "Bilanzgewinn (% des heutigen Erlöses)": Rs.stat_text(r["P"]),
                            "ÖV-Anteil (%)": Rs.stat_text(r["oev"]), "Bilanz ≥ heute in": Rs.share_text(r["zul"]) + " Netzen"} for n, r in zip(names, ct)]), width="stretch", hide_index=True)
pr = Rs.prices(sc)
st.markdown(
    f"Das **Preisniveau** (Mittel der zwei Preise relativ zu heute) liegt bei Max-R bei {Rs.stat_text(pr['R']['niveau'], 2)}, bei Max-P bei {Rs.stat_text(pr['P']['niveau'], 2)} und bei Max-D bei {Rs.stat_text(pr['D']['niveau'], 2)}; "
    f"das Verhältnis Max-P zu Max-R beträgt {Rs.stat_text(pr['ratio_P_R'], 2)}. Gewinn- und Erlösmaximum fallen in {Rs.share_text(pr['P_gleich_R'])} Netzen zusammen (Preisniveau auf 0,1 % gleich). "
    f"Max-D liegt in {Rs.share_text(pr['D_nahe_heute'])} Netzen auf 2 % am heutigen Tarif."
)
lm = Rs.loss_matrix(sc)
st.markdown(
    f"**Preise eines anderen Ziels verwenden kostet:** Mit den Preisen von Max-R fehlen gegenüber Max-D {Rs.stat_text(lm['R']['T'])} Prozentpunkte der Fahrten; mit den Preisen von Max-D fehlen gegenüber Max-R "
    f"{Rs.stat_text(lm['D']['R'])} Prozentpunkte Erlös. Zwischen Max-R und Max-P ist der Unterschied klein: {Rs.stat_text(lm['R']['P'])} Punkte Gewinn bzw. {Rs.stat_text(lm['P']['R'])} Punkte Erlös."
)

st.markdown("### Die letzten Erlöspunkte kosten viele Fahrten")
par = Rs.pareto(sc)
steps = par["schritte"]
st.plotly_chart(build_pareto_mean(par["stufen"]), width="stretch", key="series_pareto")
st.markdown(
    f"Max-D unter der Bedingung, mindestens den Anteil f des Erlösmaximums einzunehmen (f an den Punkten). Der **letzte Schritt** von f = 0,9 auf 1,0 kostet {Rs.stat_text(steps[-1]['drop'])} Prozentpunkte Fahrten, "
    f"der Schritt davor (0,8 auf 0,9) {Rs.stat_text(steps[-2]['drop'])}; der letzte Schritt ist der größte in {par['n_vollstaendig']} von {par['n_vollstaendig']} Netzen: {'ja' if par['knick_letzter_groesster'] else 'nein'}. "
    f"Die Kurve ist in allen Netzen monoton: {'ja' if par['monoton'] else 'nein'}."
)

st.markdown("### Wer Fahrgäste will, braucht Subvention")
sb = Rs.subsidy(sc)
st.plotly_chart(build_subsidy_mean(sb["raster"], sb["Stoday"]["mean"], sb["Snull"]["mean"], sb["Smin"]["mean"]), width="stretch", key="series_subsidy")
st.markdown(
    f"Die kleinste Subvention, bei der Max-D zulässig ist, liegt bei {Rs.stat_text(sb['Smin'], 2)} × Betriebskosten (in manchen Netzen negativ: der Erlös deckt die Kosten ohne Zuschuss); heute sind es {Rs.stat_text(sb['Stoday'], 2)}; "
    f"der **Nulltarif** ist ab {Rs.stat_text(sb['Snull'], 2)} zulässig, das ist, als Mittel der Verhältnisse je Netz, das {Rs.stat_text(sb['Snull_over_Stoday'], 1)}-Fache der heutigen Subvention. Kurz unter dieser Schwelle (0,97-fach) bleibt der Preis der Monatskarte noch positiv, "
    f"der Einzelkartenpreis ist schon null."
)

st.markdown("### Der optimale Preis hängt an der Annahme, der Erlös kaum")
an = Rs.assumptions()
entries = an["eintraege"]
st.plotly_chart(build_assumption_shift(entries, "R"), width="stretch", key="series_assumptions")
rows_a = []
for n in ASSUMPTION_LABEL:
    lo = next((e for e in entries if e["name"] == n and e["mode"] == "recal" and e["fac"] == 0.75), None)
    hi = next((e for e in entries if e["name"] == n and e["mode"] == "recal" and e["fac"] == 1.25), None)
    if lo is None or hi is None:
        continue
    rows_a.append({"Annahme": ASSUMPTION_LABEL[n], "Max-R × 0,75": f"{de(lo['shift']['R']['mean'], 1)} %", "Max-R × 1,25": f"{de(hi['shift']['R']['mean'], 1)} %",
                   "Max-P × 0,75": f"{de(lo['shift']['P']['mean'], 1)} %", "Max-P × 1,25": f"{de(hi['shift']['P']['mean'], 1)} %",
                   "Regret Max-R (Punkte, × 0,75 / × 1,25)": f"{de(lo['regret']['R']['mean'], 2)} / {de(hi['regret']['R']['mean'], 2)}", "Wirkung": "wirkungslos" if lo["wirkungslos"] and hi["wirkungslos"] else "wirkt"})
st.dataframe(pd.DataFrame(rows_a), width="stretch", hide_index=True)
st.markdown(
    "Jede Annahme wurde um ±25 % geändert und das Modell auf den Ausgangs-Modal-Split neu kalibriert. **Preisempfindlichkeit** verschiebt das Preisniveau des Erlösmaximums am stärksten, der „Regret“ (Erlöspunkte, die der Preis der nominalen Annahme in der anderen Welt verliert) "
    "bleibt dagegen bei wenigen Punkten. **Zeitwert** und **Auto-Kilometerkosten** haben nach der Kalibrierung keine Wirkung (größte Änderung unter 1e-6); deshalb sind sie in der App kein Regler. Kosten ändern nur Max-P, nie Max-R."
)

st.markdown("### Wie weit ist die Praxisregel „alle Preise um X % anheben“ vom Optimum?")
gp = Rs.practice_gap()
st.markdown(
    f"Bei {gp['R']['luecke_beste_anhebung_pct']['n']} Fällen (40 Netze mit je fünf zufälligen Ausgangstarifen, jeder Preis bis zu ±35 % vom heutigen entfernt) bleibt beim Erlösziel mit dem besten X eine Lücke von {Rs.stat_text(gp['R']['luecke_beste_anhebung_pct'], 1)} % des heutigen Erlöses "
    f"(ohne Anhebung {Rs.stat_text(gp['R']['luecke_ohne_anhebung_pct'], 1)} %); beim Gewinnziel {Rs.stat_text(gp['P']['luecke_beste_anhebung_pct'], 1)} % (ohne Anhebung {Rs.stat_text(gp['P']['luecke_ohne_anhebung_pct'], 1)} %). "
    f"Das beste X liegt im Mittel bei {Rs.stat_text(gp['R']['bestes_X_pct'], 0)} % (Erlös) bzw. {Rs.stat_text(gp['P']['bestes_X_pct'], 0)} % (Gewinn). Die Regel ist im Zonentarif nah am Optimum, weil das Optimum fast auf der Geraden durch den Ausgangstarif liegt; "
    "die Lücke hängt am Ausgangstarif."
)

st.markdown("---")

st.markdown("### 📄 Szenario-Bericht")
report = build_report(
    "Tarifplanung: ein Preis, drei Ziele",
    [("Haltepunkte", str(settings.stops)), ("Nachfragekonzentration", C.HUB_LABELS[settings.hub]), ("Preisempfindlichkeit", C.MU_LABELS[settings.mu_factor]),
     ("Ausgangs-Auto-Anteil", f"{int(round(100 * settings.car_share))} %"), ("Kostenstruktur", C.COST_LABELS[settings.cost]),
     ("Subvention (x Betriebskosten)", de(a.subsidy / m.C0, 2)), ("Zufalls-Seed", str(settings.seed))],
    [(name, f"Einzelkarte {de(k['x'][0], 2)}, Monatskarte {de(k['x'][1], 0)}; Erlös {de(100 * k['R'] / m.R0, 1)} %, Fahrten {de(100 * k['T'] / m.T0, 1)} %, Bilanz {de(100 * k['profit'] / m.R0, 1)} %")
     for name, k in (("heute", a.today), ("Max-R", a.goals["R"]["kpi"]), ("Max-P", a.goals["P"]["kpi"]), ("Max-D", a.goals["D"]["kpi"]))],
    text.replace("**", ""),
    ["Alle Angaben gelten für den Zonentarif dieses Modells (Logit-Wahl, synthetisches Netz) und sind Mechanismen, keine Prognose für ein reales Netz."],
)
st.download_button("PDF herunterladen", report, file_name="tarifplanung_szenario.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    tab_verf, tab_kan, tab_modell, tab_series = st.tabs(["🧮 Verfahren", "🎫 Monatskarte und Einzelkarten", "📋 Modell und Annahmen", "📈 Messreihe"])

    with tab_verf:
        st.markdown("Dasselbe Ziel, vier Wege. Alle vier rechnen auf dem gezeigten Szenario; die Zahl ist der Zielwert in Prozent des heutigen Erlöses, der Abstand ist der Rückstand auf den besten Fund in Prozentpunkten.")
        goal = st.selectbox("Ziel", ("R", "P"), format_func=lambda g: ev.GOAL_LABEL[g], key="method_goal")
        mt = ev.method_table(a, goal)
        st.dataframe(pd.DataFrame([{"Verfahren": r["name"], "Zielwert (% des heutigen Erlöses)": de(r["value"], 2), "Abstand zum besten Fund (Punkte)": de(r["gap"], 3),
                                    "Einzelkarte (GE)": de(r["x"][0], 2), "Monatskarte (GE)": de(r["x"][1], 0)} for r in mt]), width="stretch", hide_index=True)
        st.markdown(
            """
| Verfahren | Was es tut | Wann es reicht |
|---|---|---|
| lokal (SLSQP, 5 Startpunkte) | Folgt dem analytischen Gradienten des Modells bis zum Maximum, von fünf festen Startpunkten aus | Der Erlös ist in den zwei Preisen glatt und hat hier genau ein Maximum: alle Starts enden am selben Punkt |
| Gitter | Wertet die Preis-Ebene an 21 × 21 bzw. 41 × 41 Punkten aus und nimmt den besten | Findet das Maximum bis auf die Gitterweite; das feinere Gitter kostet viermal so viele Auswertungen |
| Praxisregel | Alle Preise um denselben Prozentsatz ändern; hier mit dem je Szenario besten Prozentsatz | Liegt nahe am Optimum, kennt aber das Verhältnis von Einzel- zu Monatskarte nicht |
| heutiger Tarif | Bezugspunkt | – |
"""
        )
        st.caption("Ehrlicher Befund: Es gibt ein Maximum, die Verfahren stimmen überein. Der Gewinn aus der Optimierung steckt im Verhältnis der beiden Preise, nicht im Verfahren.")

    with tab_kan:
        st.markdown("Was passiert, wenn nur die **Monatskarte** billiger wird? Heutiger Einzelkartenpreis, Monatskarte um 10, 20 oder 30 % gesenkt; die Verschiebungen sind Prozentpunkte der Fahrten.")
        cn = ev.cannibalization(a)
        st.dataframe(pd.DataFrame([{"Monatskarte": f"−{int(round(100 * r['cut']))} %", "Erlös (% von heute)": de(r["R"], 1), "Erlösverlust (Punkte)": de(r["R_loss_points"], 1),
                                    "Fahrten (% von heute)": de(r["T"], 1), "Monatskarte (Pkt.)": de(r["d_month"], 1), "Einzelkarte (Pkt.)": de(r["d_single"], 1), "Auto (Pkt.)": de(r["d_car"], 1)} for r in cn]),
                     width="stretch", hide_index=True)
        st.caption("Im Logit-Modell folgt der Wechsel aus der Struktur (Unabhängigkeit irrelevanter Alternativen): Das Verhältnis zweier unveränderter Alternativen bleibt gleich, die Monatskarte zieht Fahrgäste anteilig von Einzelkarte und Auto. Das ist eine Eigenschaft der Annahme, kein Verhaltensbefund.")

    with tab_modell:
        st.markdown(
            """
**Annahmen, die die Demo setzt** (Wer setzt an: die Demo, soweit nicht ein Regler genannt ist):

| Annahme | Wert | Wirkung |
|---|---|---|
| Netz | 20 bis 30 Haltepunkte auf einem 91 × 91-Raster, 1 Einheit = 0,2 km; Liniennetz = minimaler Spannbaum plus die zwei nächsten Nachbarn | Kulisse: die optimalen relativen Preise streuen über Netze kaum |
| Fahrzeit | 3 min je km Kante plus 1 min Halt je Kante; Auto 3 min + 1,8 min je km | Nach der Kalibrierung wirkungslos |
| Nachfrage | ganzzahlige Gewichte, zwei Zentren; 10 Personen je Nachfrageeinheit | Regler „Nachfragekonzentration“ |
| Fahrtenzahl | 2, 6, 12, 20, 30, 44 oder 60 Fahrten je Person und Monat, instanzspezifisch gewichtet (Mittel ≈ 20) | Regler „Fahrten je Person und Monat“ |
| Wahl | Logit zwischen Einzelkarte, Monatskarte und Auto; μ = 0,30 je Geldeinheit und Fahrt, Fahrpreiselastizität ≈ −0,46 | Regler „Preisempfindlichkeit“ (μ × 0,75 / 1 / 1,25): wirksamste Annahme |
| Kalibrierung | Der Auto-Zuschlag je OD-Paar wird so gesetzt, dass der Auto-Anteil beim heutigen Tarif 45 % ± 10 % (steigt mit der Entfernung) beträgt | Zeitwert (0,1 GE/min) und Auto-km-Kosten (0,15 GE) sind dadurch wirkungslos und deshalb **kein Regler** |
| Zonentarif | Preis = Zonenzahl (1 oder 2) × Preis; heute (2,50; 60) | Die zwei Preise sind die Entscheidung |
| Betrieb | je Kante Abfahrten = max(Mindestzahl, Last/80), Kosten = c · Länge · Abfahrten; drei Kostenstrukturen | Regler „Kostenstruktur“ (frei gesetzt, nicht beobachtbar) |
| Subvention | S in Vielfachen der heutigen Betriebskosten; „wie heute“ = Kosten − Erlös heute | Wirkt nur auf Max-D und die Bilanz; Max-R und Max-P ändern sich nicht |

**Was nicht gemessen ist:** Nachfragefunktionen lassen sich nicht messen (Fahrpreis-Experimente gibt es nicht); die Nutzenfunktion ist eine Annahme, und der optimale Preis hängt an ihr. Die Demo zeigt Mechanismen, keine Prognose für ein reales Netz.
"""
        )

    with tab_series:
        st.markdown(f"Alle Zahlen dieses Tabs stammen aus der vorgerechneten Messreihe ({_n} Netze, Mittel ± Standardfehler, Median mit P10 bis P90 steht in `data/tar_results.json`).")
        st.markdown("**Verfahrensvergleich über alle Netze.** Abstand zum besten Fund in Prozent des heutigen Erlöses bzw. der heutigen Fahrten; „Anteil“ = Netze, in denen der Abstand höchstens die Meldeschwelle beträgt.")
        gsel = st.selectbox("Ziel der Messreihe", ("R", "P"), format_func=lambda g: ev.GOAL_LABEL[g], key="series_goal")
        mm = Rs.methods(gsel)
        st.dataframe(pd.DataFrame([{"Verfahren": r["label"], "Abstand (%)": Rs.stat_text(r["gap_pct"], 3), "Anteil bis zur Meldeschwelle": f"{100 * r['le_thr']['p']:.0f} %".replace(".", ",")}
                                   for r in mm["methoden"].values()]), width="stretch", hide_index=True)
        op = mm["optima"]
        st.caption(f"Meldeschwelle {mm['schwelle']:.0e} (aus dem Rauschen zweier Läufe: 95. Perzentil {mm['rauschen']['p95']:.1e}). Bei 50 Starts enden die Läufe im Mittel in {de(op['endpunkte']['mean'], 2)} räumlich verschiedenen Punkten; "
                   f"Netze mit mehreren Endpunkten: {Rs.share_text(op['anteil_mehrere'])}. Rechenzeit (Median): 10 Starts {de(mm['zeit']['ms10_med'], 2)} s, Gitter 21 × 21 {de(mm['zeit']['grid_grob_med'], 2)} s, Gitter 41 × 41 {de(mm['zeit']['grid_mittel_med'], 2)} s.")
        st.markdown("**Kannibalisierung der Monatskarte** (nur die Monatskarte wird billiger):")
        kb = Rs.cannibalization()
        st.dataframe(pd.DataFrame([{"Monatskarte": f"−{int(round(100 * (1 - s['c'])))} %", "Erlös (Punkte)": Rs.stat_text(s["dR"]), "davon Einzelkarten": Rs.stat_text(s["dRs"]), "davon Monatskarten": Rs.stat_text(s["dRm"]),
                                    "Fahrten (Punkte)": Rs.stat_text(s["dT"]), "neue Monatskarten-Nutzer kamen aus Einzelkarte (%)": Rs.stat_text(s["aus_einzelkarte_pct"])} for s in kb["schnitte"]]), width="stretch", hide_index=True)
        st.caption(f"Die Einzelkarte macht heute {Rs.stat_text(kb['einzel_erloesanteil_pct'])} % des Erlöses aus. Das Verhältnis zweier unveränderter Alternativen bleibt im Modell gleich (größte Abweichung {kb['iia_max_abw']:.0e}): "
                   "die Wechselquote folgt aus der Logit-Struktur, kein Verhaltensbefund.")
        st.markdown("**Rechenzeiten der Live-Abschnitte** (Median über 10 Netze, ein Prozess; Streamlit Cloud nicht gemessen):")
        rz = Rs.load()["rechenzeit"]
        st.dataframe(pd.DataFrame([{"Haltepunkte": n, "Modellaufbau (s)": de(v["modell"]["med"], 3), "drei Ziele (s)": de(v["solve_goals"]["med"], 2), "Preis-Ebene (s)": de(v["preisebene31"]["med"], 2),
                                    "Pareto, 11 Stufen (s)": de(v["pareto"]["med"], 2), "Subventions-Kurve (s)": de(v["subventionskurve"]["med"], 2), "alles zusammen (s)": de(v["alles"]["med"], 2)} for n, v in rz["n"].items()]),
                     width="stretch", hide_index=True)

st.markdown("---")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        """
**Szenario.** Ein synthetisches Netz mit 20 bis 30 Haltepunkten und ungleich verteilter Nachfrage. Zwischen je zwei Haltepunkten gibt es Personen, die im Monat mehrmals fahren und sich
für **eine** Alternative für alle ihre Fahrten entscheiden: Einzelkarte, Monatskarte oder Auto. Die Wahl folgt einem Logit-Modell (Nutzen je Fahrt = −Preis je Fahrt − Zeitwert · Fahrzeit).

**Kalibrierung.** Das Modell wird so eingestellt, dass beim heutigen Tarif der gewählte Auto-Anteil herauskommt. Danach kommt es nur noch darauf an, wie die ÖV-Preise gegenüber dem heutigen Tarif liegen:
Zeitwert und Autokosten stecken im Zuschlag und fallen weg.

**Drei Ziele.** *Max-R* maximiert den Erlös. *Max-P* maximiert den Gewinn (Erlös minus Betriebskosten plus Subvention). *Max-D* maximiert die Fahrten im ÖV unter der Bedingung, dass die Bilanz
nicht unter null fällt; ohne diese Bedingung wäre der Nulltarif immer optimal.

**Live und Messreihe.** Ein Szenario streut von Zufalls-Seed zu Zufalls-Seed. Deshalb steht die vorgerechnete Messreihe (40 Netze) daneben.

**Preisempfindlichkeit.** Sie ist die wirksamste Verhaltensannahme: Sie verschiebt das Preisniveau des Erlösmaximums deutlich, der Erlös des nominalen Optimums sinkt dagegen nur um wenige Punkte (Zahlen: Messreihe).
"""
    )

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/tarif-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind ein Netz mit Haltepunkten, die Fahrten der Personen und der heutige Tarif. Gesucht sind die zwei Preise $x_s$ und $x_m$, die je Ziel (Erlös, Gewinn, Fahrten) am besten sind.

| Symbol | Bedeutung |
|---|---|
| $a$, $P_a$ | Alternative (Einzelkarte, Monatskarte, Auto) und Wahrscheinlichkeit, sie zu wählen |
| $V_a$ | Nutzen einer Alternative je Fahrt |
| $\mu$ | Preisempfindlichkeit (Skalierung des Nutzens je Geldeinheit) |
| $k$ | Fahrten einer Person im Monat |
| $z$ | Zonenzahl der Verbindung (1 oder 2) |
| $x_s$, $x_m$ | Preis der Einzelkarte und der Monatskarte |
| $\lambda$, $\tau$ | Zeitwert und Fahrzeit |
| $F$, $c$, $\ell$ | monatliche Fixkosten des Autos, Kosten je km, Weglänge |
| $y_{st}$ | Auto-Zuschlag des Paares, so gesetzt, dass der heutige Auto-Anteil herauskommt |
| $w_{stk}$ | Personen eines Paares mit $k$ Fahrten |
| $R$, $T$, $C$ | Erlös, Fahrten im ÖV, Betriebskosten |
| $S$ | Subvention (in Vielfachen der heutigen Betriebskosten) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/tarif-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown("Eine Person im Paar $s$–$t$ mit $k$ Fahrten im Monat wählt $a \\in \\{\\text{Einzel}, \\text{Monat}, \\text{Auto}\\}$ mit der Wahrscheinlichkeit")
    st.latex(r"P_a = \frac{e^{\mu V_a}}{\sum_b e^{\mu V_b}}")
    st.markdown("Nutzen je Fahrt für die Zonenzahl $z$ (1 oder 2), Preise $x = (x_s, x_m)$, Fahrzeit $\\tau$ und Zeitwert $\\lambda$:")
    st.latex(r"V_{\text{Einzel}} = -z\,x_s - \lambda\,\tau, \qquad V_{\text{Monat}} = -\frac{z\,x_m}{k} - \lambda\,\tau, \qquad V_{\text{Auto}} = -\frac{F}{k} - c\,\ell - \lambda\,\tau_{\text{Auto}} + y_{st}")
    st.markdown("Der Zuschlag $y_{st}$ wird je Paar so bestimmt, dass beim heutigen Tarif $x^0$ der Auto-Anteil gleich dem Ausgangs-Auto-Anteil ist. Erlös, Fahrten und Betrieb mit Personen $w_{stk}$:")
    st.latex(r"R(x) = \sum_{s,t,k} w_{stk}\left(P_{\text{Einzel}}\,k\,z\,x_s + P_{\text{Monat}}\,z\,x_m\right), \qquad T(x) = \sum_{s,t,k} w_{stk}\,k\,(1 - P_{\text{Auto}})")
    st.latex(r"C(x) = c_{\text{dep}} \sum_e \ell_e \max\!\left(d_{\min},\ \frac{\text{Last}_e(x)}{\text{Kapazität}}\right)")
    st.markdown("Die drei Ziele bei Subvention $S$ und den Preisgrenzen $0 \\le x \\le 5\\,x^0$:")
    st.latex(r"\text{Max-R: } \max_x R(x) \qquad \text{Max-P: } \max_x R(x) - C(x) + S \qquad \text{Max-D: } \max_x T(x) \ \text{ u. d. N. } R(x) - C(x) + S \ge 0")
    st.markdown("Die **Pareto-Kurve** ersetzt die Nebenbedingung durch $R(x) \\ge f \\cdot R_{\\max}$ mit $f$ von 0 bis 1.")


st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zum Thema: [ÖPNV und Fernverkehr optimieren](https://sebastianhanisch.net/oepnv-fernverkehr-optimierung.html)."
)
