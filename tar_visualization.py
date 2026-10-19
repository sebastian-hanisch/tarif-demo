"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import tar_constants as C
from tar_evaluation import de

GOAL_COLOR = {"R": C.COLOR_R, "P": C.COLOR_P, "D": C.COLOR_D, "today": C.COLOR_TODAY, "zero": C.COLOR_ZERO}
LABEL_POS = {"today": "bottom left", "R": "top center", "P": "top center", "D": "bottom right", "zero": "top right"}   # heute und Max-D liegen oft nah beieinander
GOAL_NAME = {"R": "Max-R", "P": "Max-P", "D": "Max-D", "today": "heute", "zero": "Nulltarif"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_price_plane(a, plane):
    """Preis-Ebene: Höhenlinien des Erlöses (in % von heute) über Einzelkartenpreis und Monatskartenpreis, die Grenze „Bilanz = 0“ (links davon
    ist die Bilanz negativ) und die Marken von heute, Max-R, Max-P, Max-D."""
    xs, ys, R, B = plane
    fig = go.Figure()
    fig.add_trace(go.Contour(x=list(map(float, xs)), y=list(map(float, ys)), z=R.tolist(), colorscale="Greens", ncontours=14, contours=dict(coloring="heatmap", showlabels=False),
                             colorbar=dict(title="Erlös in % von heute", thickness=12), hovertemplate="Einzelkarte %{x:.2f}<br>Monatskarte %{y:.0f}<br>Erlös %{z:.0f} %<extra></extra>", name="Erlös"))
    fig.add_trace(go.Contour(x=list(map(float, xs)), y=list(map(float, ys)), z=B.tolist(), showscale=False, contours=dict(start=0, end=0, size=1, coloring="none"),
                             line=dict(color="#222222", width=2, dash="dash"), hoverinfo="skip", name="Bilanz = 0", showlegend=True))
    marks = [("today", a.model.x0), ("R", a.goals["R"]["x"]), ("P", a.goals["P"]["x"]), ("D", a.goals["D"]["x"])]
    if a.verdict == "nulltarif":
        marks.append(("zero", (0.0, 0.0)))
    for key, x in marks:
        fig.add_trace(go.Scatter(x=[float(x[0])], y=[float(x[1])], mode="markers+text", text=[GOAL_NAME[key]], textposition=LABEL_POS[key],
                                 marker=dict(size=13, color=GOAL_COLOR[key], line=dict(width=2, color="white")), name=GOAL_NAME[key],
                                 hovertemplate=f"{GOAL_NAME[key]}<br>Einzelkarte %{{x:.2f}}, Monatskarte %{{y:.0f}}<extra></extra>"))
    fig.update_xaxes(title="Einzelkarte je Fahrt und Zone (GE)", range=[0, float(xs[-1])])
    fig.update_yaxes(title="Monatskarte je Zone (GE)", range=[0, float(ys[-1])])
    return _base(fig, 440)


def build_tariff_bars(a):
    """Vier (bei Nulltarif fünf) Tarife nebeneinander: Erlös, Fahrten und Bilanz in % des heutigen Erlöses bzw. der heutigen Fahrten."""
    m = a.model
    rows = [("today", a.today), ("R", a.goals["R"]["kpi"]), ("P", a.goals["P"]["kpi"]), ("D", a.goals["D"]["kpi"])]
    if a.verdict == "nulltarif":
        rows.append(("zero", a.zero))
    names = [GOAL_NAME[k] for k, _ in rows]
    fig = go.Figure()
    series = (("Erlös", [100.0 * r["R"] / m.R0 for _, r in rows], "% von heute"), ("Fahrten", [100.0 * r["T"] / m.T0 for _, r in rows], "% von heute"),
              ("Bilanz (Erlös − Kosten + Subvention)", [100.0 * r["profit"] / m.R0 for _, r in rows], "% des heutigen Erlöses"))
    for (name, vals, unit), color in zip(series, ("#2F6B65", "#4c78a8", "#c8872b")):
        fig.add_trace(go.Bar(x=names, y=vals, name=name, marker_color=color, text=[de(v, 0) for v in vals], textposition="outside", textfont=dict(size=10), cliponaxis=False,
                             hovertemplate=f"{name}: %{{y:.1f}} {unit}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Prozent")
    return _base(fig, 380)


def build_pareto(points, a, extra=None):
    """Pareto-Kurve: Fahrten gegen Erlös (beide in % von heute), mit den Marken heute, Max-R und Max-P."""
    m = a.model
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["R"] for p in points], y=[p["T"] for p in points], mode="lines+markers", line=dict(color=C.COLOR_D, width=3), marker=dict(size=7),
                             name="Max-D unter Mindest-Erlös", customdata=[p["f"] for p in points],
                             hovertemplate="Mindest-Erlös %{customdata:.0%} des Maximums<br>Erlös %{x:.1f} %, Fahrten %{y:.1f} %<extra></extra>"))
    for key in ("today", "R", "P"):
        k = a.today if key == "today" else a.goals[key]["kpi"]
        fig.add_trace(go.Scatter(x=[100.0 * k["R"] / m.R0], y=[100.0 * k["T"] / m.T0], mode="markers+text", text=[GOAL_NAME[key]], textposition="top right",
                                 marker=dict(size=13, color=GOAL_COLOR[key], line=dict(width=2, color="white")), name=GOAL_NAME[key]))
    fig.update_xaxes(title="Erlös (% von heute)")
    fig.update_yaxes(title="Fahrten (% von heute)")
    return _base(fig, 380)


def build_subsidy_curve(points, a):
    """Fahrten gegen Subvention (Vielfache der heutigen Betriebskosten C0): Max-D mit Kostenbilanz; senkrechte Marken für heute und den Nulltarif."""
    m = a.model
    ok = [p for p in points if p["feasible"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["s"] for p in ok], y=[p["T"] for p in ok], mode="lines+markers", line=dict(color=C.COLOR_D, width=3), marker=dict(size=7), name="Max-D",
                             hovertemplate="Subvention %{x:.1f} × C0<br>Fahrten %{y:.1f} %<extra></extra>"))
    for x, label, color in ((m.S_today / m.C0, "heute", C.COLOR_TODAY), (a.s_null / m.C0, "Nulltarif zulässig", C.COLOR_ZERO)):
        fig.add_vline(x=x, line=dict(color=color, dash="dash"), annotation_text=label, annotation_position="top", annotation_bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(title="Subvention (Vielfache der heutigen Betriebskosten)")
    fig.update_yaxes(title="Fahrten (% von heute)")
    return _base(fig, 360)


# ------------------------------------------------------------------------------------------------------------------ Messreihe
def build_pareto_mean(stufen, mark_r=None):
    """Mittlere Pareto-Kurve über die Netze: Fahrten gegen Erlös (% von heute), Fehlerbalken = Standardfehler; die Stufen f sind beschriftet."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[s["R"]["mean"] for s in stufen], y=[s["T"]["mean"] for s in stufen], mode="lines+markers+text", line=dict(color=C.COLOR_D, width=3), marker=dict(size=7),
                             error_y=dict(type="data", array=[s["T"]["se"] for s in stufen], visible=True), text=[f"{s['f']:.1f}".replace(".", ",") for s in stufen],
                             textposition="top right", name="Mittel über die Netze",
                             hovertemplate="Mindest-Erlös %{text} des Maximums<br>Erlös %{x:.1f} %, Fahrten %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=100.0, line=dict(color=C.COLOR_TODAY, dash="dot"), annotation_text="Fahrten wie heute", annotation_position="bottom left", annotation_bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(title="Erlös (% von heute)")
    fig.update_yaxes(title="Fahrten (% von heute)")
    return _base(fig, 380)


def build_subsidy_mean(raster, s_today, s_null, s_min):
    """Mittlere Fahrten (% von heute) gegen die Subvention; die Marken zeigen heutige Subvention, kleinste zulässige und Nulltarif (Mittel über die Netze)."""
    pts = [r for r in raster if r["T"]["n"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["s"] for r in pts], y=[r["T"]["mean"] for r in pts], mode="lines+markers", line=dict(color=C.COLOR_D, width=3), marker=dict(size=7),
                             error_y=dict(type="data", array=[r["T"]["se"] for r in pts], visible=True), name="Max-D (Mittel)",
                             customdata=[f"{r['zul']['k']} von {r['zul']['n']}" for r in pts],
                             hovertemplate="Subvention %{x:.1f} × C0<br>Fahrten %{y:.1f} %<br>zulässig in %{customdata} Netzen<extra></extra>"))
    for x, label, color, pos in ((s_min, "kleinste zulässige", C.COLOR_TODAY, "top left"), (s_today, "heute", C.COLOR_P, "top"), (s_null, "Nulltarif zulässig", C.COLOR_ZERO, "top")):
        fig.add_vline(x=x, line=dict(color=color, dash="dash"), annotation_text=label, annotation_position=pos, annotation_bgcolor="rgba(255,255,255,0.7)")
    fig.update_xaxes(title="Subvention (Vielfache der heutigen Betriebskosten)")
    fig.update_yaxes(title="Fahrten (% von heute)")
    return _base(fig, 380)


ASSUMPTION_LABEL = {"mu": "Preisempfindlichkeit", "car_share": "Ausgangs-Auto-Anteil", "k_mean": "Fahrten je Person", "car_fix": "Auto-Fixkosten", "cdep": "Betriebskosten je Abfahrt",
                    "dmin": "Mindest-Abfahrten", "lam": "Zeitwert", "car_km": "Auto-km-Kosten"}


def build_assumption_shift(entries, goal="R"):
    """Verschiebung des Preisniveaus des Ziels (in %) bei Annahme −25 % und +25 %, je Annahme (nur Neukalibrierung auf den Ausgangs-Modal-Split)."""
    names = [n for n in ASSUMPTION_LABEL if any(e["name"] == n and e["mode"] == "recal" for e in entries)]
    fig = go.Figure()
    for fac, color, label in ((0.75, "#c8872b", "Annahme × 0,75"), (1.25, "#2F6B65", "Annahme × 1,25")):
        vals = []
        for n in names:
            e = next(e for e in entries if e["name"] == n and e["mode"] == "recal" and e["fac"] == fac)
            vals.append(e["shift"][goal]["mean"])
        fig.add_trace(go.Bar(y=[ASSUMPTION_LABEL[n] for n in names], x=vals, orientation="h", name=label, marker_color=color,
                             hovertemplate="%{y}: %{x:.1f} %<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=f"Verschiebung des Preisniveaus von {GOAL_NAME[goal]} (%)")
    return _base(fig, 420, legend_y=-0.18)
