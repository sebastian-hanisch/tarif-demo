# Tarifplanung: ein Preis, drei Ziele – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-tarif-demo.streamlit.app/)** -->

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`, Gruppe ÖPNV), dritte Demo der Seite [ÖPNV und Fernverkehr optimieren](https://sebastianhanisch.net/oepnv-fernverkehr-optimierung.html) nach dem festen Liniennetz ([transit-demo](https://github.com/sebastian-hanisch/transit-demo)) und dem [Rufbus](https://github.com/sebastian-hanisch/rufbus-demo). Ein **Zonentarif** hat zwei Preise, die Einzelkarte und die Monatskarte. Wer den **Erlös** maximiert, wer den **Gewinn** (Erlös minus Betriebskosten) maximiert und wer **möglichst viele Fahrgäste** bei einer Bilanz wie heute fahren will, verlangt drei verschiedene Preise.

## Warum dieses Problem

Welche Preise „richtig“ sind, hängt am Ziel, und die Ziele widersprechen sich: Nahe dem Erlösmaximum ist die Erlöskurve flach, die Fahrgastzahl nicht. Auf 40 synthetischen Netzen bringt der Erlös-optimale Zonentarif 108,8 ± 0,2 % des heutigen Erlöses, verliert aber 22,5 Prozentpunkte der Fahrten (77,5 ± 0,2 % von heute). Wer Fahrgäste will, braucht Subvention: Der Nulltarif ist erst ab dem 1,23-Fachen der heutigen Betriebskosten zulässig, das ist das 4,5-Fache der heutigen Subvention. Und der optimale Preis hängt an einer Annahme, die sich nicht messen lässt (der Preisempfindlichkeit), der Erlös dagegen kaum.

## Modell

- **Netz:** 20 bis 30 Haltepunkte auf einem ganzzahligen 91 × 91-Raster (1 Einheit = 0,2 km), Nachfrage ungleich verteilt (zwei Zentren). Liniennetz = minimaler Spannbaum plus die zwei nächsten Nachbarn je Haltepunkt; Fahrzeit = kürzester Weg (3 min je km plus 1 min Halt je Kante).
- **Fahrgäste:** Je Paar Personen, die im Monat 2, 6, 12, 20, 30, 44 oder 60 Fahrten machen (instanzspezifisch gewichtet, Mittel rund 20). Jede Person wählt **eine** Alternative für alle ihre Fahrten: Einzelkarte, Monatskarte oder Auto, nach einem Logit-Modell mit Nutzen je Fahrt = −Preis je Fahrt − Zeitwert · Fahrzeit.
- **Kalibrierung:** Je Paar wird ein Zuschlag für das Auto so gesetzt, dass beim heutigen Tarif der Ausgangs-Auto-Anteil herauskommt (45 %, steigend mit der Entfernung). Dadurch haben **Zeitwert** und **Auto-Kilometerkosten** keine Wirkung mehr und sind kein Regler.
- **Zonentarif:** Preis = Zonenzahl (1 oder 2) × Preis; heute 2,50 je Fahrt und Zone und 60 für die Monatskarte (Geldeinheiten, GE).
- **Betrieb:** je Kante Abfahrten = max(Mindestzahl, Last / 80), Kosten = Kostensatz · Länge · Abfahrten; drei Kostenstrukturen (Fixkosten hoch, mittel, niedrig; Fixkostenanteil heute 0,78, 0,57 und 0,34).
- **Ziele:** *Max-R* maximiert den Erlös, *Max-P* den Gewinn (Erlös − Kosten + Subvention), *Max-D* die Fahrten im ÖV unter der Bedingung, dass die Bilanz Erlös − Kosten + Subvention nicht unter null fällt (ohne diese Bedingung wäre der Nulltarif immer optimal). Subvention in Vielfachen der heutigen Betriebskosten; „wie heute“ heißt Kosten − Erlös heute.
- Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

- **Live:** ein Szenario mit den Reglern der Seitenleiste (Haltepunkte, Nachfragekonzentration, Preisempfindlichkeit, Ausgangs-Auto-Anteil, Fahrten je Person, Kostenstruktur, Subvention): die drei Ziele mit SLSQP (analytischer Gradient, 5 feste Startpunkte), Preis-Ebene (Gitter 31 × 31), Pareto-Kurve (11 Stufen), Subventions-Kurve (9 Stützstellen), drei Welten bei niedriger, mittlerer und hoher Preisempfindlichkeit. Eine Meldung in drei Zuständen (drei Preise, Gewinn- und Erlösmaximum fallen zusammen, Nulltarif zulässig).
- **Messreihe (vorgerechnet, nie live):** 40 Netze (Zufalls-Seeds 1 bis 40, 20 bis 30 Haltepunkte), drei Kostenstrukturen, erzeugt mit `tools/sweep_tarif.py`, ausgewertet mit `tools/analyse.py` und als `data/tar_results.json` (unter 100 kB) abgelegt. Die Rohdaten (rund 4 MB) liegen nicht im Repo; die Werkzeuge schreiben sie in den Ordner der Umgebungsvariable `TAR_RAW_DIR`. `tools/pruefungen.py` enthält die Modellprüfungen, `tools/dump_results.py` schreibt die Ergebnisdatei.
- **Vorab-Reihe und Neumessung:** Die erste Messreihe lief auf einem numpy-Generator; für den Bau wurde alles auf dem ganzzahligen Generator neu gemessen (siehe „Befunde und Korrekturen gegenüber dem Plan“).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für den Zonentarif dieses Modells auf synthetischen Netzen (Mittel ± Standardfehler über 40 Netze, Kostenstruktur mittel, Subvention heute), stehen in `data/tar_results.json` und sind in `tests/test_claims.py` nachgerechnet.

| Frage | Befund |
|---|---|
| Was bringen die drei Ziele? | **Max-R:** 108,8 ± 0,2 % Erlös bei 77,5 ± 0,2 % der Fahrten. **Max-P:** 105,0 ± 0,6 % Erlös bei 62,8 ± 1,0 % der Fahrten, Bilanzgewinn 28,5 ± 1,5 % des heutigen Erlöses. **Max-D:** 100,1 ± 0,0 % der Fahrten bei Bilanz wie heute. **Nulltarif:** 142,1 ± 0,3 % der Fahrten, Bilanz −133,7 ± 1,5 % des heutigen Erlöses. |
| Wie liegen die Preise? | Preisniveau (Mittel von Einzel- und Monatskartenpreis relativ zu heute): Max-R 1,55 ± 0,01, Max-P 1,99 ± 0,03, Max-D 1,00 ± 0,00; die Rangfolge Max-D < Max-R < Max-P gilt in 40 von 40 Netzen. Max-D liegt in 39 von 40 Netzen auf 2 % am heutigen Tarif. |
| Was kostet ein falsches Ziel? | Mit den Preisen von Max-R fehlen gegenüber Max-D 22,6 ± 0,2 Prozentpunkte der Fahrten; mit den Preisen von Max-D fehlen gegenüber Max-R 8,8 ± 0,2 Prozentpunkte Erlös. Zwischen Max-R und Max-P ist der Abstand klein: 5,1 ± 0,6 Punkte Gewinn bzw. 3,9 ± 0,5 Punkte Erlös. |
| Hängt das an den Kosten? | **Ja, nur Max-P.** Das Preisniveau-Verhältnis Max-P zu Max-R beträgt 1,09 ± 0,01 bei hohen Fixkosten, 1,28 ± 0,02 bei mittleren und 1,58 ± 0,03 bei niedrigen. Bei hohen Fixkosten fallen Gewinn- und Erlösmaximum in 7 von 40 Netzen zusammen, bei mittleren in 0 von 40. Max-R hängt nie an den Kosten. |
| Wo steckt der Zielkonflikt? | Max-D unter der Bedingung, mindestens den Anteil f des Erlösmaximums einzunehmen: Bei f = 0,9 sind es 102,6 ± 0,3 % der Fahrten. **Der letzte Schritt von 0,9 auf 1,0 kostet 25,1 ± 0,1 Prozentpunkte Fahrten, der davor (0,8 auf 0,9) nur 9,4 ± 0,0**; der letzte Schritt ist in allen 40 Netzen der größte. |
| Was kostet der Nulltarif? | Die kleinste Subvention, bei der Max-D zulässig ist, liegt bei 0,14 ± 0,02 Betriebskosten (in manchen Netzen negativ), heute sind es 0,33 ± 0,02; der Nulltarif ist ab 1,23 ± 0,01 zulässig. Kurz vorher (0,97-fach) ist zuerst der Einzelkartenpreis null, die Monatskarte bleibt positiv. |
| Wie stark hängt der Preis an der Annahme? | Preisempfindlichkeit ×0,75: Preisniveau von Max-R **+25,0 %**; ×1,25: **−14,4 %**. Der „Regret“ (Erlöspunkte, die der nominale Preis in der anderen Welt verliert) liegt dabei nur bei 3,1 bzw. 1,7 Punkten. Auto-Anteil ×0,75 / ×1,25: +13,9 % / −10,7 %. |
| Welche Annahmen wirken nicht? | **Zeitwert und Auto-Kilometerkosten:** Nach der Kalibrierung auf den Ausgangs-Modal-Split ändert ein Faktor 0,75 oder 1,25 nichts (größte Änderung unter 1e-6). Ohne Neukalibrierung wirken sie mit 9 bis 17 %, sind dann aber eine Annahme über den Ausgangs-Modal-Split, nicht über das Verhalten. |
| Wie gut ist die Praxisregel „alle Preise um X % anheben“? | Mit dem je Netz besten X bleibt beim Erlös eine Lücke von 0,9 ± 0,1 % des heutigen Erlöses (global bestes X: 45 %), mit dem Anheben um 10 % noch 5,6 ± 0,2 %; das Optimum liegt fast auf der Geraden durch den heutigen Tarif. Die Lücke hängt am Ausgangstarif: bei zufälligen Ausgangstarifen (±35 % je Preis) sind es 1,9 ± 0,2 % beim Erlös und 3,5 ± 0,2 % beim Gewinn. |
| Findet das lokale Verfahren das Optimum? | Ja: Beim Erlösziel enden 50 Startpunkte alle im selben Punkt (1,0 Endpunkte im Mittel, 0 von 40 Netzen mit mehreren); beim Gewinnziel treffen im Mittel 99,5 % der Starts den besten Fund (flacher Grat, siehe Grenzen). Das Gitter 41 × 41 liegt 0,048 ± 0,005 % unter dem Optimum, das Gitter 21 × 21 0,187 ± 0,017 %. |
| Kannibalisiert die Monatskarte die Einzelkarte? | Monatskarte −10 %: Erlös −1,8 ± 0,0 Punkte, Fahrten +2,1 ± 0,0 Punkte; 41,9 ± 0,3 % der neuen Monatskarten-Nutzer kommen aus der Einzelkarte (bei −30 %: −6,8 Punkte Erlös). Das folgt aus der Logit-Struktur (Unabhängigkeit irrelevanter Alternativen), ist also eine Eigenschaft der Annahme. |
| Wie lange rechnet ein Live-Szenario? | Die drei Ziele 0,21 s, Preis-Ebene 0,17 s, Pareto-Kurve 1,33 s, Subventions-Kurve 1,09 s (Median bei 25 Haltepunkten; alles zusammen 2,87 s). Spinner und Ergebnis-Cache je Einstellung. |

## Ehrliche Grenzen

- **Nachfragefunktionen lassen sich nicht messen.** Fahrpreis-Experimente gibt es nicht; die Nutzenfunktion ist eine Annahme, und der optimale Preis hängt an ihr (+25 % bzw. −14 % bei ±25 % Preisempfindlichkeit). Die Demo zeigt Mechanismen, keine Prognose für ein reales Netz und keine Euro-Aussagen.
- **Die Kostenfunktion ist frei gesetzt.** Ob Gewinn- und Erlösmaximum zusammenfallen oder auseinanderliegen, entscheidet der Fixkostenanteil, der nicht beobachtbar ist; deshalb ist die Kostenstruktur ein sichtbarer Regler und die Meldung hat einen eigenen Zustand dafür.
- **Nur der Zonentarif.** Ein Entfernungstarif entartet im Modell zum Einheitspreis, und das Ranking der Tarifsysteme hängt am Kalibrier-Anker; deshalb gibt es keinen Vergleich der Systeme.
- **Netze sind Kulisse.** Die optimalen relativen Preise streuen über die Netze kaum; die Messreihe hat 40 Netze mit 20 bis 30 Haltepunkten, Standardfehler stehen überall dabei.
- **Die Praxisregel-Lücke hängt am Ausgangstarif**, und das Gewinnziel hat einen flachen Grat: Einzelne Startpunkte enden bis 1e-4 des heutigen Erlöses unter dem besten Fund (Max-P, in 4 von 40 Netzen bei strengem Vergleich).
- **Kannibalisierung ist eine Modelleigenschaft** (IIA des Logit), kein Verhaltensbefund. Keine Heterogenität der Fahrgäste, keine Zahlungsbereitschaft nach Entfernung, keine Liniendichte als Entscheidung.

## Verifikation

- **Kern:** Handrechnung für ein Paar mit zwei Fahrten (Wahrscheinlichkeiten, Erlös, Fahrten, Betrieb von Hand), Kalibrierung trifft den Ziel-Auto-Anteil für jedes Paar, Zeitwert und Auto-Kosten sind nach der Kalibrierung wirkungslos.
- **Orakel (`tests/test_oracle_tarif.py`):** Logit-Erwartungen gegen Monte-Carlo (Gumbel-argmax mit vektorisiertem SplitMix64, nicht aus dem Modell); der analytische Gradient gegen zentrale Differenzen (Messreihe: 105 Punkte, größter relativer Fehler 4,7e-9); SLSQP-Mehrfachstart gegen ein Gitter 41 × 41 (nie schlechter) und gegen 20 Starts (dasselbe Optimum); Struktur (IIA, Nulltarif ist das Optimum von Max-D ab der Subvention S_null).
- **Netz (`tests/test_scenario.py`):** eingefrorene Instanzen für drei Zufalls-Seeds, Spannbaum gegen Kruskal, kürzeste Wege gegen Floyd-Warshall, Gleichstände nach der dokumentierten Regel.
- **Messreihe gegen Live:** Die Mittel der drei Ziele über 40 Netze stimmen mit einer Live-Rechnung auf denselben Netzen überein (`tests/test_results.py`). In der Messreihe selbst: Monte-Carlo-Kontrolle mit 12 Läufen zu 400 000 Ziehungen (größtes |z| = 2,06), zwei Sweep-Läufe bitgleich (137 235 von 137 235 Zahlen).
- **Neumessung:** Von 638 verglichenen Kennzahlen der Vorab-Reihe bestehen 632, 0 sind verschoben, 6 brechen (siehe unten).
- **Mutationsprüfung:** `tools/mutation_check.py` ändert 41 Rechenstellen einzeln (Preise, Zonen, Kosten, Gradient, Kalibrierung, Löser, Generator, Verdikt); alle 41 werden von mindestens einem Test gefunden. Bewusst nicht in der Liste: Fahrzeit und Kilometerkosten des Autos (nach der Kalibrierung wirkungslos), der Maßstab der Auto-Anteil-Steigung und der Sonderfall f = 1 der Pareto-Kurve (liefert dasselbe Maximum).
- **Presets:** Jede Geschichte stimmt auf dem gezeigten Szenario (`tests/test_stories.py`), jede Schwelle kippt einzeln an künstlichen Werten; die Zufalls-Seeds liegen außerhalb der Messreihe (ab 500).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler), Layout (Schnellstart oben, fester Kopf, aktives Preset hervorgehoben), Footer.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `data/tar_results.json`, `data/README_data.md` | vorgerechnete Messreihe und Beschreibung ihrer Schlüssel |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tar_algorithm.py` | Löser: SLSQP-Mehrfachstart, Gitter, Praxisregel, Pareto- und Subventions-Kurve |
| `tar_constants.py` | Modellannahmen, Reglerstufen, Kostenstrukturen, Farben, Presets |
| `tar_evaluation.py` | Live-Szenario: Einstellungen, drei Ziele, Preis-Ebene, Annahmen-Tabelle, Verdikt |
| `tar_layout.py` | Fester Kopf (Stylesheet) |
| `tar_model.py` | Logit-Wahl, Kalibrierung, Erlös, Fahrten, Betriebskosten, Gradient |
| `tar_pdf_export.py` | Szenario-Bericht als PDF |
| `tar_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `tar_results.py` | Laden und Abfragen der vorgerechneten Messreihe |
| `tar_rng.py` | SplitMix64 |
| `tar_scenario.py` | ganzzahliger Netz- und Nachfragegenerator, Spannbaum, kürzeste Wege, Zonen |
| `tar_stories.py` | Abnahmekriterien der Presets |
| `tar_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/` | Sweeps, Analyse, Ergebnisdatei, Modellprüfungen, Preset-Suche, Mutationsprüfung |
| `tests/` | Handrechnung, Orakel, Netz, Auswertung, Ergebnisse, Behauptungen, Presets, Layout, Smoke-Test, Footer |

## Bewusst nicht umgesetzt

- Entfernungstarif und Vergleich der Tarifsysteme, Nested Logit und heterogene Fahrgäste, Zahlungsbereitschaft nach Entfernung.
- Linienfrequenz als Entscheidung (die Betriebskosten folgen nur der Last), Mehrziel-Suchverfahren wie NSGA, echte Tarifdaten, Euro-Wirkungen für reale Netze.
- Regler für Zeitwert und Auto-Kilometerkosten (nach der Kalibrierung wirkungslos).

## Befunde und Korrekturen gegenüber dem Plan

- **Neumessung auf dem ganzzahligen Generator:** Von 638 verglichenen Kennzahlen bestehen 632, 0 sind verschoben, 6 brechen. Die Urteilsregeln standen vor der Neumessung fest (besteht: Abstand der Mittel höchstens zwei kombinierte Standardfehler und gleiche Aussage). Die sechs Brüche: Viermal ist es der Status „wirkungslos“ für Zeitwert und Auto-Kilometerkosten, weil die alte Reihe Löser-Rauschen bis 3e-4 hatte und die neue höchstens 4e-7 (die Aussage selbst stimmt in beiden Reihen); zweimal kippt der Status „nominale Max-D-Preise halten die Bilanz“ für den Zeitwert ohne Neukalibrierung (17 von 40 gegen 21 von 40 und 22 von 40 gegen 14 von 40, beidseits nahe 50 %).
- **Subvention:** Sie verschiebt nur Max-D und die Bilanz; der optimale Preis von Max-R und Max-P bleibt gleich. Im Plan stand sie fälschlich als Regler für Max-P.
- **Meldung:** Der dritte Zustand ist „Nulltarif zulässig“ (Max-D ist der Nulltarif) statt „Max-D nur mit Subvention zulässig“, denn bei „wie heute“ ist Max-D immer zulässig.
- **Presets:** sechs, alle mit demselben Zufalls-Seed 503 (außerhalb der Messreihe 1 bis 40); das erste Preset „Standardfall“ ist die Voreinstellung. Preisempfindlichkeit „hoch“ und „niedrig“ sind selbst Presets, deshalb zeigt eine einzelne Änderung dieses Reglers genau dieses Preset hervorgehoben.
- **Messreihe:** 40 statt 30 Netze (das Rechnen ist billig).

## Verwandte Demos mit demselben mathematischen Modell

- **[transit-demo](https://github.com/sebastian-hanisch/transit-demo):** das feste Liniennetz; dort wird die Linienführung gewählt, hier der Preis auf einem gegebenen Netz.
- **[taktfahrplan-demo](https://github.com/sebastian-hanisch/taktfahrplan-demo):** Abfahrtsminuten für feste Linien, die nächste Stufe nach dem Liniennetz.
- **[rufbus-demo](https://github.com/sebastian-hanisch/rufbus-demo):** Rufbus oder Linienbus: wo kippt es? Gleiche Kulisse (Netz, Nachfrage), andere Frage (Betriebsform statt Preis).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zum Thema: [ÖPNV und Fernverkehr optimieren](https://sebastianhanisch.net/oepnv-fernverkehr-optimierung.html).
