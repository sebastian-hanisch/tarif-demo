# data/tar_results.json

Vorgerechnete Messreihe der Tarif-Demo (nie im Browser gerechnet). Erzeugt mit `tools/sweep_tarif.py` (Rohdaten), `tools/analyse.py` (Aggregation) und `tools/dump_results.py` (diese Datei). UTF-8, Zahlen auf 6 gültige Stellen gerundet. ±Unendlich stünde als Text `"inf"` / `"-inf"` in der Datei (kommt nicht vor), NaN als `null`.

Netze: Zufalls-Seeds 1 bis 40, Haltepunkte 20 + (Seed − 1) mod 11, Nachfragekonzentration mittel (50 %), nur Zonentarif, Preisempfindlichkeit mittel, Ausgangs-Auto-Anteil 45 %.

**Statistikfelder** haben die Form `{n, mean, se, med, p10, p90}` (Mittel, Standardfehler, Median, 10. und 90. Perzentil, Zahl der Netze), **Anteile** die Form `{k, n, p}`.

| Schlüssel | Inhalt |
|---|---|
| `meta` | Zahl der Aufträge je Rohdatei |
| `modell` | je Kostenstruktur (`fix`, `mittel`, `var`): Haltepunkte, Kanten, Fahrzeit, Entfernung, Fahrpreiselastizität, Kostendeckung heute, Modal Split heute, mittlere Fahrtenzahl, Fixkostenanteil, heutige Subvention (`S_today_over_C0`) |
| `kreuz` | je Kostenstruktur die Zeilen `base` (heutiger Tarif), `R`, `P`, `D` (Preise des Ziels), `null` (Nulltarif); Spalten `R` (Erlös, % von heute), `P` (Bilanzgewinn, % des heutigen Erlöses), `T` (Fahrten, % von heute), `oev` (ÖV-Anteil, %), `zul` (Bilanz mindestens wie heute) |
| `preise` | Preise der drei Ziele absolut (`x_s`, `x_m`) und relativ zu heute (`x_s_rel`, `x_m_rel`), Preisniveau, Verhältnisse `ratio_P_R`, `ratio_D_R`, `ratio_D_P`, Anteil Netze mit Max-P = Max-R, Rangfolge der Preisniveaus |
| `verlust` | „Preise des Ziels X verwenden“: Verlust gegenüber dem eigenen Optimum in Erlös (`R`), Gewinn (`P`) und Fahrten (`T`) |
| `pareto` | `stufen`: Max-D unter Erlös ≥ f · Erlösmaximum für f = 0 bis 1; `schritte`: Fahrtenverlust je Schritt von 0,1; Monotonie und „letzter Schritt am größten“ |
| `subvention` | `Smin`, `Stoday`, `Snull` (in Vielfachen der heutigen Betriebskosten), `heute`, `raster` (Subvention 0,2 bis 1,4: Fahrten, ÖV-Anteil, Preisniveau, Zulässigkeit, Anteil Nulltarif), `null_vielfache` (0,9-, 0,97-, 1,001-faches von `Snull`) |
| `verfahren` | nur Kostenstruktur mittel, je Ziel `R` und `P`: Abstand zum besten Fund je Verfahren (heutiger Tarif, Praxisregel, Gitter 21 × 21 und 41 × 41, lokal mit 10 und 50 Starts), Rauschen zweier Läufe, Meldeschwelle, Zahl der Endpunkte, Rechenzeit |
| `annahmen` | nur mittel: Annahmen ±25 % (`mu`, `lam`, `car_fix`, `car_km`, `car_share`, `k_mean`, `cdep`, `dmin`) mit Neukalibrierung (`recal`) bzw. festem Zuschlag (`fest`): Verschiebung des Preisniveaus je Ziel (`shift`), Regret, Zulässigkeit, `wirkungslos` |
| `kannibal` | nur mittel: Monatskarte auf 90, 80 und 70 % des Preises: Erlös- und Fahrtenänderung, Verschiebung des Modal Split, Herkunft der neuen Nutzer, IIA-Abweichung |
| `luecke` | nur mittel: Lücke der Praxisregel „alle Preise um X % anheben“ bei zufälligen Ausgangstarifen (200 Fälle) |
| `rechenzeit` | Median, 90. Perzentil und Maximum über 10 Netze für 20, 25 und 30 Haltepunkte: Modellaufbau, drei Ziele, Preis-Ebene, Pareto, Subventions-Kurve, Annahmen-Tabelle, alles zusammen |
| `pruefungen` | Ergebnisse der Modellprüfungen (`tools/pruefungen.py`): Monte-Carlo, Gradient, Löser gegen Gitter, Determinismus |
