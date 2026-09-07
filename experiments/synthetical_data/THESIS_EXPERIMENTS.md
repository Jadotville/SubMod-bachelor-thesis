# Synthetische Experimente: Arbeit ↔ Code

Die IDs in `run.py`, `datasets.py` und `results/<id>/` bleiben unverändert.
Die Bachelorarbeit nummeriert den Kern neu, damit keine Lücken entstehen
(kein C2 ohne C1, kein B3 ohne B2).

Zuordnung und gestrichene Texte:
`bachelor thesis/writing/archive/synthetische_experimente_entfernte_laeufe.tex`
Vollständiger Anhang vor dem Schnitt:
`bachelor thesis/writing/archive/09a_synthetische_experimente_vor_kern.tex`

## Kern (in der Arbeit)

| Arbeit | Code-ID | Funktion in `run.py` |
| --- | --- | --- |
| A Zufallsmengen | A0 | `exp_a0` |
| B Prior-Shift (AUC) | B1 | `exp_b1` |
| C1 Stückweise Regeln | C2 | `exp_c2` |
| C2 Shortcut | C3 | `exp_c3` |
| C3 Konzeptwechsel | C4 | `exp_c4` |
| D Kapazität | D1 | `exp_d1` |
| E Interaktionsterme | B3 | `exp_b3` |

B (Code B1) läuft im Code zusätzlich unter Accuracy. Die Arbeit berichtet nur die AUC.

E (Code B3) läuft im Code auf C1–C6. Die Arbeit berichtet C1, C2 und C3 (Code C2, C3, C4).

Nur die Code-IDs des Kerns ausführen:

```
python experiments/synthetical_data/run.py --group thesis
```

gleichbedeutend mit `--experiments A0 B1 B3 C2 C3 C4 D1`.

## Nicht in der Arbeit

| Code-ID | Kurz |
| --- | --- |
| A1 | Labelrauschen auf homogenem DGP |
| B2 | Feature-Shift, Regel bleibt lernbar |
| C1 | heterogene Steigung |
| C5 | Merkmal nur in der Minderheit |
| C6 | Misspezifikation plus Feature-Shift |
| D2 | Regularisierung |
| D3 | Streuung über Modell-Seeds |
| D4 | acht stückweise Regeln gegen Vorzeichenwechsel |
| E1 | Minderheitenanteil |
| E2 | extreme Klassenunbalance |
| E3 | Zahl irrelevanter Merkmale |
| F1 | Suche auf homogenen Daten |
| F2 | kleine Stichprobe, viele Rauschspalten |
| G1 | Rang der Ground Truth, Suche nur über den Indikator |
