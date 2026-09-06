# SubROC-Rangkorrelationen bei α = 0,7 und β = 0

Die Heatmap zeigt pro Datensatz und Modell die Spearman-Rangkorrelation zwischen
der Local-Retraining-Gain-Qualität `quality` und der größengewichteten
SubROC-Qualität `phi_gew`.

Die Werte stammen ausschließlich aus der vorhandenen Auswertung
`experiments/comparison_subroc/results/phi_on_adapt/summary.csv`. Diese wird
von `experiments/comparison_subroc/score_phi.py` mit dem standardmäßigen
Größengewicht `α = 0,7` und ohne Klassenbalance-Gewicht (`β = 0`) erzeugt.
Ein Gedankenstrich bedeutet, dass für diese Datensatz-Modell-Kombination keine
definierte Rangkorrelation vorliegt.

Vom Repository-Root aus:

```bash
.venv/bin/python thesis/plots/subroc_rank_correlation_alpha_0p7_beta_0/generate_plot.py
```

Das Skript erzeugt in diesem Ordner:

- `subroc_rank_correlations.png`
- `subroc_rank_correlations.csv`
