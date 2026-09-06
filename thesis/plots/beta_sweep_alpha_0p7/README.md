# β-Sweep bei α = 0,7

`generate_plot.py` verwendet ausschließlich die vorhandenen Offline-Ergebnisse
`experiments/experiments_real_world_data/results/offline/all_subgroups.csv`
und `findings.csv`. Es bildet für Kandidaten mit `quality_raw > 0`
`Q_gew = quality_raw · (|S|/n)^0.7 · cb(S)^β`, skaliert die Qualität je
Datensatz und berechnet die Spearman-Rangkorrelation zwischen `cb(S)` und
`Q_gew` je Modell.

Vom Repository-Root aus ausführen:

```bash
.venv/bin/python thesis/plots/beta_sweep_alpha_0p7/generate_plot.py
```

Das Skript erzeugt `spearman_vs_beta.png` und `spearman_vs_beta.csv` in diesem
Ordner.
