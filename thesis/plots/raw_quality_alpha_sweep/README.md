# Spearman-α-Sweep

Dieser Ordner enthält den Plot der Spearman-Rangkorrelation zwischen
Subgruppengröße und Qualitätsfunktion über den α-Sweep.

Bei `α = 0` ist `Q_gew` identisch mit der rohen Qualität `quality_raw`.
Für jeden weiteren α-Wert wird

\[
Q_{\mathrm{gew}} = Q_{\mathrm{roh}} \cdot (|S|/n)^\alpha
\]

gebildet. Berücksichtigt werden die bereits berechneten Kandidaten mit
`quality_raw > 0`. Größe und Qualität werden innerhalb jedes Datensatzes
skaliert und anschließend je Modell gepoolt korreliert.

## Reproduktion

Benötigt Python mit `pandas`, `numpy`, `scipy` und `matplotlib`. Vom Root des
Repositories aus:

```bash
python thesis/plots/raw_quality_alpha_sweep/generate_plot.py
```

Das Skript nutzt ausschließlich diese vorhandenen Ergebnisdateien:

- `experiments/experiments_real_world_data/results/offline/all_subgroups.csv`
- `experiments/experiments_real_world_data/results/offline/findings.csv`

Es erzeugt bzw. überschreibt:

- `spearman_raw_quality_vs_alpha.png`
- `spearman_raw_quality_vs_alpha.csv`
