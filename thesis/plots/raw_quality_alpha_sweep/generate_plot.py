#!/usr/bin/env python3
"""
Generate the Spearman alpha-sweep plot from completed repository results.

Run from any directory:
    python thesis/plots/raw_quality_alpha_sweep/generate_plot.py

Inputs:
    experiments/experiments_real_world_data/results/offline/all_subgroups.csv
    experiments/experiments_real_world_data/results/offline/findings.csv
Outputs (next to this script):
    spearman_raw_quality_vs_alpha.csv
    spearman_raw_quality_vs_alpha.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "experiments_real_world_data" / "results" / "offline"
SUBGROUPS_PATH = RESULTS_DIR / "all_subgroups.csv"
FINDINGS_PATH = RESULTS_DIR / "findings.csv"
OUT_DIR = Path(__file__).resolve().parent
ALPHAS = tuple(np.round(np.arange(0.0, 1.2 + 1e-9, 0.1), 10))
MODELS = ("lr", "rf", "lgbm", "mlp")
MODEL_LABELS = {
    "lr": "logistische Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
}
MODEL_COLORS = {
    "lr": "#2F6F9F",
    "rf": "#7A3E9D",
    "lgbm": "#B8860B",
    "mlp": "#C0392B",
}


def load_candidates() -> pd.DataFrame:
    """Load valid, positive-quality candidates and their relative sizes."""
    for path in (SUBGROUPS_PATH, FINDINGS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Benötigte Ergebnisdatei fehlt: {path}")

    frame = pd.read_csv(SUBGROUPS_PATH)
    needed = {"dataset", "model", "subgroup", "size_sg", "quality_raw"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{SUBGROUPS_PATH} fehlen Spalten: {sorted(missing)}")

    frame = frame[frame["subgroup"].astype(str).ne("Dataset")].copy()
    frame["size_sg"] = pd.to_numeric(frame["size_sg"], errors="coerce")
    frame["quality_raw"] = pd.to_numeric(frame["quality_raw"], errors="coerce")
    frame = frame.dropna(subset=["size_sg", "quality_raw"])
    frame = frame[(frame["size_sg"] > 0) & (frame["quality_raw"] > 0)]

    dataset_sizes = (
        pd.read_csv(FINDINGS_PATH, usecols=["dataset", "n_rows"])
        .drop_duplicates("dataset")
        .set_index("dataset")["n_rows"]
    )
    frame["size_frac"] = frame["size_sg"] / frame["dataset"].map(dataset_sizes)
    return frame.dropna(subset=["size_frac"]).reset_index(drop=True)


def calculate_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate pooled, dataset-normalized Spearman rho for every alpha."""
    rows: list[dict[str, float | int | str]] = []
    for alpha in ALPHAS:
        weighted = frame.copy()
        weighted["quality_weighted"] = (
            weighted["quality_raw"] * weighted["size_frac"].pow(alpha)
        )
        max_quality = weighted.groupby("dataset")["quality_weighted"].transform("max")
        weighted["quality_scaled"] = weighted["quality_weighted"] / max_quality

        for model in MODELS:
            cell = weighted[weighted["model"] == model]
            rho, p_value = spearmanr(cell["size_frac"], cell["quality_scaled"])
            rows.append(
                {
                    "alpha": alpha,
                    "model": model,
                    "n": len(cell),
                    "n_datasets": cell["dataset"].nunique(),
                    "spearman": rho,
                    "spearman_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def plot_sweep(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#888888", lw=1.0, zorder=1)
    for model in MODELS:
        cell = table[table["model"] == model].sort_values("alpha")
        ax.plot(
            cell["alpha"],
            cell["spearman"],
            "o-",
            color=MODEL_COLORS[model],
            lw=1.6,
            markersize=5,
            label=MODEL_LABELS[model],
            zorder=2,
        )
    mean = table.groupby("alpha", sort=True)["spearman"].mean()
    ax.plot(
        mean.index,
        mean.to_numpy(),
        ls="--",
        color="#222222",
        lw=1.8,
        label="Mittelwert",
        zorder=3,
    )
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(
        r"Spearman-Rangkorrelation $\rho$ "
        r"($|S|/n$ vs. $Q_{\mathrm{gew}}$)"
    )
    ax.set_xlim(-0.02, 1.22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticks(ALPHAS, minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        r"Abhängigkeit des Local Retraining Gain von der Subgruppengröße "
        r"($\beta = 0$)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    frame = load_candidates()
    table = calculate_sweep(frame)
    csv_path = OUT_DIR / "spearman_raw_quality_vs_alpha.csv"
    png_path = OUT_DIR / "spearman_raw_quality_vs_alpha.png"
    table.to_csv(csv_path, index=False)
    plot_sweep(table, png_path)
    print(f"{len(frame):,} Kandidaten verarbeitet")
    print(f"geschrieben: {csv_path}")
    print(f"geschrieben: {png_path}")


if __name__ == "__main__":
    main()
