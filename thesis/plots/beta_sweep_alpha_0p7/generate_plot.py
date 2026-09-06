#!/usr/bin/env python3
"""Generate the beta sweep of Local Retraining Gain at alpha = 0.7."""
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
ALPHA = 0.7
BETAS = tuple(np.round(np.arange(0.0, 1.2 + 1e-9, 0.1), 10))
MODELS = ("lr", "rf", "lgbm", "mlp")
MODEL_LABELS = {"lr": "logistische Regression", "rf": "Random Forest", "lgbm": "LightGBM", "mlp": "MLP"}
MODEL_COLORS = {"lr": "#2F6F9F", "rf": "#7A3E9D", "lgbm": "#B8860B", "mlp": "#C0392B"}


def load_candidates() -> pd.DataFrame:
    """Load valid, positive-quality candidates from completed local results."""
    for path in (SUBGROUPS_PATH, FINDINGS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Benötigte Ergebnisdatei fehlt: {path}")
    frame = pd.read_csv(SUBGROUPS_PATH)
    required = {"dataset", "model", "subgroup", "size_sg", "quality_raw", "class_balance"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{SUBGROUPS_PATH} fehlen Spalten: {sorted(missing)}")
    frame = frame[frame["subgroup"].astype(str).ne("Dataset")].copy()
    for column in ("size_sg", "quality_raw", "class_balance"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["size_sg", "quality_raw"])
    frame["class_balance"] = frame["class_balance"].fillna(0.0).clip(lower=0.0)
    frame = frame[(frame["size_sg"] > 0) & (frame["quality_raw"] > 0)]
    sizes = pd.read_csv(FINDINGS_PATH, usecols=["dataset", "n_rows"]).drop_duplicates("dataset")
    frame["size_frac"] = frame["size_sg"] / frame["dataset"].map(sizes.set_index("dataset")["n_rows"])
    return frame.dropna(subset=["size_frac"]).reset_index(drop=True)


def calculate_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for beta in BETAS:
        weighted = frame.copy()
        weighted["quality_weighted"] = (
            weighted["quality_raw"]
            * weighted["size_frac"].pow(ALPHA)
            * weighted["class_balance"].pow(beta)
        )
        maximum = weighted.groupby("dataset")["quality_weighted"].transform("max")
        weighted["quality_scaled"] = weighted["quality_weighted"] / maximum
        for model in MODELS:
            cell = weighted[weighted["model"] == model]
            rho, p_value = spearmanr(cell["class_balance"], cell["quality_scaled"])
            rows.append({"alpha": ALPHA, "beta": beta, "model": model, "n": len(cell),
                         "n_datasets": cell["dataset"].nunique(), "spearman": rho, "spearman_p": p_value})
    return pd.DataFrame(rows)


def plot_sweep(table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#888888", lw=1.0, zorder=1)
    for model in MODELS:
        cell = table[table["model"] == model].sort_values("beta")
        ax.plot(cell["beta"], cell["spearman"], "o-", color=MODEL_COLORS[model],
                lw=1.6, markersize=5, label=MODEL_LABELS[model], zorder=2)
    mean = table.groupby("beta", sort=True)["spearman"].mean()
    ax.plot(mean.index, mean.to_numpy(), ls="--", color="#222222", lw=1.8,
            label="Mittelwert", zorder=3)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"Spearman-Rangkorrelation $\rho$  ($\mathrm{cb}(S)$ vs. $Q_{\mathrm{gew}}$)")
    ax.set_xlim(-0.02, 1.22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticks(BETAS, minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(r"Abhängigkeit des Local Retraining Gain von der Klassenbalance ($\alpha = 0.7$)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    frame = load_candidates()
    table = calculate_sweep(frame)
    table.to_csv(OUT_DIR / "spearman_vs_beta.csv", index=False)
    plot_sweep(table, OUT_DIR / "spearman_vs_beta.png")
    print(f"{len(frame):,} Kandidaten verarbeitet")


if __name__ == "__main__":
    main()
