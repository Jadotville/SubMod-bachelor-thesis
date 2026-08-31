"""
Size-weight sweep (β = 0) on the pooled Q>0 cloud.

Reweighting is arithmetic on quality_raw and |S|/n — no new search.
Fine grid α ∈ {0, 0.1, …, 1.2} so zero-crossings of Spearman ρ can be read
off the grid. The line plot (α vs ρ, one line per model) is the figure that
replaces the synthetic recalibration trade-off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUBGROUPSIZE = HERE.parent / "subgroupsize"
if str(SUBGROUPSIZE) not in sys.path:
    sys.path.insert(0, str(SUBGROUPSIZE))

import plot_quality_vs_size as qvs  # noqa: E402

ALPHAS = tuple(np.round(np.arange(0.0, 1.2 + 1e-9, 0.1), 10))
POLICY_ALPHA = 0.7


def sweep_spearman(frame: pd.DataFrame, alphas: tuple[float, ...] = ALPHAS) -> pd.DataFrame:
    """Spearman ρ(|S|/n, Q_α / max_dataset Q_α) per model, Q_roh > 0 pooled."""
    pos = qvs.positive_subgroups(frame)
    if "size_frac" not in pos.columns:
        raise ValueError("size_frac missing")
    rows: list[dict] = []
    for alpha in alphas:
        weighted = pos.copy()
        weighted["quality_weighted"] = weighted["quality_raw"].to_numpy(
            dtype=float
        ) * np.power(weighted["size_frac"].to_numpy(dtype=float), float(alpha))
        corr = qvs.correlation_by_model_positive(
            weighted, models=qvs.PLOT_MODELS, quality_col="quality_weighted"
        )
        corr.insert(0, "alpha", float(alpha))
        rows.append(corr)
    return pd.concat(rows, ignore_index=True)


def zero_crossings(table: pd.DataFrame) -> pd.DataFrame:
    """
    First grid point with ρ ≥ 0, and the preceding point.

    No interpolation: the crossing sits in (alpha_before, alpha_at] on this grid.
    """
    rows = []
    for model in qvs.PLOT_MODELS:
        part = table[table["model"] == model].sort_values("alpha")
        rho = part["spearman"].to_numpy()
        alphas = part["alpha"].to_numpy()
        nonnegative = np.where(rho >= 0)[0]
        if len(nonnegative) == 0:
            rows.append(
                {
                    "model": model,
                    "alpha_before": float(alphas[-1]),
                    "rho_before": float(rho[-1]),
                    "alpha_at": np.nan,
                    "rho_at": np.nan,
                    "note": "kein Nulldurchgang bis α=1.2",
                }
            )
            continue
        i = int(nonnegative[0])
        if i == 0:
            rows.append(
                {
                    "model": model,
                    "alpha_before": np.nan,
                    "rho_before": np.nan,
                    "alpha_at": float(alphas[0]),
                    "rho_at": float(rho[0]),
                    "note": "bereits bei α=0 nichtnegativ",
                }
            )
            continue
        rows.append(
            {
                "model": model,
                "alpha_before": float(alphas[i - 1]),
                "rho_before": float(rho[i - 1]),
                "alpha_at": float(alphas[i]),
                "rho_at": float(rho[i]),
                "note": f"zwischen {alphas[i - 1]:g} und {alphas[i]:g}",
            }
        )
    return pd.DataFrame(rows)


def plot_spearman_vs_alpha(table: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.axvline(
        POLICY_ALPHA,
        color="#bbbbbb",
        lw=1.0,
        ls=":",
        zorder=1,
        label=rf"Protokoll $\alpha={POLICY_ALPHA:g}$",
    )
    for model in qvs.PLOT_MODELS:
        part = table[table["model"] == model].sort_values("alpha")
        ax.plot(
            part["alpha"],
            part["spearman"],
            "o-",
            color=qvs.MODEL_COLORS[model],
            lw=1.6,
            markersize=5,
            label=qvs.MODEL_LABELS[model],
            zorder=2,
        )
    ax.set_xlabel(r"Size-Gewicht $\alpha$   (bei $\beta=0$)")
    ax.set_ylabel(r"Spearman $\rho$  ($|S|/n$ vs. $Q_{\alpha}$, skaliert)")
    ax.set_xlim(-0.02, 1.22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticks(list(ALPHAS), minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        r"Größenpräferenz der Qualität in Abhängigkeit von $\alpha$",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    frame = qvs.load_subgroups()
    print(f"loaded {len(frame):,} rows; Q_roh>0 → {int((frame.quality_raw > 0).sum()):,}")
    table = sweep_spearman(frame, ALPHAS)
    csv_path = HERE / "correlation_by_alpha.csv"
    table.to_csv(csv_path, index=False)
    print("wrote", csv_path)

    crossings = zero_crossings(table)
    crossings.to_csv(HERE / "zero_crossings.csv", index=False)
    print("\nNulldurchgänge (Gitter, ohne Interpolation):")
    for _, row in crossings.iterrows():
        label = qvs.MODEL_LABELS.get(row["model"], row["model"])
        print(f"  {label}: {row['note']}")

    fig_path = plot_spearman_vs_alpha(table, HERE / "spearman_vs_alpha.png")
    print("\nwrote", fig_path)
    wide = table.pivot(index="alpha", columns="model", values="spearman")
    wide = wide.reindex(columns=list(qvs.PLOT_MODELS))
    print("\nSpearman ρ:")
    print(wide.to_string(float_format=lambda x: f"{x:7.3f}"))


if __name__ == "__main__":
    main()
