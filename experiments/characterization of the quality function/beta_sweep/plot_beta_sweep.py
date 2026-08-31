"""
Balance-weight sweep (α = 0) on the pooled Q>0 cloud.

Same procedure as the α-sweep, but Spearman is ρ(cb, Q_β / max_dataset Q_β):
raw quality against class balance. Reweighting is Q_roh · cb^β — no new search.
Fine grid β ∈ {0, 0.1, …, 1.2}.
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

BETAS = tuple(np.round(np.arange(0.0, 1.2 + 1e-9, 0.1), 10))
POLICY_BETA = 0.0
X_COL = "class_balance"


def sweep_spearman(frame: pd.DataFrame, betas: tuple[float, ...] = BETAS) -> pd.DataFrame:
    """Spearman ρ(cb, Q_β / max_dataset Q_β) per model, Q_roh > 0 pooled."""
    pos = qvs.positive_subgroups(frame)
    if X_COL not in pos.columns:
        raise ValueError("class_balance missing")
    rows: list[dict] = []
    for beta in betas:
        weighted = pos.copy()
        cb = weighted[X_COL].fillna(0.0).to_numpy(dtype=float)
        weighted["quality_weighted"] = weighted["quality_raw"].to_numpy(
            dtype=float
        ) * np.power(cb, float(beta))
        corr = qvs.correlation_by_model_positive(
            weighted,
            models=qvs.PLOT_MODELS,
            quality_col="quality_weighted",
            x_col=X_COL,
        )
        corr.insert(0, "beta", float(beta))
        rows.append(corr)
    return pd.concat(rows, ignore_index=True)


def zero_crossings(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in qvs.PLOT_MODELS:
        part = table[table["model"] == model].sort_values("beta")
        rho = part["spearman"].to_numpy()
        betas = part["beta"].to_numpy()
        nonnegative = np.where(rho >= 0)[0]
        if len(nonnegative) == 0:
            rows.append(
                {
                    "model": model,
                    "beta_before": float(betas[-1]),
                    "rho_before": float(rho[-1]),
                    "beta_at": np.nan,
                    "rho_at": np.nan,
                    "note": "kein Nulldurchgang bis β=1.2",
                }
            )
            continue
        i = int(nonnegative[0])
        if i == 0:
            rows.append(
                {
                    "model": model,
                    "beta_before": np.nan,
                    "rho_before": np.nan,
                    "beta_at": float(betas[0]),
                    "rho_at": float(rho[0]),
                    "note": "bereits bei β=0 nichtnegativ",
                }
            )
            continue
        rows.append(
            {
                "model": model,
                "beta_before": float(betas[i - 1]),
                "rho_before": float(rho[i - 1]),
                "beta_at": float(betas[i]),
                "rho_at": float(rho[i]),
                "note": f"zwischen {betas[i - 1]:g} und {betas[i]:g}",
            }
        )
    return pd.DataFrame(rows)


def plot_spearman_vs_beta(table: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.axvline(
        POLICY_BETA,
        color="#bbbbbb",
        lw=1.0,
        ls=":",
        zorder=1,
        label=rf"Protokoll $\beta={POLICY_BETA:g}$",
    )
    for model in qvs.PLOT_MODELS:
        part = table[table["model"] == model].sort_values("beta")
        ax.plot(
            part["beta"],
            part["spearman"],
            "o-",
            color=qvs.MODEL_COLORS[model],
            lw=1.6,
            markersize=5,
            label=qvs.MODEL_LABELS[model],
            zorder=2,
        )
    ax.set_xlabel(r"Balance-Gewicht $\beta$   (bei $\alpha=0$)")
    ax.set_ylabel(r"Spearman $\rho$  ($\mathrm{cb}$ vs. $Q_{\beta}$, skaliert)")
    ax.set_xlim(-0.02, 1.22)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticks(list(BETAS), minor=True)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        r"Balancepräferenz der Qualität in Abhängigkeit von $\beta$",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pooled_raw(frame: pd.DataFrame, out_path: Path) -> Path:
    return qvs.plot_positive_pooled_by_model(
        frame,
        out_path,
        models=qvs.PLOT_MODELS,
        quality_col="quality_raw",
        x_col=X_COL,
        title=(
            r"Alle $Q_{\mathrm{roh}}>0$:  Qualität gegen Klassenbalance "
            r"(Qualität je Datensatz skaliert)"
        ),
        xlabel=r"Klassenbalance $\mathrm{cb}(S)$",
        ylabel=r"$Q_{\mathrm{roh}}\,/\,\max_{\mathrm{Datensatz}} Q_{\mathrm{roh}}$",
        log_x=True,
    )


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    frame = qvs.load_subgroups()
    n_pos = int((frame.quality_raw > 0).sum())
    print(f"loaded {len(frame):,} rows; Q_roh>0 → {n_pos:,}")

    pooled_path = plot_pooled_raw(frame, HERE / "quality_vs_balance_positive_pooled.png")
    print("wrote", pooled_path)

    table = sweep_spearman(frame, BETAS)
    csv_path = HERE / "correlation_by_beta.csv"
    table.to_csv(csv_path, index=False)
    print("wrote", csv_path)

    crossings = zero_crossings(table)
    crossings.to_csv(HERE / "zero_crossings.csv", index=False)
    print("\nNulldurchgänge (Gitter, ohne Interpolation):")
    for _, row in crossings.iterrows():
        label = qvs.MODEL_LABELS.get(row["model"], row["model"])
        print(f"  {label}: {row['note']}")

    fig_path = plot_spearman_vs_beta(table, HERE / "spearman_vs_beta.png")
    print("\nwrote", fig_path)
    wide = table.pivot(index="beta", columns="model", values="spearman")
    wide = wide.reindex(columns=list(qvs.PLOT_MODELS))
    print("\nSpearman ρ:")
    print(wide.to_string(float_format=lambda x: f"{x:7.3f}"))


if __name__ == "__main__":
    main()
