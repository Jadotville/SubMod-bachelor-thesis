"""Save result tables and analysis-style scatter plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def subgroups_only(result_df: pd.DataFrame) -> pd.DataFrame:
    return result_df[result_df["subgroup"].astype(str) != "Dataset"].copy()


def save_top_table(
    result_df: pd.DataFrame,
    out_csv: Path,
    *,
    top_k: int = 15,
    quality_col: str = "quality",
) -> pd.DataFrame:
    """Write top-k subgroups (excl. Dataset row) as a readable CSV table."""
    sg = subgroups_only(result_df)
    if quality_col not in sg.columns:
        quality_col = "quality_ga" if "quality_ga" in sg.columns else "quality"
    sort_col = quality_col
    cols = [
        c
        for c in [
            "quality_ga",
            "quality",
            "subgroup",
            "size_sg",
            "size_sg_train",
            "size_sg_test",
            "global_test_score",
            "local_test_score",
            "local_train_score",
        ]
        if c in sg.columns
    ]
    top = sg.nlargest(min(top_k, len(sg)), sort_col)[cols] if len(sg) else sg
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_csv, index=False, encoding="utf-8")
    return top


def plot_quality_vs_size(
    result_df: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
    quality_col: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Scatter like archive/experiments_real_world/results/analysis.ipynb:
    x=size_sg, y=quality, color=local_test_score, top-5 annotated.
    """
    sg = subgroups_only(result_df)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if sg.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title + " (no subgroups)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return

    if quality_col is None:
        quality_col = "quality" if "quality" in sg.columns else "quality_ga"

    # Prefer full-cover size_sg; fall back for legacy result tables.
    size_col = "size_sg" if "size_sg" in sg.columns else "size_sg_full"

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        sg[size_col],
        sg[quality_col],
        c=sg["local_test_score"],
        cmap="viridis",
        alpha=0.55,
        s=18,
        edgecolors="none",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label=f"{quality_col} = 0")
    ax.set_xlabel("Subgruppengröße (Train + Test)")
    ax.set_ylabel(ylabel or f"Qualität ({quality_col})")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Lokaler Test-Score")

    top5 = sg.nlargest(min(5, len(sg)), quality_col)
    ax.scatter(
        top5[size_col],
        top5[quality_col],
        facecolors="none",
        edgecolors="red",
        s=80,
        linewidths=1.5,
        label="Top 5",
    )
    for _, row in top5.iterrows():
        label = str(row["subgroup"])
        if len(label) > 45:
            label = label[:42] + "…"
        ax.annotate(
            label,
            (row[size_col], row[quality_col]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            alpha=0.85,
        )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
