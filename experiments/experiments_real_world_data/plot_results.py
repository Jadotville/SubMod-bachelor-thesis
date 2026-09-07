"""Save result tables and analysis-style scatter plots."""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd

MODEL_DISPLAY_NAMES = {
    "lr": "Logistische Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "Mehrschichtiges Perzeptron",
    "tabpfn": "TabPFN",
}

DATASET_DISPLAY_NAMES = {
    "covertype": "Covertype",
    "electricity": "Electricity",
    "Diabetes130US": "Diabetes 130-US",
    "UKRoadSafety": "UK Road Safety",
    "ACSIncome": "ACS Income",
    "ACSMobility": "ACS Mobility",
    "ACSPublicCoverage": "ACS Public Coverage",
    "ACSTravelTime": "ACS Travel Time",
    "adult": "Adult",
    "bank-marketing": "Bank Marketing",
    "credit-default": "Credit-Card Default",
    "PhishingWebsites": "Phishing Websites",
    "mushroom": "Mushroom",
}


def format_plot_title(dataset: str, model: str) -> str:
    """German-style title with capitalized nouns."""
    dataset_label = DATASET_DISPLAY_NAMES.get(
        dataset,
        dataset.replace("-", " ").replace("_", " ").title(),
    )
    model_label = MODEL_DISPLAY_NAMES.get(model, model)
    return f"{dataset_label} — {model_label}"


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
    Scatter of quality against subgroup size.

    The five highest-ranked points stay marked, but as numbers. The selector
    strings sit in a list under the axes so nearby points do not stack labels.
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

    fig, (ax, ax_list) = plt.subplots(
        2,
        1,
        figsize=(10, 8.4),
        gridspec_kw={"height_ratios": [3.5, 1.55], "hspace": 0.18},
    )
    sc = ax.scatter(
        sg[size_col],
        sg[quality_col],
        c=sg["local_test_score"],
        cmap="viridis",
        alpha=0.55,
        s=18,
        edgecolors="none",
    )
    ax.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=r"$Q_{\mathrm{gew}} = 0$",
    )
    ax.set_xlabel("Subgruppengröße")
    ax.set_ylabel(ylabel or r"$Q_{\mathrm{gew}}$")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("AUC des lokalen Modells auf Subgruppe")

    top5 = sg.nlargest(min(5, len(sg)), quality_col)
    ax.scatter(
        top5[size_col],
        top5[quality_col],
        facecolors="none",
        edgecolors="red",
        s=90,
        linewidths=1.5,
        label="Top 5",
        zorder=3,
    )
    number_offsets = [
        (14 * math.cos(math.radians(-20 + (rank - 1) * 72)),
         14 * math.sin(math.radians(-20 + (rank - 1) * 72)))
        for rank in range(1, 6)
    ]
    stroke = [patheffects.withStroke(linewidth=2.4, foreground="white")]
    lines: list[str] = []
    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        dx, dy = number_offsets[rank - 1]
        ax.annotate(
            str(rank),
            (row[size_col], row[quality_col]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            fontweight="bold",
            color="0.15",
            ha="center",
            va="center",
            zorder=4,
            path_effects=stroke,
        )
        wrapped = textwrap.fill(str(row["subgroup"]), width=92, subsequent_indent="   ")
        lines.append(f"{rank}. {wrapped}")
    ax.legend(loc="best")

    ax_list.axis("off")
    ax_list.set_xlim(0, 1)
    ax_list.set_ylim(0, 1)
    ax_list.text(
        0.0,
        1.0,
        "Beschreibungen der 5 höchstbewerteten Subgruppen\n" + "\n".join(lines),
        transform=ax_list.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        family="sans-serif",
        linespacing=1.45,
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def regenerate_plots(results_dir: Path) -> int:
    """Rebuild plot_quality_size.png for every completed run directory."""
    count = 0
    candidates = sorted(
        {*results_dir.glob("*__*"), *results_dir.glob("*/*__*")}
    )
    for run_dir in candidates:
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "subgroups.csv"
        if not csv_path.is_file():
            continue
        meta_path = run_dir / "meta.json"
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            dataset = str(meta.get("dataset", run_dir.name.partition("__")[0]))
            model = str(meta.get("model", run_dir.name.rpartition("__")[2]))
        else:
            dataset, model = run_dir.name.partition("__")[0], run_dir.name.rpartition("__")[2]
        plot_quality_vs_size(
            pd.read_csv(csv_path),
            run_dir / "plot_quality_size.png",
            title=format_plot_title(dataset, model),
        )
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate plot_quality_size.png from saved subgroups.csv files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "offline",
        help="Root directory with completed run folders",
    )
    args = parser.parse_args(argv)
    if not args.results_dir.is_dir():
        print(f"{args.results_dir} existiert nicht", file=sys.stderr)
        return 1
    count = regenerate_plots(args.results_dir)
    print(f"{count} Plot(s) geschrieben unter {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
