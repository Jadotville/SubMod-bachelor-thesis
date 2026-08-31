"""
Raw adaptability quality vs subgroup size.

Uses the finished real-world offline search
(``experiments_real_world_data/results/offline/all_subgroups.csv``): every
evaluated candidate, not just the top-k. The α/β-sweep scripts import this
module for loading, reweighting, and the pooled scatter plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = Path(__file__).resolve().parent
CHAR_DIR = HERE.parent
OFFLINE_RESULTS = (
    CHAR_DIR.parent
    / "experiments_real_world_data"
    / "results"
    / "offline"
    / "all_subgroups.csv"
)
# Shipped copy from an earlier batch, only if the offline summary is missing.
_SHIPPED = CHAR_DIR / "data" / "all_subgroups.csv"
DEFAULT_RESULTS = OFFLINE_RESULTS if OFFLINE_RESULTS.is_file() else _SHIPPED

# Dataset roles for legend order; kept here so the public folder does not
# need the real-world pipeline.
_DATASET_ROLES = {
    "covertype": "candidate",
    "electricity": "candidate",
    "Diabetes130US": "candidate",
    "road-safety": "candidate",
    "ACSPublicCoverage": "candidate",
    "ACSIncome": "baseline",
    "ACSMobility": "baseline",
    "ACSTravelTime": "baseline",
    "adult": "baseline",
    "bank-marketing": "baseline",
    "default-of-credit-card-clients": "baseline",
    "PhishingWebsites": "control",
    "mushroom": "control",
    "MagicTelescope": "illustration",
}


def dataset_role(name: str) -> str:
    return _DATASET_ROLES.get(name, "unknown")

MODEL_ORDER = ("lr", "rf", "lgbm", "mlp")
MODEL_COLORS = {
    "lr": "#2F6F9F",
    "rf": "#7A3E9D",
    "lgbm": "#B8860B",
    "mlp": "#C0392B",
}
MODEL_LABELS = {
    "lr": "logistische Regression",
    "rf": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
}
DATASET_TITLES = {
    "covertype": "Covertype",
    "electricity": "Electricity",
    "Diabetes130US": "Diabetes 130 US",
    "road-safety": "Road safety",
    "ACSPublicCoverage": "ACS Public Coverage",
    "ACSIncome": "ACS Income",
    "ACSMobility": "ACS Mobility",
    "ACSTravelTime": "ACS Travel Time",
    "adult": "Adult",
    "bank-marketing": "Bank marketing",
    "default-of-credit-card-clients": "Credit default",
    "PhishingWebsites": "Phishing Websites",
    "mushroom": "Mushroom",
}
ROLE_LABELS = {
    "candidate": "Kandidat",
    "baseline": "Baseline",
    "control": "Kontrolle",
    "illustration": "Illustration",
}

# A few datasets spanning the four roles, with enough candidates to see a cloud.
PLOT_DATASETS = ("covertype", "electricity", "adult", "PhishingWebsites")
PLOT_MODELS = MODEL_ORDER

MIN_N_CORR = 8


def load_subgroups(path: Path = DEFAULT_RESULTS) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Run the real-world offline batch and "
            "summarize_runs.py --csv first."
        )
    frame = pd.read_csv(path)
    need = {"dataset", "model", "size_sg", "quality_raw"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    frame = frame.copy()
    if "subgroup" in frame.columns:
        frame = frame[frame["subgroup"].astype(str).ne("Dataset")]
    frame["size_sg"] = pd.to_numeric(frame["size_sg"], errors="coerce")
    frame["quality_raw"] = pd.to_numeric(frame["quality_raw"], errors="coerce")
    if "class_balance" in frame.columns:
        frame["class_balance"] = pd.to_numeric(
            frame["class_balance"], errors="coerce"
        ).clip(lower=0.0)
    frame = frame.dropna(subset=["size_sg", "quality_raw"])
    frame = frame[frame["size_sg"] > 0]
    if "rolle" not in frame.columns:
        frame["rolle"] = frame["dataset"].map(dataset_role)
    n_map = _dataset_sizes(path)
    if n_map:
        frame["n_rows"] = frame["dataset"].map(n_map)
        frame["size_frac"] = frame["size_sg"] / frame["n_rows"]
    return frame.reset_index(drop=True)


def _dataset_sizes(subgroups_path: Path) -> dict[str, int]:
    findings = subgroups_path.parent / "findings.csv"
    if not findings.is_file():
        return {}
    table = pd.read_csv(findings, usecols=["dataset", "n_rows"])
    return (
        table.drop_duplicates("dataset")
        .set_index("dataset")["n_rows"]
        .astype(int)
        .to_dict()
    )


def positive_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["quality_raw"] > 0].copy()


def _corr_pair(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = int(len(x))
    if n < MIN_N_CORR or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {
            "n": n,
            "spearman": np.nan,
            "spearman_p": np.nan,
            "pearson": np.nan,
            "pearson_p": np.nan,
        }
    rho, p_rho = stats.spearmanr(x, y)
    r, p_r = stats.pearsonr(x, y)
    return {
        "n": n,
        "spearman": float(rho),
        "spearman_p": float(p_rho),
        "pearson": float(r),
        "pearson_p": float(p_r),
    }


def correlation_table(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Spearman / Pearson of (size_sg, quality_raw) per dataset × model.

    ``spearman_pos`` is the same rank correlation restricted to candidates with
    ``quality_raw > 0`` — the part of the cloud that a search without size
    weight would actually promote.
    """
    sub = frame
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    if models is not None:
        sub = sub[sub["model"].isin(models)]

    rows: list[dict] = []
    for (dataset, model), part in sub.groupby(["dataset", "model"], sort=False):
        all_c = _corr_pair(part["size_sg"].to_numpy(), part["quality_raw"].to_numpy())
        pos = part[part["quality_raw"] > 0]
        pos_c = _corr_pair(pos["size_sg"].to_numpy(), pos["quality_raw"].to_numpy())
        rows.append(
            {
                "rolle": part["rolle"].iloc[0] if "rolle" in part else dataset_role(dataset),
                "dataset": dataset,
                "model": model,
                "n": all_c["n"],
                "spearman": all_c["spearman"],
                "spearman_p": all_c["spearman_p"],
                "pearson": all_c["pearson"],
                "pearson_p": all_c["pearson_p"],
                "n_pos": pos_c["n"],
                "spearman_pos": pos_c["spearman"],
                "spearman_pos_p": pos_c["spearman_p"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model"] = pd.Categorical(out["model"], list(MODEL_ORDER), ordered=True)
    role_order = {"candidate": 0, "baseline": 1, "control": 2, "illustration": 3}
    out["_role"] = out["rolle"].map(role_order).fillna(9)
    if datasets is not None:
        ds_order = {name: i for i, name in enumerate(datasets)}
        out["_ds"] = out["dataset"].map(ds_order).fillna(99)
    else:
        out["_ds"] = out["dataset"]
    return (
        out.sort_values(["_role", "_ds", "model"])
        .drop(columns=["_role", "_ds"])
        .reset_index(drop=True)
    )


def apply_balance_weight(frame: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Q_β = Q_roh · cb^β  (α = 0). β = 0 recovers raw quality."""
    if "class_balance" not in frame.columns:
        raise ValueError("class_balance missing")
    out = frame.copy()
    cb = out["class_balance"].fillna(0.0).to_numpy(dtype=float)
    raw = out["quality_raw"].to_numpy(dtype=float)
    out["quality_weighted"] = raw * np.power(cb, float(beta))
    return out


def apply_size_weight(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Q_α = Q_roh · (|S|/n)^α  (β = 0). α = 0 recovers raw quality."""
    if "size_frac" not in frame.columns:
        raise ValueError("size_frac missing")
    out = frame.copy()
    frac = out["size_frac"].to_numpy(dtype=float)
    raw = out["quality_raw"].to_numpy(dtype=float)
    out["quality_weighted"] = raw * np.power(frac, float(alpha))
    return out


def apply_size_and_balance_weight(
    frame: pd.DataFrame, alpha: float, beta: float
) -> pd.DataFrame:
    """Q = Q_roh · (|S|/n)^α · cb^β."""
    if "size_frac" not in frame.columns:
        raise ValueError("size_frac missing")
    if "class_balance" not in frame.columns:
        raise ValueError("class_balance missing")
    out = frame.copy()
    frac = out["size_frac"].to_numpy(dtype=float)
    cb = out["class_balance"].fillna(0.0).to_numpy(dtype=float)
    raw = out["quality_raw"].to_numpy(dtype=float)
    out["quality_weighted"] = (
        raw * np.power(frac, float(alpha)) * np.power(cb, float(beta))
    )
    return out


def scale_positive_per_dataset(
    frame: pd.DataFrame,
    *,
    quality_col: str = "quality_raw",
) -> pd.DataFrame:
    """
    Make Q_roh>0 candidates comparable across datasets.

    Size: relative cover |S|/n.
    Quality: divide by the dataset's maximum of ``quality_col`` among those
    candidates. Within-dataset ranks stay the same; the pooled comparison moves.
    """
    pos = positive_subgroups(frame)
    if pos.empty:
        return pos
    if quality_col not in pos.columns:
        raise ValueError(f"{quality_col!r} missing")
    out = pos.copy()
    if "size_frac" not in out.columns or out["size_frac"].isna().any():
        raise ValueError("size_frac missing; load_subgroups needs findings.csv n_rows")
    qmax = out.groupby("dataset")[quality_col].transform("max")
    out["size_scaled"] = out["size_frac"]
    out["quality_scaled"] = np.where(
        qmax.to_numpy() > 0, out[quality_col] / qmax, np.nan
    )
    return out.dropna(subset=["size_scaled", "quality_scaled"])


def correlation_by_model_positive(
    frame: pd.DataFrame,
    *,
    models: Sequence[str] | None = None,
    quality_col: str = "quality_raw",
    x_col: str = "size_scaled",
) -> pd.DataFrame:
    """Spearman / Pearson per model on (x_col, Q/max Q), Q_roh>0 pooled."""
    pos = scale_positive_per_dataset(frame, quality_col=quality_col)
    if models is not None:
        pos = pos[pos["model"].isin(models)]
    if x_col not in pos.columns:
        raise ValueError(f"{x_col!r} missing")
    rows: list[dict] = []
    for model, part in pos.groupby("model", sort=False):
        cell = part.dropna(subset=[x_col, "quality_scaled"])
        scaled = _corr_pair(
            cell[x_col].to_numpy(), cell["quality_scaled"].to_numpy()
        )
        rows.append(
            {
                "model": model,
                "n": scaled["n"],
                "n_datasets": int(cell["dataset"].nunique()) if not cell.empty else 0,
                "spearman": scaled["spearman"],
                "spearman_p": scaled["spearman_p"],
                "pearson": scaled["pearson"],
                "pearson_p": scaled["pearson_p"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["model"] = pd.Categorical(out["model"], list(MODEL_ORDER), ordered=True)
    return out.sort_values("model").reset_index(drop=True)


def _lowess_xy(size: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(size)
    x = size[order]
    y = quality[order]
    frac = 0.35 if len(x) >= 80 else 0.6
    smoothed = lowess(y, np.log10(x), frac=frac, it=1, return_sorted=True)
    return 10 ** smoothed[:, 0], smoothed[:, 1]


def _quantile_bins(
    size: np.ndarray,
    quality: np.ndarray,
    *,
    n_bins: int = 8,
    q: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    log_s = np.log10(size)
    edges = np.linspace(log_s.min(), log_s.max(), n_bins + 1)
    mids, vals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (log_s >= lo) & (log_s <= hi if hi == edges[-1] else log_s < hi)
        if mask.sum() < 5:
            continue
        mids.append(10 ** (0.5 * (lo + hi)))
        vals.append(np.quantile(quality[mask], q))
    return np.asarray(mids), np.asarray(vals)


def plot_dataset_panels(
    frame: pd.DataFrame,
    dataset: str,
    out_path: Path,
    *,
    models: Sequence[str] = PLOT_MODELS,
) -> Path:
    part = frame[frame["dataset"] == dataset]
    if part.empty:
        raise ValueError(f"no rows for dataset {dataset!r}")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), sharex=True)
    role = ROLE_LABELS.get(dataset_role(dataset), dataset_role(dataset))
    title = DATASET_TITLES.get(dataset, dataset)
    fig.suptitle(
        rf"{title} ({role}):  $Q_{{\mathrm{{roh}}}}$ gegen Subgruppengröße",
        fontsize=12,
    )

    for ax, model in zip(axes.ravel(), models):
        cell = part[part["model"] == model]
        color = MODEL_COLORS[model]
        ax.axhline(0.0, color="#888888", lw=0.8, ls=":", zorder=0)
        if cell.empty:
            ax.set_title(f"{MODEL_LABELS[model]}\nkeine Subgruppen")
            continue

        size = cell["size_sg"].to_numpy()
        quality = cell["quality_raw"].to_numpy()
        ax.scatter(
            size,
            quality,
            s=10 if len(cell) > 1500 else 14,
            alpha=0.28 if len(cell) > 800 else 0.45,
            color=color,
            edgecolors="none",
            zorder=2,
        )
        try:
            xs, ys = _lowess_xy(size, quality)
            ax.plot(xs, ys, color="black", lw=1.6, zorder=3, label="LOWESS")
        except Exception:
            pass
        q90_x, q90_y = _quantile_bins(size, quality, q=0.9)
        if len(q90_x) >= 3:
            ax.plot(
                q90_x,
                q90_y,
                color=color,
                lw=1.4,
                ls="--",
                zorder=3,
                label="90-%-Quantil",
            )

        corr = _corr_pair(size, quality)
        pos = cell[cell["quality_raw"] > 0]
        pos_c = _corr_pair(pos["size_sg"].to_numpy(), pos["quality_raw"].to_numpy())
        rho = corr["spearman"]
        rho_pos = pos_c["spearman"]
        rho_txt = "n/a" if not np.isfinite(rho) else f"{rho:.2f}"
        pos_txt = "n/a" if not np.isfinite(rho_pos) else f"{rho_pos:.2f}"
        ax.set_title(
            f"{MODEL_LABELS[model]}   n={int(corr['n'])}\n"
            rf"Spearman $\rho={rho_txt}$   ($\rho_{{Q>0}}={pos_txt}$)",
            fontsize=9,
        )
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        if ax is axes[0, 1]:
            ax.legend(fontsize=8, loc="best")

    for ax in axes[1]:
        ax.set_xlabel(r"Subgruppengröße $|S|$ (Train + Test)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"rohe Qualität $Q_{\mathrm{roh}}$")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_spearman_heatmap(
    table: pd.DataFrame,
    out_path: Path,
    *,
    value: str = "spearman",
    title: str,
    datasets: Sequence[str] | None = None,
) -> Path:
    sub = table.copy()
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    pivot = sub.pivot(index="dataset", columns="model", values=value)
    pivot = pivot.reindex(columns=[m for m in MODEL_ORDER if m in pivot.columns])
    if datasets is not None:
        pivot = pivot.reindex([d for d in datasets if d in pivot.index])

    fig, ax = plt.subplots(figsize=(6.4, 0.7 + 0.55 * max(len(pivot), 1)))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(float(vmax), 0.3)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([MODEL_LABELS.get(c, c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [
            f"{DATASET_TITLES.get(d, d)} ({ROLE_LABELS.get(dataset_role(d), '')})"
            for d in pivot.index
        ],
        fontsize=8,
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isfinite(val):
                text = "—"
            else:
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _dataset_color_map(datasets: Sequence[str]) -> dict[str, tuple]:
    names = list(datasets)
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % 20) for i, name in enumerate(names)}


def plot_positive_pooled_by_model(
    frame: pd.DataFrame,
    out_path: Path,
    *,
    models: Sequence[str] = PLOT_MODELS,
    quality_col: str = "quality_raw",
    x_col: str = "size_scaled",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    log_x: bool = True,
) -> Path:
    """
    One panel per model: Q_roh>0 candidates, all datasets, after per-dataset scaling.

    Default: x = |S|/n, y = quality_col / max(quality_col) on that dataset.
    """
    pos = scale_positive_per_dataset(frame, quality_col=quality_col)
    pos = pos[pos["model"].isin(models)]
    if x_col not in pos.columns:
        raise ValueError(f"{x_col!r} missing")
    role_rank = {"candidate": 0, "baseline": 1, "control": 2, "illustration": 3}
    datasets = sorted(
        pos["dataset"].unique(),
        key=lambda d: (role_rank.get(dataset_role(d), 9), d),
    )
    colors = _dataset_color_map(datasets)
    corr = correlation_by_model_positive(
        frame, models=models, quality_col=quality_col, x_col=x_col
    ).set_index("model")

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6), sharex=True, sharey=True)
    fig.suptitle(
        title
        or r"Alle $Q_{\mathrm{roh}}>0$, Größe und Qualität je Datensatz skaliert",
        fontsize=12,
    )

    for ax, model in zip(axes.ravel(), models):
        cell = pos[pos["model"] == model].dropna(subset=[x_col, "quality_scaled"])
        ax.axhline(0.0, color="#888888", lw=0.8, ls=":", zorder=0)
        if cell.empty:
            ax.set_title(f"{MODEL_LABELS[model]}\nkeine positiven Subgruppen")
            continue

        for dataset in datasets:
            part = cell[cell["dataset"] == dataset]
            if part.empty:
                continue
            ax.scatter(
                part[x_col],
                part["quality_scaled"],
                s=8 if len(cell) > 4000 else 12,
                alpha=0.35,
                color=colors[dataset],
                edgecolors="none",
                zorder=2,
            )

        x = cell[x_col].to_numpy()
        quality = cell["quality_scaled"].to_numpy()
        lowess_x = x[x > 0] if log_x else x
        lowess_y = quality[x > 0] if log_x else quality
        try:
            if len(lowess_x) >= 20:
                xs, ys = _lowess_xy(lowess_x, lowess_y)
                ax.plot(xs, ys, color="black", lw=1.7, zorder=4)
        except Exception:
            pass

        row = corr.loc[model] if model in corr.index else None
        if row is not None:
            rho = row["spearman"]
            r = row["pearson"]
            rho_txt = "n/a" if not np.isfinite(rho) else f"{rho:.2f}"
            r_txt = "n/a" if not np.isfinite(r) else f"{r:.2f}"
            ax.set_title(
                f"{MODEL_LABELS[model]}   n={int(row['n'])} "
                f"({int(row['n_datasets'])} Datensätze)\n"
                rf"Spearman $\rho={rho_txt}$,  Pearson $r={r_txt}$",
                fontsize=9,
            )
        if log_x:
            ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3, which="both")

    for ax in axes[1]:
        ax.set_xlabel(xlabel or r"relative Subgruppengröße $|S|/n$")
    for ax in axes[:, 0]:
        ax.set_ylabel(
            ylabel
            or r"$Q_{\mathrm{roh}}\,/\,\max_{\mathrm{Datensatz}} Q_{\mathrm{roh}}$"
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[name],
            markeredgecolor="none",
            markersize=6,
            label=DATASET_TITLES.get(name, name),
        )
        for name in datasets
    ]
    legend_handles.append(Line2D([0], [0], color="black", lw=1.7, label="LOWESS"))
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        frameon=False,
        title="Datensatz",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_plots(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    datasets: Iterable[str] = PLOT_DATASETS,
    models: Sequence[str] = PLOT_MODELS,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = tuple(datasets)
    table = correlation_table(frame, datasets=datasets, models=models)
    csv_path = out_dir / "correlation_by_cell.csv"
    table.to_csv(csv_path, index=False)

    written: dict[str, Path] = {"correlation_csv": csv_path}
    for dataset in datasets:
        written[dataset] = plot_dataset_panels(
            frame,
            dataset,
            out_dir / f"quality_vs_size_{dataset}.png",
            models=models,
        )
    written["heatmap_all"] = plot_spearman_heatmap(
        table,
        out_dir / "spearman_heatmap.png",
        value="spearman",
        title=r"Spearman $\rho$  ($|S|$ vs. $Q_{\mathrm{roh}}$, alle Kandidaten)",
        datasets=datasets,
    )
    written["heatmap_pos"] = plot_spearman_heatmap(
        table,
        out_dir / "spearman_heatmap_q_positive.png",
        value="spearman_pos",
        title=r"Spearman $\rho$  nur $Q_{\mathrm{roh}}>0$",
        datasets=datasets,
    )
    pooled = correlation_by_model_positive(frame, models=models)
    pooled_csv = out_dir / "correlation_by_model_positive.csv"
    pooled.to_csv(pooled_csv, index=False)
    written["correlation_pooled_csv"] = pooled_csv
    written["positive_pooled"] = plot_positive_pooled_by_model(
        frame,
        out_dir / "quality_vs_size_positive_pooled.png",
        models=models,
    )
    return written


def main() -> None:
    frame = load_subgroups()
    written = write_plots(frame, HERE)
    print("rows:", len(frame))
    print("wrote:")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
