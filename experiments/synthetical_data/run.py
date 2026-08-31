"""
Führt die synthetischen Experimente A0–G1 aus.

Jedes Experiment schreibt CSV und Figuren nach ``results/<exp_id>/``.
Die Datensätze kommen aus ``datasets.make_<id>``. Suche und GT-Messung
sitzen in ``lib``.

Usage::

    python experiments/synthetical_data/run.py --list
    python experiments/synthetical_data/run.py --experiments A0 A1 B2 C5
    python experiments/synthetical_data/run.py --group C --seeds 10
    python experiments/synthetical_data/run.py --all

Gruppen: A Fundament, B Nullfälle, C Mechanismen, D Kapazität,
E Modulatoren, F Kontrollen, G Suche.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import datasets as dgp  # noqa: E402
from datasets import DEFAULT_LABEL_NOISE  # noqa: E402
from lib import (  # noqa: E402
    _log,
    aggregate_sweep,
    build_search_space,
    default_min_support,
    evaluate_gt_adaptability,
    failure_report,
    gt_rows,
    make_adaptability_qf,
    matched_null_rows,
    plot_quality_heatmap,
    plot_sweep,
    reduce_gt,
    run_capacity_grid,
    run_model_kwargs_sweep,
    run_param_sweep,
    run_sd,
    run_sd_sweep_point,
)

__all__ = [
    "build_search_space",
    "default_min_support",
    "make_adaptability_qf",
    "run_sd",
    "main",
]

warnings.filterwarnings("ignore")

MODELS = ["lr", "rf", "mlp"]
N_DEFAULT = 4_000
RESULTS_DIR = _HERE / "results"


class Ctx:
    """Gemeinsame Laufkonfiguration."""

    def __init__(self, n: int, seeds: tuple[int, ...], label_noise: float, out: Path):
        self.n = n
        self.seeds = seeds
        self.label_noise = label_noise
        self.out = out

    def few(self, k: int = 5) -> tuple[int, ...]:
        return tuple(self.seeds[:k])


EXPERIMENTS: dict[str, dict[str, Any]] = {}


def experiment(exp_id: str, group: str, title: str):
    def deco(fn: Callable[[Ctx], list[tuple[str, pd.DataFrame, Optional[dict]]]]):
        EXPERIMENTS[exp_id] = {
            "id": exp_id,
            "group": group,
            "title": title,
            "fn": fn,
        }
        return fn

    return deco


DILUTION_VALUES = [0, 15, 30, 60]


def _dilution_sweep(ctx: Ctx, builder, base_kwargs: dict, title: str):
    """Verdünnung: ``n_noise`` auf der Experiment-Funktion ``make_<id>``."""
    df = run_param_sweep(
        builder,
        sweep_param="n_noise",
        sweep_values=DILUTION_VALUES,
        models=MODELS,
        base_kwargs=base_kwargs,
        seeds=ctx.few(5),
    )
    return (
        "dilution",
        df,
        {"kind": "sweep", "title": title, "xlabel": "irrelevante Spalten"},
    )


# ---------------------------------------------------------------------------
# A — Fundament
# ---------------------------------------------------------------------------


@experiment("A0", "A", "Matched null: Q auf Zufallsregionen gleicher Größe")
def exp_a0(ctx: Ctx):
    rows: list[dict] = []
    for carrier in ("piecewise", "homogeneous", "homogeneous_diluted"):
        for model in MODELS:
            for seed in ctx.few(3):
                _log(f"    A0 {carrier} {model} seed={seed}")
                spec = dgp.make_a0(
                    n=ctx.n,
                    seed=seed,
                    label_noise=ctx.label_noise,
                    carrier=carrier,
                )
                rows.extend(
                    matched_null_rows(
                        spec,
                        model,
                        seed=seed,
                        n_draws=8,
                        size_fractions=(0.1, 0.25, 0.5),
                        carrier=carrier,
                        sweep_param="size_fraction",
                        sweep_value=None,
                    )
                )
    df = pd.DataFrame(rows)
    df["sweep_value"] = df["size_fraction"]
    df["sweep_param"] = "size_fraction"
    return [
        (
            "matched_null",
            df,
            {
                "kind": "sweep",
                "title": "A0 Matched null (random subgroups)",
                "xlabel": "subgroup size fraction",
            },
        )
    ]


@experiment("A1", "A", "Labelrauschen auf dem homogenen DGP")
def exp_a1(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_a1,
        sweep_param="label_noise",
        sweep_values=[0.0, 0.05, 0.15, 0.25, 0.35],
        models=MODELS,
        base_kwargs={"n": ctx.n},
        seeds=ctx.seeds,
    )
    return [
        (
            "noise_calibration",
            df,
            {
                "kind": "global",
                "title": "A1 Global AUC vs. label noise",
                "xlabel": "label_noise",
            },
        )
    ]


# ---------------------------------------------------------------------------
# B — erwartete Nullfälle
# ---------------------------------------------------------------------------


@experiment("B1", "B", "Reiner Prior-Shift (ranginvariant → Q=0)")
def exp_b1(ctx: Ctx):
    out = []
    for metric in ("auc", "accuracy"):
        df = run_param_sweep(
            dgp.make_b1,
            sweep_param="prior1",
            sweep_values=[0.5, 0.3, 0.15, 0.05],
            models=MODELS,
            base_kwargs={"n": ctx.n, "prior0": 0.5},
            seeds=ctx.seeds,
            metric=metric,
        )
        out.append(
            (
                f"prior_shift_{metric}",
                df,
                {
                    "kind": "sweep",
                    "title": f"B1 Prior shift ({metric})",
                    "xlabel": "prior1 (prior0=0.5)",
                },
            )
        )
    return out


@experiment("B2", "B", "Feature-Shift, Regel bleibt in der Hypothesenklasse")
def exp_b2(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_b2,
        sweep_param="n",
        sweep_values=[1000, 2000, 4000, 8000],
        models=MODELS,
        base_kwargs={"label_noise": ctx.label_noise},
        seeds=ctx.seeds,
    )
    return [
        (
            "feature_shift",
            df,
            {
                "kind": "sweep",
                "title": "B2 Specified feature shift (control)",
                "xlabel": "n",
            },
        )
    ]


@experiment("B3", "B", "LR mit vs. ohne Interaktionsterme")
def exp_b3(ctx: Ctx):
    out = []
    rows = []
    for mode in ("no_interaction", "with_interaction"):
        for seed in ctx.seeds:
            spec = dgp.make_b3(
                n=ctx.n,
                seed=seed,
                label_noise=ctx.label_noise,
                include_interactions=(mode == "with_interaction"),
                carrier="sign_flip",
            )
            info = evaluate_gt_adaptability(spec, "lr", seed=seed)
            rows.extend(
                gt_rows(info, sweep_param="feature_set", sweep_value=mode, kind="gt")
            )
    out.append(
        (
            "lr_interactions",
            pd.DataFrame(rows),
            {
                "kind": "sweep",
                "title": "B3 LR with/without interaction terms",
                "xlabel": "feature set",
                "numeric_x": False,
            },
        )
    )
    carriers = (
        "C1_slope",
        "C2_piecewise",
        "C3_shortcut",
        "C4_concept_shift",
        "C5_hidden",
        "C6_misspec_shift",
    )
    rows = []
    for carrier in carriers:
        for saturated in (False, True):
            for seed in ctx.few(5):
                spec = dgp.make_b3(
                    n=ctx.n,
                    seed=seed,
                    label_noise=ctx.label_noise,
                    carrier=carrier,
                    saturated=saturated,
                )
                info = evaluate_gt_adaptability(spec, "lr", seed=seed)
                rows.extend(
                    gt_rows(
                        info,
                        sweep_param="carrier",
                        sweep_value=carrier,
                        kind="gt",
                        saturated=saturated,
                    )
                )
    out.append(("attribution_all_carriers", pd.DataFrame(rows), {"kind": "table"}))
    return out


# ---------------------------------------------------------------------------
# C — Mechanismen
# ---------------------------------------------------------------------------


@experiment("C1", "C", "Heterogene Steigung (beta0, plus Verdünnung)")
def exp_c1(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_c1,
        sweep_param="beta0",
        sweep_values=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
        models=MODELS,
        base_kwargs={"n": ctx.n, "beta1": 1.0, "label_noise": ctx.label_noise},
        seeds=ctx.seeds,
    )
    return [
        (
            "beta0_sweep",
            df,
            {
                "kind": "sweep",
                "title": "C1 Heterogeneous slope (beta1=1)",
                "xlabel": "beta0",
            },
        ),
        _dilution_sweep(
            ctx,
            dgp.make_c1,
            {
                "n": ctx.n,
                "beta0": -2.0,
                "beta1": 1.0,
                "label_noise": ctx.label_noise,
            },
            "C1 Heterogeneous slope under dilution",
        ),
    ]


@experiment("C2", "C", "Stückweise Regeln (n und Zahl der Strata)")
def exp_c2(ctx: Ctx):
    out = []
    df_n = run_param_sweep(
        dgp.make_c2,
        sweep_param="n",
        sweep_values=[1000, 2000, 4000, 8000, 16000, 32000],
        models=MODELS,
        base_kwargs={"n_strata": 4, "label_noise": ctx.label_noise},
        seeds=ctx.few(5),
    )
    out.append(
        (
            "sample_size",
            df_n,
            {"kind": "sweep", "title": "C2 Piecewise - sample size", "xlabel": "n"},
        )
    )
    df_cap = run_model_kwargs_sweep(
        lambda s: dgp.make_c2(
            n=32000, seed=s, n_strata=4, label_noise=ctx.label_noise
        ),
        model="mlp",
        kw_name="hidden_layer_sizes",
        kw_values=[(16, 8), (64, 32), (128, 64)],
        seeds=ctx.few(3),
    )
    out.append(
        (
            "mlp_capacity_at_large_n",
            df_cap,
            {
                "kind": "sweep",
                "title": "C2 MLP width at n=32000 (is the plateau capacity-bound?)",
                "xlabel": "hidden_layer_sizes",
            },
        )
    )
    df_k = run_param_sweep(
        dgp.make_c2,
        sweep_param="n_strata",
        sweep_values=[2, 4, 8, 16],
        models=MODELS,
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        seeds=ctx.few(5),
    )
    out.append(
        (
            "n_strata",
            df_k,
            {
                "kind": "sweep",
                "title": "C2 Mixture complexity (n fixed - size confounded)",
                "xlabel": "n_strata",
            },
        )
    )
    df_kc = run_param_sweep(
        dgp.make_c2,
        sweep_param="n_strata",
        sweep_values=[2, 4, 8, 16],
        models=MODELS,
        base_kwargs={"per_stratum": 500, "label_noise": ctx.label_noise},
        seeds=ctx.few(5),
    )
    out.append(
        (
            "n_strata_fixed_stratum",
            df_kc,
            {
                "kind": "sweep",
                "title": "C2 Mixture complexity (stratum size fixed at 500)",
                "xlabel": "n_strata",
            },
        )
    )
    return out


@experiment("C3", "C", "Shortcut mit Korrelationsumkehr")
def exp_c3(ctx: Ctx):
    out = []
    base = {"n": ctx.n, "label_noise": ctx.label_noise}
    df_c = run_param_sweep(
        dgp.make_c3,
        sweep_param="minority_shortcut_corr",
        sweep_values=[1.0, 0.5, 0.0, -0.5, -1.0],
        models=MODELS,
        base_kwargs={**base, "minority_fraction": 0.2, "shortcut_strength": 0.9},
        seeds=ctx.seeds,
    )
    out.append(
        (
            "shortcut_correlation",
            df_c,
            {
                "kind": "sweep",
                "title": "C3 Shortcut correlation in the minority",
                "xlabel": "minority_shortcut_corr",
            },
        )
    )
    df_s = run_param_sweep(
        dgp.make_c3,
        sweep_param="shortcut_strength",
        sweep_values=[0.0, 0.3, 0.6, 0.9],
        models=MODELS,
        base_kwargs={**base, "minority_fraction": 0.2, "minority_shortcut_corr": -1.0},
        seeds=ctx.seeds,
    )
    out.append(
        (
            "shortcut_strength",
            df_s,
            {
                "kind": "sweep",
                "title": "C3 Shortcut strength",
                "xlabel": "shortcut_strength",
            },
        )
    )
    df_m = run_param_sweep(
        dgp.make_c3,
        sweep_param="minority_fraction",
        sweep_values=[0.05, 0.1, 0.2, 0.35],
        models=MODELS,
        base_kwargs={**base, "shortcut_strength": 0.9, "minority_shortcut_corr": -1.0},
        seeds=ctx.seeds,
    )
    out.append(
        (
            "minority_fraction",
            df_m,
            {
                "kind": "sweep",
                "title": "C3 Shortcut - minority size",
                "xlabel": "minority_fraction",
            },
        )
    )
    df_o = run_param_sweep(
        dgp.make_c3,
        sweep_param="interaction_order",
        sweep_values=[2, 3, 4],
        models=MODELS,
        base_kwargs={
            **base,
            "minority_fraction": 0.2,
            "shortcut_strength": 0.9,
            "minority_shortcut_corr": -1.0,
        },
        seeds=ctx.few(5),
    )
    out.append(
        (
            "interaction_order",
            df_o,
            {
                "kind": "sweep",
                "title": "C3 Reachability of the true rule",
                "xlabel": "interaction_order",
            },
        )
    )

    def shortcut_spec(s: int):
        return dgp.make_c3(
            n=ctx.n,
            seed=s,
            label_noise=ctx.label_noise,
            minority_fraction=0.2,
            shortcut_strength=0.9,
            minority_shortcut_corr=-1.0,
        )

    for model, kw_name, kw_values in (
        ("rf", "max_depth", [1, 2, 4, None]),
        ("mlp", "hidden_layer_sizes", [(2,), (4,), (16, 8)]),
    ):
        df_k = run_model_kwargs_sweep(
            shortcut_spec,
            model=model,
            kw_name=kw_name,
            kw_values=kw_values,
            seeds=ctx.few(5),
        )
        df_k["sweep_value"] = df_k["sweep_value"].astype(str).replace(
            {"None": "unlimited"}
        )
        out.append(
            (
                f"capacity_control_{model}",
                df_k,
                {
                    "kind": "sweep",
                    "title": f"C3 Shortcut vs. {model} capacity (order 2)",
                    "xlabel": kw_name,
                },
            )
        )
    return out


@experiment("C4", "C", "Heterogenes Rauschen vs. Konzeptwechsel")
def exp_c4(ctx: Ctx):
    out = []
    for mode in ("uniform", "feature_dependent", "concept_shift"):
        df = run_param_sweep(
            dgp.make_c4,
            sweep_param="flip_out",
            sweep_values=[0.0, 0.1, 0.25, 0.4, 0.5],
            models=MODELS,
            base_kwargs={
                "n": ctx.n,
                "noise_mode": mode,
                "gt_fraction": 0.25,
                "label_noise": ctx.label_noise,
            },
            seeds=ctx.seeds,
        )
        df["noise_mode"] = mode
        out.append(
            (
                f"noise_{mode}",
                df,
                {
                    "kind": "sweep",
                    "title": f"C4 Heterogeneous noise - {mode}",
                    "xlabel": "flip_out",
                },
            )
        )
    out.append(
        _dilution_sweep(
            ctx,
            dgp.make_c4,
            {
                "n": ctx.n,
                "noise_mode": "concept_shift",
                "gt_fraction": 0.25,
                "flip_out": 0.5,
                "label_noise": ctx.label_noise,
            },
            "C4 concept_shift under dilution",
        )
    )
    return out


@experiment("C5", "C", "Hidden Stratification")
def exp_c5(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_c5,
        sweep_param="hidden_fraction",
        sweep_values=[0.05, 0.1, 0.2, 0.35],
        models=MODELS,
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        seeds=ctx.seeds,
    )
    return [
        (
            "hidden_fraction",
            df,
            {
                "kind": "sweep",
                "title": "C5 Hidden stratification",
                "xlabel": "hidden_fraction",
            },
        ),
        _dilution_sweep(
            ctx,
            dgp.make_c5,
            {"n": ctx.n, "hidden_fraction": 0.2, "label_noise": ctx.label_noise},
            "C5 Hidden stratification under dilution",
        ),
    ]


@experiment("C6", "C", "Misspezifikation plus Feature-Shift in 2D")
def exp_c6(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_c6,
        sweep_param="angle_scale",
        sweep_values=[0.0, 0.5, 1.0, 2.0],
        models=MODELS,
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        seeds=ctx.seeds,
    )
    df_cap = run_model_kwargs_sweep(
        lambda s: dgp.make_c6(
            n=ctx.n, seed=s, angle_scale=2.0, label_noise=ctx.label_noise
        ),
        model="rf",
        kw_name="max_depth",
        kw_values=[1, 2, 3, 6, None],
        seeds=ctx.few(5),
    )
    df_cap["sweep_value"] = df_cap["sweep_value"].astype(str).replace(
        {"None": "unlimited"}
    )
    return [
        (
            "angle_scale",
            df,
            {
                "kind": "sweep",
                "title": "C6 Misspecification + feature shift (2D)",
                "xlabel": "angle_scale",
            },
        ),
        _dilution_sweep(
            ctx,
            dgp.make_c6,
            {"n": ctx.n, "angle_scale": 2.0, "label_noise": ctx.label_noise},
            "C6 Misspecification + shift under dilution",
        ),
        (
            "rf_depth_control",
            df_cap,
            {
                "kind": "sweep",
                "title": "C6 RF depth (is the RF ever misspecified?)",
                "xlabel": "max_depth",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# D — Kapazität
# ---------------------------------------------------------------------------


@experiment("D1", "D", "Kapazität × Mischungskomplexität")
def exp_d1(ctx: Ctx):
    out = []
    df_rf = run_capacity_grid(
        dgp.make_d1,
        complexity_param="n_strata",
        complexity_values=[2, 4, 8, 16],
        model="rf",
        capacity_name="max_depth",
        capacity_values=[1, 2, 3, 6, None],
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        fixed_kwargs={"n_estimators": 80, "min_samples_leaf": 20},
        seeds=ctx.few(5),
    )
    out.append(
        (
            "rf_depth_x_strata",
            df_rf,
            {
                "kind": "heatmap",
                "model": "rf",
                "title": "D1 RF: depth x mixture complexity",
                "xlabel": "max_depth",
                "ylabel": "n_strata",
            },
        )
    )
    df_mlp = run_capacity_grid(
        dgp.make_d1,
        complexity_param="n_strata",
        complexity_values=[2, 4, 8, 16],
        model="mlp",
        capacity_name="hidden_layer_sizes",
        capacity_values=[(2,), (4,), (8, 4), (16, 8), (64, 32)],
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        seeds=ctx.few(3),
    )
    out.append(
        (
            "mlp_width_x_strata",
            df_mlp,
            {
                "kind": "heatmap",
                "model": "mlp",
                "title": "D1 MLP: width x mixture complexity",
                "xlabel": "hidden_layer_sizes",
                "ylabel": "n_strata",
            },
        )
    )
    return out


@experiment("D2", "D", "Regularisierung innerhalb einer Modellklasse")
def exp_d2(ctx: Ctx):
    out = []

    def carrier(seed):
        return dgp.make_d2(n=ctx.n, seed=seed, label_noise=ctx.label_noise)

    df_lr = run_model_kwargs_sweep(
        carrier,
        model="lr",
        kw_name="C",
        kw_values=[0.01, 0.1, 1.0, 10.0, 100.0],
        seeds=ctx.seeds,
        fixed_kwargs={"max_iter": 1000},
    )
    out.append(
        (
            "lr_C",
            df_lr,
            {
                "kind": "sweep",
                "title": "D2 LR regularisation C",
                "xlabel": "C",
                "numeric_x": False,
            },
        )
    )
    df_rf = run_model_kwargs_sweep(
        carrier,
        model="rf",
        kw_name="min_samples_leaf",
        kw_values=[1, 5, 10, 20, 50],
        seeds=ctx.seeds,
        fixed_kwargs={"n_estimators": 80, "max_depth": 6},
    )
    out.append(
        (
            "rf_min_samples_leaf",
            df_rf,
            {
                "kind": "sweep",
                "title": "D2 RF min_samples_leaf",
                "xlabel": "min_samples_leaf",
                "numeric_x": False,
            },
        )
    )
    df_mlp = run_model_kwargs_sweep(
        carrier,
        model="mlp",
        kw_name="alpha",
        kw_values=[1e-5, 1e-3, 1e-2, 1e-1, 1.0],
        seeds=ctx.few(5),
    )
    out.append(
        (
            "mlp_alpha",
            df_mlp,
            {
                "kind": "sweep",
                "title": "D2 MLP alpha",
                "xlabel": "alpha",
                "numeric_x": False,
            },
        )
    )
    return out


@experiment("D3", "D", "Optimierungsvarianz (nur Modell-Seed, Daten fest)")
def exp_d3(ctx: Ctx):
    rows = []
    for model in ("mlp", "rf"):
        for run_seed in range(42, 62):
            spec = dgp.make_d3(n=ctx.n, seed=7, label_noise=ctx.label_noise)
            info = evaluate_gt_adaptability(spec, model, seed=run_seed)
            rows.extend(
                gt_rows(
                    info,
                    sweep_param="run_seed",
                    sweep_value=run_seed,
                    data_seed=7,
                    kind="gt",
                )
            )
    df = pd.DataFrame(rows)
    return [
        (
            "seed_variance",
            df,
            {
                "kind": "hist",
                "title": "D3 Q across model seeds (data fixed)",
                "xlabel": "GT quality Q",
            },
        )
    ]


@experiment("D4", "D", "Induktiver Bias: zackige vs. glatte Zielregel")
def exp_d4(ctx: Ctx):
    out = []
    df = run_param_sweep(
        dgp.make_d4,
        sweep_param="n",
        sweep_values=[1000, 2000, 4000, 8000],
        models=["rf", "mlp", "lr"],
        base_kwargs={
            "variant": "jagged",
            "n_strata": 8,
            "label_noise": ctx.label_noise,
        },
        seeds=ctx.seeds,
    )
    out.append(
        (
            "jagged_piecewise",
            df,
            {
                "kind": "sweep",
                "title": "D4 Jagged piecewise target (K=8)",
                "xlabel": "n",
            },
        )
    )
    df_smooth = run_param_sweep(
        dgp.make_d4,
        sweep_param="n",
        sweep_values=[1000, 2000, 4000, 8000],
        models=["rf", "mlp", "lr"],
        base_kwargs={
            "variant": "smooth",
            "beta0": -1.0,
            "beta1": 1.0,
            "label_noise": ctx.label_noise,
        },
        seeds=ctx.seeds,
    )
    out.append(
        (
            "smooth_linear",
            df_smooth,
            {
                "kind": "sweep",
                "title": "D4 Smooth linear target",
                "xlabel": "n",
            },
        )
    )
    return out


# ---------------------------------------------------------------------------
# E — Modulatoren
# ---------------------------------------------------------------------------


@experiment("E1", "E", "Subgruppengröße")
def exp_e1(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_e1,
        sweep_param="minority_fraction",
        sweep_values=[0.02, 0.05, 0.1, 0.2, 0.4],
        models=MODELS,
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        seeds=ctx.seeds,
    )
    return [
        (
            "subgroup_size",
            df,
            {
                "kind": "sweep",
                "title": "E1 Subgroup size",
                "xlabel": "minority_fraction",
            },
        )
    ]


@experiment("E2", "E", "Klassenunbalance (Kontrolle)")
def exp_e2(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_e2,
        sweep_param="n",
        sweep_values=[2000, 4000, 8000],
        models=MODELS,
        seeds=ctx.seeds,
    )
    return [
        (
            "class_imbalance",
            df,
            {
                "kind": "sweep",
                "title": "E2 Class-imbalance control",
                "xlabel": "n",
            },
        )
    ]


@experiment("E3", "E", "Irrelevante Merkmale / Dimensionalität")
def exp_e3(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_e3,
        sweep_param="n_noise",
        sweep_values=[0, 10, 50, 100],
        models=MODELS,
        base_kwargs={"n": ctx.n, "label_noise": ctx.label_noise},
        seeds=ctx.few(5),
    )
    return [
        (
            "noise_features",
            df,
            {"kind": "sweep", "title": "E3 Noise features", "xlabel": "n_noise"},
        )
    ]


# ---------------------------------------------------------------------------
# F — Kontrollen
# ---------------------------------------------------------------------------


@experiment("F1", "F", "Homogene Kontrolle (volle Suche)")
def exp_f1(ctx: Ctx):
    rows = []
    for seed in ctx.few(3):
        spec = dgp.make_f1(
            n=min(ctx.n, 3000), seed=seed, label_noise=ctx.label_noise
        )
        for model in MODELS:
            _log(f"    F1 search {model} seed={seed}")
            info = run_sd_sweep_point(spec, model, seed=seed, max_workers=2)
            rows.append(
                {
                    "sweep_param": "model",
                    "sweep_value": model,
        "model": model,
        "seed": seed,
                    "kind": "search",
                    "max_quality": info.get("max_quality"),
                    "max_quality_ga": info.get("max_quality_ga"),
                    "max_quality_raw": info.get("max_quality_raw"),
                    "n_interesting": info.get("n_interesting"),
                    "n_subgroups": info.get("n_subgroups"),
                    "global_test_score": info.get("global_test_score"),
                    "runtime_sec": info.get("runtime_sec"),
                }
            )
    df = pd.DataFrame(rows)
    return [("homogeneous_control", df, {"kind": "table"})]


@experiment("F2", "F", "Overfitting-Kontrolle")
def exp_f2(ctx: Ctx):
    df = run_param_sweep(
        dgp.make_f2,
        sweep_param="n_noise",
        sweep_values=[10, 20, 40, 60],
        models=MODELS,
        base_kwargs={"n": 1500, "label_noise": ctx.label_noise},
        seeds=ctx.seeds,
    )
    return [
        (
            "overfitting_control",
            df,
            {
                "kind": "sweep",
                "title": "F2 Overfitting control",
                "xlabel": "n_noise",
            },
        )
    ]


# ---------------------------------------------------------------------------
# G — Suche
# ---------------------------------------------------------------------------


@experiment("G1", "G", "Suche: Rang der Ground-Truth im Mini-Suchraum")
def exp_g1(ctx: Ctx):
    rows = []
    n = min(ctx.n, 3000)
    carriers = {
        "piecewise": {"carrier": "piecewise", "n_noise": 0},
        "hidden": {"carrier": "hidden", "n_noise": 0},
        "hidden_diluted": {"carrier": "hidden", "n_noise": DILUTION_VALUES[-1]},
        "concept_shift_diluted": {
            "carrier": "concept_shift",
            "n_noise": DILUTION_VALUES[-1],
        },
    }
    for name, kwargs in carriers.items():
        for seed in ctx.few(3):
            spec = dgp.make_g1(
                n=n,
                seed=seed,
                label_noise=ctx.label_noise,
                **kwargs,
            )
            for model in MODELS:
                _log(f"    G1 search {name} {model} seed={seed}")
                info = run_sd_sweep_point(spec, model, seed=seed, max_workers=2)
                rows.append(
                    {
                        "carrier": name,
                        "model": model,
                        "seed": seed,
                        "kind": "search",
                        "best_gt_quality": info.get("best_ground_truth_quality"),
                        "best_gt_rank": info.get("best_ground_truth_rank"),
        "max_quality": info.get("max_quality"),
        "max_quality_ga": info.get("max_quality_ga"),
                        "max_quality_raw": info.get("max_quality_raw"),
        "n_interesting": info.get("n_interesting"),
                        "n_subgroups": info.get("n_subgroups"),
                        "global_test_score": info.get("global_test_score"),
                        "runtime_sec": info.get("runtime_sec"),
                        "gt_subgroups": ";".join(
                            info.get("ground_truth_subgroups", [])
                        ),
                    }
                )
    return [("discovery_demo", pd.DataFrame(rows), {"kind": "table"})]


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------


def _plot_global(df: pd.DataFrame, spec: dict, path: Path):
    agg = (
        df.groupby(["sweep_value", "model"], dropna=False)["global_test_full"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in dict.fromkeys(agg["model"]):
        sub = agg[agg["model"] == model].sort_values("sweep_value")
        ax.plot(sub["sweep_value"], sub["global_test_full"], marker="o", label=model)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_title(spec["title"])
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel("global test AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_hist(df: pd.DataFrame, spec: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    red = reduce_gt(df, mode="mean")
    for model in dict.fromkeys(red["model"]):
        vals = red[red["model"] == model]["quality"].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=12, alpha=0.6, edgecolor="black", label=model)
    ax.axvline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_title(spec["title"])
    ax.set_xlabel(spec.get("xlabel", "Q"))
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_outputs(exp_id: str, name: str, df: pd.DataFrame, spec: Optional[dict], out: Path):
    exp_dir = out / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(exp_dir / f"{name}.csv", index=False, encoding="utf-8")

    if "quality" in df.columns and "gt" in df.columns:
        summary = aggregate_sweep(df)
        if len(summary):
            summary.to_csv(exp_dir / f"{name}__summary.csv", index=False)
        fails = failure_report(df)
        if len(fails) and (fails["status"] != "ok").any():
            fails.to_csv(exp_dir / f"{name}__failures.csv", index=False)

    if not spec:
        return
    kind = spec.get("kind")
    path = exp_dir / f"{name}.png"
    try:
        if kind == "sweep":
            fig, _ = plot_sweep(
                df,
                title=spec["title"],
                xlabel=spec.get("xlabel"),
                numeric_x=spec.get("numeric_x", True),
            )
            fig.savefig(path, dpi=130)
            plt.close(fig)
        elif kind == "heatmap":
            fig, _, pivot = plot_quality_heatmap(
                df,
                model=spec["model"],
                title=spec["title"],
                xlabel=spec.get("xlabel"),
                ylabel=spec.get("ylabel"),
            )
            fig.savefig(path, dpi=130)
            plt.close(fig)
            pivot.to_csv(exp_dir / f"{name}__pivot.csv")
        elif kind == "global":
            _plot_global(df, spec, path)
        elif kind == "hist":
            _plot_hist(df, spec, path)
    except Exception as exc:
        print(f"    [warn] plot failed for {exp_id}/{name}: {type(exc).__name__}: {exc}")


def run_experiment(exp_id: str, ctx: Ctx) -> None:
    meta = EXPERIMENTS[exp_id]
    _log(f"[{exp_id}] {meta['title']}")
    t0 = time.perf_counter()
    outputs = meta["fn"](ctx)
    for name, df, spec in outputs:
        _save_outputs(exp_id, name, df, spec, ctx.out)
        if "quality" in df.columns and df["quality"].notna().any():
            red = reduce_gt(df, mode="mean")
            per_model = red.groupby("model")["quality"].agg(["mean", "std", "count"])
            for model, row in per_model.iterrows():
                _log(
                    f"    {name:28s} {model:4s} Q={row['mean']:+.4f} "
                    f"+/-{(row['std'] if np.isfinite(row['std']) else 0):.4f} "
                    f"(n={int(row['count'])})"
                )
        elif "global_test_full" in df.columns:
            per_model = df.groupby("model")["global_test_full"].mean()
            for model, val in per_model.items():
                _log(f"    {name:28s} {model:4s} global AUC={val:.4f}")
    else:
            _log(f"    {name:28s} {len(df)} rows")
    _log(f"    done in {time.perf_counter() - t0:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="*", help="IDs, z. B. C1 C2")
    parser.add_argument("--group", help="ganze Gruppe (A/B/C/D/E/F/G)")
    parser.add_argument("--all", action="store_true", help="alles ausführen")
    parser.add_argument("--list", action="store_true", help="Experimente auflisten")
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--label-noise", type=float, default=DEFAULT_LABEL_NOISE)
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    if args.list:
        for exp_id, meta in EXPERIMENTS.items():
            print(f"{exp_id:4s} [{meta['group']}] {meta['title']}")
        return 0

    if args.all:
        ids = list(EXPERIMENTS)
    elif args.group:
        ids = [i for i, m in EXPERIMENTS.items() if m["group"] == args.group.upper()]
    elif args.experiments:
        ids = args.experiments
    else:
        parser.error("choose --all, --group or --experiments (or --list)")

    unknown = [i for i in ids if i not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiment ids: {unknown}")

    ctx = Ctx(
        n=args.n,
        seeds=tuple(range(42, 42 + args.seeds)),
        label_noise=args.label_noise,
        out=Path(args.out),
    )
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    ctx.out.mkdir(parents=True, exist_ok=True)
    _log(
        f"n={ctx.n} seeds={len(ctx.seeds)} label_noise={ctx.label_noise} "
        f"out={ctx.out}"
    )
    for exp_id in ids:
        run_experiment(exp_id, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
