"""
Unified real-world model-adaptability pipeline.

Defaults: stratified 50/50 split, α=0.7, β=0, generalization-aware,
min_support = max(50, floor(0.02 * n)).

Example:
    python run.py --datasets adult bank-marketing --models lr lgbm --depth 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Must precede the first numpy import: BLAS reads these once, at load time.
# Setting them further down (as this module used to) left the parent running a
# multi-threaded BLAS while the single-threaded search workers used one thread,
# which changed the reduction order and made the local fits disagree in the fifth
# decimal between algorithms. It also oversubscribes the machine.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT.parent
PROJECT_ROOT = ROOT.parents[1]
for _path in (PROJECT_ROOT, EXPERIMENTS_DIR, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
import pysubgroup as ps
from pysubgroup.model_adaptability_target import ModelAdaptabilityDiscoveryResult
from pysubgroup.parallel_model_adaptability_dfs import (
    ParallelModelAdaptabilityDFS,
    ProcessModelAdaptabilityDFS,
    fork_available,
)

from datasets import (
    DATASET_SPECS,
    MAIN_DATASETS,
    SMOKE_DATASETS,
    available_local_datasets,
    feature_columns,
    prepare_dataset,
    stratified_split_fn,
)
from models import (
    ALL_MODELS,
    OFFLINE_MODELS,
    SMOKE_MODELS,
    TabPFNBudgetExceeded,
    TabPFNCallCounter,
    configure_tabpfn,
    resolve_model_hooks,
    tabpfn_settings,
)
from plot_results import format_plot_title, plot_quality_vs_size, save_top_table

# Characterization defaults, see notes/6_gewichtung_alpha_beta.md and
# writing/chapters/03_methode.tex.
#
# alpha corrects an estimator artifact: without it the ranking is dominated by
# subgroup size (Spearman -0.41 to -0.57 across models), because small subgroups
# carry fewer test rows and the search takes the maximum over thousands of
# candidates. 0.7 sits inside the plateau [0.6, 0.8] where the residual
# correlation stays below 0.20 for every model; it is not a point optimum.
#
# beta is 0 on purpose. The ROC AUC already ignores prevalence, and after the
# size correction the residual dependence on class balance is positive for
# LightGBM and the MLP — a positive beta would over-correct there.
DEFAULT_SIZE_WEIGHT = 0.7
DEFAULT_BALANCE_WEIGHT = 0.0
MIN_SUPPORT_FLOOR = 50
MIN_SUPPORT_FRAC = 0.02

# Processes rather than threads: see _make_algorithm for the measured reason.
DEFAULT_ALGORITHM = "processes"
# Measured on PhishingWebsites at depth 3: 13x at 16 workers, 21x at 32, 24x at 48.
# Scaling is close to linear up to 32 and flattens afterwards.
DEFAULT_MAX_WORKERS = min(32, os.cpu_count() or 1)

RESULTS_DIR = ROOT / "results"


def default_min_support(n: int) -> int:
    return max(MIN_SUPPORT_FLOOR, int(n * MIN_SUPPORT_FRAC))


def build_search_space(
    df: pd.DataFrame,
    min_support: int,
    max_selectors: int | None = None,
) -> list:
    """
    Selectors for ``df``, dropping those that cannot possibly survive min_support.

    This is exact, not a heuristic: a subgroup is only scored if it covers at least
    ``min_support`` rows, so a single selector below that threshold can never appear
    in any accepted description — neither alone nor in a conjunction, since adding a
    condition only shrinks the cover. Generating such selectors is pure waste.

    It matters because nominal columns produce one selector per level. Once the coded
    columns of `Diabetes130US` are treated as nominal (which they are), its 716-788
    ICD-9 diagnosis levels push the search space to 2259 selectors — roughly 2.5
    million candidates at depth 2. Filtering by support brings it back to a size that
    reflects what the search can actually accept.
    """
    selectors = ps.create_selectors(df, ignore=["target", "prediction"])
    kept = [s for s in selectors if s.covers(df).sum() >= min_support]
    if max_selectors is not None:
        kept = kept[:max_selectors]
    return kept


def environment_info() -> dict[str, Any]:
    """Machine and library versions, so runtimes stay interpretable later."""
    import platform

    versions = {}
    for module in ("numpy", "pandas", "sklearn", "lightgbm", "tabpfn", "torch"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - a missing version is not fatal
            versions[module] = "unbekannt"
    cuda: dict[str, Any] | None
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception:  # noqa: BLE001
        cuda = None
    return {
        "host": platform.node(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "thread_limits": {
            var: os.environ.get(var)
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "versions": versions,
        "cuda": cuda,
    }


def is_complete(results_dir: Path, dataset: str, model: str) -> bool:
    """Whether a previous batch already finished this cell."""
    meta_path = run_dir(results_dir, dataset, model) / "meta.json"
    if not meta_path.is_file():
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return "runtime_sec" in meta


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("real_world_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_algorithm(
    algorithm: str,
    max_workers: int,
    logger: logging.Logger,
    tabpfn_n_estimators: int = 1,
):
    """
    Resolve the search algorithm, defaulting to process parallelism.

    Threads only help when the local fit releases the GIL, which of these models
    only LightGBM does; measured on PhishingWebsites the thread pool reaches
    1.30x for LightGBM but 0.70x for logistic regression and 0.64x for the MLP,
    i.e. it makes two of three models slower than no parallelism at all. Forked
    processes reach 8.0x to 9.7x for the same three models at 16 workers.

    ``spawn`` is the CUDA-safe TabPFN path: fork after a parent GPU fit cannot
    re-init CUDA in the children.
    """
    if algorithm == "dfs":
        return ps.DFS()
    if algorithm == "threads":
        return ParallelModelAdaptabilityDFS(max_workers=max_workers)
    if algorithm == "spawn":
        from tabpfn_spawn import TabPFNSpawnProcessDFS

        return TabPFNSpawnProcessDFS(
            max_workers=max_workers,
            n_estimators=tabpfn_n_estimators,
        )
    if algorithm == "processes":
        if fork_available():
            return ProcessModelAdaptabilityDFS(max_workers=max_workers)
        logger.warning(
            "fork is unavailable on this platform; falling back to thread parallelism"
        )
        return ParallelModelAdaptabilityDFS(max_workers=max_workers)
    raise ValueError(f"Unknown algorithm {algorithm!r}")


def run_dir(results_dir: Path, dataset: str, model: str) -> Path:
    return results_dir / f"{dataset}__{model}"


def annotate_thesis_criterion(result_df: pd.DataFrame) -> pd.DataFrame:
    """Kapitel 3: bewertbar, Q_roh > 0, Q_ga >= 0. Zeile 0 ist keine Subgruppe."""
    out = result_df.copy()
    out["quality_raw"] = pd.to_numeric(
        out["local_test_score"], errors="coerce"
    ) - pd.to_numeric(out["global_test_score"], errors="coerce")
    ga = (
        pd.to_numeric(out["quality_ga"], errors="coerce")
        if "quality_ga" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    ok = (
        out["status"].astype(str) == "ok"
        if "status" in out.columns
        else pd.Series(True, index=out.index)
    )
    is_subgroup = (
        out["subgroup"].astype(str) != "Dataset"
        if "subgroup" in out.columns
        else pd.Series(True, index=out.index)
    )
    flags = (
        is_subgroup & ok & (out["quality_raw"] > 0) & (ga >= 0)
    ).fillna(False).to_numpy()
    if len(flags):
        flags[0] = False
    out["interesting"] = flags
    return out


def run_single(
    dataset_name: str,
    model_name: str,
    *,
    seed: int,
    depth: int,
    result_set_size: int,
    min_support: int | None,
    max_workers: int,
    sample_frac: float | None,
    row_limit: int | None,
    train_cap: int | None = None,
    test_cap: int | None = None,
    max_search_selectors: int | None,
    size_weight: float,
    balance_weight: float,
    generalization_awareness: bool,
    results_dir: Path,
    logger: logging.Logger,
    algorithm: str = DEFAULT_ALGORITHM,
    tabpfn_max_fits: int | None = None,
    tabpfn_n_estimators: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out = run_dir(results_dir, dataset_name, model_name)
    out.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "depth": depth,
        "result_set_size": result_set_size,
        "max_workers": max_workers,
        "sample_frac": sample_frac,
        "row_limit": row_limit,
        "train_cap": train_cap,
        "test_cap": test_cap,
        "max_search_selectors": max_search_selectors,
        "size_weight": size_weight,
        "balance_weight": balance_weight,
        "generalization_awareness": generalization_awareness,
        "status": "ok",
        "error": None,
        "out_dir": str(out),
    }
    if model_name == "tabpfn":
        meta["tabpfn"] = tabpfn_settings(seed, tabpfn_n_estimators)

    logger.info("=== %s | model=%s ===", dataset_name, model_name)

    df, spec = prepare_dataset(
        dataset_name, seed=seed, sample_frac=sample_frac, row_limit=row_limit
    )
    meta["n_rows_source"] = int(len(df))
    meta["openml_id"] = spec.openml_id
    meta["sample_frac_effective"] = (
        spec.sample_frac if sample_frac is None else sample_frac
    )

    split_fn = stratified_split_fn(
        seed=seed, test_size=0.5, train_cap=train_cap, test_cap=test_cap
    )
    train_for_search, test_heldout = split_fn(df)
    if train_cap is not None or test_cap is not None:
        # Size weight and min_support must use the rows the QF actually sees.
        n_train = len(train_for_search)
        df = pd.concat([train_for_search, test_heldout], ignore_index=True)

        def split_fn(data: pd.DataFrame, _n: int = n_train) -> tuple[pd.DataFrame, pd.DataFrame]:
            return data.iloc[:_n].copy(), data.iloc[_n:].copy()

        train_for_search = df.iloc[:n_train].copy()

    meta["shape"] = list(df.shape)
    meta["n_train"] = int(len(train_for_search))
    meta["n_test"] = int(len(df) - len(train_for_search))

    ms = default_min_support(len(df)) if min_support is None else min_support
    meta["min_support"] = ms

    fc = feature_columns(df)
    meta["n_features"] = len(fc)

    logger.info(
        "    %s/%s: n=%d (train=%d test=%d) min_support=%d",
        dataset_name,
        model_name,
        len(df),
        meta["n_train"],
        meta["n_test"],
        ms,
    )
    # min_support counts rows of the full dataset, while selectors are built on the
    # training half; halve the threshold so the filter stays conservative.
    search_space = build_search_space(
        train_for_search, max(1, ms // 2), max_search_selectors
    )
    meta["search_space_size"] = len(search_space)
    logger.info("    search space: %d selectors", len(search_space))

    # Only TabPFN is metered; for the offline models the counter stays unused.
    counter = (
        TabPFNCallCounter(tabpfn_max_fits) if model_name == "tabpfn" else None
    )
    builder, train_g, train_l, pred_g, pred_l = resolve_model_hooks(
        model_name, seed, counter, tabpfn_n_estimators
    )

    def make_qf() -> ps.LocalSoftClassifierPerformanceQF:
        return ps.LocalSoftClassifierPerformanceQF(
            model_builder_global=builder,
            training_global=train_g,
            training_local=train_l,
            prediction_global=pred_g,
            prediction_local=pred_l,
            model_builder_local=builder,
            split_fn=split_fn,
            random_state=seed,
            subgroup_size_weight=size_weight,
            subgroup_class_balance_weight=balance_weight,
        )

    qf = make_qf()
    target = ps.ModelAdaptabilityTarget(label_column="target", feature_columns=fc)
    task = ps.ModelAdaptabilityDiscoveryTask(
        data=df,
        target=target,
        search_space=search_space,
        qf=qf,
        depth=depth,
        result_set_size=result_set_size,
        constraints=[ps.MinSupportConstraint(ms)],
        generalization_awareness=generalization_awareness,
    )

    algo = _make_algorithm(
        algorithm,
        max_workers,
        logger,
        tabpfn_n_estimators=tabpfn_n_estimators or 1,
    )
    meta["algorithm"] = type(algo).__name__
    logger.info("    fitting global %s on %d rows...", model_name, meta["n_train"])
    t_global = time.perf_counter()
    qf.calculate_constant_statistics(df, target)
    logger.info(
        "    global AUC train=%.4f test=%.4f (%.1fs); starting search",
        float(qf.global_train_score or 0.0),
        float(qf.global_test_score or 0.0),
        time.perf_counter() - t_global,
    )
    t_sd = time.perf_counter()
    raw = algo.execute(task)
    if hasattr(algo, "n_local_fits"):
        meta["n_local_fits"] = int(algo.n_local_fits)
    result = ModelAdaptabilityDiscoveryResult.from_discovery_result(raw)
    runtime_sd = time.perf_counter() - t_sd

    result_df = annotate_thesis_criterion(result.to_dataframe())
    meta["n_interesting"] = int(result_df["interesting"].sum())

    csv_path = out / "subgroups.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")

    table_path = out / "table_top15.csv"
    save_top_table(result_df, table_path)

    plot_path = out / "plot_quality_size.png"
    plot_quality_vs_size(
        result_df,
        plot_path,
        title=format_plot_title(dataset_name, model_name),
    )

    if counter is not None:
        meta.update(counter.as_dict())

    meta["runtime_sec"] = time.perf_counter() - t0
    meta["runtime_sd_sec"] = runtime_sd
    meta["n_subgroups"] = max(0, len(result_df) - 1)
    meta["global_train_score"] = float(
        result.global_model_scores.get("global_train_score") or 0
    )
    meta["global_test_score"] = float(
        result.global_model_scores.get("global_test_score") or 0
    )
    meta["paths"] = {
        "subgroups_csv": str(csv_path),
        "table_top15": str(table_path),
        "plot": str(plot_path),
    }

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Done %s/%s in %.1fs (SD %.1fs); subgroups=%d; global AUC test=%.4f",
        dataset_name,
        model_name,
        meta["runtime_sec"],
        runtime_sd,
        meta["n_subgroups"],
        meta["global_test_score"],
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-world model-adaptability SD")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Default: locally available CSVs in data/ (the thirteen datasets of chapter 4).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(OFFLINE_MODELS),
        help="Default: lr rf lgbm mlp. TabPFN: bash run_tabpfn.sh (local GPU).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--result-set-size", type=int, default=0)
    p.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="Override; default max(50, 0.02*n)",
    )
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--row-limit", type=int, default=None)
    p.add_argument(
        "--train-cap",
        type=int,
        default=None,
        help="Max training rows after the 50/50 split (stratified). Default: no cap.",
    )
    p.add_argument(
        "--test-cap",
        type=int,
        default=None,
        help="Max test rows after the 50/50 split (stratified). Default: no cap.",
    )
    p.add_argument("--max-search-selectors", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    p.add_argument(
        "--algorithm",
        choices=("processes", "threads", "dfs", "spawn"),
        default=DEFAULT_ALGORITHM,
        help="Search parallelism; processes are 8-10x faster than threads here. "
        "spawn is the CUDA-safe TabPFN process pool.",
    )
    p.add_argument("--size-weight", type=float, default=DEFAULT_SIZE_WEIGHT)
    p.add_argument("--balance-weight", type=float, default=DEFAULT_BALANCE_WEIGHT)
    p.add_argument("--no-ga", action="store_true", help="Disable generalization awareness")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dataset/model cells that already have a finished meta.json, so an "
        "interrupted batch can be restarted with the same command",
    )
    p.add_argument(
        "--tabpfn-token",
        default=None,
        help="Prior Labs license key for the one-time Hugging Face weight download. "
        "Also: env TABPFN_TOKEN or tabpfn_api_token.txt (gitignored).",
    )
    p.add_argument(
        "--tabpfn-max-fits",
        type=int,
        default=None,
        help="Optional local fit cap. Default: no cap. An unfinished cell is retried "
        "with --skip-existing.",
    )
    p.add_argument(
        "--tabpfn-n-estimators",
        type=int,
        default=None,
        help="TabPFN ensemble size. Package default is 8; the thesis GPU batch "
        "(run_tabpfn.sh) uses 1.",
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke datasets/models and small search settings",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    default_datasets = available_local_datasets() or list(MAIN_DATASETS)
    if args.smoke:
        results_dir = results_dir / "smoke"
        datasets = list(args.datasets) if args.datasets else list(SMOKE_DATASETS)
        models = (
            list(SMOKE_MODELS)
            if set(args.models) == set(OFFLINE_MODELS)
            else list(args.models)
        )
        depth = min(args.depth, 1)
        row_limit = args.row_limit if args.row_limit is not None else 800
        max_sel = args.max_search_selectors if args.max_search_selectors is not None else 12
        max_workers = min(args.max_workers, 2)
        sample_frac = 1.0 if args.sample_frac is None else args.sample_frac
    else:
        datasets = list(args.datasets) if args.datasets else default_datasets
        models = list(args.models)
        depth = args.depth
        row_limit = args.row_limit
        max_sel = args.max_search_selectors
        max_workers = args.max_workers
        sample_frac = args.sample_frac

    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir / "run.log")

    if "tabpfn" in models:
        configure_tabpfn(args.tabpfn_token)

    run_meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "datasets": datasets,
        "models": models,
        # Recorded so a runtime comparison months later can still be attributed:
        # the same batch on a different core count is not the same measurement.
        "environment": environment_info(),
        "tabpfn": (
            tabpfn_settings(args.seed, args.tabpfn_n_estimators)
            if "tabpfn" in models
            else None
        ),
        "config": {
            k: v
            for k, v in vars(args).items()
            if k not in ("tabpfn_token", "datasets", "models")
        },
        "runs": [],
    }
    logger.info(
        "Starting batch: %d datasets × %d models → %s",
        len(datasets),
        len(models),
        results_dir,
    )

    failed = 0
    skipped = 0
    budget_exhausted = False
    for ds in datasets:
        if budget_exhausted:
            break
        if ds not in DATASET_SPECS:
            logger.error("Unknown dataset %s", ds)
            failed += 1
            continue
        for model in models:
            if model not in ALL_MODELS and model not in SMOKE_MODELS:
                logger.error("Unknown model %s", model)
                failed += 1
                continue
            if args.skip_existing and is_complete(results_dir, ds, model):
                logger.info("Skipping %s/%s (already complete)", ds, model)
                skipped += 1
                continue
            # Local TabPFN shares one GPU. The official batch uses dfs / one
            # worker. ``spawn`` is the CUDA-safe process pool (uncapped runs);
            # fork ``processes`` cannot re-init CUDA after the parent fit.
            is_tabpfn = model == "tabpfn"
            if is_tabpfn and args.algorithm == "spawn":
                algorithm = "spawn"
            elif is_tabpfn and args.algorithm == "threads":
                algorithm = "threads"
            elif is_tabpfn:
                algorithm = "dfs"
            else:
                algorithm = args.algorithm
            try:
                meta = run_single(
                    ds,
                    model,
                    seed=args.seed,
                    depth=depth,
                    result_set_size=args.result_set_size,
                    min_support=args.min_support,
                    max_workers=(
                        max_workers if (not is_tabpfn or algorithm in ("spawn", "threads")) else 1
                    ),
                    sample_frac=sample_frac,
                    row_limit=row_limit,
                    train_cap=args.train_cap,
                    test_cap=args.test_cap,
                    max_search_selectors=max_sel,
                    size_weight=args.size_weight,
                    balance_weight=args.balance_weight,
                    generalization_awareness=not args.no_ga,
                    algorithm=algorithm,
                    results_dir=results_dir,
                    logger=logger,
                    tabpfn_max_fits=args.tabpfn_max_fits,
                    tabpfn_n_estimators=args.tabpfn_n_estimators,
                )
            except TabPFNBudgetExceeded as exc:
                # Optional --tabpfn-max-fits. No meta.json, so --skip-existing retries.
                logger.warning("TabPFN fit cap reached at %s/%s: %s", ds, model, exc)
                run_meta["runs"].append(
                    {
                        "dataset": ds,
                        "model": model,
                        "status": "tabpfn_budget_exceeded",
                        "message": str(exc),
                    }
                )
                logger.warning(
                    "Stopping the batch; rerun the same command after raising "
                    "--tabpfn-max-fits or omitting the cap."
                )
                budget_exhausted = True
                break
            except Exception as exc:
                failed += 1
                logger.error("FAILED %s/%s: %s", ds, model, exc)
                logger.error(traceback.format_exc())
                meta = {
                    "dataset": ds,
                    "model": model,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            run_meta["runs"].append(meta)

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["n_failed"] = failed
    run_meta["n_skipped"] = skipped
    run_meta["tabpfn_budget_exhausted"] = budget_exhausted
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        # default=str: the config block mirrors argparse, which holds Path objects
        # (and numpy scalars arrive via the run metas). A batch that has just spent
        # hours computing must not die while writing its own summary.
        json.dump(run_meta, f, indent=2, ensure_ascii=False, default=str)
    logger.info(
        "Batch finished (%d failures, %d skipped). Summary: %s",
        failed,
        skipped,
        results_dir / "summary.json",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
