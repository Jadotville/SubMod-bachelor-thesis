"""
Worker sweep for local GPU TabPFN (package ``tabpfn``, not the API client).

Same dataset as the offline sweep (Phishing Websites, depth 2), same train/test
caps and ensemble size as the real-world TabPFN batch (1024/1024, one
estimator). Sequential DFS is the baseline; threads and processes are timed at
each worker count. Processes use ``TabPFNSpawnProcessDFS``, the same CUDA-safe
pool as the real-world runs. Several processes each load the foundation model
onto the same GPU — that is part of the measurement, not a bug.

    python worker_sweep_tabpfn.py
    bash run_worker_sweep_tabpfn.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import pandas as pd

_HERE = Path(__file__).resolve().parent
_RW_DIR = _HERE.parent / "experiments_real_world_data"
for _p in (_HERE, _RW_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pysubgroup as ps  # noqa: E402

from models import TabPFNCallCounter, configure_tabpfn  # noqa: E402
from profile_time_breakdown import ProfileConfig, build_task  # noqa: E402
from pysubgroup import ParallelModelAdaptabilityDFS  # noqa: E402
from tabpfn_spawn import TabPFNSpawnProcessDFS  # noqa: E402
from worker_sweep import (  # noqa: E402
    RESULTS_DIR,
    _append_row,
    _cell_key,
    _load_done,
    plot_speedup,
    summarize,
)

DEFAULT_WORKERS = (1, 2, 4, 8, 16)
DEFAULT_TRAIN_CAP = 1024
DEFAULT_TEST_CAP = 1024
DEFAULT_N_ESTIMATORS = 1
DEFAULT_DEPTH = 2


def _space_meta(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ProfileConfig(
        dataset=args.dataset,
        model="lr",
        algorithm="dfs",
        seed=args.seed,
        depth=args.depth,
        row_limit=args.row_limit,
        max_search_selectors=args.max_search_selectors,
        train_cap=args.train_cap,
        test_cap=args.test_cap,
    )
    _, meta = build_task(cfg)
    n_selectors = int(meta["n_selectors"])
    n_cells = 1 + len(args.algorithms) * len(args.workers)
    return {
        **meta,
        "n_cells": n_cells,
        "n_selectors": n_selectors,
    }


def print_plan(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    print("=== TabPFN-Worker-Sweep (lokal, CUDA)")
    print(f"  Datensatz     : {args.dataset}")
    print(f"  n             : {meta['n_rows']} (train={meta['n_train']} test={meta['n_test']})")
    print(f"  Caps          : train={args.train_cap} test={args.test_cap}")
    print(f"  Estimators    : {args.tabpfn_n_estimators}")
    print(f"  Tiefe         : {args.depth}")
    print(f"  Selektoren    : {meta['n_selectors']}")
    cells = " + ".join(f"{len(args.workers)}× {a}" for a in args.algorithms)
    print(f"  Zellen        : {meta['n_cells']}  (1× dfs + {cells})")
    print(f"  Worker        : {args.workers}")


def _make_algorithm(cfg: ProfileConfig):
    """
    TabPFN needs the ``spawn`` pool, not the fork pool of the offline models:
    a forked child cannot re-initialise CUDA after a GPU fit in the parent, so
    fork silently drops the local fits instead of running them.
    """
    if cfg.algorithm == "dfs":
        return ps.DFS()
    if cfg.algorithm == "threads":
        return ParallelModelAdaptabilityDFS(max_workers=cfg.max_workers)
    if cfg.algorithm == "processes":
        return TabPFNSpawnProcessDFS(
            max_workers=cfg.max_workers,
            n_estimators=cfg.tabpfn_n_estimators or 1,
        )
    raise ValueError(f"Unknown algorithm {cfg.algorithm!r}")


def time_once(cfg: ProfileConfig) -> dict[str, Any]:
    counter = TabPFNCallCounter()
    task, meta = build_task(cfg, counter=counter)
    algorithm = _make_algorithm(cfg)
    t0 = time.perf_counter()
    result = task.execute(algorithm=algorithm)
    elapsed = time.perf_counter() - t0
    workers = 1 if cfg.algorithm == "dfs" else cfg.max_workers
    # The call counter only sees the parent, so the worker-side work is read off
    # the algorithm instead; both numbers together show that every cell of the
    # sweep evaluates the same candidates.
    n_local_fits = getattr(algorithm, "n_local_fits", None)
    if n_local_fits is None and cfg.algorithm == "dfs":
        n_local_fits = max(0, int(counter.as_dict().get("tabpfn_fits", 1)) - 1)
    meta = {
        **meta,
        "n_local_fits": "" if n_local_fits is None else int(n_local_fits),
        "n_result": int(len(result.results)),
    }
    return {
        "dataset": cfg.dataset,
        "model": cfg.model,
        "algorithm": cfg.algorithm,
        "max_workers": workers,
        "depth": cfg.depth,
        "row_limit": cfg.row_limit if cfg.row_limit is not None else "",
        "max_search_selectors": cfg.max_search_selectors
        if cfg.max_search_selectors is not None
        else "",
        "seed": cfg.seed,
        "runtime_sec": elapsed,
        "status": "ok",
        "error": "",
        **meta,
        **counter.as_dict(),
    }


def _configs(args: argparse.Namespace) -> list[ProfileConfig]:
    base = dict(
        dataset=args.dataset,
        model="tabpfn",
        seed=args.seed,
        depth=args.depth,
        row_limit=args.row_limit,
        max_search_selectors=args.max_search_selectors,
        train_cap=args.train_cap,
        test_cap=args.test_cap,
        tabpfn_n_estimators=args.tabpfn_n_estimators,
    )
    configs = [ProfileConfig(algorithm="dfs", max_workers=1, **base)]
    for algorithm in args.algorithms:
        for workers in args.workers:
            configs.append(
                ProfileConfig(
                    algorithm=algorithm,
                    max_workers=workers,
                    **base,
                )
            )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PhishingWebsites")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["threads", "processes"],
        choices=["threads", "processes"],
    )
    parser.add_argument("--workers", nargs="+", type=int, default=list(DEFAULT_WORKERS))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--max-search-selectors", type=int, default=None)
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP)
    parser.add_argument("--test-cap", type=int, default=DEFAULT_TEST_CAP)
    parser.add_argument(
        "--tabpfn-n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="worker_sweep_tabpfn")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print the cell plan and exit (no TabPFN fits).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    space = _space_meta(args)
    print_plan(args, space)
    if args.estimate_only:
        return 0

    configure_tabpfn()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / f"{args.tag}_raw.csv"
    summary_path = RESULTS_DIR / f"{args.tag}_summary.csv"
    png_path = RESULTS_DIR / f"{args.tag}.png"
    done = _load_done(raw_path) if args.skip_existing else set()
    configs = _configs(args)
    started = time.perf_counter()
    skip_processes = False
    for index, cfg in enumerate(configs, start=1):
        workers = 1 if cfg.algorithm == "dfs" else cfg.max_workers
        probe = {
            "dataset": cfg.dataset,
            "model": cfg.model,
            "algorithm": cfg.algorithm,
            "max_workers": workers,
            "seed": cfg.seed,
            "repeat": 0,
        }
        label = f"{cfg.model}/{cfg.algorithm}_w{workers}"
        if _cell_key(probe) in done:
            print(f"[{index}/{len(configs)}] skip {label}", flush=True)
            continue
        if skip_processes and cfg.algorithm == "processes":
            print(f"[{index}/{len(configs)}] skip {label} (earlier process cell failed)", flush=True)
            continue
        print(f"[{index}/{len(configs)}] {label}", flush=True)
        try:
            row = time_once(cfg)
        except Exception as exc:  # noqa: BLE001 - record GPU/fork failures
            traceback.print_exc()
            row = {
                "dataset": cfg.dataset,
                "model": cfg.model,
                "algorithm": cfg.algorithm,
                "max_workers": workers,
                "depth": cfg.depth,
                "row_limit": cfg.row_limit if cfg.row_limit is not None else "",
                "seed": cfg.seed,
                "runtime_sec": float("nan"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if cfg.algorithm == "processes":
                skip_processes = True
        row["repeat"] = 0
        _append_row(raw_path, row)
        done.add(_cell_key(row))
        calls = row.get("tabpfn_calls_total", "")
        runtime = row.get("runtime_sec")
        runtime_txt = f"{runtime:.2f}s" if isinstance(runtime, float) and runtime == runtime else row.get("status")
        print(f"    {runtime_txt}  calls={calls}", flush=True)

    if not raw_path.is_file():
        print("keine Messungen", file=sys.stderr)
        return 1
    raw = pd.read_csv(raw_path)
    if "status" in raw.columns:
        ok = raw[raw["status"].fillna("ok") == "ok"].copy()
    else:
        ok = raw.copy()
    ok = ok.dropna(subset=["runtime_sec"])
    if ok.empty:
        print("keine erfolgreichen Zellen", file=sys.stderr)
        return 1
    summary = summarize(ok)
    summary.to_csv(summary_path, index=False)
    plot_speedup(
        summary,
        png_path,
        subtitle=(
            f"TabPFN lokal, Tiefe {args.depth}, "
            f"train/test-cap {args.train_cap}/{args.test_cap}, "
            f"{args.tabpfn_n_estimators} estimator"
        ),
    )
    meta = {
        "dataset": args.dataset,
        "model": "tabpfn",
        "backend": "local",
        "workers": args.workers,
        "depth": args.depth,
        "train_cap": args.train_cap,
        "test_cap": args.test_cap,
        "tabpfn_n_estimators": args.tabpfn_n_estimators,
        "plan": space,
        "elapsed_sec": time.perf_counter() - started,
        "files": [str(p) for p in (raw_path, summary_path, png_path)],
    }
    (RESULTS_DIR / f"{args.tag}_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== Median runtime and speedup over sequential DFS")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"{time.perf_counter() - started:.0f}s -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
