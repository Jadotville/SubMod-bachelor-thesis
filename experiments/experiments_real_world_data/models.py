"""Model builders and train/predict hooks for real-world adaptability runs."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from preprocess import EncodedModelHooks

# One model per inductive bias, in increasing flexibility: linear, bagged trees,
# boosted trees, neural net, in-context learner. Random forest is not redundant
# with LightGBM here — it averages instead of boosting, so it overfits the global
# fit far less (LightGBM's train-test AUC gap reaches 0.12 on these datasets),
# which changes how much headroom a local model can still claim.
CORE_MODELS = ("lr", "rf", "lgbm", "mlp", "tabpfn")
#: Everything except TabPFN. TabPFN runs locally on one GPU (``run_tabpfn.sh``).
OFFLINE_MODELS = ("lr", "rf", "lgbm", "mlp")
SMOKE_MODELS = ("lr", "lgbm")
ALL_MODELS = CORE_MODELS

# Package ``tabpfn`` (not ``tabpfn_client``). ``TABPFN_N_ESTIMATORS`` matches the
# 8.x classifier default (eight). The thesis GPU batch passes ``--tabpfn-n-estimators 1``.
# ``ignore_pretraining_limits`` allows the 50/50 split of the 100k-row cap
# (50k training rows), which exceeds the original 10k context.
TABPFN_N_ESTIMATORS = 8
TABPFN_DEVICE = "cuda"
TABPFN_IGNORE_PRETRAINING_LIMITS = True


def tabpfn_settings(seed: int, n_estimators: int | None = None) -> dict[str, object]:
    """Frozen constructor arguments, recorded in meta.json for later repeats."""
    return {
        "backend": "local",
        "package": "tabpfn",
        "n_estimators": TABPFN_N_ESTIMATORS if n_estimators is None else n_estimators,
        "device": TABPFN_DEVICE,
        "ignore_pretraining_limits": TABPFN_IGNORE_PRETRAINING_LIMITS,
        "random_state": seed,
        "memory_saving_mode": "auto",
    }


def _load_tabpfn_license_token(access_token: Optional[str] = None) -> None:
    """Load TABPFN_TOKEN for the one-time weight-download license, not API inference."""
    if access_token and access_token.strip():
        os.environ["TABPFN_TOKEN"] = access_token.strip()
        return
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return
    home = Path.home()
    root = Path(__file__).resolve().parents[2]
    for path in (
        home / ".cache" / "tabpfn" / "auth_token",
        home / ".tabpfn" / "token",
        root / "tabpfn_api_token.txt",
    ):
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                os.environ["TABPFN_TOKEN"] = token
                return


def configure_tabpfn(access_token: Optional[str] = None) -> None:
    """Require a local CUDA build of ``tabpfn``. ``access_token`` is a license key."""
    _load_tabpfn_license_token(access_token)
    try:
        import torch
        from tabpfn import TabPFNClassifier  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Lokales TabPFN fehlt. Nach einem CUDA-PyTorch: "
            "pip install -r requirements.txt"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Lokales TabPFN braucht CUDA. "
            '`python -c "import torch; print(torch.cuda.get_device_name(0))"` '
            "muss eine GPU nennen."
        )


def _tabpfn_builder(seed: int, n_estimators: int | None = None) -> Callable:
    from tabpfn import TabPFNClassifier

    n_ens = TABPFN_N_ESTIMATORS if n_estimators is None else n_estimators

    def build() -> TabPFNClassifier:
        return TabPFNClassifier(
            n_estimators=n_ens,
            device=TABPFN_DEVICE,
            random_state=seed,
            ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS,
            memory_saving_mode="auto",
        )

    return build


def _sklearn_builder(name: str, seed: int) -> Callable:
    if name == "lr":
        return lambda: LogisticRegression(
            max_iter=1000, n_jobs=1, random_state=seed
        )
    if name == "rf":
        # Depth is capped rather than left unlimited: an unbounded forest memorises
        # the training half and its global AUC stops being a meaningful reference
        # point for the local models.
        return lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=seed,
        )
    if name == "lgbm":
        return lambda: LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    if name == "mlp":
        return lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=300,
            early_stopping=True,
            random_state=seed,
            alpha=0.01,
        )
    raise ValueError(f"Unknown sklearn-family model {name!r}")


class TabPFNBudgetExceeded(RuntimeError):
    """Optional fit cap (``--tabpfn-max-fits``). Unused in the default local batch."""


_LOG = logging.getLogger("real_world_data")


class TabPFNCallCounter:
    """Count TabPFN fit/predict calls and stop before a budget is exhausted."""

    def __init__(self, max_fits: Optional[int] = None, log_every: int = 10):
        self.max_fits = max_fits
        self.log_every = log_every
        self.fits = 0
        self.predicts = 0
        self._t0 = time.perf_counter()

    @property
    def total(self) -> int:
        return self.fits + self.predicts

    def note_fit(self) -> None:
        if self.max_fits is not None and self.fits >= self.max_fits:
            raise TabPFNBudgetExceeded(
                f"TabPFN fit budget of {self.max_fits} reached "
                f"({self.predicts} predictions so far). Reduce --depth, "
                "--max-search-selectors, or raise --tabpfn-max-fits."
            )
        self.fits += 1
        if self.fits == 1 or self.fits % self.log_every == 0:
            elapsed = time.perf_counter() - self._t0
            rate = self.fits / elapsed if elapsed else 0.0
            _LOG.info(
                "    tabpfn: %d fits in %.0fs (%.2f/s)",
                self.fits,
                elapsed,
                rate,
            )

    def note_predict(self) -> None:
        self.predicts += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "tabpfn_fits": self.fits,
            "tabpfn_predicts": self.predicts,
            "tabpfn_calls_total": self.total,
            "tabpfn_max_fits": self.max_fits,
        }


def resolve_model_hooks(
    model_name: str,
    seed: int,
    counter: Optional[TabPFNCallCounter] = None,
    tabpfn_n_estimators: int | None = None,
) -> tuple[Callable, Callable, Callable, Callable, Callable]:
    """
    Return (model_builder, train_global, train_local, predict_global, predict_local).

    For TabPFN an optional ``counter`` records every local fit.
    """
    if model_name == "tabpfn":
        builder = _tabpfn_builder(seed, tabpfn_n_estimators)

        def train(model, X: pd.DataFrame, y: pd.Series) -> None:
            if counter is not None:
                counter.note_fit()
            model.fit(X, y)

        def predict(model, X: pd.DataFrame) -> np.ndarray:
            if counter is not None:
                counter.note_predict()
            return model.predict_proba(X)[:, 1]

        return (builder, train, train, predict, predict)

    if model_name in {"lr", "rf", "lgbm", "mlp"}:
        hooks = EncodedModelHooks(_sklearn_builder(model_name, seed))
        return (
            hooks.build,
            hooks.train_global,
            hooks.train_local,
            hooks.predict,
            hooks.predict,
        )

    raise ValueError(
        f"Unknown model {model_name!r}; choose from {ALL_MODELS}"
    )
