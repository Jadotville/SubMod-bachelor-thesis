"""Re-export. Canonical module: experiments_real_world_data/tabpfn_spawn.py."""
from __future__ import annotations

import sys
from pathlib import Path

_RW = Path(__file__).resolve().parents[1] / "experiments_real_world_data"
if str(_RW) not in sys.path:
    sys.path.insert(0, str(_RW))

from tabpfn_spawn import (  # noqa: E402
    TabPFNSpawnProcessDFS,
    _eval_cover,
    _init_worker,
    _predict_proba_positive,
)

__all__ = [
    "TabPFNSpawnProcessDFS",
    "_eval_cover",
    "_init_worker",
    "_predict_proba_positive",
]
