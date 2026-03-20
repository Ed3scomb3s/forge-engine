from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


_WORKER_RE = re.compile(r"_worker\d+$")
_NON_FINAL_RE = re.compile(r"(?:^|_)(smoke|test|debug|tmp|temp)(?:_|$)")


def is_primary_rl_result_file(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    name = path.stem.lower()
    if not name.startswith("rl_"):
        return False
    if name.endswith("_checkpoint"):
        return False
    if _WORKER_RE.search(name):
        return False
    if _NON_FINAL_RE.search(name):
        return False
    return True


def iter_primary_rl_result_files(results_dir: Path) -> Iterable[Path]:
    for path in sorted(results_dir.glob("rl_*.json")):
        if is_primary_rl_result_file(path):
            yield path


def _returns_from_equities(equities: List[float]) -> np.ndarray:
    eq = np.asarray(equities, dtype=np.float64)
    if len(eq) < 2:
        return np.array([], dtype=np.float64)
    prev = np.maximum(eq[:-1], 1e-10)
    return eq[1:] / prev - 1.0


def aggregate_rl_holdout_returns(rl_data: Dict[str, object]) -> Optional[np.ndarray]:
    holdout = rl_data.get("holdout") or {}
    if not isinstance(holdout, dict):
        return None
    seeds = holdout.get("seeds") or {}
    if not isinstance(seeds, dict) or not seeds:
        return None

    curves: List[np.ndarray] = []
    for seed_data in seeds.values():
        if not isinstance(seed_data, dict):
            continue
        equities = seed_data.get("equities")
        if not isinstance(equities, list) or len(equities) < 2:
            continue
        curves.append(np.asarray(equities, dtype=np.float64))

    if not curves:
        return None

    min_len = min(len(curve) for curve in curves)
    if min_len < 2:
        return None

    aligned = np.vstack([curve[:min_len] for curve in curves])
    mean_equity = np.mean(aligned, axis=0)
    return _returns_from_equities(mean_equity.tolist())


def extract_baseline_holdout_returns(baseline_data: Dict[str, object], strategy_name: str) -> Optional[np.ndarray]:
    strategies = baseline_data.get("strategies") or {}
    if not isinstance(strategies, dict):
        return None
    strategy = strategies.get(strategy_name) or {}
    if not isinstance(strategy, dict):
        return None
    holdout_metrics = strategy.get("holdout_metrics") or {}
    if not isinstance(holdout_metrics, dict):
        return None
    equities = holdout_metrics.get("equities")
    if not isinstance(equities, list) or len(equities) < 2:
        return None
    return _returns_from_equities(equities)
