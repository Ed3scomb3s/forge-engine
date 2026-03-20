"""
Monte Carlo bootstrap significance tests for thesis Phase A.

Tests whether Sharpe ratio differences between RL agents and human baselines
are statistically significant using bootstrap resampling (10,000 samples).
Also computes multi-seed stability and WFA fold consistency metrics.

Usage:
    python -m evaluation.statistical_tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evaluation.result_utils import (
    aggregate_rl_holdout_returns,
    extract_baseline_holdout_returns,
    iter_primary_rl_result_files,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ANNUALIZATION_FACTOR = np.sqrt(8760)  # hourly bars, 8760 hours/year


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        if f != f or abs(f) == float("inf"):
            return default
        return f
    except Exception:
        return default


def _extract_metric(data: dict, path: str, default: float = 0.0) -> float:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return _safe_float(cur, default)


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe from hourly returns array."""
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns) / std * ANNUALIZATION_FACTOR)


def _returns_from_equities(equities: list) -> np.ndarray:
    """Convert equity curve to returns array."""
    eq = np.array(equities, dtype=float)
    if len(eq) < 2:
        return np.array([], dtype=float)
    mask = eq[:-1] > 0
    rets = np.zeros(len(eq) - 1)
    rets[mask] = eq[1:][mask] / eq[:-1][mask] - 1.0
    return rets


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------

def bootstrap_sharpe_difference(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Monte Carlo bootstrap test for Sharpe ratio difference.

    Tests H0: Sharpe(A) - Sharpe(B) = 0
    Uses paired resampling (same indices for both series).

    Args:
        returns_a: Hourly returns of strategy A.
        returns_b: Hourly returns of strategy B.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for reproducibility.

    Returns:
        dict with: observed_diff, mean_diff, ci_lower, ci_upper, p_value, significant
    """
    n = min(len(returns_a), len(returns_b))
    if n < 10:
        return {
            "observed_diff": 0.0,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "significant": False,
            "error": f"Insufficient data: {n} observations",
        }

    ra = returns_a[:n]
    rb = returns_b[:n]

    observed_diff = _compute_sharpe(ra) - _compute_sharpe(rb)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled_a = ra[idx]
        sampled_b = rb[idx]
        diffs[i] = _compute_sharpe(sampled_a) - _compute_sharpe(sampled_b)

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided p-value: fraction of bootstrap samples on the other side of zero
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": float(observed_diff),
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": bool(ci_lower > 0 or ci_upper < 0),  # CI excludes zero
    }


# ---------------------------------------------------------------------------
# Multi-seed stability
# ---------------------------------------------------------------------------

def multi_seed_stability(seed_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute stability metrics across seeds.

    Args:
        seed_results: dict of seed_key -> {sharpe, sortino, max_dd_pct, pnl_pct}

    Returns:
        dict with mean, std, cv for each metric
    """
    metrics: Dict[str, Dict[str, float]] = {}

    for metric_name in ("sharpe", "sortino", "max_dd_pct", "pnl_pct"):
        values = []
        for seed_data in seed_results.values():
            v = seed_data.get(metric_name)
            if v is not None:
                values.append(_safe_float(v))

        if not values:
            metrics[metric_name] = {"mean": 0.0, "std": 0.0, "cv": float("inf"), "n_seeds": 0}
            continue

        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        cv = std / abs(mean) if abs(mean) > 1e-10 else float("inf")
        metrics[metric_name] = {
            "mean": mean,
            "std": std,
            "cv": cv,
            "n_seeds": len(values),
            "values": [float(v) for v in values],
        }

    return metrics


# ---------------------------------------------------------------------------
# WFA fold consistency
# ---------------------------------------------------------------------------

def wfa_fold_consistency(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze how consistent OOS results are across WFA folds.

    Args:
        fold_results: list of fold dicts with test_metrics.

    Returns:
        dict with per-fold Sharpe, spread, consistency metrics
    """
    sharpes = []
    pnls = []
    dds = []

    for fr in fold_results:
        tm = fr.get("test_metrics") or {}
        perf = tm.get("performance") or {}
        risk = tm.get("risk") or {}

        if "smart_sharpe" in perf:
            sharpes.append(_safe_float(perf["smart_sharpe"]))
        if "pnl_pct" in perf:
            pnls.append(_safe_float(perf["pnl_pct"]))
        if "max_drawdown_pct" in risk:
            dds.append(_safe_float(risk["max_drawdown_pct"]))

    result: Dict[str, Any] = {
        "n_folds": len(fold_results),
    }

    for name, values in [("sharpe", sharpes), ("pnl_pct", pnls), ("max_dd_pct", dds)]:
        if not values:
            result[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
            continue
        arr = np.array(values)
        result[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "values": [float(v) for v in values],
        }
        # Positive-fold ratio for Sharpe
        if name == "sharpe":
            result[name]["positive_ratio"] = float(np.mean(arr > 0))

    return result


# ---------------------------------------------------------------------------
# Load data and run tests
# ---------------------------------------------------------------------------

def _load_rl_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load an RL results file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_baseline_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load a baseline results file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_baseline_equities(baseline_data: dict, strat_name: str) -> Optional[np.ndarray]:
    """Try to extract equity curve or returns from baseline results.

    Baseline optimizer results may include full metrics with equity data.
    If not available, we cannot run bootstrap tests on baselines.
    """
    strat = baseline_data.get("strategies", {}).get(strat_name, {})
    # Check if holdout_metrics contains equities array
    hold = strat.get("holdout_metrics", {})
    equities = hold.get("equities")
    if equities:
        return _returns_from_equities(equities)

    # Check best_oos_metrics
    oos = strat.get("best_oos_metrics", {})
    equities = oos.get("equities")
    if equities:
        return _returns_from_equities(equities)

    return None


def _get_rl_returns(rl_data: dict, period: str = "holdout") -> Optional[np.ndarray]:
    """Extract returns from RL results (best seed or first available).

    Args:
        rl_data: RL result file content
        period: "holdout" or "oos"
    """
    seeds = rl_data.get("seeds", {})
    for seed_key, seed_data in seeds.items():
        if period == "holdout":
            hold = seed_data.get("holdout_eval", seed_data.get("holdout_metrics", {}))
            equities = hold.get("equities")
            if equities:
                return _returns_from_equities(equities)
        else:
            # Concatenate OOS fold equities
            folds = seed_data.get("fold_results", [])
            all_equities = []
            for fold in folds:
                tm = fold.get("test_metrics", {})
                eq = tm.get("equities")
                if eq:
                    all_equities.extend(eq)
            if all_equities:
                return _returns_from_equities(all_equities)
    return None


def run_all_tests() -> Dict[str, Any]:
    """Run all statistical tests on available results.

    Parses the actual train_rl.py output format:
    {algorithm, symbol, folds: [{fold, seeds: {seed: {sharpe, ...}}, mean, std}],
     holdout: {seeds: {...}, mean, std}, aggregate_oos: {mean_sharpe, ...}}
    """
    results: Dict[str, Any] = {
        "bootstrap_tests": [],
        "seed_stability": {},
        "fold_consistency": {},
    }

    if not RESULTS_DIR.exists():
        print(f"[stats] Results directory not found: {RESULTS_DIR}")
        return results

    rl_files = list(iter_primary_rl_result_files(RESULTS_DIR))
    baseline_files = sorted(RESULTS_DIR.glob("baselines_*.json"))

    print(f"[stats] Found {len(baseline_files)} baseline files, {len(rl_files)} RL files")

    # --- Multi-seed stability analysis ---
    # For each RL agent, compute stability of holdout metrics across seeds
    print("\n=== MULTI-SEED STABILITY ===")
    for rl_path in rl_files:
        rl_data = _load_rl_file(rl_path)
        if not rl_data:
            continue

        agent = rl_data.get("algorithm", rl_path.stem)
        symbol = rl_data.get("symbol", "?")

        # Extract holdout per-seed metrics
        holdout = rl_data.get("holdout") or {}
        ho_seeds = holdout.get("seeds", {}) if holdout else {}
        if len(ho_seeds) < 2:
            # Also try aggregate from folds
            pass

        # Collect per-seed aggregate OOS metrics from all folds
        seed_list = rl_data.get("config", {}).get("seeds", [])
        folds = rl_data.get("folds", [])

        if seed_list and folds:
            # Compute per-seed aggregate: average each seed's Sharpe across all folds
            seed_agg_metrics = {}
            for seed in [str(s) for s in seed_list]:
                seed_sharpes = []
                seed_pnls = []
                seed_dds = []
                seed_sortinos = []
                for fold in folds:
                    sm = fold.get("seeds", {}).get(seed, {})
                    if "error" in sm:
                        continue
                    if "sharpe" in sm:
                        seed_sharpes.append(sm["sharpe"])
                    if "pnl_pct" in sm:
                        seed_pnls.append(sm["pnl_pct"])
                    if "max_dd_pct" in sm:
                        seed_dds.append(sm["max_dd_pct"])
                    if "sortino" in sm:
                        seed_sortinos.append(sm["sortino"])
                if seed_sharpes:
                    seed_agg_metrics[seed] = {
                        "sharpe": float(np.mean(seed_sharpes)),
                        "sortino": float(np.mean(seed_sortinos)) if seed_sortinos else 0.0,
                        "max_dd_pct": float(np.mean(seed_dds)) if seed_dds else 0.0,
                        "pnl_pct": float(np.mean(seed_pnls)) if seed_pnls else 0.0,
                    }

            if len(seed_agg_metrics) >= 2:
                stability = multi_seed_stability(seed_agg_metrics)
                key = f"{agent}_{symbol}"
                results["seed_stability"][key] = stability

                print(f"\n  {agent} ({symbol}) -- {len(seed_agg_metrics)} seeds:")
                for metric_name, stats in stability.items():
                    print(f"    {metric_name}: mean={stats['mean']:.3f}, "
                          f"std={stats['std']:.3f}, CV={stats['cv']:.2f}")
            else:
                print(f"  {agent} ({symbol}): Only {len(seed_agg_metrics)} usable seed(s)")

    # --- WFA fold consistency ---
    # Use fold-level mean metrics (averaged across seeds per fold)
    print("\n=== WFA FOLD CONSISTENCY ===")
    for rl_path in rl_files:
        rl_data = _load_rl_file(rl_path)
        if not rl_data:
            continue

        agent = rl_data.get("algorithm", rl_path.stem)
        symbol = rl_data.get("symbol", "?")
        folds = rl_data.get("folds", [])

        if not folds:
            continue

        # Build fold_results with the mean metrics per fold
        fold_sharpes = []
        fold_pnls = []
        fold_dds = []
        for fold in folds:
            mean = fold.get("mean", {})
            if "sharpe" in mean:
                fold_sharpes.append(mean["sharpe"])
            if "pnl_pct" in mean:
                fold_pnls.append(mean["pnl_pct"])
            if "max_dd_pct" in mean:
                fold_dds.append(mean["max_dd_pct"])

        if fold_sharpes:
            arr = np.array(fold_sharpes)
            consistency = {
                "n_folds": len(folds),
                "sharpe": {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "range": float(np.max(arr) - np.min(arr)),
                    "values": [float(v) for v in arr],
                    "positive_ratio": float(np.mean(arr > 0)),
                },
            }
            if fold_pnls:
                parr = np.array(fold_pnls)
                consistency["pnl_pct"] = {
                    "mean": float(np.mean(parr)),
                    "std": float(np.std(parr, ddof=1)) if len(parr) > 1 else 0.0,
                    "values": [float(v) for v in parr],
                }

            key = f"{agent}_{symbol}"
            results["fold_consistency"][key] = consistency

            sharpe_info = consistency["sharpe"]
            print(f"\n  {agent} ({symbol}) -- {consistency['n_folds']} folds:")
            print(f"    Sharpe: mean={sharpe_info['mean']:.3f}, "
                  f"std={sharpe_info['std']:.3f}, "
                  f"range=[{sharpe_info['min']:.3f}, {sharpe_info['max']:.3f}], "
                  f"positive folds={sharpe_info['positive_ratio']:.0%}")

    # --- Bootstrap significance tests ---
    print("\n=== BOOTSTRAP SIGNIFICANCE TESTS ===")
    baseline_payloads = []
    for baseline_path in baseline_files:
        payload = _load_baseline_file(baseline_path)
        if payload:
            baseline_payloads.append((baseline_path, payload))

    for rl_path in rl_files:
        rl_data = _load_rl_file(rl_path)
        if not rl_data:
            continue

        agent = rl_data.get("algorithm", rl_path.stem)
        rl_symbol = str(rl_data.get("symbol", "?"))
        rl_returns = aggregate_rl_holdout_returns(rl_data)

        matching = [
            payload for _path, payload in baseline_payloads
            if rl_symbol.upper() == str(payload.get("symbol", "")).upper()
        ]

        if not matching:
            results["bootstrap_tests"].append({
                "rl_agent": agent,
                "symbol": rl_symbol,
                "error": "no_matching_baseline_file",
            })
            continue

        baseline_data = matching[0]
        strategies = baseline_data.get("strategies", {})
        if not isinstance(strategies, dict):
            continue

        for strategy_name, strategy_payload in strategies.items():
            if not isinstance(strategy_payload, dict):
                continue

            test_entry: Dict[str, Any] = {
                "rl_agent": agent,
                "baseline_strategy": strategy_name,
                "symbol": rl_symbol,
                "period": "holdout",
            }

            baseline_returns = extract_baseline_holdout_returns(baseline_data, strategy_name)
            if rl_returns is None:
                test_entry["error"] = "rl_holdout_curve_missing"
            elif baseline_returns is None:
                test_entry["error"] = "baseline_holdout_curve_missing"
            else:
                test_entry.update(
                    bootstrap_sharpe_difference(
                        rl_returns,
                        baseline_returns,
                        n_bootstrap=10_000,
                        seed=42,
                    )
                )

            results["bootstrap_tests"].append(test_entry)

            if "error" in test_entry:
                print(f"  {agent} vs {strategy_name} ({rl_symbol}): {test_entry['error']}")
            else:
                print(
                    f"  {agent} vs {strategy_name} ({rl_symbol}): "
                    f"diff={test_entry['observed_diff']:+.3f}, "
                    f"CI=[{test_entry['ci_lower']:+.3f}, {test_entry['ci_upper']:+.3f}], "
                    f"p={test_entry['p_value']:.3f}"
                )

    if not results["bootstrap_tests"]:
        results["bootstrap_tests"] = [{
            "note": "No RL or baseline result files were available for bootstrap testing.",
        }]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[stats] Running statistical tests...")
    results = run_all_tests()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "statistical_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[stats] Results saved to {out_path}")

    # Summary
    bootstrap = results.get("bootstrap_tests", [])
    sig_count = sum(1 for t in bootstrap if t.get("significant"))
    total = len(bootstrap)
    tested = sum(1 for t in bootstrap if "error" not in t)

    print(f"\n=== STATISTICAL TESTS SUMMARY ===")
    print(f"  Bootstrap comparisons:  {total} total, {tested} with data")
    print(f"  Significant at 95%:     {sig_count}/{tested}")
    print(f"  Seed stability entries: {len(results.get('seed_stability', {}))}")
    print(f"  Fold consistency:       {len(results.get('fold_consistency', {}))}")

    if not bootstrap or tested == 0:
        print("\n  NOTE: Bootstrap tests require equity curve data in result files.")
        print("  Ensure RL training scripts save equities arrays in their output.")

    print("\n[stats] Done.")


if __name__ == "__main__":
    main()
