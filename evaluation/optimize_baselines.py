"""
Headless Optuna HPO for all 4 human-designed baseline strategies.

Usage:
    python -m evaluation.optimize_baselines --asset btc
    python -m evaluation.optimize_baselines --asset zec
    python -m evaluation.optimize_baselines --asset all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forge_engine import (
    OptunaOptimizer,
    WalkForwardConfig,
    HoldoutConfig,
    IntRange,
    FloatRange,
    MetricSpec,
    Constraint,
    create_session,
)
from forge_engine.optuna_optimizer import AntiOverfitConfig, WalkForwardSplitter
from evaluation.artifacts import run_strategy_with_artifacts

from examples.sma_cross import SMACrossStrategy
from examples.rsi_reversal import RSIReversalStrategy
from examples.bb_reversion import BollingerBandStrategy
from examples.macd_momentum import MACDMomentumStrategy

# ---------------------------------------------------------------------------
# Strategy configurations
# ---------------------------------------------------------------------------

STRATEGIES = {
    "SMACross": {
        "ctor": SMACrossStrategy,
        "param_space": {
            "fast": IntRange(5, 15, step=2),
            "slow": IntRange(20, 50, step=5),
            "margin_pct": FloatRange(0.05, 0.2, step=0.05),
            "sl_pct": FloatRange(0.0, 0.05, step=0.01),
            "tp_pct": FloatRange(0.0, 0.10, step=0.02),
        },
    },
    "RSIReversal": {
        "ctor": RSIReversalStrategy,
        "param_space": {
            "rsi_period": IntRange(7, 21, step=2),
            "oversold_threshold": FloatRange(20.0, 35.0, step=5.0),
            "overbought_threshold": FloatRange(65.0, 80.0, step=5.0),
            "margin_pct": FloatRange(0.05, 0.2, step=0.05),
            "sl_pct": FloatRange(0.0, 0.05, step=0.01),
            "tp_pct": FloatRange(0.0, 0.10, step=0.02),
        },
    },
    "BollingerBand": {
        "ctor": BollingerBandStrategy,
        "param_space": {
            "period": IntRange(10, 30, step=5),
            "multiplier": FloatRange(1.5, 3.0, step=0.5),
            "margin_pct": FloatRange(0.05, 0.2, step=0.05),
            "sl_pct": FloatRange(0.0, 0.05, step=0.01),
            "tp_pct": FloatRange(0.0, 0.10, step=0.02),
        },
    },
    "MACDMomentum": {
        "ctor": MACDMomentumStrategy,
        "param_space": {
            "fast": IntRange(8, 16, step=2),
            "slow": IntRange(20, 30, step=2),
            "signal_period": IntRange(7, 12, step=1),
            "atr_period": IntRange(10, 20, step=2),
            "atr_min_pct": FloatRange(0.005, 0.03, step=0.005),
            "margin_pct": FloatRange(0.05, 0.2, step=0.05),
            "sl_pct": FloatRange(0.01, 0.05, step=0.01),
            "tp_pct": FloatRange(0.02, 0.10, step=0.02),
        },
    },
}

# ---------------------------------------------------------------------------
# Asset-specific session configs
# ---------------------------------------------------------------------------

ASSET_CONFIGS = {
    "btc": {
        "symbol": "BTCUSDT_PERP",
        "slippage_pct": 0.0005,
        "leverage": 2,
        "output_file": "baselines_btc.json",
    },
    "zec": {
        "symbol": "ZECUSDT_PERP",
        "slippage_pct": 0.0015,
        "leverage": 5,
        "output_file": "baselines_zec.json",
    },
}

COMMON_SESSION = {
    "start_date": "2020-01-01T00:00:00Z",
    "end_date": "2026-02-12T00:00:00Z",
    "starting_cash": 100000,
    "margin_mode": "cross",
    "warmup_candles": 50,
    "timeframe": "1h",
    "close_at_end": True,
}

N_TRIALS = 200
N_WFA_SPLITS = 5
HOLDOUT_RATIO = 0.15
WFA_MODE = "anchored"

METRICS = [
    MetricSpec("performance.smart_sharpe", "max", 1.0),
    MetricSpec("performance.pnl", "max", 1.0),
    MetricSpec("performance.win_rate_pct", "max", 1.0),
]

CONSTRAINTS = [
    Constraint("risk.max_drawdown_pct", "<=", 25),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        if abs(f) == float("inf"):
            return default
        return f
    except Exception:
        return default


def _extract_metric(metrics: dict, path: str, default=0.0):
    cur = metrics
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return _safe_float(cur, default)


def _check_data_exists(symbol: str) -> bool:
    data_dir = PROJECT_ROOT / "data"
    pattern = f"{symbol}_1m.csv"
    return (data_dir / pattern).exists()


def _parse_utc_z(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _with_trace(metrics: dict, trace: dict) -> dict:
    enriched = dict(metrics or {})
    enriched["equities"] = trace.get("equities", [])
    enriched["equity_times"] = trace.get("equity_times", [])
    enriched["returns"] = trace.get("returns", [])
    enriched["events"] = trace.get("events", [])
    return enriched


def _trace_strategy_run(session_kwargs: dict, ctor, params: dict) -> dict:
    session = create_session(enable_visual=False, **session_kwargs)
    strategy = ctor(**params)
    return run_strategy_with_artifacts(session, strategy)


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def optimize_asset(asset_key: str) -> dict:
    cfg = ASSET_CONFIGS[asset_key]
    symbol = cfg["symbol"]

    print(f"\n{'='*70}")
    print(f"  OPTIMIZING BASELINES FOR {symbol}")
    print(f"{'='*70}\n")

    if not _check_data_exists(symbol):
        print(f"  WARNING: Data file for {symbol} not found in data/. Skipping.")
        return {"symbol": symbol, "strategies": {}, "error": "data_not_found"}

    session_kwargs = {
        **COMMON_SESSION,
        "symbol": symbol,
        "slippage_pct": cfg["slippage_pct"],
        "leverage": cfg["leverage"],
    }

    results: dict = {"symbol": symbol, "strategies": {}}
    total_start = time.perf_counter()

    splitter = WalkForwardSplitter(WalkForwardConfig(
        n_splits=N_WFA_SPLITS,
        test_ratio=0.2,
        mode=WFA_MODE,
    ))
    _splits, holdout_period = splitter.generate_splits(
        full_start=_parse_utc_z(COMMON_SESSION["start_date"]),
        full_end=_parse_utc_z(COMMON_SESSION["end_date"]),
        holdout_ratio=HOLDOUT_RATIO,
        timeframe_minutes=60,
    )

    for strat_name, strat_cfg in STRATEGIES.items():
        print(f"\n--- {strat_name} ({symbol}) ---")
        strat_start = time.perf_counter()

        try:
            optimizer = OptunaOptimizer(
                session_kwargs=session_kwargs,
                strategy_ctor=strat_cfg["ctor"],
                param_space=strat_cfg["param_space"],
                metrics=METRICS,
                constraints=CONSTRAINTS,
                wfa_config=WalkForwardConfig(
                    n_splits=N_WFA_SPLITS,
                    test_ratio=0.2,
                    mode=WFA_MODE,
                ),
                holdout_config=HoldoutConfig(holdout_ratio=HOLDOUT_RATIO),
                anti_overfit=AntiOverfitConfig(min_trades_per_fold=3),
            )

            result = optimizer.optimize(
                n_trials=N_TRIALS,
                n_jobs=1,
                verbose=True,
            )

            strat_elapsed = time.perf_counter() - strat_start

            oos_m = result.best_oos_metrics or {}
            hold_m = result.holdout_metrics or {}
            best_trial = result.leaderboard[0] if result.leaderboard else None

            holdout_trace = {}
            if holdout_period is not None:
                holdout_trace = _trace_strategy_run(
                    {
                        **session_kwargs,
                        "start_date": holdout_period[0].isoformat().replace("+00:00", "Z"),
                        "end_date": holdout_period[1].isoformat().replace("+00:00", "Z"),
                    },
                    strat_cfg["ctor"],
                    result.best_params,
                )
                if holdout_trace.get("metrics"):
                    hold_m = _with_trace(holdout_trace["metrics"], holdout_trace)

            entry = {
                "best_params": result.best_params,
                "oos_sharpe": _extract_metric(oos_m, "performance.smart_sharpe"),
                "oos_sortino": _extract_metric(oos_m, "performance.smart_sortino"),
                "oos_max_dd_pct": _extract_metric(oos_m, "risk.max_drawdown_pct"),
                "oos_pnl_pct": _safe_float(
                    _extract_metric(oos_m, "performance.pnl") / COMMON_SESSION["starting_cash"] * 100
                ),
                "holdout_sharpe": _extract_metric(hold_m, "performance.smart_sharpe"),
                "holdout_sortino": _extract_metric(hold_m, "performance.smart_sortino"),
                "holdout_max_dd_pct": _extract_metric(hold_m, "risk.max_drawdown_pct"),
                "holdout_pnl_pct": _safe_float(
                    _extract_metric(hold_m, "performance.pnl") / COMMON_SESSION["starting_cash"] * 100
                ),
                "best_oos_metrics": oos_m,
                "holdout_metrics": hold_m,
                "best_fold_results": [asdict(fr) for fr in (best_trial.fold_results if best_trial else [])],
                "n_trials": result.completed_trials,
                "pruned_trials": result.pruned_trials,
                "duration_sec": round(strat_elapsed, 1),
            }

            results["strategies"][strat_name] = entry

            print(f"\n  [DONE] {strat_name}: OOS sharpe={entry['oos_sharpe']:.3f}, "
                  f"holdout sharpe={entry['holdout_sharpe']:.3f}, "
                  f"DD={entry['oos_max_dd_pct']:.1f}%, "
                  f"time={strat_elapsed:.0f}s")

        except Exception as exc:
            strat_elapsed = time.perf_counter() - strat_start
            print(f"\n  [ERROR] {strat_name} failed after {strat_elapsed:.0f}s: {exc}")
            traceback.print_exc()
            results["strategies"][strat_name] = {
                "error": str(exc),
                "duration_sec": round(strat_elapsed, 1),
            }

    total_elapsed = time.perf_counter() - total_start
    results["total_duration_sec"] = round(total_elapsed, 1)

    # Save results
    out_dir = PROJECT_ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["output_file"]

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to {out_path}")
    print(f"  Total time for {symbol}: {total_elapsed:.0f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Optuna HPO on baseline strategies")
    parser.add_argument(
        "--asset",
        choices=["btc", "zec", "all"],
        default="btc",
        help="Which asset to optimize (default: btc)",
    )
    args = parser.parse_args()

    assets = ["btc", "zec"] if args.asset == "all" else [args.asset]
    all_results = {}

    for asset in assets:
        all_results[asset] = optimize_asset(asset)

    print("\n" + "=" * 70)
    print("  ALL OPTIMIZATIONS COMPLETE")
    print("=" * 70)

    for asset, res in all_results.items():
        symbol = res.get("symbol", asset.upper())
        strats = res.get("strategies", {})
        print(f"\n  {symbol}:")
        for name, data in strats.items():
            if "error" in data and "best_params" not in data:
                print(f"    {name}: FAILED ({data['error']})")
            else:
                print(f"    {name}: OOS sharpe={data.get('oos_sharpe', 0):.3f}, "
                      f"holdout sharpe={data.get('holdout_sharpe', 0):.3f}")


if __name__ == "__main__":
    main()
