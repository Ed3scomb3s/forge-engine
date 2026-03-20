"""
Comparative evaluation harness for thesis Phase A.

Loads all baseline and RL result JSON files from evaluation/results/
and produces unified comparison tables (printed + saved as JSON/CSV).

Usage:
    python -m evaluation.compare
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.result_utils import iter_primary_rl_result_files

RESULTS_DIR = Path(__file__).resolve().parent / "results"

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


# ---------------------------------------------------------------------------
# Result loaders
# ---------------------------------------------------------------------------

def load_baseline_results(path: Path) -> List[Dict[str, Any]]:
    """Load a baselines_{asset}.json file and return list of row dicts."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    symbol = data.get("symbol", "UNKNOWN")
    asset = "BTC" if "BTC" in symbol.upper() else ("ZEC" if "ZEC" in symbol.upper() else symbol)
    rows = []

    for strat_name, strat_data in data.get("strategies", {}).items():
        if "error" in strat_data and "best_params" not in strat_data:
            continue

        oos_metrics = strat_data.get("best_oos_metrics", {})
        hold_metrics = strat_data.get("holdout_metrics", {})

        # Prefer backfilled holdout_metrics (fresh-start evaluation) when
        # available.  This matches how RL agents are evaluated on holdout
        # (no position carry from OOS).  Fall back to the optimizer's
        # top-level convenience fields only when holdout_metrics is absent.
        hold_perf = hold_metrics.get("performance", {}) if isinstance(hold_metrics, dict) else {}
        hold_risk = hold_metrics.get("risk", {}) if isinstance(hold_metrics, dict) else {}
        has_backfill = bool(hold_perf.get("sharpe") is not None
                           and isinstance(hold_metrics.get("equities"), list))

        if has_backfill:
            h_sharpe = _safe_float(hold_perf.get("sharpe"))
            h_sortino = _safe_float(hold_perf.get("sortino"))
            h_max_dd = _safe_float(hold_risk.get("max_drawdown_pct"))
            h_pnl = _safe_float(hold_perf.get("pnl_pct"))
        else:
            h_sharpe = _safe_float(strat_data.get("holdout_sharpe"))
            h_sortino = _safe_float(strat_data.get("holdout_sortino"))
            h_max_dd = _safe_float(strat_data.get("holdout_max_dd_pct"))
            h_pnl = _safe_float(strat_data.get("holdout_pnl_pct"))

        row = {
            "strategy": strat_name,
            "type": "Human",
            "symbol": asset,
            "oos_sharpe": _safe_float(strat_data.get("oos_sharpe")),
            "oos_sortino": _safe_float(strat_data.get("oos_sortino")),
            "oos_max_dd_pct": _safe_float(strat_data.get("oos_max_dd_pct")),
            "oos_pnl_pct": _safe_float(strat_data.get("oos_pnl_pct")),
            "oos_win_rate": _extract_metric(oos_metrics, "performance.win_rate_pct"),
            "oos_trade_count": int(_extract_metric(oos_metrics, "trade.total_trades")),
            "holdout_sharpe": h_sharpe,
            "holdout_sortino": h_sortino,
            "holdout_max_dd_pct": h_max_dd,
            "holdout_pnl_pct": h_pnl,
            "holdout_win_rate": _extract_metric(hold_metrics, "performance.win_rate_pct"),
            "holdout_trade_count": int(_extract_metric(hold_metrics, "trade.total_trades")),
            "n_trials": strat_data.get("n_trials", 0),
        }
        rows.append(row)

    return rows


def load_rl_results(path: Path) -> List[Dict[str, Any]]:
    """Load an RL result JSON file produced by evaluation/train_rl.py.

    Format: {algorithm, symbol, config, folds: [{fold, seeds: {seed: metrics}, mean, std}],
             holdout: {seeds: {seed: metrics}, mean, std}, aggregate_oos: {mean_sharpe, ...}}
    """
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    agent = data.get("algorithm", path.stem)
    symbol_raw = data.get("symbol", "UNKNOWN")
    asset = "BTC" if "BTC" in symbol_raw.upper() else ("ZEC" if "ZEC" in symbol_raw.upper() else symbol_raw)

    agg = data.get("aggregate_oos", {})
    holdout = data.get("holdout", {})
    ho_mean = holdout.get("mean", {}) if holdout else {}
    ho_std = holdout.get("std", {}) if holdout else {}
    # Try config.seeds first (PPO_v6, newer formats), fall back to counting
    # holdout seed keys for older formats (SAC_v4 etc.) that lack a config block.
    n_seeds = len(data.get("config", {}).get("seeds", []))
    if n_seeds == 0:
        holdout_seeds = (data.get("holdout") or {}).get("seeds", {})
        if isinstance(holdout_seeds, dict):
            n_seeds = len(holdout_seeds)

    row = {
        "strategy": agent,
        "type": "RL",
        "symbol": asset,
        "oos_sharpe": _safe_float(agg.get("mean_sharpe")),
        "oos_sortino": _safe_float(agg.get("mean_sortino")),
        "oos_max_dd_pct": _safe_float(agg.get("mean_max_dd_pct")),
        "oos_pnl_pct": _safe_float(agg.get("mean_pnl_pct")),
        "oos_win_rate": 0.0,
        "oos_trade_count": 0,
        "holdout_sharpe": _safe_float(ho_mean.get("sharpe")),
        "holdout_sortino": _safe_float(ho_mean.get("sortino")),
        "holdout_max_dd_pct": _safe_float(ho_mean.get("max_dd_pct")),
        "holdout_pnl_pct": _safe_float(ho_mean.get("pnl_pct")),
        "holdout_win_rate": 0.0,
        "holdout_trade_count": 0,
        "n_trials": n_seeds,
    }

    return [row]


def load_passive_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []
    for symbol_raw, payload in data.items():
        if not isinstance(payload, dict):
            continue
        asset = "BTC" if "BTC" in symbol_raw.upper() else ("ZEC" if "ZEC" in symbol_raw.upper() else symbol_raw)
        metrics = payload.get("metrics", {})
        perf = metrics.get("performance", {}) if isinstance(metrics, dict) else {}
        risk = metrics.get("risk", {}) if isinstance(metrics, dict) else {}
        trade = metrics.get("trade", {}) if isinstance(metrics, dict) else {}

        rows.append({
            "strategy": payload.get("strategy", "Passive"),
            "type": "Passive",
            "symbol": asset,
            "oos_sharpe": 0.0,
            "oos_sortino": 0.0,
            "oos_max_dd_pct": 0.0,
            "oos_pnl_pct": 0.0,
            "oos_win_rate": 0.0,
            "oos_trade_count": 0,
            "holdout_sharpe": _extract_metric(metrics, "performance.sharpe"),
            "holdout_sortino": _extract_metric(metrics, "performance.sortino"),
            "holdout_max_dd_pct": _extract_metric(metrics, "risk.max_drawdown_pct"),
            "holdout_pnl_pct": _extract_metric(metrics, "performance.pnl_pct"),
            "holdout_win_rate": _extract_metric(metrics, "performance.win_rate_pct"),
            "holdout_trade_count": int(_extract_metric(metrics, "trade.total_trades")),
            "n_trials": 1,
        })

    return rows


# ---------------------------------------------------------------------------
# Load all results
# ---------------------------------------------------------------------------

def load_all_results() -> List[Dict[str, Any]]:
    """Scan evaluation/results/ and load all baseline and RL result files."""
    if not RESULTS_DIR.exists():
        print(f"[compare] Results directory not found: {RESULTS_DIR}")
        return []

    rows: List[Dict[str, Any]] = []

    for fname in sorted(RESULTS_DIR.glob("baselines_*.json")):
        loaded = load_baseline_results(fname)
        print(f"  Loaded {len(loaded)} strategies from {fname.name}")
        rows.extend(loaded)

    for fname in iter_primary_rl_result_files(RESULTS_DIR):
        loaded = load_rl_results(fname)
        print(f"  Loaded {len(loaded)} RL entries from {fname.name}")
        rows.extend(loaded)

    passive_path = RESULTS_DIR / "passive_benchmarks.json"
    if passive_path.exists():
        loaded = load_passive_results(passive_path)
        print(f"  Loaded {len(loaded)} passive entries from {passive_path.name}")
        rows.extend(loaded)

    return rows


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

COLUMNS = [
    ("Strategy",        "strategy",           "<30s"),
    ("Type",            "type",               "<6s"),
    ("Symbol",          "symbol",             "<5s"),
    ("OOS Sharpe",      "oos_sharpe",         ">10.3f"),
    ("OOS Sortino",     "oos_sortino",        ">11.3f"),
    ("OOS MaxDD%",      "oos_max_dd_pct",     ">10.1f"),
    ("OOS PnL%",        "oos_pnl_pct",        ">9.1f"),
    ("OOS WinR%",       "oos_win_rate",       ">9.1f"),
    ("OOS Trades",      "oos_trade_count",    ">10d"),
    ("Hold Sharpe",     "holdout_sharpe",     ">11.3f"),
    ("Hold PnL%",       "holdout_pnl_pct",   ">10.1f"),
    ("Hold MaxDD%",     "holdout_max_dd_pct", ">10.1f"),
]


def print_comparison_table(rows: List[Dict[str, Any]]) -> str:
    """Print and return a formatted comparison table."""
    if not rows:
        msg = "[compare] No results to compare."
        print(msg)
        return msg

    # Sort: Human first, then RL; within each group by symbol then strategy
    rows_sorted = sorted(rows, key=lambda r: (0 if r["type"] == "Human" else 1, r["symbol"], r["strategy"]))

    # Header
    header_parts = []
    sep_parts = []
    for title, _key, fmt in COLUMNS:
        width = max(len(title), int("".join(c for c in fmt if c.isdigit()) or "10"))
        header_parts.append(f"{title:>{width}s}")
        sep_parts.append("-" * width)

    header = " | ".join(header_parts)
    sep = "-+-".join(sep_parts)

    lines = [header, sep]

    for row in rows_sorted:
        parts = []
        for _title, key, fmt in COLUMNS:
            val = row.get(key, 0)
            try:
                parts.append(f"{val:{fmt}}")
            except (ValueError, TypeError):
                width = max(10, int("".join(c for c in fmt if c.isdigit()) or "10"))
                parts.append(f"{str(val):>{width}s}")
        lines.append(" | ".join(parts))

    table = "\n".join(lines)
    print("\n=== COMPARATIVE EVALUATION ===\n")
    print(table)
    print()
    return table


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_comparison(rows: List[Dict[str, Any]], table_str: str):
    """Save comparison as JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = RESULTS_DIR / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    # CSV
    csv_path = RESULTS_DIR / "comparison.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  Saved CSV:  {csv_path}")

    # Plain text table
    txt_path = RESULTS_DIR / "comparison.txt"
    with open(txt_path, "w") as f:
        f.write("COMPARATIVE EVALUATION TABLE\n")
        f.write("=" * 40 + "\n\n")
        f.write(table_str)
        f.write("\n")
    print(f"  Saved TXT:  {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[compare] Loading results from", RESULTS_DIR)
    rows = load_all_results()

    if not rows:
        print("[compare] No result files found. Ensure other tasks have completed.")
        print(f"  Expected files in: {RESULTS_DIR}")
        print("  Baseline format: baselines_btc.json, baselines_zec.json")
        print("  RL format: rl_ppo_btc.json, rl_dqn_btc.json, etc.")
        sys.exit(1)

    table_str = print_comparison_table(rows)
    save_comparison(rows, table_str)

    # Summary statistics
    human_rows = [r for r in rows if r["type"] == "Human"]
    rl_rows = [r for r in rows if r["type"] == "RL"]

    print(f"\n=== SUMMARY ===")
    print(f"  Human strategies: {len(human_rows)}")
    print(f"  RL strategies:    {len(rl_rows)}")

    if human_rows:
        best_human = max(human_rows, key=lambda r: r["oos_sharpe"])
        print(f"  Best human OOS Sharpe:  {best_human['strategy']} ({best_human['symbol']}) = {best_human['oos_sharpe']:.3f}")

    if rl_rows:
        best_rl = max(rl_rows, key=lambda r: r["oos_sharpe"])
        print(f"  Best RL OOS Sharpe:     {best_rl['strategy']} ({best_rl['symbol']}) = {best_rl['oos_sharpe']:.3f}")

    if human_rows and rl_rows:
        best_h_sharpe = max(r["oos_sharpe"] for r in human_rows)
        best_r_sharpe = max(r["oos_sharpe"] for r in rl_rows)
        diff = best_r_sharpe - best_h_sharpe
        print(f"  RL - Human gap:         {diff:+.3f}")

    print("\n[compare] Done.")


if __name__ == "__main__":
    main()
