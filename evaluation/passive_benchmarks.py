from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from forge_engine import WalkForwardConfig, create_session
from forge_engine.optuna_optimizer import WalkForwardSplitter

from evaluation.artifacts import run_strategy_with_artifacts
from examples.buy_and_hold import BuyAndHoldLongStrategy


RESULTS_DIR = Path(__file__).resolve().parent / "results"

COMMON = {
    "start_date": "2020-01-01T00:00:00Z",
    "end_date": "2026-02-12T00:00:00Z",
    "starting_cash": 100000,
    "margin_mode": "cross",
    "warmup_candles": 50,
    "timeframe": "1h",
    "close_at_end": True,
}

ASSETS = {
    "BTCUSDT_PERP": {
        "slippage_pct": 0.0005,
        "leverage": 2,
    },
    "ZECUSDT_PERP": {
        "slippage_pct": 0.0015,
        "leverage": 5,
    },
}


def _parse_utc_z(value: str) -> datetime:
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _holdout_period():
    splitter = WalkForwardSplitter(WalkForwardConfig(
        n_splits=5,
        test_ratio=0.2,
        mode="anchored",
    ))
    _splits, holdout = splitter.generate_splits(
        full_start=_parse_utc_z(COMMON["start_date"]),
        full_end=_parse_utc_z(COMMON["end_date"]),
        holdout_ratio=0.15,
        timeframe_minutes=60,
    )
    if holdout is None:
        raise RuntimeError("Expected holdout period")
    return holdout


def run_passive_benchmark(symbol: str) -> dict:
    cfg = ASSETS[symbol]
    holdout_start, holdout_end = _holdout_period()
    session = create_session(
        symbol=symbol,
        start_date=holdout_start.isoformat().replace("+00:00", "Z"),
        end_date=holdout_end.isoformat().replace("+00:00", "Z"),
        starting_cash=COMMON["starting_cash"],
        leverage=cfg["leverage"],
        margin_mode=COMMON["margin_mode"],
        warmup_candles=COMMON["warmup_candles"],
        timeframe=COMMON["timeframe"],
        close_at_end=COMMON["close_at_end"],
        slippage_pct=cfg["slippage_pct"],
        enable_visual=False,
    )
    trace = run_strategy_with_artifacts(session, BuyAndHoldLongStrategy(margin_pct=0.95))
    metrics = dict(trace.get("metrics") or {})
    metrics["equities"] = trace.get("equities", [])
    metrics["equity_times"] = trace.get("equity_times", [])
    metrics["returns"] = trace.get("returns", [])
    metrics["events"] = trace.get("events", [])
    return {
        "symbol": symbol,
        "holdout_period": [
            holdout_start.isoformat().replace("+00:00", "Z"),
            holdout_end.isoformat().replace("+00:00", "Z"),
        ],
        "leverage": cfg["leverage"],
        "strategy": "BuyAndHoldLong",
        "metrics": metrics,
    }


def main() -> None:
    results = {symbol: run_passive_benchmark(symbol) for symbol in ASSETS}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "passive_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved passive benchmarks to {out_path}")


if __name__ == "__main__":
    main()
