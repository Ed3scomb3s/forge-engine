from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from forge_engine import create_session, WalkForwardConfig
from forge_engine.optuna_optimizer import WalkForwardSplitter

from evaluation.artifacts import run_strategy_with_artifacts
from examples.bb_reversion import BollingerBandStrategy
from examples.macd_momentum import MACDMomentumStrategy
from examples.rsi_reversal import RSIReversalStrategy
from examples.sma_cross import SMACrossStrategy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

COMMON_SESSION = {
    "start_date": "2020-01-01T00:00:00Z",
    "end_date": "2026-02-12T00:00:00Z",
    "starting_cash": 100000,
    "margin_mode": "cross",
    "warmup_candles": 50,
    "timeframe": "1h",
    "close_at_end": True,
}

ASSET_CONFIGS = {
    "baselines_btc.json": {
        "symbol": "BTCUSDT_PERP",
        "slippage_pct": 0.0005,
        "leverage": 2,
    },
    "baselines_zec.json": {
        "symbol": "ZECUSDT_PERP",
        "slippage_pct": 0.0015,
        "leverage": 5,
    },
}

STRATEGY_CTORS = {
    "SMACross": SMACrossStrategy,
    "RSIReversal": RSIReversalStrategy,
    "BollingerBand": BollingerBandStrategy,
    "MACDMomentum": MACDMomentumStrategy,
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
        full_start=_parse_utc_z(COMMON_SESSION["start_date"]),
        full_end=_parse_utc_z(COMMON_SESSION["end_date"]),
        holdout_ratio=0.15,
        timeframe_minutes=60,
    )
    if holdout is None:
        raise RuntimeError("Expected holdout period for baseline backfill")
    return holdout


def _trace_holdout(session_kwargs: dict, ctor, params: dict) -> dict:
    session = create_session(enable_visual=False, **session_kwargs)
    strategy = ctor(**params)
    return run_strategy_with_artifacts(session, strategy)


def backfill_file(filename: str) -> None:
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"skip {filename}: file not found")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    asset_cfg = ASSET_CONFIGS[filename]
    holdout_start, holdout_end = _holdout_period()

    changed = False
    for strat_name, strat_payload in (data.get("strategies") or {}).items():
        if not isinstance(strat_payload, dict) or "best_params" not in strat_payload:
            continue
        ctor = STRATEGY_CTORS.get(strat_name)
        if ctor is None:
            continue

        holdout_metrics = strat_payload.get("holdout_metrics") or {}
        if isinstance(holdout_metrics, dict) and isinstance(holdout_metrics.get("equities"), list):
            continue

        trace = _trace_holdout(
            {
                **COMMON_SESSION,
                **asset_cfg,
                "start_date": holdout_start.isoformat().replace("+00:00", "Z"),
                "end_date": holdout_end.isoformat().replace("+00:00", "Z"),
            },
            ctor,
            strat_payload["best_params"],
        )

        metrics = dict(trace.get("metrics") or {})
        metrics["equities"] = trace.get("equities", [])
        metrics["equity_times"] = trace.get("equity_times", [])
        metrics["returns"] = trace.get("returns", [])
        metrics["events"] = trace.get("events", [])
        strat_payload["holdout_metrics"] = metrics
        changed = True
        print(f"backfilled {filename}:{strat_name}")

    if changed:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"updated {path}")
    else:
        print(f"no changes for {path}")


def main() -> None:
    backfill_file("baselines_btc.json")
    backfill_file("baselines_zec.json")


if __name__ == "__main__":
    main()
