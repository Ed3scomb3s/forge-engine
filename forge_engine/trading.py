"""Thin wrappers around the compiled Rust TradingEngine (forge_engine._rust_core).

Phase 7: the pure-Python TradingEngine has been deleted.  All state-machine
logic, rounding, order execution, trigger resolution and indicator updates now
live in compiled Rust.  This module exposes the same public API surface so that
existing callers (step_session, strategy, UI) work unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from uuid import UUID

from forge_engine._rust_core import RustTradingEngine

# ---------------------------------------------------------------------------
# Module-level registry – one engine per session, keyed by session.id
# ---------------------------------------------------------------------------

_ENGINES: Dict[UUID, RustTradingEngine] = {}


def _load_intra_candles(session) -> list:
    """Read all base-timeframe candles for the session and return as
    ``[(unix_ts_i64, low_f64, high_f64), ...]`` for Rust binary-search
    trigger resolution.
    """
    from .engine import get_intra_candles, _parse_iso8601_utc, _isoformat_z

    start_iso = _isoformat_z(session.start_date)
    end_iso = _isoformat_z(session.end_date)

    raw = get_intra_candles(session, start_iso, end_iso)
    tuples = []
    for c in raw:
        ts_str = c.get("open_time", "")
        if not ts_str:
            continue
        dt = _parse_iso8601_utc(ts_str)
        tuples.append((int(dt.timestamp()), float(c["low"]), float(c["high"])))
    return tuples


def ensure_engine(session) -> RustTradingEngine:
    """Return (or lazily create) the Rust engine for *session*."""
    eng = _ENGINES.get(session.id)
    if eng is not None:
        return eng

    eng = RustTradingEngine(
        symbol=session.symbol,
        margin_mode=session.margin_mode,
        leverage=float(session.leverage),
        starting_cash=float(session.starting_cash),
        slippage_pct=float(getattr(session, "slippage_pct", 0.0)),
        timeframe_minutes=int(session.timeframe_minutes),
        funding_data=getattr(session, "funding_data", None),
    )

    # Pre-load 1 m candles for intra-candle trigger resolution
    if int(session.timeframe_minutes) > 1:
        try:
            intra = _load_intra_candles(session)
            if intra:
                eng.load_intra_candles(intra)
        except Exception:
            pass  # graceful degradation – priority fallback

    _ENGINES[session.id] = eng
    return eng


def clear_engine(session) -> None:
    """Remove engine from registry to free memory after backtest completes."""
    _ENGINES.pop(session.id, None)


# ---------------------------------------------------------------------------
# Public helpers that the rest of the codebase calls
# ---------------------------------------------------------------------------

def get_state(session) -> Dict[str, object]:
    return ensure_engine(session).snapshot()


def create_order(
    session,
    side: str,
    price: float,
    margin_pct: float,
    tp: Optional[float] = None,
    sl: Optional[float] = None,
) -> Dict[str, object]:
    return ensure_engine(session).create_order(side.upper(), price, margin_pct, tp, sl)


def close_order(session, price: float) -> Dict[str, object]:
    return ensure_engine(session).close_order(price)


def cancel_order(session, order_id) -> Dict[str, object]:
    return ensure_engine(session).cancel_order(str(order_id))


def on_candle(
    session, candle: Dict[str, object], index: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    return ensure_engine(session).on_candle(candle, index)


def compute_open_capacity(session, side: str, price: float) -> Dict[str, float]:
    return ensure_engine(session).compute_open_capacity(side.upper(), price)
