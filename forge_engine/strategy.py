from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .engine import get_warmup_candles, step_session, step_session_single_pass
from .indicators import Indicator, register_indicators, clear_indicators as _clear_indicators
from .trading import (
    cancel_order as _cancel_order,
    close_order as _close_order,
    create_order as _create_order,
    clear_engine as _clear_engine,
)
from .metrics import compute_metrics
from .indicators import bootstrap_indicators_for_session as _bootstrap_inds


class Strategy:
    """Base class for candle-driven strategies.

    Subclass and implement:
      - indicators(self) -> Iterable[Indicator]: optional, to auto-register
      - on_warmup(self, warmups: List[dict], meta: Dict[str, object]) -> None: optional
      - on_candle(self, candle: dict, state: dict, events: list) -> None: required
    """

    def __init__(self) -> None:
        self.session = None

    # --- Customization points ---
    def indicators(self) -> Iterable[Indicator]:
        return []

    def on_warmup(self, warmups: List[dict], meta: Dict[str, object]) -> None:  # noqa: D401
        return None

    def on_candle(self, candle: dict, state: dict, events: list) -> None:  # noqa: D401
        raise NotImplementedError

    # --- Helpers ---
    def attach(self, session) -> None:
        self.session = session

    def create_order(
        self,
        side: str,
        price: float,
        margin_pct: float,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
    ) -> Dict[str, object]:
        if self.session is None:
            return {"status": "rejected", "reason": "Strategy not attached to a session"}
        return _create_order(self.session, side=side, price=price, margin_pct=margin_pct, tp=tp, sl=sl)

    def close_order(self, price: float) -> Dict[str, object]:
        if self.session is None:
            return {"status": "rejected", "reason": "Strategy not attached to a session"}
        return _close_order(self.session, price=price)

    def cancel_order(self, order_id) -> Dict[str, object]:
        if self.session is None:
            return {"status": "rejected", "reason": "Strategy not attached to a session"}
        return _cancel_order(self.session, order_id)


class VectorStrategy(Strategy):
    """Vectorized strategy for fast optimizer backtesting.

    Instead of ``on_candle()`` (called per candle), implement ``signals()``
    which receives **numpy arrays** of close prices and indicator values and
    returns a signal array for the entire period in one call.

    The optimizer detects ``VectorStrategy`` subclasses and uses a Rust-native
    backtest loop — no per-candle Python↔Rust FFI overhead.

    Signal codes (class constants):
        HOLD  = 0 — do nothing
        LONG  = 1 — open long (ignored if already in a position)
        SHORT = 2 — open short (ignored if already in a position)
        CLOSE = 3 — close current position
    """

    HOLD: int = 0
    LONG: int = 1
    SHORT: int = 2
    CLOSE: int = 3

    def __init__(self) -> None:
        super().__init__()
        # Buffers for the on_candle compatibility fallback
        self._close_buf: List[float] = []
        self._ind_bufs: Dict[str, List[float]] = {}

    def signals(self, close: np.ndarray, indicators: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a signal array for the full candle range.

        Args:
            close: 1-D float64 numpy array of close prices.
            indicators: dict mapping indicator label (e.g. ``"SMA(20)[close]"``)
                        to a 1-D float64 numpy array. NaN where the indicator
                        has not warmed up yet.
        Returns:
            1-D int8 numpy array, same length as *close*, using the signal
            constants ``HOLD``, ``LONG``, ``SHORT``, ``CLOSE``.
        """
        raise NotImplementedError

    def signal_params(self) -> Dict[str, Any]:
        """Fixed trading parameters applied to every trade by the Rust engine.

        Returns:
            dict with keys:
                margin_pct  — fraction of equity per trade (0, 1]
                sl_pct      — stop-loss distance as fraction of entry price (0 = disabled)
                tp_pct      — take-profit distance as fraction of entry price (0 = disabled)
        """
        return {"margin_pct": 0.1, "sl_pct": 0.0, "tp_pct": 0.0}

    # ------------------------------------------------------------------
    # on_candle compatibility fallback
    # ------------------------------------------------------------------
    # This is NOT the fast path — it is called by run_strategy(), the
    # chart generator, and example __main__ blocks.  The optimizer
    # bypasses this entirely via _run_vectorized_backtest.
    # ------------------------------------------------------------------

    def on_candle(self, candle: dict, state: dict, events: list) -> None:  # noqa: D401
        if not candle or not state:
            return

        close_px = float(candle.get("close", 0))
        self._close_buf.append(close_px)

        # Accumulate indicator values from engine state
        inds = state.get("indicators") or {}
        for k, v in inds.items():
            if k not in self._ind_bufs:
                # Back-fill with NaN for candles before this indicator appeared
                self._ind_bufs[k] = [float("nan")] * (len(self._close_buf) - 1)
            self._ind_bufs[k].append(float(v) if v is not None else float("nan"))

        # Pad any indicator buffers that didn't emit a value this candle
        for k, buf in self._ind_bufs.items():
            if len(buf) < len(self._close_buf):
                buf.append(float("nan"))

        # Need at least 2 data points for cross-detection to work
        if len(self._close_buf) < 2:
            return

        # Build numpy arrays and call signals()
        close_arr = np.array(self._close_buf, dtype=np.float64)
        ind_dict = {k: np.array(v, dtype=np.float64) for k, v in self._ind_bufs.items()}

        try:
            sig_arr = self.signals(close_arr, ind_dict)
        except Exception:
            return

        current_signal = int(sig_arr[-1])
        pos = state.get("position")
        params = self.signal_params()
        margin_pct = float(params.get("margin_pct", 0.1))
        sl_pct = float(params.get("sl_pct", 0.0))
        tp_pct = float(params.get("tp_pct", 0.0))

        if current_signal == self.LONG and not pos:
            sl = close_px * (1.0 - sl_pct) if sl_pct > 0 else None
            tp = close_px * (1.0 + tp_pct) if tp_pct > 0 else None
            self.create_order("LONG", price=close_px, margin_pct=margin_pct, sl=sl, tp=tp)

        elif current_signal == self.SHORT and not pos:
            sl = close_px * (1.0 + sl_pct) if sl_pct > 0 else None
            tp = close_px * (1.0 - tp_pct) if tp_pct > 0 else None
            self.create_order("SHORT", price=close_px, margin_pct=margin_pct, sl=sl, tp=tp)

        elif current_signal == self.CLOSE and pos:
            self.close_order(price=close_px)


def run_strategy(session, strategy: Strategy, auto_register_indicators: bool = True) -> Optional[dict]:
    """Run a strategy over the session.

    Fast-path: single-pass warmup + session iteration (uses step_session_single_pass),
    with automatic fallback to legacy two-pass (get_warmup_candles + step_session).
    """
    strategy.attach(session)

    if auto_register_indicators:
        try:
            inds = list(strategy.indicators() or [])
            if inds:
                register_indicators(session, inds)
        except Exception:
            # Ignore indicator registration errors to keep runner robust
            pass

    meta = {
        "symbol": getattr(session, "symbol", None),
        "timeframe": getattr(session, "timeframe", None),
        "timeframe_minutes": getattr(session, "timeframe_minutes", None),
        "warmup_candles": getattr(session, "warmup_candles", 0),
    }

    last_state: Optional[dict] = None
    _equities: List[float] = []
    _times: List[str] = []
    _flat_events: List[Dict[str, object]] = []

    used_single_pass = False

    # --- Attempt single-pass path ---
    try:
        gen = step_session_single_pass(session)
        first = next(gen, None)
        warmups = []
        if isinstance(first, dict) and "warmups" in first:
            try:
                warmups = list(first.get("warmups") or [])
            except Exception:
                warmups = []
        try:
            strategy.on_warmup(warmups, meta)
        except Exception:
            # Strategy warmup errors should not crash the runner
            pass
        # Bootstrap indicators to avoid a second CSV scan
        try:
            _bootstrap_inds(session, warmups, meta)
        except Exception:
            pass

        used_single_pass = True

        for tick in gen:
            try:
                strategy.on_candle(tick.get("candle"), tick.get("state"), tick.get("events"))
            except Exception:
                pass
            last_state = tick.get("state") or last_state
            st = tick.get("state") or {}
            cd = tick.get("candle") or {}
            eq = st.get("equity")
            if eq is not None:
                try:
                    _equities.append(float(eq))
                    _times.append(str(cd.get("open_time")))
                except Exception:
                    pass
            evs = tick.get("events") or []
            if evs:
                _flat_events.extend(list(evs))
    except Exception:
        used_single_pass = False

    # --- Fallback to legacy two-pass if single-pass failed ---
    if not used_single_pass:
        warmups = []
        try:
            warmups = get_warmup_candles(session)
        except Exception:
            pass

        try:
            strategy.on_warmup(warmups, meta)
        except Exception:
            # Strategy warmup errors should not crash the runner
            pass

        for tick in step_session(session):
            try:
                strategy.on_candle(tick.get("candle"), tick.get("state"), tick.get("events"))
            except Exception:
                pass
            last_state = tick.get("state") or last_state
            st = tick.get("state") or {}
            cd = tick.get("candle") or {}
            eq = st.get("equity")
            if eq is not None:
                try:
                    _equities.append(float(eq))
                    _times.append(str(cd.get("open_time")))
                except Exception:
                    pass
            evs = tick.get("events") or []
            if evs:
                _flat_events.extend(list(evs))

    if last_state is not None:
        try:
            metrics = compute_metrics(session, _times, _equities, _flat_events, last_state)
            last_state["metrics"] = metrics
            try:
                # push metrics to WebUI
                from .visual import publish_metrics
                publish_metrics(session, metrics)
            except Exception:
                pass
        except Exception:
            pass

    # Clean up engine and indicators to prevent memory leaks
    try:
        _clear_engine(session)
    except Exception:
        pass
    try:
        _clear_indicators(session)
    except Exception:
        pass

    return last_state

