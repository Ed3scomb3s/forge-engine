"""Indicator framework with per-session registry.

Phase 7: indicator *computation* now happens inside the Rust engine during
``on_candle``.  The Python ``Indicator`` subclasses are kept as lightweight
configuration objects so that callers can still write::

    register_indicators(session, [SMA(20), RSI(14)])

Registration forwards to the Rust engine via ``eng.register_sma(...)`` etc.
``update_indicators_for_session`` simply reads the latest values from the
engine snapshot.  ``bootstrap_indicators_for_session`` feeds warmup candles
through the Rust engine to seed the indicators.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Iterable
from uuid import UUID


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Indicator:
    def name(self) -> str:
        raise NotImplementedError

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        return None

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete indicator configs (kept for public API / config objects)
# ---------------------------------------------------------------------------

@dataclass
class SMA(Indicator):
    period: int = 20
    source: str = "close"
    label: Optional[str] = None

    def __post_init__(self):
        self._buf: deque[float] = deque(maxlen=int(self.period))
        self._sum: float = 0.0

    def name(self) -> str:
        if self.label:
            return self.label
        return f"SMA({self.period})[{self.source}]"

    def _push(self, value: float) -> Optional[float]:
        if len(self._buf) == self._buf.maxlen:
            self._sum -= self._buf[0]
        self._buf.append(value)
        self._sum += value
        if len(self._buf) == self._buf.maxlen:
            return self._sum / float(self._buf.maxlen)
        return None

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        for c in warmups:
            try:
                v = float(c.get(self.source))
            except Exception:
                continue
            self._push(v)

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        try:
            v = float(candle.get(self.source))
        except Exception:
            return None
        out = self._push(v)
        if out is None:
            return None
        return {self.name(): round(out, 8)}


@dataclass
class EMA(Indicator):
    period: int = 20
    source: str = "close"
    label: Optional[str] = None

    def __post_init__(self):
        p = int(self.period)
        if p <= 0:
            p = 1
        self._period: int = p
        self._alpha: float = 2.0 / (p + 1.0)
        self._ema: Optional[float] = None
        self._seed_sum: float = 0.0
        self._seed_count: int = 0

    def name(self) -> str:
        if self.label:
            return self.label
        return f"EMA({self._period})[{self.source}]"

    def _push(self, value: float) -> Optional[float]:
        if self._ema is None:
            if self._seed_count < self._period:
                self._seed_sum += value
                self._seed_count += 1
                if self._seed_count == self._period:
                    self._ema = self._seed_sum / float(self._period)
                return None
            if self._period == 1:
                self._ema = value
                return self._ema
        else:
            self._ema = (value - self._ema) * self._alpha + self._ema
        return self._ema

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        for c in warmups:
            try:
                v = float(c.get(self.source))
            except Exception:
                continue
            self._push(v)

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        try:
            v = float(candle.get(self.source))
        except Exception:
            return None
        out = self._push(v)
        if out is None:
            return None
        return {self.name(): round(out, 8)}


@dataclass
class RSI(Indicator):
    period: int = 14
    source: str = "close"
    label: Optional[str] = None

    def __post_init__(self):
        p = int(self.period)
        if p <= 0:
            p = 1
        self._period: int = p
        self._prev: Optional[float] = None
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self._seed_g: float = 0.0
        self._seed_l: float = 0.0
        self._seed_count: int = 0

    def name(self) -> str:
        if self.label:
            return self.label
        return f"RSI({self._period})[{self.source}]"

    def _push(self, value: float) -> Optional[float]:
        if self._prev is None:
            self._prev = value
            return None
        delta = value - self._prev
        self._prev = value
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0

        if self._avg_gain is None or self._avg_loss is None:
            if self._seed_count < self._period:
                self._seed_g += gain
                self._seed_l += loss
                self._seed_count += 1
                if self._seed_count == self._period:
                    self._avg_gain = self._seed_g / float(self._period)
                    self._avg_loss = self._seed_l / float(self._period)
                return None
        else:
            self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / float(self._period)
            self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / float(self._period)

        if self._avg_gain is None or self._avg_loss is None:
            return None
        if self._avg_loss == 0:
            rsi = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        for c in warmups:
            try:
                v = float(c.get(self.source))
            except Exception:
                continue
            self._push(v)

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        try:
            v = float(candle.get(self.source))
        except Exception:
            return None
        rsi = self._push(v)
        if rsi is None:
            return None
        return {self.name(): round(rsi, 8)}


@dataclass
class ATR(Indicator):
    period: int = 14
    label: Optional[str] = None

    def __post_init__(self):
        p = int(self.period)
        if p <= 0:
            p = 1
        self._period: int = p
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None
        self._seed_sum: float = 0.0
        self._seed_count: int = 0

    def name(self) -> str:
        if self.label:
            return self.label
        return f"ATR({self._period})"

    def _tr(self, high: float, low: float, close: float) -> float:
        if self._prev_close is None:
            tr = float(high - low)
        else:
            pc = self._prev_close
            tr = max(high - low, abs(high - pc), abs(low - pc))
        self._prev_close = float(close)
        return float(tr)

    def _push(self, high: float, low: float, close: float) -> Optional[float]:
        tr = self._tr(high, low, close)
        if self._atr is None:
            if self._seed_count < self._period:
                self._seed_sum += tr
                self._seed_count += 1
                if self._seed_count == self._period:
                    self._atr = self._seed_sum / float(self._period)
                return None
        else:
            self._atr = (self._atr * (self._period - 1) + tr) / float(self._period)
        return self._atr

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        for c in warmups:
            try:
                h = float(c.get("high"))
                l = float(c.get("low"))
                cl = float(c.get("close"))
            except Exception:
                continue
            self._push(h, l, cl)

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        try:
            h = float(candle.get("high"))
            l = float(candle.get("low"))
            cl = float(candle.get("close"))
        except Exception:
            return None
        atr = self._push(h, l, cl)
        if atr is None:
            return None
        return {self.name(): round(atr, 8)}


@dataclass
class BollingerBands(Indicator):
    period: int = 20
    multiplier: float = 2.0
    source: str = "close"
    label: Optional[str] = None

    def __post_init__(self):
        p = int(self.period)
        if p <= 0:
            p = 1
        self._period: int = p
        self._buf: deque[float] = deque(maxlen=p)
        self._sum: float = 0.0
        self._sumsq: float = 0.0

    def _basename(self) -> str:
        if self.label:
            return self.label
        m = int(self.multiplier) if abs(self.multiplier - int(self.multiplier)) < 1e-12 else self.multiplier
        return f"BB({self._period},{m})[{self.source}]"

    def name(self) -> str:
        return self._basename()

    def _push(self, value: float) -> Optional[Dict[str, float]]:
        if len(self._buf) == self._buf.maxlen:
            old = self._buf[0]
            self._sum -= old
            self._sumsq -= old * old
        self._buf.append(value)
        self._sum += value
        self._sumsq += value * value
        if len(self._buf) < self._buf.maxlen:
            return None
        n = float(self._buf.maxlen)
        mean = self._sum / n
        var = max(0.0, (self._sumsq - (self._sum * self._sum) / n) / n)
        std = sqrt(var)
        upper = mean + self.multiplier * std
        lower = mean - self.multiplier * std
        base = self._basename()
        return {
            f"{base}.upper": round(upper, 8),
            f"{base}.mid": round(mean, 8),
            f"{base}.lower": round(lower, 8),
        }

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        for c in warmups:
            try:
                v = float(c.get(self.source))
            except Exception:
                continue
            self._push(v)

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        try:
            v = float(candle.get(self.source))
        except Exception:
            return None
        return self._push(v)


@dataclass
class MACD(Indicator):
    fast: int = 12
    slow: int = 26
    signal: int = 9
    source: str = "close"
    label: Optional[str] = None

    def __post_init__(self):
        f = max(1, int(self.fast))
        s = max(1, int(self.slow))
        g = max(1, int(self.signal))
        self._fast_p = f
        self._slow_p = s
        self._sig_p = g
        self._alpha_f = 2.0 / (f + 1.0)
        self._alpha_s = 2.0 / (s + 1.0)
        self._alpha_g = 2.0 / (g + 1.0)
        self._ema_f: Optional[float] = None
        self._ema_s: Optional[float] = None
        self._seed_f_sum: float = 0.0
        self._seed_s_sum: float = 0.0
        self._seed_f_count: int = 0
        self._seed_s_count: int = 0
        self._ema_sig: Optional[float] = None
        self._sig_seed_sum: float = 0.0
        self._sig_seed_count: int = 0

    def _basename(self) -> str:
        if self.label:
            return self.label
        return f"MACD({self._fast_p},{self._slow_p},{self._sig_p})[{self.source}]"

    def name(self) -> str:
        return self._basename()

    def _push_value(self, v: float) -> Optional[Dict[str, float]]:
        if self._ema_f is None:
            if self._seed_f_count < self._fast_p:
                self._seed_f_sum += v
                self._seed_f_count += 1
                if self._seed_f_count == self._fast_p:
                    self._ema_f = self._seed_f_sum / float(self._fast_p)
        else:
            self._ema_f = (v - self._ema_f) * self._alpha_f + self._ema_f

        if self._ema_s is None:
            if self._seed_s_count < self._slow_p:
                self._seed_s_sum += v
                self._seed_s_count += 1
                if self._seed_s_count == self._slow_p:
                    self._ema_s = self._seed_s_sum / float(self._slow_p)
        else:
            self._ema_s = (v - self._ema_s) * self._alpha_s + self._ema_s

        if self._ema_f is None or self._ema_s is None:
            return None

        macd_line = self._ema_f - self._ema_s

        if self._ema_sig is None:
            if self._sig_seed_count < self._sig_p:
                self._sig_seed_sum += macd_line
                self._sig_seed_count += 1
                if self._sig_seed_count == self._sig_p:
                    self._ema_sig = self._sig_seed_sum / float(self._sig_p)
            return None
        else:
            self._ema_sig = (macd_line - self._ema_sig) * self._alpha_g + self._ema_sig

        signal_val = self._ema_sig
        hist = macd_line - signal_val
        base = self._basename()
        return {
            f"{base}.line": round(macd_line, 8),
            f"{base}.signal": round(signal_val, 8),
            f"{base}.hist": round(hist, 8),
        }

    def bootstrap(self, warmups: List[dict], meta: Dict[str, object]) -> None:
        for c in warmups:
            try:
                v = float(c.get(self.source))
            except Exception:
                continue
            self._push_value(v)

    def on_candle(self, candle: dict, index: int) -> Optional[Dict[str, float]]:
        try:
            v = float(candle.get(self.source))
        except Exception:
            return None
        return self._push_value(v)


# ---------------------------------------------------------------------------
# Rust-engine registration helper
# ---------------------------------------------------------------------------

def _register_indicator_on_engine(eng, ind: Indicator) -> None:
    """Forward a Python Indicator config to the Rust engine's native impl."""
    label = ind.name()
    if isinstance(ind, SMA):
        eng.register_sma(label, ind.period, ind.source)
    elif isinstance(ind, EMA):
        eng.register_ema(label, ind._period, ind.source)
    elif isinstance(ind, RSI):
        eng.register_rsi(label, ind._period, ind.source)
    elif isinstance(ind, ATR):
        eng.register_atr(label, ind._period)
    elif isinstance(ind, BollingerBands):
        eng.register_bollinger_bands(label, ind._period, ind.multiplier, ind.source)
    elif isinstance(ind, MACD):
        eng.register_macd(label, ind._fast_p, ind._slow_p, ind._sig_p, ind.source)


# ---------------------------------------------------------------------------
# Per-session registry and update helpers
# ---------------------------------------------------------------------------

_REGISTRY: Dict[UUID, List[Indicator]] = {}
_BOOTSTRAPPED: Dict[UUID, bool] = {}
_LAST_VALUES: Dict[UUID, Dict[str, float]] = {}


def register_indicators(session, indicators: Iterable[Indicator]) -> None:
    indicator_list = list(indicators)
    _REGISTRY[session.id] = indicator_list
    _BOOTSTRAPPED[session.id] = False
    _LAST_VALUES.pop(session.id, None)

    # Also register on the Rust engine so on_candle computes them natively
    from .trading import ensure_engine
    try:
        eng = ensure_engine(session)
        for ind in indicator_list:
            _register_indicator_on_engine(eng, ind)
    except Exception:
        pass


def clear_indicators(session) -> None:
    """Clear all indicator state for a session to prevent memory leaks."""
    _REGISTRY.pop(session.id, None)
    _BOOTSTRAPPED.pop(session.id, None)
    _LAST_VALUES.pop(session.id, None)


def update_indicators_for_session(session, candle: dict, index: int) -> Dict[str, float]:
    """Return latest indicator values from the Rust engine snapshot."""
    from .trading import ensure_engine
    try:
        eng = ensure_engine(session)
        snap = eng.snapshot()
        out = snap.get("indicators", {})
        if out:
            _LAST_VALUES[session.id] = out
        return out
    except Exception:
        return _LAST_VALUES.get(session.id, {})


def bootstrap_indicators_for_session(
    session, warmups: List[dict], meta: Dict[str, object],
) -> None:
    """Feed warmup candles through the Rust engine to seed indicators."""
    from .trading import ensure_engine
    try:
        eng = ensure_engine(session)
        for candle in warmups:
            eng.on_candle(candle, -1)
    except Exception:
        pass
    _BOOTSTRAPPED[session.id] = True


__all__ = [
    "Indicator",
    "SMA",
    "EMA",
    "RSI",
    "ATR",
    "BollingerBands",
    "MACD",
    "register_indicators",
    "update_indicators_for_session",
    "bootstrap_indicators_for_session",
    "clear_indicators",
]
