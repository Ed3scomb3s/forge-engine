"""Observation space builders for RL environments.

Each observation feature knows how to:
  1. Declare its size (number of floats it contributes)
  2. Compute its value from (candle, state, history)
  3. Define its bounds for the gymnasium Box space

Users compose observations by listing feature names or ObsFeature instances.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ObsFeature(ABC):
    """A single observation feature (may produce 1 or more floats)."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def size(self) -> int:
        """Number of floats this feature contributes to the observation vector."""
        ...

    @abstractmethod
    def low(self) -> np.ndarray:
        """Lower bounds (shape = (size,))."""
        ...

    @abstractmethod
    def high(self) -> np.ndarray:
        """Upper bounds (shape = (size,))."""
        ...

    def reset(self) -> None:
        """Called on env.reset() to clear any internal state."""
        pass

    @abstractmethod
    def observe(self, candle: dict, state: dict, history: "_ObsHistory") -> np.ndarray:
        """Return observation values (shape = (size,))."""
        ...


# ---------------------------------------------------------------------------
# History buffer shared across all features
# ---------------------------------------------------------------------------


class _ObsHistory:
    """Rolling buffer of recent candle/state data for lookback features."""

    def __init__(self, maxlen: int = 200):
        self.candles: deque[dict] = deque(maxlen=maxlen)
        self.states: deque[dict] = deque(maxlen=maxlen)
        self._initial_equity: Optional[float] = None

    def push(self, candle: dict, state: dict) -> None:
        self.candles.append(candle)
        self.states.append(state)
        if self._initial_equity is None:
            eq = state.get("equity")
            if eq is not None:
                self._initial_equity = float(eq)

    def reset(self) -> None:
        self.candles.clear()
        self.states.clear()
        self._initial_equity = None

    @property
    def initial_equity(self) -> float:
        return self._initial_equity or 1.0


# ---------------------------------------------------------------------------
# Concrete features
# ---------------------------------------------------------------------------


@dataclass
class OHLCV(ObsFeature):
    """Normalized OHLCV using log-returns relative to the current close.

    Produces 5 values: [log(open/close), log(high/close), log(low/close), 0.0, norm_volume]
    This is price-level invariant — the agent sees relative structure, not absolute price.
    """

    def name(self) -> str:
        return "ohlcv"

    def size(self) -> int:
        return 5

    def low(self) -> np.ndarray:
        return np.array([-1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        c = float(candle.get("close", 1.0) or 1.0)
        o = float(candle.get("open", c) or c)
        h = float(candle.get("high", c) or c)
        lo = float(candle.get("low", c) or c)
        v = float(candle.get("volume", 0.0) or 0.0)

        # Log-returns relative to close (safe against zero)
        eps = 1e-10
        log_oc = math.log(max(o, eps) / max(c, eps))
        log_hc = math.log(max(h, eps) / max(c, eps))
        log_lc = math.log(max(lo, eps) / max(c, eps))

        # Normalize volume: divide by rolling mean volume
        vol_sum = 0.0
        vol_count = 0
        for cd in history.candles:
            vv = float(cd.get("volume", 0.0) or 0.0)
            if vv > 0:
                vol_sum += vv
                vol_count += 1
        mean_vol = (vol_sum / vol_count) if vol_count > 0 else max(v, 1.0)
        norm_v = v / max(mean_vol, 1e-10)

        return np.array([log_oc, log_hc, log_lc, 0.0, norm_v], dtype=np.float32)


@dataclass
class Returns(ObsFeature):
    """Log-return of close price: log(close_t / close_{t-1}).

    Produces 1 value.
    """

    def name(self) -> str:
        return "returns"

    def size(self) -> int:
        return 1

    def low(self) -> np.ndarray:
        return np.array([-1.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        if len(history.candles) < 2:
            return np.array([0.0], dtype=np.float32)
        prev_c = float(history.candles[-2].get("close", 1.0) or 1.0)
        curr_c = float(candle.get("close", 1.0) or 1.0)
        eps = 1e-10
        lr = math.log(max(curr_c, eps) / max(prev_c, eps))
        return np.array([np.clip(lr, -1.0, 1.0)], dtype=np.float32)


@dataclass
class PositionInfo(ObsFeature):
    """Position state: [has_position, side_encoding, unrealized_pnl_pct, margin_usage_pct].

    - has_position: 0.0 or 1.0
    - side_encoding: -1.0 (short), 0.0 (flat), 1.0 (long)
    - unrealized_pnl_pct: upnl / initial_equity (clipped to [-1, 1])
    - margin_usage_pct: used_margin / equity (clipped to [0, 1])
    """

    def name(self) -> str:
        return "position_info"

    def size(self) -> int:
        return 4

    def low(self) -> np.ndarray:
        return np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        pos = state.get("position")
        has_pos = 1.0 if pos else 0.0

        side = 0.0
        if pos:
            s = pos.get("side", "")
            if s == "LONG":
                side = 1.0
            elif s == "SHORT":
                side = -1.0

        upnl = float(state.get("unrealized_pnl", 0.0) or 0.0)
        init_eq = history.initial_equity
        upnl_pct = np.clip(upnl / max(init_eq, 1e-10), -1.0, 1.0)

        eq = float(state.get("equity", 1.0) or 1.0)
        used_im = float(state.get("used_initial_margin", 0.0) or 0.0)
        margin_pct = np.clip(used_im / max(eq, 1e-10), 0.0, 1.0)

        return np.array([has_pos, side, upnl_pct, margin_pct], dtype=np.float32)


@dataclass
class EquityCurve(ObsFeature):
    """Equity relative to initial: equity / initial_equity - 1.0.

    Produces 1 value clipped to [-1, 5].
    """

    def name(self) -> str:
        return "equity_curve"

    def size(self) -> int:
        return 1

    def low(self) -> np.ndarray:
        return np.array([-1.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([5.0], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        eq = float(state.get("equity", 1.0) or 1.0)
        init_eq = history.initial_equity
        rel = eq / max(init_eq, 1e-10) - 1.0
        return np.array([np.clip(rel, -1.0, 5.0)], dtype=np.float32)


@dataclass
class Drawdown(ObsFeature):
    """Current drawdown from equity high-water mark.

    Produces 1 value in [0, 1] (0 = at peak, 1 = total loss).
    """

    _hwm: float = field(default=0.0, repr=False)

    def name(self) -> str:
        return "drawdown"

    def size(self) -> int:
        return 1

    def low(self) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def reset(self) -> None:
        self._hwm = 0.0

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        eq = float(state.get("equity", 1.0) or 1.0)
        if eq > self._hwm:
            self._hwm = eq
        dd = 0.0 if self._hwm <= 0 else (self._hwm - eq) / self._hwm
        return np.array([np.clip(dd, 0.0, 1.0)], dtype=np.float32)


@dataclass
class IndicatorObs(ObsFeature):
    """Wraps any engine indicator value from state['indicators'].

    The user specifies the indicator key (e.g., 'RSI(14)[close]') and the
    normalization bounds. The raw value is linearly scaled to [0, 1].

    For convenience, shorthand names are expanded:
      - 'rsi_14'     -> RSI(14)[close]
      - 'sma_20'     -> SMA(20)[close]
      - 'ema_50'     -> EMA(50)[close]
      - 'atr_14'     -> ATR(14)
    """

    key: str = ""
    raw_low: float = 0.0
    raw_high: float = 100.0
    label: Optional[str] = None

    def __post_init__(self):
        self._resolved_key = self._expand_shorthand(self.key)

    @staticmethod
    def _expand_shorthand(key: str) -> str:
        k = key.strip()
        low = k.lower()
        # rsi_14 -> RSI(14)[close]
        if low.startswith("rsi_"):
            period = low.split("_", 1)[1]
            return f"RSI({period})[close]"
        # sma_20 -> SMA(20)[close]
        if low.startswith("sma_"):
            period = low.split("_", 1)[1]
            return f"SMA({period})[close]"
        # ema_50 -> EMA(50)[close]
        if low.startswith("ema_"):
            period = low.split("_", 1)[1]
            return f"EMA({period})[close]"
        # atr_14 -> ATR(14)
        if low.startswith("atr_"):
            period = low.split("_", 1)[1]
            return f"ATR({period})"
        return k

    def name(self) -> str:
        return self.label or f"ind:{self.key}"

    def size(self) -> int:
        return 1

    def low(self) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        inds = state.get("indicators") or {}
        raw = inds.get(self._resolved_key)
        if raw is None:
            return np.array([0.5], dtype=np.float32)  # neutral default
        val = float(raw)
        # Linear scale to [0, 1]
        span = self.raw_high - self.raw_low
        if span <= 0:
            normed = 0.5
        else:
            normed = (val - self.raw_low) / span
        return np.array([np.clip(normed, 0.0, 1.0)], dtype=np.float32)


@dataclass
class SMA_Ratio(ObsFeature):
    """Ratio of two SMA indicators: SMA(fast) / SMA(slow) - 1.0.

    Requires the corresponding SMA indicators to be registered on the strategy.
    Produces 1 value clipped to [-0.5, 0.5].
    """

    fast: int = 20
    slow: int = 50

    def name(self) -> str:
        return f"sma_ratio_{self.fast}_{self.slow}"

    def size(self) -> int:
        return 1

    def low(self) -> np.ndarray:
        return np.array([-0.5], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([0.5], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        inds = state.get("indicators") or {}
        fast_key = f"SMA({self.fast})[close]"
        slow_key = f"SMA({self.slow})[close]"
        f_val = inds.get(fast_key)
        s_val = inds.get(slow_key)
        if f_val is None or s_val is None or float(s_val) == 0:
            return np.array([0.0], dtype=np.float32)
        ratio = float(f_val) / float(s_val) - 1.0
        return np.array([np.clip(ratio, -0.5, 0.5)], dtype=np.float32)


@dataclass
class VolumeProfile(ObsFeature):
    """Volume features: [taker_buy_ratio, volume_change_ratio].

    - taker_buy_ratio: taker_buy_volume / total_volume (0.5 = neutral)
    - volume_change: current_volume / rolling_mean_volume - 1 (clipped)
    """

    def name(self) -> str:
        return "volume_profile"

    def size(self) -> int:
        return 2

    def low(self) -> np.ndarray:
        return np.array([0.0, -1.0], dtype=np.float32)

    def high(self) -> np.ndarray:
        return np.array([1.0, 5.0], dtype=np.float32)

    def observe(self, candle: dict, state: dict, history: _ObsHistory) -> np.ndarray:
        v = float(candle.get("volume", 0.0) or 0.0)
        tb = float(candle.get("taker_buy_base_asset_volume", 0.0) or 0.0)

        buy_ratio = tb / max(v, 1e-10) if v > 0 else 0.5

        vol_sum = 0.0
        vol_count = 0
        for cd in history.candles:
            vv = float(cd.get("volume", 0.0) or 0.0)
            if vv > 0:
                vol_sum += vv
                vol_count += 1
        mean_vol = (vol_sum / vol_count) if vol_count > 0 else max(v, 1.0)
        vol_change = v / max(mean_vol, 1e-10) - 1.0

        return np.array(
            [
                np.clip(buy_ratio, 0.0, 1.0),
                np.clip(vol_change, -1.0, 5.0),
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Registry: string name -> feature class + defaults
# ---------------------------------------------------------------------------

_FEATURE_REGISTRY: Dict[str, Type[ObsFeature]] = {
    "ohlcv": OHLCV,
    "returns": Returns,
    "position_info": PositionInfo,
    "equity_curve": EquityCurve,
    "drawdown": Drawdown,
    "volume_profile": VolumeProfile,
}


def resolve_features(specs: Sequence[Any]) -> List[ObsFeature]:
    """Convert a list of feature specs into ObsFeature instances.

    Each spec can be:
      - A string name from the registry (e.g., "ohlcv", "position_info")
      - A string like "rsi_14", "sma_20", "atr_14" (creates IndicatorObs)
      - A string like "sma_ratio_20_50" (creates SMA_Ratio)
      - An ObsFeature instance (used as-is)
    """
    features: List[ObsFeature] = []
    for spec in specs:
        if isinstance(spec, ObsFeature):
            features.append(spec)
            continue

        if not isinstance(spec, str):
            raise ValueError(
                f"Invalid observation spec: {spec!r}. Expected str or ObsFeature."
            )

        s = spec.strip().lower()

        # Check registry first
        if s in _FEATURE_REGISTRY:
            features.append(_FEATURE_REGISTRY[s]())
            continue

        # SMA ratio: "sma_ratio_20_50"
        if s.startswith("sma_ratio_"):
            parts = s.split("_")
            if len(parts) == 4:
                try:
                    fast = int(parts[2])
                    slow = int(parts[3])
                    features.append(SMA_Ratio(fast=fast, slow=slow))
                    continue
                except ValueError:
                    pass

        # Indicator shorthand: "rsi_14", "sma_20", "ema_50", "atr_14"
        if "_" in s:
            prefix, rest = s.split("_", 1)
            if prefix in ("rsi", "sma", "ema", "atr"):
                try:
                    period = int(rest)
                except ValueError:
                    raise ValueError(f"Invalid indicator period in '{spec}'")

                if prefix == "rsi":
                    features.append(IndicatorObs(key=spec, raw_low=0.0, raw_high=100.0))
                elif prefix in ("sma", "ema"):
                    # Price-level indicator: normalize relative to current close
                    # We'll use a wide range; the IndicatorObs will clip
                    features.append(
                        IndicatorObs(key=spec, raw_low=0.0, raw_high=200000.0)
                    )
                elif prefix == "atr":
                    features.append(
                        IndicatorObs(key=spec, raw_low=0.0, raw_high=10000.0)
                    )
                continue

        # Direct indicator key (e.g., "RSI(14)[close]")
        features.append(IndicatorObs(key=spec, raw_low=0.0, raw_high=100.0))

    return features


def build_observation_space(
    features: List[ObsFeature],
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Build combined observation space from features.

    Returns (total_size, low_bounds, high_bounds).
    """
    total = sum(f.size() for f in features)
    lows = (
        np.concatenate([f.low() for f in features])
        if features
        else np.array([], dtype=np.float32)
    )
    highs = (
        np.concatenate([f.high() for f in features])
        if features
        else np.array([], dtype=np.float32)
    )
    return total, lows, highs
