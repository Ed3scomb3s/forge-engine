"""Agent-to-Strategy adapter.

Wraps a trained RL model into a forge_engine.Strategy subclass so it can be
run through the exact same run_strategy() / metrics / visualization pipeline
as human-designed strategies. This enables fair comparison for the thesis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..strategy import Strategy
from ..indicators import Indicator
from .observations import (
    ObsFeature,
    _ObsHistory,
    resolve_features,
    build_observation_space,
)
from .actions import ActionSpace, DiscreteActions


class AgentStrategy(Strategy):
    """Run a trained RL model as a Strategy.

    This adapter lets you use a trained stable-baselines3 model (or any model
    with a predict(obs) -> action interface) in the same run_strategy() pipeline
    as human-designed strategies like SMACrossStrategy.

    The strategy:
      1. Builds observations from candle + state using the same features as training
      2. Feeds the observation to model.predict()
      3. Translates the predicted action into trading commands

    Parameters:
        model: Trained SB3 model (or anything with .predict(obs, deterministic=bool))
        observations: Same observation specs used during training
        actions: Same ActionSpace used during training
        indicators_list: List of Indicator instances (same as training)
        deterministic: Use deterministic predictions (default True for evaluation)
    """

    def __init__(
        self,
        model: Any,
        observations: Optional[Sequence[Any]] = None,
        actions: Optional[ActionSpace] = None,
        indicators_list: Optional[List[Indicator]] = None,
        deterministic: bool = True,
    ):
        super().__init__()
        self._model = model
        self._obs_specs = observations or ["ohlcv", "position_info"]
        self._obs_features: List[ObsFeature] = resolve_features(self._obs_specs)
        self._action_space_def: ActionSpace = actions or DiscreteActions()
        self._indicators_list: List[Indicator] = indicators_list or []
        self._deterministic = deterministic

        # Build observation space for clipping
        self._obs_size, self._obs_low, self._obs_high = build_observation_space(
            self._obs_features
        )

        # Internal state
        self._history = _ObsHistory()
        self._auto_indicators: List[Indicator] = []

    def indicators(self):
        """Return indicators needed for observations."""
        self._auto_indicators = self._infer_indicators()
        return list(self._indicators_list) + list(self._auto_indicators)

    def _infer_indicators(self) -> List[Indicator]:
        """Auto-create engine Indicator objects from observation specs."""
        from ..indicators import SMA, EMA, RSI, ATR, BollingerBands

        inds: List[Indicator] = []
        seen: set = set()

        for spec in self._obs_specs:
            if not isinstance(spec, str):
                continue
            s = spec.strip().lower()

            if s.startswith("rsi_") and s not in seen:
                period = int(s.split("_")[1])
                inds.append(RSI(period=period))
                seen.add(s)
            elif (
                s.startswith("sma_") and not s.startswith("sma_ratio") and s not in seen
            ):
                period = int(s.split("_")[1])
                inds.append(SMA(period=period))
                seen.add(s)
            elif s.startswith("ema_") and s not in seen:
                period = int(s.split("_")[1])
                inds.append(EMA(period=period))
                seen.add(s)
            elif s.startswith("atr_") and s not in seen:
                period = int(s.split("_")[1])
                inds.append(ATR(period=period))
                seen.add(s)
            elif s.startswith("bb_context_") and s not in seen:
                parts = s.split("_")
                if len(parts) == 4:
                    period = int(parts[2])
                    multiplier = int(parts[3])
                    inds.append(BollingerBands(period=period, multiplier=multiplier))
                    seen.add(s)
            elif s.startswith("sma_ratio_"):
                parts = s.split("_")
                if len(parts) == 4:
                    fast, slow = int(parts[2]), int(parts[3])
                    key_f = f"sma_{fast}"
                    key_s = f"sma_{slow}"
                    if key_f not in seen:
                        inds.append(SMA(period=fast))
                        seen.add(key_f)
                    if key_s not in seen:
                        inds.append(SMA(period=slow))
                        seen.add(key_s)

        return inds

    def on_warmup(self, warmups, meta):
        """Reset observation history on warmup."""
        self._history.reset()
        for f in self._obs_features:
            f.reset()

    def on_candle(self, candle, state, events):
        """Observe -> predict -> act."""
        if not candle or not state:
            return

        # Update history
        self._history.push(candle, state)

        # Build observation vector
        parts = []
        for f in self._obs_features:
            try:
                val = f.observe(candle, state, self._history)
                parts.append(val)
            except Exception:
                parts.append(np.zeros(f.size(), dtype=np.float32))

        obs = (
            np.concatenate(parts).astype(np.float32)
            if parts
            else np.array([], dtype=np.float32)
        )
        obs = np.clip(obs, self._obs_low, self._obs_high)

        # Predict action
        action, _ = self._model.predict(obs, deterministic=self._deterministic)

        # Execute action
        self._action_space_def.translate(action, candle, state, self)
