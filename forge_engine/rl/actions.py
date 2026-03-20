"""Action space definitions for RL environments.

Each ActionSpace defines:
  1. The gymnasium space (discrete or continuous)
  2. How to translate an RL action into engine trading commands
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym


class ActionSpace(ABC):
    """Base class for action spaces."""

    @abstractmethod
    def gymnasium_space(self) -> gym.Space:
        """Return the gymnasium action space."""
        ...

    @abstractmethod
    def translate(self, action: Any, candle: dict, state: dict, strategy: Any) -> None:
        """Execute the action using strategy helpers (create_order, close_order, etc.)."""
        ...

    @abstractmethod
    def action_labels(self) -> List[str]:
        """Human-readable labels for each action (for logging/debugging)."""
        ...


# ---------------------------------------------------------------------------
# Discrete action spaces
# ---------------------------------------------------------------------------


@dataclass
class DiscreteActions(ActionSpace):
    """Configurable discrete action space.

    Default actions: ["hold", "open_long", "open_short", "close"]

    Supported action names:
      - "hold"       : do nothing
      - "open_long"  : open a long position at current close
      - "open_short" : open a short position at current close
      - "close"      : close current position at current close

    Parameters:
      actions: list of action name strings
      margin_pct: fraction of equity to use per trade (default 0.1 = 10%)
      sl_pct: stop-loss distance as fraction of entry price (0 = disabled)
      tp_pct: take-profit distance as fraction of entry price (0 = disabled)
    """

    actions: List[str] = field(
        default_factory=lambda: ["hold", "open_long", "open_short", "close"]
    )
    margin_pct: float = 0.1
    sl_pct: float = 0.0
    tp_pct: float = 0.0

    def gymnasium_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self.actions))

    def action_labels(self) -> List[str]:
        return list(self.actions)

    def translate(self, action: Any, candle: dict, state: dict, strategy: Any) -> None:
        if isinstance(action, np.ndarray):
            val = action.item() if action.size == 1 else action[0]
        elif isinstance(action, (list, tuple)):
            val = action[0]
        else:
            val = action

        idx = int(val)
        if idx < 0 or idx >= len(self.actions):
            return  # invalid -> hold

        name = self.actions[idx]
        close_px = float(candle.get("close", 0.0) or 0.0)
        pos = state.get("position")

        if name == "hold":
            return

        elif name == "open_long":
            if pos:
                return  # already in a position
            sl = close_px * (1.0 - self.sl_pct) if self.sl_pct > 0 else None
            tp = close_px * (1.0 + self.tp_pct) if self.tp_pct > 0 else None
            strategy.create_order(
                "LONG", price=close_px, margin_pct=self.margin_pct, sl=sl, tp=tp
            )

        elif name == "open_short":
            if pos:
                return
            sl = close_px * (1.0 + self.sl_pct) if self.sl_pct > 0 else None
            tp = close_px * (1.0 - self.tp_pct) if self.tp_pct > 0 else None
            strategy.create_order(
                "SHORT", price=close_px, margin_pct=self.margin_pct, sl=sl, tp=tp
            )

        elif name == "close":
            if not pos:
                return
            strategy.close_order(price=close_px)


@dataclass
class DiscreteActionsWithSizing(ActionSpace):
    """Discrete actions with multiple position sizes.

    Actions: hold, long_small, long_medium, long_large, short_small, short_medium, short_large, close

    Parameters:
      sizes: mapping of size label to margin_pct (e.g., {"small": 0.05, "medium": 0.1, "large": 0.2})
      sl_pct: stop-loss distance (0 = disabled)
      tp_pct: take-profit distance (0 = disabled)
    """

    sizes: Dict[str, float] = field(
        default_factory=lambda: {
            "small": 0.05,
            "medium": 0.1,
            "large": 0.2,
        }  # Small = 5% margin, Medium = 10%, Large = 20%
    )
    sl_pct: float = 0.0
    tp_pct: float = 0.0

    def __post_init__(self):
        self._actions: List[str] = ["hold"]
        self._margin_map: Dict[str, float] = {}
        for label, mpct in self.sizes.items():
            for side in ("long", "short"):
                name = f"{side}_{label}"
                self._actions.append(name)
                self._margin_map[name] = mpct
        self._actions.append("close")

    def gymnasium_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self._actions))

    def action_labels(self) -> List[str]:
        return list(self._actions)

    def translate(self, action: Any, candle: dict, state: dict, strategy: Any) -> None:
        if isinstance(action, np.ndarray):
            val = action.item() if action.size == 1 else action[0]
        elif isinstance(action, (list, tuple)):
            val = action[0]
        else:
            val = action

        idx = int(val)
        if idx < 0 or idx >= len(self._actions):
            return

        name = self._actions[idx]
        close_px = float(candle.get("close", 0.0) or 0.0)
        pos = state.get("position")

        if name == "hold":
            return
        elif name == "close":
            if pos:
                strategy.close_order(price=close_px)
            return

        # long_small, short_medium, etc.
        margin_pct = self._margin_map.get(name, 0.1)
        if pos:
            return  # already in position

        if name.startswith("long_"):
            sl = close_px * (1.0 - self.sl_pct) if self.sl_pct > 0 else None
            tp = close_px * (1.0 + self.tp_pct) if self.tp_pct > 0 else None
            strategy.create_order(
                "LONG", price=close_px, margin_pct=margin_pct, sl=sl, tp=tp
            )
        elif name.startswith("short_"):
            sl = close_px * (1.0 + self.sl_pct) if self.sl_pct > 0 else None
            tp = close_px * (1.0 - self.tp_pct) if self.tp_pct > 0 else None
            strategy.create_order(
                "SHORT", price=close_px, margin_pct=margin_pct, sl=sl, tp=tp
            )


# ---------------------------------------------------------------------------
# Continuous action space
# ---------------------------------------------------------------------------


@dataclass
class ContinuousActions(ActionSpace):
    """Continuous action space for PPO / SAC style agents.

    Action is a single float in [-1, 1]:
      - [-1, -threshold): open short with margin_pct = |action| * max_margin_pct
      - [-threshold, threshold]: hold (dead zone)
      - (threshold, 1]: open long with margin_pct = |action| * max_margin_pct

    If already in a position:
      - Action in opposite direction triggers close
      - Action in same direction is ignored (already positioned)

    Parameters:
      max_margin_pct: maximum margin percentage when |action| = 1.0
      threshold: dead zone around 0 where agent holds (default 0.1)
      sl_pct: stop-loss (0 = disabled)
      tp_pct: take-profit (0 = disabled)
    """

    max_margin_pct: float = 0.2
    threshold: float = 0.1
    sl_pct: float = 0.0
    tp_pct: float = 0.0

    def gymnasium_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action_labels(self) -> List[str]:
        return ["position_signal"]

    def translate(self, action: Any, candle: dict, state: dict, strategy: Any) -> None:
        if isinstance(action, np.ndarray):
            val = action.item() if action.size == 1 else action[0]
        elif isinstance(action, (list, tuple)):
            val = action[0]
        else:
            val = action

        signal = float(np.clip(val, -1.0, 1.0))

        close_px = float(candle.get("close", 0.0) or 0.0)
        pos = state.get("position")

        # Dead zone: do nothing (hold current state).
        # Positions exit via SL/TP or opposite-direction signal.
        if abs(signal) <= self.threshold:
            return

        if signal > self.threshold:
            # Want long
            if pos:
                if pos.get("side") == "SHORT":
                    strategy.close_order(price=close_px)
                return  # already long or just closed short
            margin = abs(signal) * self.max_margin_pct
            sl = close_px * (1.0 - self.sl_pct) if self.sl_pct > 0 else None
            tp = close_px * (1.0 + self.tp_pct) if self.tp_pct > 0 else None
            strategy.create_order(
                "LONG", price=close_px, margin_pct=margin, sl=sl, tp=tp
            )

        else:  # signal < -threshold
            # Want short
            if pos:
                if pos.get("side") == "LONG":
                    strategy.close_order(price=close_px)
                return  # already short or just closed long
            margin = abs(signal) * self.max_margin_pct
            sl = close_px * (1.0 + self.sl_pct) if self.sl_pct > 0 else None
            tp = close_px * (1.0 - self.tp_pct) if self.tp_pct > 0 else None
            strategy.create_order(
                "SHORT", price=close_px, margin_pct=margin, sl=sl, tp=tp
            )
