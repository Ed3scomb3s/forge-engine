"""Reward functions for RL environments.

Each reward function computes the scalar reward given the current and previous state.
Designed per the thesis literature review recommendations: risk-adjusted rewards
are preferred over raw PnL to encourage stable, risk-aware behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import math
import numpy as np


class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def name(self) -> str: ...

    def reset(self) -> None:
        """Called on env.reset() to clear internal state."""
        pass

    @abstractmethod
    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        """Compute reward for this step."""
        ...


# ---------------------------------------------------------------------------
# Concrete reward functions
# ---------------------------------------------------------------------------


@dataclass
class PnLReward(RewardFunction):
    """Simple PnL-based reward: change in equity scaled by initial equity.

    reward = (equity_t - equity_{t-1}) / initial_equity

    Simple but known to encourage excessive risk-taking.
    Included as a baseline — prefer risk-adjusted rewards.
    """

    _initial_equity: float = field(default=0.0, repr=False)

    def name(self) -> str:
        return "pnl"

    def reset(self) -> None:
        self._initial_equity = 0.0

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        eq_prev = float(prev_state.get("equity", 0.0) or 0.0)
        eq_curr = float(curr_state.get("equity", 0.0) or 0.0)

        if self._initial_equity == 0.0:
            self._initial_equity = eq_prev if eq_prev > 0 else 1.0

        return (eq_curr - eq_prev) / max(self._initial_equity, 1e-10)


@dataclass
class LogReturnReward(RewardFunction):
    """Log-return reward: log(equity_t / equity_{t-1}).

    More numerically stable than simple PnL for multiplicative returns.
    Still not risk-adjusted — use as alternative baseline.
    """

    def name(self) -> str:
        return "log_return"

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        eq_prev = float(prev_state.get("equity", 1.0) or 1.0)
        eq_curr = float(curr_state.get("equity", 1.0) or 1.0)
        eps = 1e-10
        return math.log(max(eq_curr, eps) / max(eq_prev, eps))


@dataclass
class DifferentialSharpeReward(RewardFunction):
    """Differential Sharpe ratio reward (Moody & Saffell, 2001).

    Updates a running estimate of Sharpe ratio and rewards the marginal
    improvement at each step. This directly optimizes risk-adjusted returns.

    The differential Sharpe is:
      dS_t = (B_{t-1} * delta_A_t - 0.5 * A_{t-1} * delta_B_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

    where A_t and B_t are exponential moving averages of returns and squared returns.

    Parameters:
      eta: decay rate for exponential moving averages (default 0.01)
      scale: output scaling factor (default 1.0)
    """

    eta: float = 0.01
    scale: float = 1.0
    _A: float = field(default=0.0, repr=False)  # EMA of returns
    _B: float = field(default=0.0, repr=False)  # EMA of squared returns

    def name(self) -> str:
        return "differential_sharpe"

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        eq_prev = float(prev_state.get("equity", 1.0) or 1.0)
        eq_curr = float(curr_state.get("equity", 1.0) or 1.0)

        if eq_prev <= 0:
            return 0.0

        r = eq_curr / eq_prev - 1.0  # simple return

        # Compute differential Sharpe before updating A, B
        delta_A = r - self._A
        delta_B = r * r - self._B

        denom = self._B - self._A * self._A
        if denom <= 1e-12:
            # Not enough variance yet — use simple return as fallback
            ds = r
        else:
            ds = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom**1.5)

        # Update running estimates
        self._A += self.eta * delta_A
        self._B += self.eta * delta_B

        return float(np.clip(ds * self.scale, -10.0, 10.0))


@dataclass
class RiskAdjustedReward(RewardFunction):
    """Risk-adjusted reward combining return and drawdown penalty.

    reward = return_pct - drawdown_penalty * max(drawdown_increase, 0)

    This penalizes the agent for increasing drawdown, encouraging smoother equity curves.

    Parameters:
      drawdown_penalty: multiplier for drawdown increase penalty (default 2.0)
    """

    drawdown_penalty: float = 2.0
    _hwm: float = field(default=0.0, repr=False)
    _prev_dd: float = field(default=0.0, repr=False)
    _initial_equity: float = field(default=0.0, repr=False)

    def name(self) -> str:
        return "risk_adjusted"

    def reset(self) -> None:
        self._hwm = 0.0
        self._prev_dd = 0.0
        self._initial_equity = 0.0

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        eq_prev = float(prev_state.get("equity", 1.0) or 1.0)
        eq_curr = float(curr_state.get("equity", 1.0) or 1.0)

        if self._initial_equity == 0.0:
            self._initial_equity = eq_prev if eq_prev > 0 else 1.0

        # Return component
        ret = (eq_curr - eq_prev) / max(self._initial_equity, 1e-10)

        # Drawdown component
        if eq_curr > self._hwm:
            self._hwm = eq_curr
        dd = 0.0 if self._hwm <= 0 else (self._hwm - eq_curr) / self._hwm
        dd_increase = max(0.0, dd - self._prev_dd)
        self._prev_dd = dd

        reward = ret - self.drawdown_penalty * dd_increase
        return float(np.clip(reward, -10.0, 10.0))


@dataclass
class SortinoReward(RewardFunction):
    """Online Sortino-inspired reward: penalizes downside deviation only.

    reward = return / max(downside_std, min_std)

    Uses exponential moving average of squared negative returns for the
    downside deviation estimate.

    Parameters:
      eta: decay rate for EMA (default 0.01)
      min_std: floor for downside std to avoid division by zero (default 1e-4)
    """

    eta: float = 0.01
    min_std: float = 1e-4
    _ema_neg_sq: float = field(default=0.0, repr=False)

    def name(self) -> str:
        return "sortino"

    def reset(self) -> None:
        self._ema_neg_sq = 0.0

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        eq_prev = float(prev_state.get("equity", 1.0) or 1.0)
        eq_curr = float(curr_state.get("equity", 1.0) or 1.0)

        if eq_prev <= 0:
            return 0.0

        r = eq_curr / eq_prev - 1.0
        neg_r = min(r, 0.0)
        self._ema_neg_sq += self.eta * (neg_r * neg_r - self._ema_neg_sq)

        downside_std = math.sqrt(max(self._ema_neg_sq, 0.0))
        denom = max(downside_std, self.min_std)

        return float(np.clip(r / denom, -10.0, 10.0))


@dataclass
class CustomReward(RewardFunction):
    """User-provided reward function.

    Parameters:
      fn: callable(prev_state, curr_state, candle, action, done) -> float
      label: name for this reward (default "custom")
    """

    fn: Callable = field(default=lambda ps, cs, c, a, d: 0.0)
    label: str = "custom"

    def name(self) -> str:
        return self.label

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        return float(self.fn(prev_state, curr_state, candle, action, done))


@dataclass
class AdvancedReward(RewardFunction):
    """Highly configurable reward function. Rewards PnL but heavily penalizes
    drawdowns, SL hits, and lingering in trades too long.
    """

    pnl_weight: float = 1.0
    drawdown_penalty: float = 3.0
    time_penalty: float = 0.001
    sl_penalty: float = 2.0
    tp_bonus: float = 1.5
    liq_penalty: float = 10.0

    _hwm: float = field(default=0.0, repr=False)
    _prev_dd: float = field(default=0.0, repr=False)
    _initial_equity: float = field(default=0.0, repr=False)

    def name(self) -> str:
        return "advanced"

    def reset(self) -> None:
        self._hwm = 0.0
        self._prev_dd = 0.0
        self._initial_equity = 0.0

    def compute(
        self,
        prev_state: dict,
        curr_state: dict,
        candle: dict,
        action: Any,
        done: bool,
        events: Optional[list] = None,
    ) -> float:
        eq_prev = float(prev_state.get("equity", 1.0) or 1.0)
        eq_curr = float(curr_state.get("equity", 1.0) or 1.0)

        if self._initial_equity == 0.0:
            self._initial_equity = eq_prev if eq_prev > 0 else 1.0

        # 1. Base PnL
        ret = (eq_curr - eq_prev) / max(self._initial_equity, 1e-10)
        reward = ret * self.pnl_weight

        # 2. Drawdown Penalty
        if eq_curr > self._hwm:
            self._hwm = eq_curr
        dd = 0.0 if self._hwm <= 0 else (self._hwm - eq_curr) / self._hwm
        dd_increase = max(0.0, dd - self._prev_dd)
        self._prev_dd = dd
        reward -= dd_increase * self.drawdown_penalty

        # 3. Time Penalty (encourages closing out early)
        if curr_state.get("position"):
            reward -= self.time_penalty

        # 4. Event-driven actions (SL / TP / Liq)
        if events:
            for ev in events:
                if isinstance(ev, dict):
                    t = ev.get("type")
                    if t == "tp":
                        reward += self.tp_bonus
                    elif t in ("sl", "sl_invalid"):
                        reward -= self.sl_penalty
                    elif t == "liquidation":
                        reward -= self.liq_penalty

        return float(np.clip(reward, -15.0, 15.0))


# ---------------------------------------------------------------------------
# Registry: string name -> reward class
# ---------------------------------------------------------------------------

_REWARD_REGISTRY: Dict[str, type] = {
    "advanced": AdvancedReward,
    "pnl": PnLReward,
    "log_return": LogReturnReward,
    "differential_sharpe": DifferentialSharpeReward,
    "risk_adjusted": RiskAdjustedReward,
    "sortino": SortinoReward,
}


def resolve_reward(spec: Any, **kwargs) -> RewardFunction:
    """Resolve a reward spec into a RewardFunction instance.

    Accepts:
      - A string name from the registry
      - A RewardFunction instance
      - A callable (wrapped in CustomReward)
      - Additional **kwargs are passed to the constructor (for string specs)
        or set as attributes (for RewardFunction instances).
    """
    if isinstance(spec, RewardFunction):
        for k, v in kwargs.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        return spec
    if callable(spec) and not isinstance(spec, str):
        return CustomReward(fn=spec)
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in _REWARD_REGISTRY:
            return _REWARD_REGISTRY[s](**kwargs)
        raise ValueError(
            f"Unknown reward function: '{spec}'. Available: {list(_REWARD_REGISTRY.keys())}"
        )
    raise ValueError(f"Invalid reward spec: {spec!r}")
