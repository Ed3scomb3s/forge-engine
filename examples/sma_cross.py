import time
import numpy as np
import forge_engine as fe


class SMACrossStrategy(fe.VectorStrategy):
    optimizer_defaults = {
        "param_space": {
            "fast": {"kind": "int_range", "start": 5, "stop": 15, "step": 2},
            "slow": {"kind": "int_range", "start": 20, "stop": 50, "step": 5},
            "margin_pct": {"kind": "float_range", "start": 0.05, "stop": 0.2, "step": 0.05},
            # Stop Loss and Take Profit as percentages (0 = disabled)
            "sl_pct": {"kind": "float_range", "start": 0.0, "stop": 0.05, "step": 0.01},
            "tp_pct": {"kind": "float_range", "start": 0.0, "stop": 0.10, "step": 0.02},
        },
        "session": {
            "symbol": "BTCUSDT_PERP",
            "timeframe": "1h",
            "start_date": "2020-01-01T00:00:00Z",
            "end_date": "2026-02-12T00:00:00Z",
            "starting_cash": 100000,
            "leverage": 2,
            "margin_mode": "cross",
            "warmup_candles": 50,
        },
        "optimization": {
            "n_trials": 500,
            "n_jobs": -1,
            "wfa_splits": 20,
            "test_ratio": 0.25,
            "holdout_ratio": 0.15,
            "min_trades_per_fold": 3,
            "wfa_mode": "anchored",
        },
        "metrics": [
            {"path": "performance.smart_sharpe", "goal": "max", "weight": 1.0},
            {"path": "performance.pnl", "goal": "max", "weight": 1.0},
            {"path": "performance.win_rate_pct", "goal": "max", "weight": 1.0},
        ],
        "constraints": [
            {"path": "risk.max_drawdown_pct", "op": "<=", "value": 25},
        ],
    }

    def __init__(
        self,
        fast: int = 10,
        slow: int = 30,
        margin_pct: float = 0.1,
        sl_pct: float = 0.0,
        tp_pct: float = 0.0,
    ):
        super().__init__()
        self.fast = int(fast)
        self.slow = int(slow)
        self._margin_pct = float(margin_pct)
        self._sl_pct = float(sl_pct)
        self._tp_pct = float(tp_pct)

    def indicators(self):
        return [
            fe.SMA(self.fast, source="close"),
            fe.SMA(self.slow, source="close"),
        ]

    def signal_params(self):
        return {
            "margin_pct": self._margin_pct,
            "sl_pct": self._sl_pct,
            "tp_pct": self._tp_pct,
        }

    def signals(self, close, indicators):
        fast = indicators[f"SMA({self.fast})[close]"]
        slow = indicators[f"SMA({self.slow})[close]"]

        signals = np.zeros(len(close), dtype=np.int8)

        # Vectorized cross detection
        # Cross up: fast was <= slow, now fast > slow -> LONG
        valid = ~(np.isnan(fast) | np.isnan(slow))
        cross_up = np.zeros(len(close), dtype=bool)
        cross_dn = np.zeros(len(close), dtype=bool)
        cross_up[1:] = (fast[1:] > slow[1:]) & (fast[:-1] <= slow[:-1]) & valid[1:] & valid[:-1]
        cross_dn[1:] = (fast[1:] < slow[1:]) & (fast[:-1] >= slow[:-1]) & valid[1:] & valid[:-1]

        signals[cross_up] = self.LONG
        signals[cross_dn] = self.CLOSE

        return signals


if __name__ == "__main__":
    # Example: run a backtest for a period with SMA cross strategy
    session = fe.create_session(
        symbol="BTCUSDT_PERP",
        start_date="2025-01-01T00:00:00Z",
        end_date="2025-12-31T23:59:00Z",
        starting_cash=10000.0,
        leverage=2.0,
        margin_mode="cross",
        warmup_candles=50,
        timeframe="1h",
        close_at_end=True,
        enable_visual=False,
    )

    # Example with SL=2% and TP=5%
    strat = SMACrossStrategy(fast=10, slow=30, margin_pct=0.1, sl_pct=0.02, tp_pct=0.05)

    # Measure runtime of the full strategy run (includes metrics computation inside run_strategy)
    t0 = time.perf_counter()
    final_state = fe.run_strategy(session, strat)
    elapsed = time.perf_counter() - t0

    print("Final equity:", (final_state or {}).get("equity"))
    print(f"Runtime: {elapsed:.3f}s")
