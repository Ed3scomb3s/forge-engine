import time
import numpy as np
import forge_engine as fe


class MACDMomentumStrategy(fe.VectorStrategy):
    """MACD momentum strategy with ATR volatility filter.

    Enters LONG when MACD histogram crosses above zero (bullish momentum).
    Enters SHORT when MACD histogram crosses below zero (bearish momentum).
    Uses ATR as a volatility filter: only enters when ATR exceeds a
    minimum threshold relative to price, avoiding low-volatility chop.
    """

    optimizer_defaults = {
        "param_space": {
            "fast": {"kind": "int_range", "start": 8, "stop": 16, "step": 2},
            "slow": {"kind": "int_range", "start": 20, "stop": 30, "step": 2},
            "signal_period": {"kind": "int_range", "start": 7, "stop": 12, "step": 1},
            "atr_period": {"kind": "int_range", "start": 10, "stop": 20, "step": 2},
            "atr_min_pct": {"kind": "float_range", "start": 0.005, "stop": 0.03, "step": 0.005},
            "margin_pct": {"kind": "float_range", "start": 0.05, "stop": 0.2, "step": 0.05},
            "sl_pct": {"kind": "float_range", "start": 0.01, "stop": 0.05, "step": 0.01},
            "tp_pct": {"kind": "float_range", "start": 0.02, "stop": 0.10, "step": 0.02},
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
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
        atr_period: int = 14,
        atr_min_pct: float = 0.01,
        margin_pct: float = 0.1,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
    ):
        super().__init__()
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal_period = int(signal_period)
        self.atr_period = int(atr_period)
        self.atr_min_pct = float(atr_min_pct)
        self._margin_pct = float(margin_pct)
        self._sl_pct = float(sl_pct)
        self._tp_pct = float(tp_pct)

    def indicators(self):
        return [
            fe.MACD(self.fast, self.slow, self.signal_period, source="close"),
            fe.ATR(self.atr_period),
        ]

    def signal_params(self):
        return {
            "margin_pct": self._margin_pct,
            "sl_pct": self._sl_pct,
            "tp_pct": self._tp_pct,
        }

    def signals(self, close, indicators):
        macd_line = indicators[f"MACD({self.fast},{self.slow},{self.signal_period})[close].line"]
        macd_signal = indicators[f"MACD({self.fast},{self.slow},{self.signal_period})[close].signal"]
        histogram = indicators[f"MACD({self.fast},{self.slow},{self.signal_period})[close].hist"]
        atr = indicators[f"ATR({self.atr_period})"]

        signals = np.zeros(len(close), dtype=np.int8)

        valid = ~(np.isnan(macd_line) | np.isnan(macd_signal) | np.isnan(histogram) | np.isnan(atr))

        # ATR volatility filter: only trade when ATR/close > threshold
        vol_ok = (atr / close) > self.atr_min_pct

        # Histogram zero-cross detection (MACD crosses signal line)
        cross_up = np.zeros(len(close), dtype=bool)
        cross_dn = np.zeros(len(close), dtype=bool)
        cross_up[1:] = (histogram[1:] > 0) & (histogram[:-1] <= 0) & valid[1:] & valid[:-1]
        cross_dn[1:] = (histogram[1:] < 0) & (histogram[:-1] >= 0) & valid[1:] & valid[:-1]

        # LONG on bullish cross with sufficient volatility
        signals[cross_up & vol_ok] = self.LONG
        # SHORT on bearish cross with sufficient volatility
        signals[cross_dn & vol_ok] = self.SHORT

        return signals


if __name__ == "__main__":
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

    strat = MACDMomentumStrategy(
        fast=12, slow=26, signal_period=9, atr_period=14,
        atr_min_pct=0.01, margin_pct=0.1, sl_pct=0.02, tp_pct=0.04,
    )

    t0 = time.perf_counter()
    final_state = fe.run_strategy(session, strat)
    elapsed = time.perf_counter() - t0

    print("Final equity:", (final_state or {}).get("equity"))
    print(f"Runtime: {elapsed:.3f}s")
