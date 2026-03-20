import time
import numpy as np
import forge_engine as fe


class RSIReversalStrategy(fe.VectorStrategy):
    """Simple RSI mean reversion strategy with SL/TP.

    Enters LONG when RSI drops below oversold_threshold.
    Exits when RSI rises above overbought_threshold or SL/TP is hit.

    Parameters:
        rsi_period: RSI calculation period (default: 14)
        oversold_threshold: RSI level to trigger entry (default: 30)
        overbought_threshold: RSI level to trigger exit (default: 70)
        margin_pct: Percentage of available margin to use (default: 0.1)
        sl_pct: Stop loss percentage from entry (default: 0.02 = 2%)
        tp_pct: Take profit percentage from entry (default: 0.04 = 4%)
    """

    optimizer_defaults = {
         
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
            "test_ratio": 0.2,
            "holdout_ratio": 0.15,
            "min_trades_per_fold": 5,
            "wfa_mode": "anchored",
        },
        "metrics": [
            {"path": "performance.smart_sharpe", "goal": "max", "weight": 1.0},
        ],
        "constraints": [
            {"path": "risk.max_drawdown_pct", "op": "<=", "value": 30},
        ],
    }

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        margin_pct: float = 0.1,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
    ):
        super().__init__()
        self.rsi_period = int(rsi_period)
        self.oversold_threshold = float(oversold_threshold)
        self.overbought_threshold = float(overbought_threshold)
        self._margin_pct = float(margin_pct)
        self._sl_pct = float(sl_pct)
        self._tp_pct = float(tp_pct)

    def indicators(self):
        return [fe.RSI(self.rsi_period, source="close")]

    def signal_params(self):
        return {
            "margin_pct": self._margin_pct,
            "sl_pct": self._sl_pct,
            "tp_pct": self._tp_pct,
        }

    def signals(self, close, indicators):
        rsi = indicators[f"RSI({self.rsi_period})[close]"]

        signals = np.zeros(len(close), dtype=np.int8)

        valid = ~np.isnan(rsi)
        signals[valid & (rsi < self.oversold_threshold)] = self.LONG
        signals[valid & (rsi > self.overbought_threshold)] = self.CLOSE

        return signals


if __name__ == "__main__":
    # Example: run a backtest with RSI reversal strategy
    session = fe.create_session(
        symbol="BTCUSDT_PERP",
        start_date="2024-01-01T00:00:00Z",
        end_date="2025-12-31T23:59:00Z",
        starting_cash=10000.0,
        leverage=20.0,
        margin_mode="cross",
        warmup_candles=50,
        timeframe="1h",
        close_at_end=True,
        enable_visual=False,
    )

    strat = RSIReversalStrategy(
        rsi_period=14,
        oversold_threshold=30.0,
        overbought_threshold=70.0,
        margin_pct=0.1,
        sl_pct=0.02,
        tp_pct=0.04,
    )

    t0 = time.perf_counter()
    final_state = fe.run_strategy(session, strat)
    elapsed = time.perf_counter() - t0

    print("Final equity:", (final_state or {}).get("equity"))
    print(f"Runtime: {elapsed:.3f}s")

    input("Press Enter to exit...")
