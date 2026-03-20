import time
import numpy as np
import forge_engine as fe


class BollingerBandStrategy(fe.VectorStrategy):
    """Bollinger Band mean reversion strategy.

    Enters LONG when close drops below the lower band.
    Enters SHORT when close rises above the upper band.
    Exits (CLOSE) when price crosses back to the middle band.
    """

    optimizer_defaults = {
        "param_space": {
            "period": {"kind": "int_range", "start": 10, "stop": 30, "step": 5},
            "multiplier": {"kind": "float_range", "start": 1.5, "stop": 3.0, "step": 0.5},
            "margin_pct": {"kind": "float_range", "start": 0.05, "stop": 0.2, "step": 0.05},
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
        period: int = 20,
        multiplier: float = 2.0,
        margin_pct: float = 0.1,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
    ):
        super().__init__()
        self.period = int(period)
        self.multiplier = float(multiplier)
        self._margin_pct = float(margin_pct)
        self._sl_pct = float(sl_pct)
        self._tp_pct = float(tp_pct)

    def indicators(self):
        return [fe.BollingerBands(self.period, self.multiplier, source="close")]

    def signal_params(self):
        return {
            "margin_pct": self._margin_pct,
            "sl_pct": self._sl_pct,
            "tp_pct": self._tp_pct,
        }

    def signals(self, close, indicators):
        upper = indicators[f"BB({self.period},{self.multiplier})[close].upper"]
        mid = indicators[f"BB({self.period},{self.multiplier})[close].mid"]
        lower = indicators[f"BB({self.period},{self.multiplier})[close].lower"]

        signals = np.zeros(len(close), dtype=np.int8)

        valid = ~(np.isnan(upper) | np.isnan(mid) | np.isnan(lower))

        # LONG when close drops below lower band (mean reversion: expect bounce up)
        signals[valid & (close < lower)] = self.LONG
        # SHORT when close rises above upper band (mean reversion: expect drop)
        signals[valid & (close > upper)] = self.SHORT
        # CLOSE when price crosses back to mid band (take profit at mean)
        # Detect cross through mid: previous bar on one side, current bar on the other
        cross_mid_up = np.zeros(len(close), dtype=bool)
        cross_mid_dn = np.zeros(len(close), dtype=bool)
        cross_mid_up[1:] = (close[1:] >= mid[1:]) & (close[:-1] < mid[:-1]) & valid[1:] & valid[:-1]
        cross_mid_dn[1:] = (close[1:] <= mid[1:]) & (close[:-1] > mid[:-1]) & valid[1:] & valid[:-1]
        signals[cross_mid_up | cross_mid_dn] = self.CLOSE

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

    strat = BollingerBandStrategy(period=20, multiplier=2.0, margin_pct=0.1, sl_pct=0.02, tp_pct=0.04)

    t0 = time.perf_counter()
    final_state = fe.run_strategy(session, strat)
    elapsed = time.perf_counter() - t0

    print("Final equity:", (final_state or {}).get("equity"))
    print(f"Runtime: {elapsed:.3f}s")
