"""Rust Engine Edge-Case Tests.

Refactored from the former test_shadow_parity.py.  The pure-Python
TradingEngine no longer exists — all logic lives in _rust_core.  These
tests exercise the engine through the Python wrapper API (trading.py)
under the most demanding configurations: cross-margin, high leverage,
slippage, and combinations thereof.

Run:
    uv run pytest tests/test_rust_engine.py -v
"""

from __future__ import annotations

import csv
import os
import random
import sys
from typing import Any, Dict, List, Optional

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from forge_engine._rust_core import RustTradingEngine

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust extension not compiled — run `uv run maturin develop --release`",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_candles(n: int = 10_000) -> List[Dict[str, Any]]:
    """Load *n* 1-minute candles from the BTCUSDT_PERP_1m CSV."""
    csv_path = os.path.join(ROOT, "data", "BTCUSDT_PERP_1m.csv")
    if not os.path.exists(csv_path):
        pytest.skip(f"Data file not found: {csv_path}")

    candles: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            candles.append(
                {
                    "open_time": row["open_time"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                    "quote_asset_volume": float(row.get("quote_asset_volume", 0)),
                    "number_of_trades": int(float(row.get("number_of_trades", 0))),
                    "taker_buy_base_asset_volume": float(
                        row.get("taker_buy_base_asset_volume", 0)
                    ),
                    "taker_buy_quote_asset_volume": float(
                        row.get("taker_buy_quote_asset_volume", 0)
                    ),
                }
            )
    return candles


@pytest.fixture(scope="module")
def candles() -> List[Dict[str, Any]]:
    return load_candles(10_000)


# ═══════════════════════════════════════════════════════════════════════════════
# Engine Runner
# ═══════════════════════════════════════════════════════════════════════════════

class EngineRunner:
    """Drive the Rust engine through a candle sequence with random orders."""

    def __init__(
        self,
        candles: List[Dict[str, Any]],
        *,
        margin_mode: str = "isolated",
        leverage: float = 10.0,
        starting_cash: float = 10_000.0,
        slippage_pct: float = 0.0,
        seed: int = 42,
    ):
        self.candles = candles
        self.rng = random.Random(seed)
        self.eng = RustTradingEngine(
            symbol="BTCUSDT_PERP",
            margin_mode=margin_mode,
            leverage=leverage,
            starting_cash=starting_cash,
            slippage_pct=slippage_pct,
            timeframe_minutes=1,
        )
        self.order_ids: List[str] = []
        self.orders_created = 0
        self.fills = 0

    def run(self) -> Dict[str, Any]:
        """Process all candles, return the final snapshot."""
        for idx, candle in enumerate(self.candles):
            r = self.rng.random()
            if r < 0.05:
                self._inject_create(candle)
            elif r < 0.08:
                self._inject_close(candle)
            elif r < 0.10:
                self._inject_cancel()

            events, state = self.eng.on_candle(candle, idx)
            for ev in events:
                if isinstance(ev, dict) and ev.get("type", "").startswith("fill"):
                    self.fills += 1

        return dict(self.eng.snapshot())

    def _inject_create(self, candle: Dict[str, Any]) -> None:
        close = float(candle["close"])
        side = self.rng.choice(["LONG", "SHORT"])
        margin_pct = round(self.rng.uniform(0.05, 0.5), 6)
        atr = abs(float(candle["high"]) - float(candle["low"]))
        if atr < 1.0:
            atr = close * 0.005

        tp: Optional[float] = None
        sl: Optional[float] = None
        if self.rng.random() < 0.7:
            offset = atr * self.rng.uniform(1, 5)
            tp = round(close + offset if side == "LONG" else close - offset, 2)
        if self.rng.random() < 0.7:
            offset = atr * self.rng.uniform(0.5, 3)
            sl = round(close - offset if side == "LONG" else close + offset, 2)

        price = round(close, 2)
        res = dict(self.eng.create_order(side, price, margin_pct, tp=tp, sl=sl))
        if res.get("status") == "accepted":
            self.order_ids.append(str(res["order_id"]))
            self.orders_created += 1

    def _inject_close(self, candle: Dict[str, Any]) -> None:
        price = round(float(candle["close"]), 2)
        self.eng.close_order(price)

    def _inject_cancel(self) -> None:
        if self.order_ids:
            self.eng.cancel_order(self.order_ids[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# Test Suite
# ═══════════════════════════════════════════════════════════════════════════════

class TestRustEngine:
    """Edge-case stress tests on the Rust engine via Python wrappers."""

    def _assert_valid_state(self, state: Dict[str, Any], runner: EngineRunner) -> None:
        """Common assertions on any final engine snapshot."""
        assert isinstance(state["cash"], (int, float))
        assert isinstance(state["equity"], (int, float))
        assert state["equity"] > 0 or state.get("position") is not None
        assert state["candle_index"] == len(runner.candles) - 1
        assert runner.orders_created > 0, "No orders created in 10k candles"

    def test_isolated_margin(self, candles: List[Dict[str, Any]]) -> None:
        """Isolated margin, 10x leverage, default params."""
        runner = EngineRunner(
            candles, margin_mode="isolated", leverage=10.0, seed=42,
        )
        state = runner.run()
        self._assert_valid_state(state, runner)

    def test_cross_margin(self, candles: List[Dict[str, Any]]) -> None:
        """Cross margin, 10x leverage."""
        runner = EngineRunner(
            candles, margin_mode="cross", leverage=10.0, seed=42,
        )
        state = runner.run()
        self._assert_valid_state(state, runner)

    def test_high_leverage(self, candles: List[Dict[str, Any]]) -> None:
        """50x leverage — higher liquidation risk."""
        runner = EngineRunner(
            candles, margin_mode="isolated", leverage=50.0, seed=123,
        )
        state = runner.run()
        self._assert_valid_state(state, runner)

    def test_with_slippage(self, candles: List[Dict[str, Any]]) -> None:
        """Slippage enabled (affects SL and liquidation fills)."""
        runner = EngineRunner(
            candles,
            margin_mode="isolated",
            leverage=10.0,
            slippage_pct=0.001,
            seed=99,
        )
        state = runner.run()
        self._assert_valid_state(state, runner)

    def test_cross_high_leverage(self, candles: List[Dict[str, Any]]) -> None:
        """Cross margin + 50x leverage — toughest combo."""
        runner = EngineRunner(
            candles, margin_mode="cross", leverage=50.0, seed=777,
        )
        state = runner.run()
        self._assert_valid_state(state, runner)

    def test_snapshot_fields(self, candles: List[Dict[str, Any]]) -> None:
        """Verify that the snapshot contains expected keys with correct types."""
        runner = EngineRunner(
            candles, margin_mode="isolated", leverage=10.0, seed=42,
        )
        state = runner.run()
        for key in ("cash", "available_cash", "equity", "realized_pnl", "unrealized_pnl"):
            assert key in state, f"Missing key: {key}"
            assert isinstance(state[key], (int, float)), f"{key} is not numeric"
        assert state["margin_mode"] == "isolated"
        assert state["leverage"] == 10.0

    def test_create_and_cancel(self) -> None:
        """Create an order then cancel it — verify engine returns to initial state."""
        eng = RustTradingEngine(
            symbol="BTCUSDT_PERP",
            margin_mode="isolated",
            leverage=10.0,
            starting_cash=10_000.0,
            slippage_pct=0.0,
            timeframe_minutes=1,
        )
        candle = {
            "open_time": "2024-01-01T00:00:00Z",
            "open": 40000.0, "high": 40100.0,
            "low": 39900.0, "close": 40000.0,
            "volume": 100.0,
            "quote_asset_volume": 0.0,
            "number_of_trades": 10,
            "taker_buy_base_asset_volume": 0.0,
            "taker_buy_quote_asset_volume": 0.0,
        }
        eng.on_candle(candle, 0)
        res = dict(eng.create_order("LONG", 40000.0, 0.1))
        assert res["status"] == "accepted"
        oid = str(res["order_id"])

        cancel_res = dict(eng.cancel_order(oid))
        assert cancel_res["status"] == "accepted"

        snap = eng.snapshot()
        assert snap["cash"] == 10_000.0
        assert snap.get("position") is None
