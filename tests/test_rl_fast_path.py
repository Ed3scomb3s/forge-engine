"""Phase 6: RL Fast-Path smoke tests.

Validates that setup_rl_env, step_rl, reset_rl_env, and get_obs_bounds
work correctly from Python, and that the observation/reward math is sane.
"""

import numpy as np
import pytest

from forge_engine._rust_core import RustTradingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(cash=10_000.0, leverage=10.0, margin_mode="isolated"):
    """Create a fresh RustTradingEngine for testing."""
    return RustTradingEngine(
        symbol="BTCUSDT_PERP",
        margin_mode=margin_mode,
        leverage=leverage,
        starting_cash=cash,
        slippage_pct=0.0,
        timeframe_minutes=60,
    )


def make_candle(open_t, o, h, l, c, vol=100.0, tbv=50.0):
    """Build a [f64; 10] candle array."""
    return [
        float(open_t),  # open_time_unix
        float(o),       # open
        float(h),       # high
        float(l),       # low
        float(c),       # close
        float(vol),     # volume
        float(vol * c), # quote_asset_volume
        100.0,          # trades
        float(tbv),     # taker_buy_base
        float(tbv * c), # taker_buy_quote
    ]


# ---------------------------------------------------------------------------
# Test: setup_rl_env returns correct shapes
# ---------------------------------------------------------------------------

class TestSetupRlEnv:
    def test_discrete_default(self):
        eng = make_engine()
        result = eng.setup_rl_env(
            "discrete",
            ["ohlcv", "position_info"],
            "differential_sharpe",
        )
        assert result == [9, 4]  # ohlcv=5 + position_info=4, 4 discrete actions

    def test_discrete_with_sizing(self):
        eng = make_engine()
        result = eng.setup_rl_env(
            "discrete_with_sizing",
            ["ohlcv"],
            "pnl",
        )
        assert result[0] == 5  # ohlcv=5
        assert result[1] == 8  # hold + 3 sizes * 2 sides + close = 8

    def test_continuous(self):
        eng = make_engine()
        result = eng.setup_rl_env(
            "continuous",
            ["ohlcv", "returns", "equity_curve"],
            "risk_adjusted",
        )
        assert result[0] == 7  # 5+1+1
        assert result[1] == 1  # continuous = Box(1,)

    def test_indicator_obs(self):
        eng = make_engine()
        eng.register_rsi("RSI(14)[close]", 14)
        result = eng.setup_rl_env(
            "discrete",
            ["ohlcv", "rsi_14"],
            "sortino",
        )
        assert result[0] == 6  # ohlcv=5 + indicator=1

    def test_sma_ratio(self):
        eng = make_engine()
        eng.register_sma("SMA(20)[close]", 20)
        eng.register_sma("SMA(50)[close]", 50)
        result = eng.setup_rl_env(
            "discrete",
            ["sma_ratio_20_50"],
            "pnl",
        )
        assert result[0] == 1

    def test_drawdown_and_volume(self):
        eng = make_engine()
        result = eng.setup_rl_env(
            "discrete",
            ["drawdown", "volume_profile"],
            "advanced",
        )
        assert result[0] == 3  # drawdown=1 + volume_profile=2

    def test_invalid_action_type(self):
        eng = make_engine()
        with pytest.raises(ValueError, match="Unknown action_type"):
            eng.setup_rl_env("invalid", ["ohlcv"], "pnl")

    def test_invalid_reward_type(self):
        eng = make_engine()
        with pytest.raises(ValueError, match="Unknown reward_type"):
            eng.setup_rl_env("discrete", ["ohlcv"], "invalid")


# ---------------------------------------------------------------------------
# Test: get_obs_bounds returns correct numpy arrays
# ---------------------------------------------------------------------------

class TestGetObsBounds:
    def test_bounds_shape(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["ohlcv", "position_info"], "pnl")
        low, high = eng.get_obs_bounds()
        assert low.shape == (9,)
        assert high.shape == (9,)
        # OHLCV lows: [-1,-1,-1,-1,0], PositionInfo lows: [0,-1,-1,0]
        np.testing.assert_array_equal(
            low, np.array([-1,-1,-1,-1,0, 0,-1,-1,0], dtype=np.float32)
        )


# ---------------------------------------------------------------------------
# Test: step_rl basic functionality
# ---------------------------------------------------------------------------

class TestStepRl:
    def test_step_returns_correct_types(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["ohlcv", "position_info"], "pnl")

        candle = make_candle(1000000, 50000, 50100, 49900, 50000)
        obs, reward, terminated, truncated, info = eng.step_rl(0.0, candle)  # hold

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == (9,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "step" in info
        assert "equity" in info

    def test_hold_no_position(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["ohlcv", "position_info"], "pnl")

        candle = make_candle(1000000, 50000, 50100, 49900, 50000)
        obs, reward, term, trunc, info = eng.step_rl(0.0, candle)

        # No position: position_info should be [0, 0, 0, 0]
        assert obs[5] == 0.0  # has_pos
        assert obs[6] == 0.0  # side
        assert info["equity"] == 10000.0

    def test_open_long_and_close(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["position_info"], "pnl")

        # Step 1: Open long (action=1)
        c1 = make_candle(1000000, 50000, 50100, 49900, 50000)
        obs1, r1, _, _, info1 = eng.step_rl(1.0, c1)  # open_long

        # Step 2: Candle moves up -> position should be filled
        c2 = make_candle(1000060, 50000, 50200, 49800, 50100)
        obs2, r2, _, _, info2 = eng.step_rl(0.0, c2)  # hold

        # After step 2, the order from step 1 should have been promoted and filled
        assert obs2[0] == 1.0  # has_pos
        assert obs2[1] == 1.0  # side = LONG

        # Step 3: Close (action=3)
        c3 = make_candle(1000120, 50100, 50300, 50000, 50200)
        obs3, r3, _, _, info3 = eng.step_rl(3.0, c3)  # close

        # Step 4: Candle to fill the close order
        c4 = make_candle(1000180, 50200, 50400, 50100, 50300)
        obs4, r4, _, _, info4 = eng.step_rl(0.0, c4)  # hold

        # After step 4, close should have filled -> no position
        assert obs4[0] == 0.0  # no position

    def test_multiple_steps_equity_changes(self):
        """Run several steps and ensure equity changes propagate to reward."""
        eng = make_engine()
        eng.setup_rl_env("discrete", ["equity_curve"], "log_return")

        prices = [50000, 50100, 50200, 50300, 50400]
        for i, p in enumerate(prices):
            candle = make_candle(1000000 + i * 60, p, p + 50, p - 50, p)
            obs, reward, term, trunc, info = eng.step_rl(0.0, candle)
            assert not term
            assert not trunc

    def test_continuous_action(self):
        eng = make_engine()
        eng.setup_rl_env(
            "continuous",
            ["ohlcv"],
            "differential_sharpe",
            {"max_margin_pct": 0.2, "threshold": 0.1},
            {"eta": 0.01, "scale": 1.0},
        )

        # Dead zone -> hold
        c1 = make_candle(1000000, 50000, 50100, 49900, 50000)
        obs, r, _, _, _ = eng.step_rl(0.05, c1)  # within threshold -> hold
        assert obs.shape == (5,)

        # Strong long signal
        c2 = make_candle(1000060, 50000, 50200, 49800, 50100)
        obs, r, _, _, _ = eng.step_rl(0.8, c2)  # long

    def test_reset_rl_env(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["ohlcv"], "pnl")

        # Step a few candles
        for i in range(5):
            c = make_candle(1000000 + i * 60, 50000, 50100, 49900, 50000)
            eng.step_rl(0.0, c)

        # Reset
        eng.reset_rl_env()

        # Step again - should work fine
        c = make_candle(2000000, 50000, 50100, 49900, 50000)
        obs, r, _, _, info = eng.step_rl(0.0, c)
        assert info["step"] == 1


# ---------------------------------------------------------------------------
# Test: Reward functions produce expected values
# ---------------------------------------------------------------------------

class TestRewards:
    def _run_steps(self, reward_type, prices, reward_config=None):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["ohlcv"], reward_type, None, reward_config)
        rewards = []
        for i, p in enumerate(prices):
            c = make_candle(1000000 + i * 60, p, p + 10, p - 10, p)
            _, r, _, _, _ = eng.step_rl(0.0, c)
            rewards.append(r)
        return rewards

    def test_pnl_reward_no_position(self):
        rewards = self._run_steps("pnl", [50000, 50100, 50200])
        # No position -> equity doesn't change -> all rewards should be 0
        for r in rewards:
            assert r == 0.0

    def test_log_return_no_position(self):
        rewards = self._run_steps("log_return", [50000, 50100, 50200])
        for r in rewards:
            assert abs(r) < 1e-10

    def test_differential_sharpe_stable(self):
        rewards = self._run_steps(
            "differential_sharpe",
            [50000, 50000, 50000],
            {"eta": 0.01, "scale": 1.0},
        )
        for r in rewards:
            assert abs(r) < 1e-6

    def test_risk_adjusted_stable(self):
        rewards = self._run_steps(
            "risk_adjusted",
            [50000, 50000, 50000],
            {"drawdown_penalty": 2.0},
        )
        for r in rewards:
            assert abs(r) < 1e-6

    def test_sortino_stable(self):
        rewards = self._run_steps("sortino", [50000, 50000, 50000])
        for r in rewards:
            assert abs(r) < 1e-6

    def test_advanced_stable(self):
        rewards = self._run_steps("advanced", [50000, 50000, 50000])
        for r in rewards:
            assert abs(r) < 1e-6


# ---------------------------------------------------------------------------
# Test: Observation features produce valid values
# ---------------------------------------------------------------------------

class TestObservations:
    def test_ohlcv_values(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["ohlcv"], "pnl")

        # Feed a candle with known values
        c = make_candle(1000000, 50000, 50100, 49900, 50000, vol=1000)
        obs, _, _, _, _ = eng.step_rl(0.0, c)

        # log(open/close) = log(50000/50000) = 0
        assert abs(obs[0]) < 1e-6
        # log(high/close) = log(50100/50000) > 0
        assert obs[1] > 0
        # log(low/close) = log(49900/50000) < 0
        assert obs[2] < 0
        # 4th element is always 0
        assert obs[3] == 0.0
        # norm_volume: first candle, mean_vol = vol itself -> norm_v = 1.0
        assert abs(obs[4] - 1.0) < 1e-4

    def test_returns_first_is_zero(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["returns"], "pnl")

        c = make_candle(1000000, 50000, 50100, 49900, 50000)
        obs, _, _, _, _ = eng.step_rl(0.0, c)
        assert abs(obs[0]) < 1e-6  # first step, no prev_close

    def test_returns_second_is_nonzero(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["returns"], "pnl")

        c1 = make_candle(1000000, 50000, 50100, 49900, 50000)
        eng.step_rl(0.0, c1)

        c2 = make_candle(1000060, 50000, 50200, 49800, 50100)
        obs, _, _, _, _ = eng.step_rl(0.0, c2)
        # log(50100/50000) ≈ 0.002
        assert obs[0] > 0

    def test_drawdown_starts_at_zero(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["drawdown"], "pnl")

        c = make_candle(1000000, 50000, 50100, 49900, 50000)
        obs, _, _, _, _ = eng.step_rl(0.0, c)
        assert obs[0] == 0.0  # no drawdown initially

    def test_volume_profile_shape(self):
        eng = make_engine()
        eng.setup_rl_env("discrete", ["volume_profile"], "pnl")

        c = make_candle(1000000, 50000, 50100, 49900, 50000, vol=1000, tbv=400)
        obs, _, _, _, _ = eng.step_rl(0.0, c)
        assert obs.shape == (2,)
        # buy_ratio = 400/1000 = 0.4
        assert abs(obs[0] - 0.4) < 1e-4


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_setup_before_step(self):
        eng = make_engine()
        c = make_candle(1000000, 50000, 50100, 49900, 50000)
        with pytest.raises(RuntimeError, match="setup_rl_env not called"):
            eng.step_rl(0.0, c)

    def test_all_features_combined(self):
        eng = make_engine()
        eng.register_rsi("RSI(14)[close]", 14)
        eng.register_sma("SMA(20)[close]", 20)
        eng.register_sma("SMA(50)[close]", 50)
        result = eng.setup_rl_env(
            "discrete",
            [
                "ohlcv",
                "returns",
                "position_info",
                "equity_curve",
                "drawdown",
                "volume_profile",
                "rsi_14",
                "sma_ratio_20_50",
            ],
            "advanced",
        )
        expected_size = 5 + 1 + 4 + 1 + 1 + 2 + 1 + 1  # = 16
        assert result[0] == expected_size

        # Run a few steps
        for i in range(10):
            c = make_candle(1000000 + i * 60, 50000 + i * 10, 50100 + i * 10, 49900 + i * 10, 50000 + i * 10)
            obs, r, t, tr, info = eng.step_rl(0.0, c)
            assert obs.shape == (expected_size,)
            assert not np.isnan(obs).any()
            assert not np.isinf(obs).any()
