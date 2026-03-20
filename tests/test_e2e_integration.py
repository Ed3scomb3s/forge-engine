"""End-to-end integration tests for the three core pillars of Forge Engine.

1. Classical strategy execution (SMACrossStrategy via run_strategy)
2. Optuna optimizer with walk-forward analysis
3. RL environment (ForgeEnv with Gymnasium step loop)

These tests use real CSV data (BTCUSDT_PERP_1m.csv) with small windows
to keep execution fast while proving end-to-end correctness.

Run:
    uv run pytest tests/test_e2e_integration.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "BTCUSDT_PERP_1m.csv")

try:
    from forge_engine._rust_core import RustTradingEngine  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="Rust extension not compiled — run `uv run maturin develop --release`",
)

needs_data = pytest.mark.skipif(
    not os.path.exists(CSV_PATH),
    reason=f"Data file not found: {CSV_PATH}",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Classical Strategy: SMACrossStrategy via run_strategy
# ═══════════════════════════════════════════════════════════════════════════════

@needs_data
class TestClassicalStrategy:
    """Run SMACrossStrategy end-to-end and validate the final state dict."""

    def test_sma_cross_produces_valid_state(self) -> None:
        import forge_engine as fe
        from examples.sma_cross import SMACrossStrategy

        # Small 1-month window on 1h timeframe (fast)
        session = fe.create_session(
            symbol="BTCUSDT_PERP",
            start_date="2024-06-01T00:00:00Z",
            end_date="2024-07-01T00:00:00Z",
            starting_cash=10_000.0,
            leverage=2.0,
            margin_mode="cross",
            warmup_candles=50,
            timeframe="1h",
            close_at_end=True,
            enable_visual=False,
            data_dir=DATA_DIR,
        )

        strat = SMACrossStrategy(fast=10, slow=30, margin_pct=0.1)
        final_state = fe.run_strategy(session, strat)

        assert final_state is not None, "run_strategy returned None"
        assert isinstance(final_state.get("equity"), (int, float)), "equity missing or non-numeric"
        assert final_state["equity"] > 0, "equity should be positive"
        assert isinstance(final_state.get("realized_pnl"), (int, float)), "realized_pnl missing"

        metrics = final_state.get("metrics")
        assert metrics is not None, "metrics dict not populated"
        assert isinstance(metrics, dict), "metrics is not a dict"

    def test_sma_cross_with_sl_tp(self) -> None:
        """Strategy with stop-loss and take-profit enabled."""
        import forge_engine as fe
        from examples.sma_cross import SMACrossStrategy

        session = fe.create_session(
            symbol="BTCUSDT_PERP",
            start_date="2024-06-01T00:00:00Z",
            end_date="2024-07-01T00:00:00Z",
            starting_cash=10_000.0,
            leverage=2.0,
            margin_mode="cross",
            warmup_candles=50,
            timeframe="1h",
            close_at_end=True,
            enable_visual=False,
            data_dir=DATA_DIR,
        )

        strat = SMACrossStrategy(fast=10, slow=30, margin_pct=0.1, sl_pct=0.02, tp_pct=0.05)
        final_state = fe.run_strategy(session, strat)
        assert final_state is not None
        assert final_state["equity"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Optuna Optimizer with Walk-Forward Analysis
# ═══════════════════════════════════════════════════════════════════════════════

@needs_data
class TestOptunaOptimizer:
    """Run a minimal Optuna optimization with WFA and validate the result."""

    @pytest.mark.timeout(120)
    def test_optimizer_completes(self) -> None:
        import forge_engine as fe

        optimizer = fe.OptunaOptimizer(
            session_kwargs=dict(
                symbol="BTCUSDT_PERP",
                start_date="2024-06-01T00:00:00Z",
                end_date="2024-09-01T00:00:00Z",
                starting_cash=10_000.0,
                leverage=2.0,
                margin_mode="cross",
                warmup_candles=30,
                timeframe="1h",
                close_at_end=True,
                enable_visual=False,
                data_dir=DATA_DIR,
            ),
            strategy_ctor="examples.sma_cross.SMACrossStrategy",
            param_space={
                "fast": fe.Choice([10, 15]),
                "slow": fe.Choice([30, 40]),
                "margin_pct": fe.Fixed(0.1),
            },
            metrics=[fe.MetricSpec("performance.smart_sharpe", "max", 1.0)],
            wfa_config=fe.WalkForwardConfig(n_splits=2, test_ratio=0.2),
            holdout_config=fe.HoldoutConfig(enabled=False),
            anti_overfit=fe.AntiOverfitConfig(
                min_trades_per_fold=0,
                use_pruning=False,
                n_startup_trials=1,
            ),
        )

        result = optimizer.optimize(
            n_trials=2,
            n_jobs=1,
            show_progress_bar=False,
            verbose=False,
        )

        assert isinstance(result, fe.OptunaOptimizationResult)
        assert result.n_trials == 2
        assert isinstance(result.best_oos_score, (int, float))
        assert isinstance(result.best_params, dict)
        assert "fast" in result.best_params
        assert "slow" in result.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RL Environment: ForgeEnv with Gymnasium step loop
# ═══════════════════════════════════════════════════════════════════════════════

@needs_data
class TestRLEnvironment:
    """Step ForgeEnv with random actions and validate observations / rewards."""

    def test_env_random_steps(self) -> None:
        from forge_engine.rl.env import ForgeEnv
        import numpy as np

        env = ForgeEnv(
            symbol="BTCUSDT_PERP",
            start_date="2024-06-01T00:00:00Z",
            end_date="2024-07-01T00:00:00Z",
            starting_cash=10_000.0,
            leverage=10.0,
            margin_mode="isolated",
            timeframe="1h",
            warmup_candles=30,
            max_steps=100,
            close_at_end=True,
            data_dir=DATA_DIR,
        )

        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray), "obs is not a numpy array"
        assert obs.dtype == np.float32, f"obs dtype should be float32, got {obs.dtype}"
        assert not np.any(np.isnan(obs)), "initial obs contains NaN"

        total_reward = 0.0
        steps = 0

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray), f"step {steps}: obs is not ndarray"
            assert isinstance(reward, float), f"step {steps}: reward is not float"
            assert isinstance(terminated, bool), f"step {steps}: terminated is not bool"
            assert isinstance(truncated, bool), f"step {steps}: truncated is not bool"

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0, "Environment didn't step at all"
        env.close()

    def test_env_reset_multiple_episodes(self) -> None:
        """Verify the environment can be reset and run multiple episodes."""
        from forge_engine.rl.env import ForgeEnv
        import numpy as np

        env = ForgeEnv(
            symbol="BTCUSDT_PERP",
            start_date="2024-06-01T00:00:00Z",
            end_date="2024-07-01T00:00:00Z",
            starting_cash=10_000.0,
            leverage=10.0,
            margin_mode="isolated",
            timeframe="1h",
            warmup_candles=30,
            max_steps=50,
            data_dir=DATA_DIR,
        )

        for episode in range(3):
            obs, info = env.reset(seed=episode)
            assert isinstance(obs, np.ndarray)

            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        env.close()
