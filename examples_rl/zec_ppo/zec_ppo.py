"""PPO agent on ZECUSDT_PERP — low-liquidity, high-volatility thesis experiment.

Zcash (ZEC) models the lower-liquidity, high-volatility environment
mentioned in the thesis proposal. Uses risk-adjusted reward with heavy
drawdown penalty to force survival in volatile conditions.

Run directly:   uv run python examples_rl/zec_ppo/zec_ppo.py
Or via launcher: uv run python examples_rl/launch_rl.py
"""

from pathlib import Path

from forge_engine.rl import ContinuousActions

# ============================================================================
# Configuration — importable as RL_CONFIG by the launcher
# ============================================================================

SCRIPT_DIR = Path(__file__).parent

RL_CONFIG = dict(
    name="ZEC PPO (Low Liquidity)",
    algorithm="PPO",
    model_path=str(SCRIPT_DIR / "zec_ppo_model"),
    symbol="ZECUSDT_PERP",
    train_start="2024-01-01T00:00:00Z",
    train_end="2024-07-01T00:00:00Z",
    test_start="2024-07-01T00:00:00Z",
    test_end="2025-01-01T00:00:00Z",
    observations=["ohlcv", "returns", "rsi_14", "atr_14", "position_info", "drawdown"],
    actions=ContinuousActions(
        max_margin_pct=0.10,  # Lower margin for higher volatility
        threshold=0.15,
        sl_pct=0.05,  # Wider stop loss for altcoins
        tp_pct=0.10,
    ),
    # Heavily penalize drawdown to force survival in volatile conditions
    reward="risk_adjusted",
    reward_kwargs=dict(drawdown_penalty=3.0),
    engine_kwargs=dict(
        starting_cash=10_000,
        leverage=5,  # Lower leverage for altcoins
        margin_mode="isolated",
        slippage_pct=0.0015,  # Higher assumed slippage for low liquidity
        close_at_end=True,
        timeframe="1h",
        warmup_candles=50,
    ),
    train_steps=150_000,
    retrain_steps=50_000,
    learning_rate=2e-4,
    seed=123,
    algo_kwargs=dict(n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.02),
)

