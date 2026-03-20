from pathlib import Path
from forge_engine.rl import DiscreteActions

SCRIPT_DIR = Path(__file__).parent

RL_CONFIG = dict(
    name="BTC DQN (Pro)",
    algorithm="DQN",
    model_path=str(SCRIPT_DIR / "btc_dqn_model"),
    symbol="BTCUSDT_PERP",
    train_start="2020-01-01T00:00:00Z",
    train_end="2025-01-01T00:00:00Z",
    test_start="2025-01-01T00:00:00Z",
    test_end="2026-02-12T00:00:00Z",
    observations=["ohlcv", "rsi_14", "atr_14", "position_info", "drawdown"],
    actions=DiscreteActions(
        actions=["hold", "open_long", "open_short", "close"],
        margin_pct=0.15,
        sl_pct=0.03,
        tp_pct=0.06,
    ),
    reward="differential_sharpe",
    engine_kwargs=dict(
        starting_cash=100_000,
        leverage=2,
        margin_mode="cross",
        slippage_pct=0.0005,
        close_at_end=True,
        timeframe="1h",
        warmup_candles=50,
    ),
    train_steps=1_000_000,
    retrain_steps=250_000,
    learning_rate=1e-4,
    seed=42,
    algo_kwargs=dict(
        buffer_size=100_000,
        learning_starts=10_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        batch_size=128,
        policy_kwargs=dict(net_arch=[256, 256]),
    ),
)
