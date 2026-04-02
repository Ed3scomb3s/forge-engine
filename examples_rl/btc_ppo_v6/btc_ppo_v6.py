from pathlib import Path

from forge_engine.rl import DiscreteActions

SCRIPT_DIR = Path(__file__).parent


RL_CONFIG = dict(
    name="BTC PPO V6 (Discrete WFA)",
    algorithm="PPO",
    model_path=str(SCRIPT_DIR / "btc_ppo_v6_model"),
    symbol="BTCUSDT_PERP",
    start_date="2020-01-01T00:00:00Z",
    end_date="2026-02-12T00:00:00Z",
    wfa_splits=5,
    wfa_test_ratio=0.2,
    wfa_holdout_ratio=0.15,
    wfa_mode="anchored",
    observations=[
        "ohlcv",
        "returns",
        "rsi_14",
        "atr_14",
        "sma_ratio_20_50",
        "sma_ratio_10_30",
        "macd_line_12_26_9",
        "macd_signal_12_26_9",
        "macd_hist_12_26_9",
        "position_info",
        "drawdown",
    ],
    actions=DiscreteActions(
        actions=["hold", "open_long", "open_short", "close"],
        margin_pct=0.10,
        sl_pct=0.03,
        tp_pct=0.09,
    ),
    reward="differential_sharpe",
    reward_kwargs=dict(
        eta=0.005,
        scale=1.0,
    ),
    engine_kwargs=dict(
        starting_cash=100_000,
        leverage=2,
        margin_mode="cross",
        slippage_pct=0.0005,
        close_at_end=True,
        timeframe="1h",
        warmup_candles=50,
        max_steps=720,
    ),
    train_steps=1_000_000,
    retrain_steps=250_000,
    learning_rate=3e-4,
    seed=42,
    algo_kwargs=dict(
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    ),
)
