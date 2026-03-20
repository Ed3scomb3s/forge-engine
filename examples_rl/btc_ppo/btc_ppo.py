from pathlib import Path
from forge_engine.rl import ContinuousActions

SCRIPT_DIR = Path(__file__).parent

# Hyperparameters found by Optuna HPO (tune_btc_ppo.py, 50 trials)
# Sortino reward dominated the search (8 of top 10 configs).
# Standard obs preset (ohlcv + returns + rsi + sma_ratio + position + drawdown)
# beat both minimal and heavy-indicator sets.
# Key fix: dead zone now means "hold" not "close" — agent can hold positions.

RL_CONFIG = dict(
    name="BTC PPO (WFA)",
    algorithm="PPO",
    model_path=str(SCRIPT_DIR / "btc_ppo_model"),
    symbol="BTCUSDT_PERP",
    # Walk-Forward Analysis: 5 anchored folds + 15% holdout.
    # The splitter generates train/test periods automatically.
    start_date="2022-01-01T00:00:00Z",
    end_date="2026-02-12T00:00:00Z",
    wfa_splits=5,
    wfa_test_ratio=0.2,
    wfa_holdout_ratio=0.15,
    wfa_mode="anchored",
    observations=[
        "ohlcv",
        "returns",
        "rsi_14",
        "sma_ratio_20_50",
        "position_info",
        "drawdown",
    ],
    actions=ContinuousActions(
        max_margin_pct=0.10,
        threshold=0.23,       # HPO optimal: 0.23 (was 0.5 — way too high)
        sl_pct=0.06,          # HPO optimal: ~6% SL
        tp_pct=0.12,          # HPO optimal: ~12% TP (2:1 reward/risk)
    ),
    reward="sortino",         # HPO winner — penalizes only downside deviation
    reward_kwargs=dict(
        eta=0.021,
        min_std=0.005,        # Raised from 1e-4 — dampens reward spikes in early episode steps
    ),
    engine_kwargs=dict(
        starting_cash=100_000,
        leverage=2,
        margin_mode="cross",
        slippage_pct=0.0005,
        close_at_end=True,
        timeframe="1h",
        warmup_candles=50,
        max_steps=720,        # 30 days — shorter episodes = less variance per episode
    ),
    # HPO showed diminishing returns after 300K steps. 1M is a good balance.
    train_steps=1_000_000,
    retrain_steps=250_000,
    learning_rate=9e-5,       # HPO optimal: ~9e-5 (lower = more stable)
    seed=42,
    algo_kwargs=dict(
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        ent_coef=0.008,       # HPO optimal: ~0.008 (moderate exploration)
        gamma=0.99,           # Raised from 0.978 — agent values longer-term rewards
        gae_lambda=0.953,
        clip_range=0.30,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    ),
)
