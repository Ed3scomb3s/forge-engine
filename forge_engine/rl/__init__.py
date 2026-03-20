"""Reinforcement Learning module for Forge Engine.

Provides Gymnasium environment wrappers, configurable observation/action/reward
spaces, and a training pipeline built on stable-baselines3.

Quick start:
    from forge_engine.rl import ForgeEnv, DiscreteActions, train_agent

    env = ForgeEnv(
        symbol="BTCUSDT_PERP",
        start_date="2024-01-01T00:00:00Z",
        end_date="2025-01-01T00:00:00Z",
        observations=["ohlcv", "rsi_14", "position_info"],
        actions=DiscreteActions(["hold", "open_long", "open_short", "close"], margin_pct=0.1),
        reward="differential_sharpe",
    )
    model = train_agent(env, algorithm="DQN", total_timesteps=100_000)
"""

# Environment
from .env import ForgeEnv

# Action spaces
from .actions import (
    ActionSpace,
    DiscreteActions,
    DiscreteActionsWithSizing,
    ContinuousActions,
)

# Observation features
from .observations import (
    ObsFeature,
    OHLCV,
    Returns,
    PositionInfo,
    EquityCurve,
    Drawdown,
    IndicatorObs,
    SMA_Ratio,
    VolumeProfile,
    resolve_features,
)

# Reward functions
from .rewards import (
    RewardFunction,
    PnLReward,
    LogReturnReward,
    DifferentialSharpeReward,
    RiskAdjustedReward,
    SortinoReward,
    CustomReward,
    AdvancedReward,
    resolve_reward,
)

# Training
from .train import (
    train_agent,
    load_agent,
    evaluate_agent,
    train_multi_seed,
)

# Strategy adapter
from .agent_strategy import AgentStrategy

__all__ = [
    # Environment
    "ForgeEnv",
    # Actions
    "ActionSpace",
    "DiscreteActions",
    "DiscreteActionsWithSizing",
    "ContinuousActions",
    # Observations
    "ObsFeature",
    "OHLCV",
    "Returns",
    "PositionInfo",
    "EquityCurve",
    "Drawdown",
    "IndicatorObs",
    "SMA_Ratio",
    "VolumeProfile",
    "resolve_features",
    # Rewards
    "RewardFunction",
    "PnLReward",
    "LogReturnReward",
    "DifferentialSharpeReward",
    "RiskAdjustedReward",
    "SortinoReward",
    "CustomReward",
    "AdvancedReward",
    "resolve_reward",
    # Training
    "train_agent",
    "load_agent",
    "evaluate_agent",
    "train_multi_seed",
    # Strategy adapter
    "AgentStrategy",
]
