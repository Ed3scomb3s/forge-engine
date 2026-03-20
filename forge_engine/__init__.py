"""Public package exports for forge_engine."""

from .engine import (
    Session,
    create_session,
    step_session,
    get_warmup_candles,
    # Pre-aggregated data for fast optimization
    CandleDataAggregated,
    preload_candle_data_aggregated,
    iter_candles_from_aggregated,
)
from .trading import create_order, cancel_order, close_order, get_state, compute_open_capacity
from .strategy import Strategy, VectorStrategy, run_strategy
from .indicators import (
    register_indicators,
    SMA,
    EMA,
    RSI,
    ATR,
    BollingerBands,
    MACD,
)
from .metrics import compute_metrics, format_metrics_report  # new
from .optuna_optimizer import (
    # Parameter space classes
    Param, Fixed, Choice, IntRange, FloatRange, UniformFloat, LogUniform,
    # Metrics and constraints
    MetricSpec, metric, Constraint,
    # WFA configuration
    WalkForwardConfig,
    HoldoutConfig,
    AntiOverfitConfig,
    # Optimizer and results
    OptunaOptimizer,
    OptunaOptimizationResult,
    WFATrialResult,
    FoldResult,
    optimize_with_wfa,
)
__all__ = [
    # Engine
    "Session",
    "create_session",
    "step_session",
    "get_warmup_candles",
    # Pre-aggregated data for fast optimization
    "CandleDataAggregated",
    "preload_candle_data_aggregated",
    "iter_candles_from_aggregated",
    # Trading
    "create_order",
    "cancel_order",
    "close_order",
    "get_state",
    "compute_open_capacity",
    # Strategy
    "Strategy",
    "VectorStrategy",
    "run_strategy",
    # Indicators
    "register_indicators",
    "SMA",
    "EMA",
    "RSI",
    "ATR",
    "BollingerBands",
    "MACD",
    # Metrics
    "compute_metrics",
    "format_metrics_report",
    # Optimizer parameter types
    "Param", "Fixed", "Choice", "IntRange", "FloatRange", "UniformFloat", "LogUniform",
    "MetricSpec", "metric", "Constraint",
    # Optuna optimizer with walk-forward analysis
    "OptunaOptimizer",
    "WalkForwardConfig",
    "HoldoutConfig",
    "AntiOverfitConfig",
    "OptunaOptimizationResult",
    "WFATrialResult",
    "FoldResult",
    "optimize_with_wfa",
]
