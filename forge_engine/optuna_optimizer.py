# forge_engine/optuna_optimizer.py
"""
Optuna-based optimizer with walk-forward analysis and anti-overfitting measures.

Features:
- Walk-forward analysis (anchored/rolling windows)
- Out-of-sample holdout validation
- Optuna TPE sampler with pruning
- Trial count penalty for data snooping mitigation
- Minimum trade count filtering
- Parallel fold execution for faster optimization
"""

from __future__ import annotations

import importlib
import inspect
import math
import random
import time
import multiprocessing
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast


import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .engine import create_session, _parse_iso8601_utc, preload_candle_data, preload_candle_data_memmap, preload_candle_data_aggregated, CandleData, CandleDataMemmap, CandleDataAggregated, iter_candles_from_aggregated

from .strategy import run_strategy


# ---------------------------------------------------------------------------
# Parameter Space Primitives
# ---------------------------------------------------------------------------

class Param:
    """Base parameter spec. Provide grid_values() for grid search and sample(rng) for random search."""
    def grid_values(self) -> Sequence[Any]:
        raise NotImplementedError

    def sample(self, rng: random.Random) -> Any:
        vals = list(self.grid_values())
        if not vals:
            raise ValueError(f"{self.__class__.__name__} has no values to sample")
        return rng.choice(vals)


@dataclass(frozen=True)
class Fixed(Param):
    value: Any
    def grid_values(self) -> Sequence[Any]:
        return [self.value]
    def sample(self, rng: random.Random) -> Any:
        return self.value


@dataclass(frozen=True)
class Choice(Param):
    options: Sequence[Any]
    def grid_values(self) -> Sequence[Any]:
        return list(self.options)


@dataclass(frozen=True)
class IntRange(Param):
    """Inclusive integer range."""
    start: int
    stop: int
    step: int = 1
    def grid_values(self) -> Sequence[int]:
        if self.step <= 0:
            raise ValueError("IntRange.step must be positive")
        if self.start > self.stop:
            return []
        n = (self.stop - self.start) // self.step + 1
        return [self.start + i * self.step for i in range(max(0, n))]


@dataclass(frozen=True)
class FloatRange(Param):
    """Inclusive float range with fixed step."""
    start: float
    stop: float
    step: float
    ndigits: int = 12
    def grid_values(self) -> Sequence[float]:
        if self.step <= 0:
            raise ValueError("FloatRange.step must be positive")
        if self.start > self.stop:
            return []
        out: List[float] = []
        count = int(math.floor((self.stop - self.start) / self.step + 0.5)) + 1
        for i in range(count):
            out.append(round(self.start + i * self.step, self.ndigits))
        if out and abs(out[-1] - self.stop) > 10 ** (-self.ndigits):
            out[-1] = round(self.stop, self.ndigits)
        return out


@dataclass(frozen=True)
class UniformFloat(Param):
    """Continuous uniform draw in [low, high]. For random/Optuna search."""
    low: float
    high: float
    def grid_values(self) -> Sequence[Any]:
        return []
    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True)
class LogUniform(Param):
    """Draw from a log-uniform distribution between [low, high] (both > 0)."""
    low: float
    high: float
    def grid_values(self) -> Sequence[Any]:
        return []
    def sample(self, rng: random.Random) -> float:
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniform bounds must be > 0")
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))


# ---------------------------------------------------------------------------
# Metrics & Constraints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """
    path: dot path inside final_state['metrics'] (e.g., 'performance.smart_sharpe')
    goal: 'max' to maximize, 'min' to minimize
    weight: weight applied in the aggregated score
    """
    path: str
    goal: str = "max"
    weight: float = 1.0


def metric(path: str, goal: str = "max", weight: float = 1.0) -> MetricSpec:
    return MetricSpec(path=path, goal=goal, weight=weight)


@dataclass(frozen=True)
class Constraint:
    """
    Simple threshold constraint evaluated on metrics.
    op: one of '<=', '<', '>=', '>', '=='
    Example: Constraint('risk.max_drawdown_pct', '<=', 25.0)
    """
    path: str
    op: str
    value: float


ParamSpec = Union[Param, Sequence[Any], Any]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _get_metric_value(metrics: Mapping[str, Any], path: str, default: float = 0.0) -> float:
    cur: Any = metrics
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    try:
        return float(cur)
    except Exception:
        return default


def _isoformat_utc_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_metric(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    return metrics if isinstance(metrics, Mapping) else {}



def _satisfies_constraints(metrics: Mapping[str, Any], constraints: Sequence[Constraint]) -> bool:
    for c in constraints or []:
        v = _get_metric_value(metrics, c.path, default=float("nan"))
        if math.isnan(v):
            return False
        if c.op == "<=" and not (v <= c.value): return False
        if c.op == "<"  and not (v <  c.value): return False
        if c.op == ">=" and not (v >= c.value): return False
        if c.op == ">"  and not (v >  c.value): return False
        if c.op == "==" and not (abs(v - c.value) < 1e-12): return False
    return True


def _score_from_specs(metrics: Mapping[str, Any], specs: Sequence[MetricSpec]) -> float:
    """Aggregate into a single scalar score."""
    score = 0.0
    for spec in specs or []:
        val = _get_metric_value(metrics, spec.path, default=0.0)
        if spec.goal.lower() == "min":
            val = -val
        score += spec.weight * val
    return float(score)


def _resolve_strategy_ctor(strategy_ctor: Union[str, Callable]):
    """Accepts a dotted path or a callable constructor/class."""
    if callable(strategy_ctor):
        return strategy_ctor
    if isinstance(strategy_ctor, str):
        mod_path, cls_name = strategy_ctor.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls
    raise TypeError("strategy_ctor must be a callable or dotted path string")


def _filter_kwargs_for_ctor(ctor: Callable[..., Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop unknown keys when instantiating the strategy."""
    try:
        sig = inspect.signature(ctor)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in params.items() if k in allowed}
    except Exception:
        return dict(params)


def _normalize_param_space(space: Mapping[str, ParamSpec]) -> Dict[str, Param]:
    norm: Dict[str, Param] = {}
    for k, v in (space or {}).items():
        if isinstance(v, Param):
            norm[k] = v
        elif isinstance(v, (list, tuple)):
            norm[k] = Choice(list(v))
        else:
            norm[k] = Fixed(v)
    return norm


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    n_splits: int = 5
    test_ratio: float = 0.2
    mode: str = "anchored"  # "anchored" (expanding) or "rolling" (fixed)
    gap_candles: int = 0  # Gap between train and test to avoid warmup leakage


@dataclass
class HoldoutConfig:
    """Configuration for final holdout validation."""
    holdout_ratio: float = 0.15
    enabled: bool = True


@dataclass
class AntiOverfitConfig:
    """Anti-overfitting measures configuration."""
    min_trades_per_fold: int = 10
    trial_penalty_factor: float = 0.0  # Penalty per sqrt(n_trials), 0 = disabled
    use_pruning: bool = True
    pruning_warmup_steps: int = 2  # Min folds before pruning can trigger
    n_startup_trials: int = 10  # Random trials before TPE kicks in


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Result from a single train/test fold."""
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    test_score: float
    trade_count: int
    valid: bool


@dataclass
class WFATrialResult:
    """Extended trial result with WFA details."""
    trial_id: int
    params: Dict[str, Any]
    oos_score: float
    oos_metrics: Dict[str, Any]
    fold_results: List[FoldResult]
    folds_valid: int
    holdout_score: Optional[float] = None
    holdout_metrics: Optional[Dict[str, Any]] = None
    runtime_sec: float = 0.0
    pruned: bool = False
    error: Optional[str] = None


@dataclass
class OptunaOptimizationResult:
    """Final optimization result."""
    study_name: str
    n_trials: int
    completed_trials: int
    pruned_trials: int
    duration_sec: float
    best_params: Dict[str, Any]
    best_oos_score: float
    best_oos_metrics: Dict[str, Any]
    holdout_score: Optional[float]
    holdout_metrics: Optional[Dict[str, Any]]
    leaderboard: List[WFATrialResult]
    walk_forward_config: WalkForwardConfig
    trial_penalty_applied: float


# ---------------------------------------------------------------------------
# Walk-Forward Splitter
# ---------------------------------------------------------------------------

class WalkForwardSplitter:
    """Generates train/test date splits for walk-forward analysis."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_splits(
        self,
        full_start: datetime,
        full_end: datetime,
        holdout_ratio: float = 0.0,
        timeframe_minutes: int = 60,
    ) -> Tuple[List[Tuple[datetime, datetime, datetime, datetime]], Optional[Tuple[datetime, datetime]]]:
        """
        Generate train/test splits for walk-forward analysis.

        Returns:
            - List of (train_start, train_end, test_start, test_end) tuples
            - (holdout_start, holdout_end) tuple or None if holdout_ratio=0
        """
        total_duration = full_end - full_start
        total_seconds = total_duration.total_seconds()

        # Reserve holdout at the end
        holdout_seconds = total_seconds * holdout_ratio
        optimization_end = full_end - timedelta(seconds=holdout_seconds)

        holdout_period = None
        if holdout_ratio > 0:
            holdout_period = (optimization_end, full_end)

        opt_start = full_start
        opt_end = optimization_end
        opt_seconds = (opt_end - opt_start).total_seconds()

        gap_seconds = self.config.gap_candles * timeframe_minutes * 60
        n = self.config.n_splits

        splits: List[Tuple[datetime, datetime, datetime, datetime]] = []

        if self.config.mode == "anchored":
            # Anchored (expanding window): train grows, test windows are sequential
            # Each fold's test period is roughly (opt_seconds * test_ratio / n_splits)
            test_seconds = opt_seconds * self.config.test_ratio / n

            for i in range(n):
                # Test period ends at fraction (i+1)/n of optimization period
                test_end_offset = opt_seconds * (i + 1) / n
                test_end = opt_start + timedelta(seconds=test_end_offset)
                test_start = test_end - timedelta(seconds=test_seconds)

                # Train goes from start to test_start minus gap
                train_start = opt_start
                train_end = test_start - timedelta(seconds=gap_seconds)

                if train_end <= train_start:
                    continue  # Skip invalid fold

                splits.append((train_start, train_end, test_start, test_end))

        else:  # rolling
            # Rolling (fixed window): both train and test slide forward
            window_seconds = opt_seconds / n
            train_seconds = window_seconds * (1 - self.config.test_ratio)
            test_seconds = window_seconds * self.config.test_ratio

            for i in range(n):
                train_start = opt_start + timedelta(seconds=window_seconds * i)
                train_end = train_start + timedelta(seconds=train_seconds) - timedelta(seconds=gap_seconds)
                test_start = train_start + timedelta(seconds=train_seconds)
                test_end = test_start + timedelta(seconds=test_seconds)

                if train_end <= train_start:
                    continue

                splits.append((train_start, train_end, test_start, test_end))

        return splits, holdout_period


# ---------------------------------------------------------------------------
# Helper: convert Python Indicator objects to Rust spec tuples
# ---------------------------------------------------------------------------

def _indicator_to_spec(ind) -> tuple:
    """Convert a Python Indicator config object to a (type, label, params) tuple
    for ``_rust_core.compute_indicators_bulk``."""
    from .indicators import SMA, EMA, RSI, ATR, BollingerBands, MACD

    label = ind.name()
    if isinstance(ind, SMA):
        return ("sma", label, {"period": ind.period, "source": ind.source})
    elif isinstance(ind, EMA):
        return ("ema", label, {"period": ind._period, "source": ind.source})
    elif isinstance(ind, RSI):
        return ("rsi", label, {"period": ind._period, "source": ind.source})
    elif isinstance(ind, ATR):
        return ("atr", label, {"period": ind._period})
    elif isinstance(ind, BollingerBands):
        return ("bollinger_bands", label, {"period": ind._period, "source": ind.source, "multiplier": ind.multiplier})
    elif isinstance(ind, MACD):
        return ("macd", label, {"fast_period": ind._fast_p, "slow_period": ind._slow_p, "signal_period": ind._sig_p, "source": ind.source})
    else:
        raise ValueError(f"Unsupported indicator type for vectorized path: {type(ind)}")


# ---------------------------------------------------------------------------
# Vectorized Fast-Path Backtest (for VectorStrategy)
# ---------------------------------------------------------------------------

def _run_vectorized_backtest(
    candle_data: CandleDataAggregated,
    session_kwargs: Dict[str, Any],
    strategy_ctor_path: str,
    params: Dict[str, Any],
    start_date: str,
    end_date: str,
    warmup_end: Optional[str] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a backtest using the vectorized signal fast path.

    Instead of calling ``strategy.on_candle()`` per candle, this:
    1. Computes all indicators over the full range in Rust (one call)
    2. Calls ``strategy.signals()`` once with numpy arrays
    3. Runs the full backtest loop in Rust with the signal array (one call)

    Returns the same dict format as ``_run_fast_backtest``.
    """
    import numpy as np
    from ._rust_core import compute_indicators_bulk, run_signals_backtest
    from .metrics import compute_metrics_from_result

    try:
        # Instantiate strategy with trial params
        ctor = _resolve_strategy_ctor(strategy_ctor_path)
        init_params = _filter_kwargs_for_ctor(ctor, params)
        strategy = ctor(**init_params)
    except Exception as exc:
        return {"metrics": {}, "error": f"strategy_init_failed:{exc!r}"}

    # Determine the full range including warmup
    warmup_count = int(session_kwargs.get("warmup_candles", 0))
    timeframe_minutes = int(session_kwargs.get("timeframe_minutes", 60))

    # We need warmup candles before the run start for indicator seeding.
    # Use the train_start as the logical start if available, else start_date.
    train_only = bool(
        train_end
        and _parse_iso8601_utc(train_end) > _parse_iso8601_utc(start_date)
        and (not train_start or _parse_iso8601_utc(train_end) > _parse_iso8601_utc(train_start or start_date))
    )
    logical_start_str = train_start if (train_only and train_start) else start_date

    # Compute warmup start
    warmup_end_dt = _parse_iso8601_utc(warmup_end) if warmup_end else _parse_iso8601_utc(logical_start_str)
    warmup_start_dt = warmup_end_dt - timedelta(minutes=warmup_count * timeframe_minutes)
    warmup_start_unix = int(warmup_start_dt.timestamp())
    end_unix = int(_parse_iso8601_utc(end_date).timestamp())

    # Get the raw numpy arrays from CandleDataAggregated
    start_idx, end_idx = candle_data.find_range_indices(warmup_start_unix, end_unix)
    if end_idx <= start_idx:
        return {"metrics": {}, "error": "no_candles_in_range"}

    # Build 2D candle array: [timestamp, open, high, low, close, vol, qav, trades, tbbav, tbqav]
    timestamps = candle_data.timestamps_unix[start_idx:end_idx]
    values = candle_data.values[start_idx:end_idx]
    n_candles = len(timestamps)

    candle_array = np.empty((n_candles, 10), dtype=np.float64)
    candle_array[:, 0] = timestamps.astype(np.float64)
    candle_array[:, 1:] = values[:, :9]  # OHLCV + extras (up to 9 columns)
    # Pad if values has fewer than 9 columns
    ncols = values.shape[1]
    if ncols < 9:
        candle_array[:, 1 + ncols:] = 0.0

    # Build indicator specs
    try:
        inds = list(strategy.indicators() or [])
        specs = [_indicator_to_spec(ind) for ind in inds]
    except Exception as exc:
        return {"metrics": {}, "error": f"indicator_spec_failed:{exc!r}"}

    # 1. Compute indicators in Rust (one call)
    try:
        indicators = compute_indicators_bulk(candle_array, specs)
    except Exception as exc:
        return {"metrics": {}, "error": f"indicator_compute_failed:{exc!r}"}

    # 2. Generate signals in Python (one call)
    close_array = candle_array[:, 4]
    try:
        signal_array = strategy.signals(close_array, indicators)
        signal_array = np.asarray(signal_array, dtype=np.int8)
    except Exception as exc:
        return {"metrics": {}, "error": f"signals_failed:{exc!r}"}

    # 3. Run the backtest loop in Rust (one call)
    sig_params = strategy.signal_params()
    starting_cash = float(session_kwargs.get("starting_cash", 10000.0))
    try:
        result = run_signals_backtest(
            candle_data=candle_array,
            signals=signal_array,
            starting_cash=starting_cash,
            leverage=float(session_kwargs.get("leverage", 10.0)),
            margin_mode=str(session_kwargs.get("margin_mode", "cross")),
            slippage_pct=float(session_kwargs.get("slippage_pct", 0.0005)),
            margin_pct=float(sig_params.get("margin_pct", 0.1)),
            sl_pct=float(sig_params.get("sl_pct", 0.0)),
            tp_pct=float(sig_params.get("tp_pct", 0.0)),
            close_at_end=True,
            warmup_count=warmup_count,
            timeframe_minutes=timeframe_minutes,
            symbol=str(session_kwargs.get("symbol", "BTCUSDT_PERP")),
        )
    except Exception as exc:
        return {"metrics": {}, "error": f"backtest_failed:{exc!r}"}

    # 4. Compute metrics
    try:
        metrics = compute_metrics_from_result(
            result,
            starting_cash=starting_cash,
            timeframe_minutes=timeframe_minutes,
        )
    except Exception as exc:
        return {"metrics": {}, "error": f"metrics_failed:{exc!r}"}

    return {"metrics": metrics}


# ---------------------------------------------------------------------------
# Fast In-Memory Backtest
# ---------------------------------------------------------------------------

def _run_fast_backtest(
    candle_data: Union[CandleData, CandleDataMemmap, CandleDataAggregated],
    session_kwargs: Dict[str, Any],
    strategy_ctor_path: str,
    params: Dict[str, Any],
    start_date: str,
    end_date: str,
    warmup_end: Optional[str] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> Dict[str, Any]:

    """
    Run a backtest using preloaded candle data.
    This function is designed to be picklable for multiprocessing.

    Args:
        candle_data: Preloaded CandleData, CandleDataMemmap, or CandleDataAggregated object
        session_kwargs: Session configuration
        strategy_ctor_path: Dotted path to strategy class
        params: Strategy parameters
        start_date: Backtest start (ISO8601)
        end_date: Backtest end (ISO8601)

    Returns:
        Final state dict with metrics
    """
    from .engine import iter_candles_from_preloaded
    from . import trading

    # -----------------------------------------------------------------------
    # Dispatch: if the strategy is a VectorStrategy, use the fast path
    # -----------------------------------------------------------------------
    from .strategy import VectorStrategy
    try:
        _ctor = _resolve_strategy_ctor(strategy_ctor_path)
        if isinstance(candle_data, CandleDataAggregated) and issubclass(_ctor, VectorStrategy):
            return _run_vectorized_backtest(
                candle_data, session_kwargs, strategy_ctor_path, params,
                start_date, end_date, warmup_end, train_start, train_end,
            )
    except Exception:
        pass  # Fall through to legacy path on any error

    # Check if we're using the optimized aggregated data
    use_aggregated = isinstance(candle_data, CandleDataAggregated)


    # Create session (but we won't use its file-based iteration)
    sess_kwargs = {
        **session_kwargs,
        "start_date": start_date,
        "end_date": end_date,
        "enable_visual": False,
        "close_at_end": True,
    }
    sess_kwargs.pop("timeframe_minutes", None)
    session = create_session(**sess_kwargs)


    try:
        # Import and instantiate strategy
        ctor = _resolve_strategy_ctor(strategy_ctor_path)
        init_params = _filter_kwargs_for_ctor(ctor, params)
        strategy = ctor(**init_params)

        # Attach session to strategy (required for trading operations)
        if hasattr(strategy, "attach"):
            strategy.attach(session)

        # Build meta info for warmup
        meta = {
            "symbol": session.symbol,
            "timeframe": session.timeframe,
            "timeframe_minutes": session.timeframe_minutes,
            "warmup_candles": session.warmup_candles,
        }

        # Get warmup candles from preloaded data (respect optional warmup_end)
        warmup_count = session.warmup_candles
        warmup_candles: List[Dict[str, Any]] = []
        warmup_end_dt = _parse_iso8601_utc(warmup_end) if warmup_end else _parse_iso8601_utc(start_date)
        if warmup_count > 0:
            warmup_minutes = warmup_count * session.timeframe_minutes
            warmup_start = warmup_end_dt - timedelta(minutes=warmup_minutes)
            # Use fast aggregated iterator if available
            if use_aggregated:
                for item in iter_candles_from_aggregated(
                    candle_data,
                    warmup_start,
                    warmup_end_dt,
                ):
                    warmup_candles.append(cast(Dict[str, Any], item["candle"]))
            else:
                for item in iter_candles_from_preloaded(
                    candle_data,
                    warmup_start,
                    warmup_end_dt,
                    session.timeframe_minutes,
                ):
                    warmup_candles.append(cast(Dict[str, Any], item["candle"]))
            warmup_candles = warmup_candles[-warmup_count:]

        # Register and bootstrap indicators
        try:
            from .indicators import register_indicators
            inds = list(strategy.indicators() or [])
            if inds:
                register_indicators(session, inds)
                # Bootstrap indicators with warmup
                from .strategy import _bootstrap_inds
                _bootstrap_inds(session, warmup_candles, meta)
        except Exception as exc:
            return {"metrics": {}, "error": f"indicator_bootstrap_failed:{exc!r}"}

        # Call strategy warmup callback
        if hasattr(strategy, "on_warmup"):
            try:
                strategy.on_warmup(warmup_candles, meta)
            except Exception as exc:
                return {"metrics": {}, "error": f"strategy_warmup_failed:{exc!r}"}


        # Run backtest using preloaded data
        eng = trading.ensure_engine(session)

        # Try to get indicator update function
        try:
            from .indicators import update_indicators_for_session as _update_inds
        except Exception:
            _update_inds = None

        # Track equity curve and events for metrics
        _equities: List[float] = []
        _times: List[str] = []
        _flat_events: List[Dict[str, Any]] = []

        candle_index = 0
        last_candle: Optional[Dict[str, Any]] = None
        last_state: Optional[Dict[str, Any]] = None

        def _stream_range(range_start: str, range_end: str) -> Iterable[Dict[str, Any]]:
            # Use fast aggregated iterator if available (O(log n) lookup vs O(n))
            if use_aggregated:
                for item in iter_candles_from_aggregated(
                    candle_data,
                    range_start,
                    range_end,
                ):
                    candle = item["candle"]
                    if isinstance(candle, dict):
                        yield cast(Dict[str, Any], candle)
            else:
                for item in iter_candles_from_preloaded(
                    candle_data,
                    range_start,
                    range_end,
                    session.timeframe_minutes,
                ):
                    candle = item["candle"]
                    if isinstance(candle, dict):
                        yield cast(Dict[str, Any], candle)

        train_start_str = train_start or start_date
        train_end_str = train_end or start_date
        train_only = bool(
            train_end
            and _parse_iso8601_utc(train_end) > _parse_iso8601_utc(start_date)
            and _parse_iso8601_utc(train_end) > _parse_iso8601_utc(train_start_str)
        )

        if train_only:
            for candle in _stream_range(train_start_str, train_end_str):
                events, state = trading.on_candle(session, candle, candle_index)
                if _update_inds:
                    try:
                        ind_vals = _update_inds(session, candle, candle_index)
                        if ind_vals:
                            state["indicators"] = ind_vals
                    except Exception:
                        pass
                if hasattr(strategy, "on_candle"):
                    try:
                        strategy.on_candle(candle, state, events)
                    except Exception as exc:
                        return {"metrics": {}, "error": f"strategy_train_failed:{exc!r}"}
                candle_index += 1
                last_candle = candle
                last_state = state

        run_start = start_date
        if train_only:
            run_start_dt = _parse_iso8601_utc(run_start)
            train_end_dt = _parse_iso8601_utc(train_end_str)
            if train_end_dt > run_start_dt:
                run_start = _isoformat_utc_z(train_end_dt)

        for candle in _stream_range(run_start, end_date):
            events, state = trading.on_candle(session, candle, candle_index)

            # Update indicators
            if _update_inds:
                try:
                    ind_vals = _update_inds(session, candle, candle_index)
                    if ind_vals:
                        state["indicators"] = ind_vals
                except Exception:
                    pass

            if hasattr(strategy, "on_candle"):
                try:
                    strategy.on_candle(candle, state, events)
                except Exception as exc:
                    return {"metrics": {}, "error": f"strategy_run_failed:{exc!r}"}

            # Track equity and events for metrics
            eq = state.get("equity")
            if isinstance(eq, (int, float)):
                _equities.append(float(eq))
                _times.append(str(candle.get("open_time")))
            if events:
                _flat_events.extend(list(events))

            candle_index += 1
            last_candle = candle
            last_state = state


        # Force close at end if position is open
        if session.close_at_end and last_candle is not None:
            try:
                snap = eng.snapshot()
                if snap.get("position") is not None:
                    close_val = last_candle.get("close")
                    close_price = float(close_val) if isinstance(close_val, (int, float)) else 0.0

                    eng.close_order(close_price)
                    close_events, close_state = eng.on_candle(last_candle, candle_index)
                    if close_events:
                        _flat_events.extend(close_events)
                    last_state = close_state
            except Exception:
                pass

        # Compute metrics using the same approach as run_strategy
        final_state = last_state or eng.snapshot()
        try:
            from .metrics import compute_metrics
            metrics = compute_metrics(session, _times, _equities, _flat_events, final_state)
            final_state["metrics"] = metrics
        except Exception:
            final_state["metrics"] = {}

        return final_state

    finally:
        # Clean up engine and indicators to prevent memory leaks (always runs)
        try:
            trading.clear_engine(session)
        except Exception:
            pass
        try:
            from .indicators import clear_indicators
            clear_indicators(session)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Optuna Objective
# ---------------------------------------------------------------------------

class OptunaObjective:
    def __init__(
        self,
        session_kwargs: Dict[str, Any],
        strategy_ctor: Union[str, Callable],
        param_space: Dict[str, ParamSpec],
        metric_specs: Sequence[MetricSpec],
        constraints: Sequence[Constraint],
        wfa_config: WalkForwardConfig,
        anti_overfit: AntiOverfitConfig,
        splits: List[Tuple[datetime, datetime, datetime, datetime]],
        candle_data: Optional[Union[CandleData, CandleDataMemmap, CandleDataAggregated]] = None,
    ):

        self.session_kwargs = session_kwargs
        self.strategy_ctor = strategy_ctor
        self.param_space = param_space
        self.metric_specs = list(metric_specs)
        self.constraints = list(constraints)
        self.wfa_config = wfa_config
        self.anti_overfit = anti_overfit
        self.splits = splits
        self.candle_data = candle_data

        # Resolve strategy constructor once
        if callable(strategy_ctor):
            self.ctor_path = f"{strategy_ctor.__module__}.{strategy_ctor.__qualname__}"
        else:
            self.ctor_path = strategy_ctor

    def __call__(self, trial: optuna.Trial) -> float:
        """Run WFA for this trial and return aggregated OOS score."""
        t0 = time.perf_counter()

        try:
            params = self._sample_params(trial)
            return self._run_sequential_folds(trial, params, t0)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            trial.set_user_attr("error", repr(e))
            return float("-inf")

    def _run_sequential_folds(self, trial: optuna.Trial, params: Dict[str, Any], t0: float) -> float:
        """Run folds sequentially (fallback when no preloaded data)."""
        fold_results: List[FoldResult] = []
        oos_scores: List[float] = []

        gap_minutes = self.wfa_config.gap_candles * int(self.session_kwargs.get("timeframe_minutes", 60))

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(self.splits):
            warmup_end = train_end if train_end < test_start else test_start
            if gap_minutes > 0:
                warmup_end = min(warmup_end, test_start - timedelta(minutes=gap_minutes))

            train_start_str = _isoformat_utc_z(train_start)
            train_end_str = _isoformat_utc_z(train_end)
            test_start_str = _isoformat_utc_z(test_start)
            test_end_str = _isoformat_utc_z(test_end)
            warmup_end_str = _isoformat_utc_z(warmup_end)

            # Use fast backtest if we have preloaded data, else fallback
            if self.candle_data is not None:
                test_result = _run_fast_backtest(
                    self.candle_data,
                    self.session_kwargs,
                    self.ctor_path,
                    params,
                    test_start_str,
                    test_end_str,
                    warmup_end=warmup_end_str,
                    train_start=train_start_str,
                    train_end=train_end_str,
                )
            else:
                train_session_kwargs = {
                    **self.session_kwargs,
                    "start_date": train_start_str,
                    "end_date": train_end_str,
                    "enable_visual": False,
                    "close_at_end": True,
                }
                test_session_kwargs = {
                    **self.session_kwargs,
                    "start_date": test_start_str,
                    "end_date": test_end_str,
                    "enable_visual": False,
                    "close_at_end": True,
                }
                train_result = self._run_backtest(train_session_kwargs, params)
                test_result = self._run_backtest(test_session_kwargs, params)
                if train_result.get("error"):
                    test_result = {"metrics": {}, "error": train_result.get("error")}

            error = test_result.get("error")
            test_metrics = _safe_metric(test_result.get("metrics", {}))

            # Check constraints and minimum trades
            trade_count = int(_get_metric_value(test_metrics, "trade.total_trades", 0))
            meets_constraints = _satisfies_constraints(test_metrics, self.constraints)
            meets_trades = trade_count >= self.anti_overfit.min_trades_per_fold
            valid = error is None and meets_constraints and meets_trades

            if error is None and not meets_trades:
                trial.set_user_attr("error", f"min_trades_not_met:{trade_count}")
            elif error is None and not meets_constraints:
                trial.set_user_attr("error", "constraints_not_met")
            elif error is not None:
                trial.set_user_attr("error", error)

            if valid:
                test_score = _score_from_specs(test_metrics, self.metric_specs)
            else:
                test_score = float("-inf")
                if error is None:
                    trial.set_user_attr("runtime_sec", time.perf_counter() - t0)

            fold_result = FoldResult(
                fold_index=fold_idx,
                train_start=train_start_str,
                train_end=train_end_str,
                test_start=test_start_str,
                test_end=test_end_str,
                train_metrics={},  # Skip train metrics to save time
                test_metrics=dict(test_metrics),
                test_score=test_score,
                trade_count=trade_count,
                valid=valid,
            )
            fold_results.append(fold_result)

            if valid:
                oos_scores.append(test_score)

            # Report intermediate value for pruning
            if self.anti_overfit.use_pruning and len(oos_scores) > 0:
                intermediate = sum(oos_scores) / len(oos_scores)
                trial.report(intermediate, fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()


        # Aggregate OOS scores
        if len(oos_scores) == 0:
            trial.set_user_attr("fold_results", [asdict(fr) for fr in fold_results])
            trial.set_user_attr("folds_valid", 0)
            return float("-inf")

        aggregated_score = sum(oos_scores) / len(oos_scores)

        # Store fold results for later analysis
        trial.set_user_attr("fold_results", [asdict(fr) for fr in fold_results])
        trial.set_user_attr("folds_valid", len(oos_scores))
        trial.set_user_attr("oos_metrics", self._aggregate_oos_metrics(fold_results))
        trial.set_user_attr("runtime_sec", time.perf_counter() - t0)

        return aggregated_score

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Map existing Param classes to Optuna's suggest_* methods."""
        params = {}
        for name, param in self.param_space.items():
            if isinstance(param, Fixed):
                params[name] = param.value
            elif isinstance(param, Choice):
                params[name] = trial.suggest_categorical(name, list(param.options))
            elif isinstance(param, IntRange):
                params[name] = trial.suggest_int(name, param.start, param.stop, step=param.step)
            elif isinstance(param, FloatRange):
                params[name] = trial.suggest_float(name, param.start, param.stop, step=param.step)
            elif isinstance(param, UniformFloat):
                params[name] = trial.suggest_float(name, param.low, param.high)
            elif isinstance(param, LogUniform):
                params[name] = trial.suggest_float(name, param.low, param.high, log=True)
            else:
                # Fallback: treat as fixed
                params[name] = param
        return params

    def _run_backtest(self, session_kwargs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest with given params."""
        sess_kwargs = dict(session_kwargs)
        sess_kwargs.pop("timeframe_minutes", None)
        session = create_session(**sess_kwargs)
        ctor = _resolve_strategy_ctor(self.ctor_path)
        init_params = _filter_kwargs_for_ctor(ctor, params)
        strategy = ctor(**init_params)
        try:
            final_state = run_strategy(session, strategy) or {}
        except Exception as exc:
            return {"metrics": {}, "error": f"strategy_run_failed:{exc!r}"}
        return final_state


    def _aggregate_oos_metrics(self, fold_results: List[FoldResult]) -> Dict[str, Any]:
        """Aggregate metrics across all valid OOS folds."""
        valid_folds = [fr for fr in fold_results if fr.valid]
        if not valid_folds:
            return {}

        # Collect metrics
        perf_values: Dict[str, List[float]] = {}
        risk_values: Dict[str, List[float]] = {}
        trade_values: Dict[str, List[float]] = {}

        for fr in valid_folds:
            m = fr.test_metrics
            for key, val in (m.get("performance") or {}).items():
                if isinstance(val, (int, float)) and math.isfinite(val):
                    perf_values.setdefault(key, []).append(val)
            for key, val in (m.get("risk") or {}).items():
                if isinstance(val, (int, float)) and math.isfinite(val):
                    risk_values.setdefault(key, []).append(val)
            for key, val in (m.get("trade") or {}).items():
                if isinstance(val, (int, float)) and math.isfinite(val):
                    trade_values.setdefault(key, []).append(val)

        def mean(vs: List[float]) -> float:
            return sum(vs) / len(vs) if vs else 0.0

        def total(vs: List[float]) -> float:
            return sum(vs)

        def worst(vs: List[float]) -> float:
            return max(vs) if vs else 0.0  # For max_drawdown, higher is worse

        return {
            "performance": {
                "pnl": total(perf_values.get("pnl", [])),
                "win_rate_pct": mean(perf_values.get("win_rate_pct", [])),
                "sharpe": mean(perf_values.get("sharpe", [])),
                "smart_sharpe": mean(perf_values.get("smart_sharpe", [])),
                "sortino": mean(perf_values.get("sortino", [])),
                "smart_sortino": mean(perf_values.get("smart_sortino", [])),
                "calmar": mean(perf_values.get("calmar", [])),
                "omega": mean(perf_values.get("omega", [])),
                "serenity": mean(perf_values.get("serenity", [])),
            },
            "risk": {
                "max_drawdown_pct": worst(risk_values.get("max_drawdown_pct", [])),
                "expectancy": mean(risk_values.get("expectancy", [])),
            },
            "trade": {
                "total_trades": int(total(trade_values.get("total_trades", []))),
                "fee_total": total(trade_values.get("fee_total", [])),
            },
            "_meta": {
                "valid_folds": len(valid_folds),
                "total_folds": len(fold_results),
            }
        }


# ---------------------------------------------------------------------------
# Main Optimizer Class
# ---------------------------------------------------------------------------

class OptunaOptimizer:
    """
    Main optimizer class orchestrating Optuna-based WFA optimization.

    Example usage:
        optimizer = OptunaOptimizer(
            session_kwargs={...},
            strategy_ctor="examples.rsi_reversal.RSIReversalStrategy",
            param_space={"rsi_period": IntRange(7, 21, step=2), ...},
            metrics=[MetricSpec("performance.smart_sharpe", "max", 1.0)],
            wfa_config=WalkForwardConfig(n_splits=5),
            holdout_config=HoldoutConfig(holdout_ratio=0.15),
        )
        result = optimizer.optimize(n_trials=100)
    """

    def __init__(
        self,
        session_kwargs: Dict[str, Any],
        strategy_ctor: Union[str, Callable],
        param_space: Dict[str, ParamSpec],
        metrics: Sequence[MetricSpec] = (MetricSpec("performance.smart_sharpe", "max", 1.0),),
        constraints: Sequence[Constraint] = (),
        wfa_config: Optional[WalkForwardConfig] = None,
        holdout_config: Optional[HoldoutConfig] = None,
        anti_overfit: Optional[AntiOverfitConfig] = None,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
    ):
        self.session_kwargs = dict(session_kwargs)
        self.strategy_ctor = strategy_ctor

        self.param_space = _normalize_param_space(param_space)
        self.metrics = list(metrics)
        self.constraints = list(constraints)
        self.wfa_config = wfa_config or WalkForwardConfig()
        self.holdout_config = holdout_config or HoldoutConfig()
        self.anti_overfit = anti_overfit or AntiOverfitConfig()
        self.storage = storage
        self.study_name = study_name or f"wfa_study_{int(time.time())}"

        # Parse date range from session_kwargs
        self.full_start = _parse_iso8601_utc(session_kwargs["start_date"])
        self.full_end = _parse_iso8601_utc(session_kwargs["end_date"])
        self.timeframe_minutes = int(session_kwargs.get("timeframe_minutes", 60))

        # If timeframe is specified as string, parse it
        if "timeframe" in session_kwargs and "timeframe_minutes" not in session_kwargs:
            from .engine import _parse_timeframe_to_minutes
            self.timeframe_minutes = _parse_timeframe_to_minutes(session_kwargs["timeframe"])

        self.session_kwargs.pop("timeframe_minutes", None)
        self.session_kwargs["timeframe_minutes"] = self.timeframe_minutes


        # Generate splits
        splitter = WalkForwardSplitter(self.wfa_config)
        holdout_ratio = self.holdout_config.holdout_ratio if self.holdout_config.enabled else 0.0
        self.splits, self.holdout_period = splitter.generate_splits(
            self.full_start,
            self.full_end,
            holdout_ratio=holdout_ratio,
            timeframe_minutes=self.timeframe_minutes,
        )

        if not self.splits:
            raise ValueError("No valid walk-forward splits could be generated. Check date range and config.")

    def _create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner."""
        pruner = None
        if self.anti_overfit.use_pruning:
            pruner = MedianPruner(
                n_startup_trials=self.anti_overfit.n_startup_trials,
                n_warmup_steps=self.anti_overfit.pruning_warmup_steps,
            )

        sampler = TPESampler(
            n_startup_trials=self.anti_overfit.n_startup_trials,
            multivariate=True,
        )

        return optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True,
        )

    def optimize(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True,
        verbose: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ) -> OptunaOptimizationResult:
        """
        Run the optimization.

        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
            n_jobs: Number of parallel jobs (1 = sequential)
            show_progress_bar: Show Optuna progress bar
            verbose: Print progress messages
            callbacks: Optional list of Optuna callbacks

        Returns:
            OptunaOptimizationResult with best params and metrics
        """
        t0 = time.perf_counter()

        if verbose:
            print(f"[OptunaOptimizer] Starting optimization")
            print(f"  Study: {self.study_name}")
            print(f"  Walk-forward splits: {len(self.splits)}")
            print(f"  Mode: {self.wfa_config.mode}")
            print(f"  Holdout: {self.holdout_config.holdout_ratio*100:.1f}%" if self.holdout_config.enabled else "  Holdout: disabled")
            print(f"  Trials: {n_trials}")
            print(f"  Workers: {n_jobs if n_jobs > 0 else f'all cores ({multiprocessing.cpu_count()})'}")

        # Normalize n_jobs: Optuna expects -1 for all cores, not 0 or other negatives
        if n_jobs <= 0:
            n_jobs = -1

        candle_data: Optional[Union[CandleData, CandleDataMemmap, CandleDataAggregated]] = None
        warmup_candles = int(self.session_kwargs.get("warmup_candles", 0))
        base_timeframe = str(self.session_kwargs.get("base_timeframe", "1m"))
        data_dir = self.session_kwargs.get("data_dir")

        # ALWAYS use pre-aggregated data - it's faster for both sequential and parallel
        # Pre-aggregation happens ONCE here, not on every trial/fold
        if verbose:
            print("  Pre-aggregating candle data (one-time cost)...")

        import time as _time
        _agg_start = _time.perf_counter()

        candle_data = preload_candle_data_aggregated(
            symbol=self.session_kwargs["symbol"],
            start_date=self.full_start,
            end_date=self.full_end,
            base_timeframe=base_timeframe,
            target_timeframe_minutes=self.timeframe_minutes,
            data_dir=data_dir,
            warmup_candles=warmup_candles,
            session_start=self.full_start,
        )

        _agg_elapsed = _time.perf_counter() - _agg_start

        if verbose:
            print(f"  Pre-aggregated {len(candle_data):,} candles to {self.timeframe_minutes}m in {_agg_elapsed:.2f}s")
            print(f"  Using binary search for O(log n) time-range lookups")
            print()


        # Create study
        study = self._create_study()
        self.study = study

        # Create objective with preloaded data
        objective = OptunaObjective(
            session_kwargs=self.session_kwargs,
            strategy_ctor=self.strategy_ctor,
            param_space=self.param_space,
            metric_specs=self.metrics,
            constraints=self.constraints,
            wfa_config=self.wfa_config,
            anti_overfit=self.anti_overfit,
            splits=self.splits,
            candle_data=candle_data,
        )



        # Suppress Optuna logs if not verbose
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks,
        )

        # Gather results
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        if not completed_trials:
            raise RuntimeError("No trials completed successfully")

        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_oos_score = best_trial.value
        best_oos_metrics = best_trial.user_attrs.get("oos_metrics", {})

        # Apply trial penalty if configured
        trial_penalty = 0.0
        if self.anti_overfit.trial_penalty_factor > 0:
            trial_penalty = self.anti_overfit.trial_penalty_factor * math.sqrt(len(study.trials))
            best_oos_score -= trial_penalty

        # Evaluate best params on holdout (using fast backtest with preloaded data)
        holdout_score = None
        holdout_metrics = None
        if self.holdout_config.enabled and self.holdout_period:
            if verbose:
                print(f"\n[OptunaOptimizer] Evaluating best params on holdout...")

            holdout_start, holdout_end = self.holdout_period
            holdout_start_str = _isoformat_utc_z(holdout_start)
            holdout_end_str = _isoformat_utc_z(holdout_end)

            if candle_data is not None:
                holdout_result = _run_fast_backtest(
                    candle_data,
                    self.session_kwargs,
                    objective.ctor_path,
                    best_params,
                    holdout_start_str,
                    holdout_end_str,
                )
            else:
                holdout_session_kwargs = {
                    **self.session_kwargs,
                    "start_date": holdout_start_str,
                    "end_date": holdout_end_str,
                    "enable_visual": False,
                    "close_at_end": True,
                }
                holdout_result = objective._run_backtest(holdout_session_kwargs, best_params)

            holdout_metrics = holdout_result.get("metrics", {})

            if _satisfies_constraints(holdout_metrics, self.constraints):
                holdout_score = _score_from_specs(holdout_metrics, self.metrics)
            else:
                holdout_score = float("-inf")


            if verbose:
                print(f"  Holdout score: {holdout_score:.4f}")
                if holdout_score != float("-inf") and best_oos_score != float("-inf"):
                    ratio = holdout_score / (best_oos_score + trial_penalty) if (best_oos_score + trial_penalty) != 0 else 0
                    print(f"  Holdout/OOS ratio: {ratio:.2f}")
                    if ratio < 0.7:
                        print("  WARNING: Significant performance degradation on holdout!")

        # Build leaderboard
        leaderboard: List[WFATrialResult] = []
        for trial in sorted(completed_trials, key=lambda t: t.value or float("-inf"), reverse=True)[:20]:
            fold_results_raw = trial.user_attrs.get("fold_results", [])
            fold_results = [FoldResult(**fr) for fr in fold_results_raw]

            leaderboard.append(WFATrialResult(
                trial_id=trial.number,
                params=dict(trial.params),
                oos_score=trial.value or float("-inf"),
                oos_metrics=trial.user_attrs.get("oos_metrics", {}),
                fold_results=fold_results,
                folds_valid=trial.user_attrs.get("folds_valid", 0),
                runtime_sec=trial.user_attrs.get("runtime_sec", 0.0),
                pruned=False,
                error=trial.user_attrs.get("error"),
            ))

        duration = time.perf_counter() - t0

        if verbose:
            print(f"\n[OptunaOptimizer] Optimization complete")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Completed trials: {len(completed_trials)}")
            print(f"  Pruned trials: {len(pruned_trials)}")
            print(f"  Best OOS score: {best_oos_score:.4f}" + (f" (penalty: {trial_penalty:.4f})" if trial_penalty > 0 else ""))
            print(f"  Best params: {best_params}")

        return OptunaOptimizationResult(
            study_name=self.study_name,
            n_trials=n_trials,
            completed_trials=len(completed_trials),
            pruned_trials=len(pruned_trials),
            duration_sec=duration,
            best_params=best_params,
            best_oos_score=best_oos_score,
            best_oos_metrics=best_oos_metrics,
            holdout_score=holdout_score,
            holdout_metrics=holdout_metrics,
            leaderboard=leaderboard,
            walk_forward_config=self.wfa_config,
            trial_penalty_applied=trial_penalty,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def optimize_with_wfa(
    session_kwargs: Dict[str, Any],
    strategy_ctor: Union[str, Callable],
    param_space: Dict[str, ParamSpec],
    metrics: Sequence[MetricSpec] = (MetricSpec("performance.smart_sharpe", "max", 1.0),),
    constraints: Sequence[Constraint] = (),
    n_splits: int = 5,
    test_ratio: float = 0.2,
    holdout_ratio: float = 0.15,
    mode: str = "anchored",
    n_trials: int = 100,
    n_jobs: int = 1,
    storage: Optional[str] = None,
    verbose: bool = True,
) -> OptunaOptimizationResult:
    """
    Convenience function for quick WFA optimization.

    Example:
        result = optimize_with_wfa(
            session_kwargs={
                "symbol": "BTCUSDT_PERP",
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2025-01-01T00:00:00Z",
                "starting_cash": 10000.0,
                "leverage": 10.0,
                "margin_mode": "cross",
                "warmup_candles": 50,
                "timeframe": "1h",
            },
            strategy_ctor="examples.sma_cross.SMACrossStrategy",
            param_space={
                "fast": IntRange(5, 20, step=1),
                "slow": IntRange(15, 50, step=5),
                "margin_pct": FloatRange(0.1, 0.5, step=0.1),
            },
            n_trials=50,
        )
    """
    optimizer = OptunaOptimizer(
        session_kwargs=session_kwargs,
        strategy_ctor=strategy_ctor,
        param_space=param_space,
        metrics=metrics,
        constraints=constraints,
        wfa_config=WalkForwardConfig(
            n_splits=n_splits,
            test_ratio=test_ratio,
            mode=mode,
        ),
        holdout_config=HoldoutConfig(
            holdout_ratio=holdout_ratio,
            enabled=holdout_ratio > 0,
        ),
        storage=storage,
    )
    return optimizer.optimize(n_trials=n_trials, n_jobs=n_jobs, verbose=verbose)
