"""ForgeEnv — Gymnasium environment backed by the Rust fast-path engine.

Phase 7: the entire candle loop, observation computation, reward calculation,
and action translation now happen in compiled Rust via ``step_rl()``.  This
gives a ~50x throughput gain over the previous pure-Python generator approach.

The Python side is responsible only for:
  * Pre-loading and caching the candle dataset (once)
  * Creating a fresh ``RustTradingEngine`` on each ``reset()``
  * Feeding warmup candles to seed indicators
  * Forwarding the flat ``[f64; 10]`` candle arrays to ``step_rl()``
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from forge_engine._rust_core import RustTradingEngine
from forge_engine.engine import (
    create_session,
    get_intra_candles,
    preload_candle_data_aggregated,
    _parse_iso8601_utc,
    _parse_timeframe_to_minutes,
    _isoformat_z,
)
from forge_engine.indicators import (
    Indicator,
    SMA,
    EMA,
    RSI,
    ATR,
    BollingerBands,
    MACD,
    _register_indicator_on_engine,
)

from .actions import (
    ActionSpace,
    ContinuousActions,
    DiscreteActions,
    DiscreteActionsWithSizing,
)
from .observations import ObsFeature, build_observation_space, resolve_features
from .rewards import RewardFunction, resolve_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_type_and_config(asd: ActionSpace) -> Tuple[str, dict]:
    """Convert a Python ActionSpace to the (type_str, config_dict) pair that
    Rust's ``setup_rl_env`` expects."""
    if isinstance(asd, DiscreteActions):
        return "discrete", {
            "margin_pct": asd.margin_pct,
            "sl_pct": asd.sl_pct,
            "tp_pct": asd.tp_pct,
            "actions": list(asd.actions),
        }
    if isinstance(asd, DiscreteActionsWithSizing):
        return "discrete_with_sizing", {
            "sizes": list(asd.sizes.values()),
            "sl_pct": asd.sl_pct,
            "tp_pct": asd.tp_pct,
        }
    if isinstance(asd, ContinuousActions):
        return "continuous", {
            "max_margin_pct": asd.max_margin_pct,
            "threshold": asd.threshold,
            "sl_pct": asd.sl_pct,
            "tp_pct": asd.tp_pct,
        }
    raise ValueError(f"Unsupported ActionSpace type: {type(asd)}")


def _reward_type_and_config(spec: Any) -> Tuple[str, dict]:
    """Convert a reward specification (string, RewardFunction, or callable) to
    the (type_str, config_dict) pair that Rust's ``setup_rl_env`` expects."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, RewardFunction):
        name = spec.name()
        cfg: dict = {}
        for attr in (
            "eta", "scale", "drawdown_penalty", "min_std",
            "pnl_weight", "time_penalty", "sl_penalty",
            "tp_bonus", "liq_penalty",
        ):
            if hasattr(spec, attr):
                cfg[attr] = getattr(spec, attr)
        return name, cfg
    # Fallback
    return "differential_sharpe", {}


def _obs_spec_strings(observations: Sequence[Any]) -> List[str]:
    """Normalise a user-facing observation spec list into plain strings."""
    out: List[str] = []
    for s in observations:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, ObsFeature):
            out.append(s.name())
        else:
            out.append(str(s))
    return out


def _extract_action_val(action: Any) -> float:
    """Pull a scalar float from whatever Gymnasium hands us."""
    if isinstance(action, np.ndarray):
        return float(action.item()) if action.size == 1 else float(action[0])
    if isinstance(action, (list, tuple)):
        return float(action[0])
    return float(action)


def _parse_bb_context_spec(spec: str) -> Optional[Tuple[int, int]]:
    parts = spec.strip().lower().split("_")
    if len(parts) == 4 and parts[0] == "bb" and parts[1] == "context":
        try:
            return int(parts[2]), int(parts[3])
        except ValueError:
            return None
    return None


def _parse_macd_spec(spec: str) -> Optional[Tuple[int, int, int]]:
    parts = spec.strip().lower().split("_")
    if len(parts) == 2 and parts[0] == "macd" and parts[1] in ("line", "signal", "hist"):
        return 12, 26, 9
    if (
        len(parts) == 5
        and parts[0] == "macd"
        and parts[1] in ("line", "signal", "hist")
    ):
        try:
            return int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# ForgeEnv
# ---------------------------------------------------------------------------

class ForgeEnv(gym.Env):
    """Gymnasium environment for RL trading with the Rust-accelerated engine.

    All heavy computation (state machine, indicators, observations, reward)
    lives in compiled Rust.  The Python wrapper is a thin orchestrator that
    pre-loads data, spins up a ``RustTradingEngine`` per episode, and streams
    flat candle arrays through ``step_rl()``.

    Parameters are identical to the previous pure-Python ``ForgeEnv``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol: str = "BTCUSDT_PERP",
        start_date: str = "2024-01-01T00:00:00Z",
        end_date: str = "2025-01-01T00:00:00Z",
        observations: Optional[Sequence[Any]] = None,
        actions: Optional[ActionSpace] = None,
        reward: Any = "differential_sharpe",
        starting_cash: float = 10000.0,
        leverage: float = 10.0,
        margin_mode: str = "isolated",
        timeframe: str = "1h",
        warmup_candles: int = 50,
        indicators: Optional[List[Indicator]] = None,
        max_steps: int = 0,
        slippage_pct: float = 0.0,
        close_at_end: bool = True,
        data_dir: Optional[str] = None,
        base_timeframe: str = "1m",
    ):
        super().__init__()

        # ---- session-level config ----
        self._symbol = symbol
        self._start_date = start_date
        self._end_date = end_date
        self._starting_cash = starting_cash
        self._leverage = leverage
        self._margin_mode = margin_mode
        self._timeframe = timeframe
        self._tf_minutes = _parse_timeframe_to_minutes(timeframe)
        self._warmup_candles = warmup_candles
        self._slippage_pct = slippage_pct
        self._close_at_end = close_at_end
        self._data_dir = data_dir
        self._base_timeframe = base_timeframe
        self._max_steps = max_steps
        self._indicators_list: List[Indicator] = list(indicators or [])

        # ---- resolve specs for Rust ----
        raw_obs = observations or ["ohlcv", "position_info"]
        self._obs_strings = _obs_spec_strings(raw_obs)
        self._action_space_def: ActionSpace = actions or DiscreteActions()
        self._action_type, self._action_config = _action_type_and_config(
            self._action_space_def
        )
        self._reward_type, self._reward_config = _reward_type_and_config(reward)

        # ---- gymnasium spaces (set from Python feature specs) ----
        obs_features = resolve_features(raw_obs)
        obs_size, obs_low, obs_high = build_observation_space(obs_features)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(obs_size,), dtype=np.float32,
        )
        self.action_space = self._action_space_def.gymnasium_space()

        # ---- cached dataset (lazy-loaded on first reset) ----
        self._candle_data = None          # CandleDataAggregated
        self._intra_tuples = None         # List[(int, float, float)] | None
        self._funding_data = None         # Dict[str, float] | None
        self._resolved_data_dir = None    # str

        # ---- per-episode state ----
        self._engine: Optional[RustTradingEngine] = None
        self._ep_timestamps: Optional[np.ndarray] = None  # int64
        self._ep_values: Optional[np.ndarray] = None       # float64 (n, 9)
        self._n_candles: int = 0
        self._current_step: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------
    # Data loading (called once, cached across episodes)
    # ------------------------------------------------------------------

    def _ensure_data_loaded(self) -> None:
        if self._candle_data is not None:
            return

        # Create a throwaway session to resolve data_dir and funding CSV
        cfg = dict(
            symbol=self._symbol,
            start_date=self._start_date,
            end_date=self._end_date,
            starting_cash=self._starting_cash,
            leverage=self._leverage,
            margin_mode=self._margin_mode,
            timeframe=self._timeframe,
            warmup_candles=self._warmup_candles,
            slippage_pct=self._slippage_pct,
            close_at_end=self._close_at_end,
            enable_visual=False,
            base_timeframe=self._base_timeframe,
        )
        if self._data_dir is not None:
            cfg["data_dir"] = self._data_dir
        tmp_session = create_session(**cfg)
        self._funding_data = getattr(tmp_session, "funding_data", None)
        self._resolved_data_dir = tmp_session.data_dir

        # Pre-aggregate candles for the full date range (includes warmup buffer)
        self._candle_data = preload_candle_data_aggregated(
            symbol=self._symbol,
            start_date=self._start_date,
            end_date=self._end_date,
            base_timeframe=self._base_timeframe,
            target_timeframe_minutes=self._tf_minutes,
            data_dir=self._resolved_data_dir,
            warmup_candles=self._warmup_candles,
        )

        # Pre-load 1 m intra-candle data for trigger resolution (if TF > 1 m)
        if self._tf_minutes > 1:
            try:
                start_iso = _isoformat_z(_parse_iso8601_utc(self._start_date))
                end_iso = _isoformat_z(_parse_iso8601_utc(self._end_date))
                raw = get_intra_candles(tmp_session, start_iso, end_iso)
                self._intra_tuples = [
                    (
                        int(_parse_iso8601_utc(c["open_time"]).timestamp()),
                        float(c["low"]),
                        float(c["high"]),
                    )
                    for c in raw
                    if c.get("open_time")
                ]
            except Exception:
                self._intra_tuples = None

    # ------------------------------------------------------------------
    # Indicator registration
    # ------------------------------------------------------------------

    def _register_indicators(self, eng: RustTradingEngine) -> None:
        """Register user-provided + auto-inferred indicators on *eng*."""
        seen: set = set()

        # User-provided
        for ind in self._indicators_list:
            label = ind.name()
            if label not in seen:
                seen.add(label)
                _register_indicator_on_engine(eng, ind)

        # Auto-infer from obs spec strings
        for spec in self._obs_strings:
            s = spec.strip().lower()

            if s.startswith("rsi_") and s not in seen:
                period = int(s.split("_")[1])
                eng.register_rsi(f"RSI({period})[close]", period, "close")
                seen.add(s)

            elif s.startswith("ema_") and s not in seen:
                period = int(s.split("_")[1])
                eng.register_ema(f"EMA({period})[close]", period, "close")
                seen.add(s)

            elif s.startswith("atr_") and s not in seen:
                period = int(s.split("_")[1])
                eng.register_atr(f"ATR({period})", period)
                seen.add(s)

            elif s.startswith("sma_ratio_"):
                parts = s.split("_")
                if len(parts) == 4:
                    fast, slow = int(parts[2]), int(parts[3])
                    k_f, k_s = f"sma_{fast}", f"sma_{slow}"
                    if k_f not in seen:
                        eng.register_sma(f"SMA({fast})[close]", fast, "close")
                        seen.add(k_f)
                    if k_s not in seen:
                        eng.register_sma(f"SMA({slow})[close]", slow, "close")
                        seen.add(k_s)

            elif s.startswith("sma_") and s not in seen:
                period = int(s.split("_")[1])
                eng.register_sma(f"SMA({period})[close]", period, "close")
                seen.add(s)

            else:
                bb_spec = _parse_bb_context_spec(s)
                if bb_spec:
                    period, multiplier = bb_spec
                    key = f"bb_{period}_{multiplier}"
                    if key not in seen:
                        eng.register_bollinger_bands(
                            f"BB({period},{multiplier})[close]",
                            period,
                            float(multiplier),
                            "close",
                        )
                        seen.add(key)
                    continue

                macd_spec = _parse_macd_spec(s)
                if macd_spec:
                    fast, slow, signal = macd_spec
                    key = f"macd_{fast}_{slow}_{signal}"
                    if key not in seen:
                        eng.register_macd(
                            f"MACD({fast},{slow},{signal})[close]",
                            fast,
                            slow,
                            signal,
                            "close",
                        )
                        seen.add(key)

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._ensure_data_loaded()

        cd = self._candle_data
        if cd is None or len(cd) == 0:
            self._done = True
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"step": 0}

        # Determine episode window (optionally randomised)
        start_dt = _parse_iso8601_utc(self._start_date)
        end_dt = _parse_iso8601_utc(self._end_date)

        if self._max_steps > 0:
            ep_dur = timedelta(minutes=self._max_steps * self._tf_minutes)
            if (end_dt - start_dt) > ep_dur:
                max_start = end_dt - ep_dur
                rand_start = start_dt + (max_start - start_dt) * random.random()
                start_dt = rand_start
                end_dt = rand_start + ep_dur

        start_unix = int(start_dt.timestamp())
        end_unix = int(end_dt.timestamp())

        # Binary-search the preloaded data for warmup / trading indices
        trade_s, trade_e = cd.find_range_indices(start_unix, end_unix)
        warmup_s = max(0, trade_s - self._warmup_candles)

        self._ep_timestamps = cd.timestamps_unix[trade_s:trade_e]
        self._ep_values = cd.values[trade_s:trade_e]
        self._n_candles = len(self._ep_timestamps)

        if self._n_candles == 0:
            self._done = True
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"step": 0}

        warmup_ts = cd.timestamps_unix[warmup_s:trade_s]
        warmup_vals = cd.values[warmup_s:trade_s]

        # ---- fresh Rust engine ----
        self._engine = RustTradingEngine(
            symbol=self._symbol,
            margin_mode=self._margin_mode,
            leverage=self._leverage,
            starting_cash=self._starting_cash,
            slippage_pct=self._slippage_pct,
            timeframe_minutes=self._tf_minutes,
            funding_data=self._funding_data,
        )

        if self._intra_tuples:
            self._engine.load_intra_candles(self._intra_tuples)

        self._register_indicators(self._engine)

        # ---- seed indicators with warmup candles ----
        for i in range(len(warmup_ts)):
            row = warmup_vals[i]
            ts = int(warmup_ts[i])
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            warmup_dict = {
                "open_time": dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "open": float(row[0]),
                "high": float(row[1]),
                "low": float(row[2]),
                "close": float(row[3]),
                "volume": float(row[4]),
                "quote_asset_volume": float(row[5]),
                "number_of_trades": int(row[6]),
                "taker_buy_base_asset_volume": float(row[7]),
                "taker_buy_quote_asset_volume": float(row[8]),
            }
            self._engine.on_candle(warmup_dict, -1)

        # ---- configure RL fast-path ----
        self._engine.setup_rl_env(
            self._action_type,
            self._obs_strings,
            self._reward_type,
            self._action_config or None,
            self._reward_config or None,
        )
        self._engine.reset_rl_env()

        # Update observation space from Rust bounds (authoritative)
        low, high = self._engine.get_obs_bounds()
        self.observation_space = gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

        # ---- initial observation (hold on first candle) ----
        self._current_step = 0
        self._done = False

        candle_arr = self._candle_array(0)
        obs, _reward, _term, _trunc, info = self._engine.step_rl(0.0, candle_arr)

        return (
            np.asarray(obs, dtype=np.float32),
            {"step": 0, "equity": self._starting_cash},
        )

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                True,
                False,
                {},
            )

        self._current_step += 1

        # Dataset exhaustion → truncate
        if self._current_step >= self._n_candles:
            self._done = True
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0,
                False,
                True,
                {"step": self._current_step, "truncated_reason": "dataset_exhausted"},
            )

        action_val = _extract_action_val(action)
        candle_arr = self._candle_array(self._current_step)

        obs, reward, terminated, truncated, info = self._engine.step_rl(
            action_val, candle_arr,
        )

        # Max-steps truncation
        if self._max_steps > 0 and self._current_step >= self._max_steps:
            truncated = True

        if terminated or truncated:
            self._done = True

        info_out: dict = {"step": self._current_step}
        if isinstance(info, dict):
            info_out.update(info)

        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            info_out,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candle_array(self, idx: int) -> list:
        """Build the ``[f64; 10]`` flat array from preloaded numpy data."""
        ts = float(self._ep_timestamps[idx])
        row = self._ep_values[idx]
        return [
            ts,             # open_time (unix seconds)
            float(row[0]),  # open
            float(row[1]),  # high
            float(row[2]),  # low
            float(row[3]),  # close
            float(row[4]),  # volume
            float(row[5]),  # quote_asset_volume
            float(row[6]),  # number_of_trades
            float(row[7]),  # taker_buy_base_asset_volume
            float(row[8]),  # taker_buy_quote_asset_volume
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self._engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
