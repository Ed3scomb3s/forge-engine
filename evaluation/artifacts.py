from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from forge_engine.engine import (
    _isoformat_z,
    _parse_iso8601_utc,
    _parse_timeframe_to_minutes,
    get_warmup_candles,
    step_session,
    step_session_single_pass,
)
from forge_engine.indicators import (
    clear_indicators as _clear_indicators,
    register_indicators,
    bootstrap_indicators_for_session as _bootstrap_inds,
)
from forge_engine.metrics import compute_metrics
from forge_engine.trading import clear_engine as _clear_engine


def _round_list(values: List[float], ndigits: int) -> List[float]:
    return [round(float(v), ndigits) for v in values]


def _returns_from_equities(equities: List[float]) -> List[float]:
    if len(equities) < 2:
        return []
    eq = np.asarray(equities, dtype=np.float64)
    prev = np.maximum(eq[:-1], 1e-10)
    rets = eq[1:] / prev - 1.0
    return [float(x) for x in rets]


def _equity_metrics(equities: List[float], timeframe: str) -> Dict[str, float]:
    if not equities:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "pnl_pct": 0.0,
            "final_equity": 0.0,
            "n_active_steps": 0,
        }

    eq = np.asarray(equities, dtype=np.float64)
    returns = np.asarray(_returns_from_equities(equities), dtype=np.float64)
    minutes = _parse_timeframe_to_minutes(timeframe)
    periods_per_year = (365.25 * 24.0 * 60.0) / float(minutes)
    ann = np.sqrt(periods_per_year)

    if len(returns) < 10 or float(np.std(returns)) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(np.mean(returns) / np.std(returns) * ann)

    neg_returns = returns[returns < 0]
    if len(neg_returns) > 0 and float(np.std(neg_returns)) > 1e-12:
        sortino = float(np.mean(returns) / np.std(neg_returns) * ann)
    else:
        sortino = sharpe

    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / np.maximum(running_max, 1e-10)
    max_dd_pct = float(np.max(drawdowns) * 100.0) if len(drawdowns) else 0.0
    pnl_pct = float((eq[-1] / eq[0] - 1.0) * 100.0) if eq[0] > 0 else 0.0
    n_active = int(np.count_nonzero(np.abs(np.diff(eq)) > 0.01))

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd_pct, 2),
        "pnl_pct": round(pnl_pct, 2),
        "final_equity": round(float(eq[-1]), 2),
        "n_active_steps": n_active,
    }


def build_equity_times(start_iso: str, periods: int, timeframe: str) -> List[str]:
    if periods <= 0:
        return []
    start_dt = _parse_iso8601_utc(start_iso)
    minutes = _parse_timeframe_to_minutes(timeframe)
    return [
        _isoformat_z(start_dt + timedelta(minutes=i * minutes))
        for i in range(periods)
    ]


def evaluate_rl_model_on_period(
    model,
    env,
    *,
    starting_cash: float,
    start_iso: str,
    end_iso: str,
    timeframe: str,
    seed: int = 42,
    deterministic: bool = True,
    max_steps_guard: int = 200_000,
) -> Dict[str, Any]:
    obs, _info = env.reset(seed=seed)

    equities: List[float] = [float(starting_cash)]
    done = False
    steps = 0
    liquidated = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        equities.append(float(info.get("equity", equities[-1])))
        liquidated = liquidated or bool(info.get("is_liquidated"))
        done = bool(terminated or truncated)
        steps += 1
        if steps > max_steps_guard:
            break

    metrics = _equity_metrics(equities, timeframe)
    metrics["total_steps"] = steps
    metrics["liquidated"] = liquidated
    metrics["period"] = [start_iso, end_iso]
    metrics["equities"] = _round_list(equities, 6)
    metrics["returns"] = _round_list(_returns_from_equities(equities), 8)
    metrics["equity_times"] = build_equity_times(start_iso, len(equities), timeframe)
    return metrics


def run_strategy_with_artifacts(session, strategy, auto_register_indicators: bool = True) -> Dict[str, Any]:
    strategy.attach(session)

    if auto_register_indicators:
        try:
            inds = list(strategy.indicators() or [])
            if inds:
                register_indicators(session, inds)
        except Exception:
            pass

    meta = {
        "symbol": getattr(session, "symbol", None),
        "timeframe": getattr(session, "timeframe", None),
        "timeframe_minutes": getattr(session, "timeframe_minutes", None),
        "warmup_candles": getattr(session, "warmup_candles", 0),
    }

    last_state: Optional[dict] = None
    equities: List[float] = []
    times: List[str] = []
    flat_events: List[Dict[str, Any]] = []
    used_single_pass = False

    try:
        gen = step_session_single_pass(session)
        first = next(gen, None)
        warmups = []
        if isinstance(first, dict) and "warmups" in first:
            warmups = list(first.get("warmups") or [])
        try:
            strategy.on_warmup(warmups, meta)
        except Exception:
            pass
        try:
            _bootstrap_inds(session, warmups, meta)
        except Exception:
            pass

        used_single_pass = True
        for tick in gen:
            try:
                strategy.on_candle(tick.get("candle"), tick.get("state"), tick.get("events"))
            except Exception:
                pass
            last_state = tick.get("state") or last_state
            state = tick.get("state") or {}
            candle = tick.get("candle") or {}
            eq = state.get("equity")
            if eq is not None:
                equities.append(float(eq))
                times.append(str(candle.get("open_time")))
            events = tick.get("events") or []
            if events:
                flat_events.extend(list(events))
    except Exception:
        used_single_pass = False

    if not used_single_pass:
        warmups = []
        try:
            warmups = get_warmup_candles(session)
        except Exception:
            pass
        try:
            strategy.on_warmup(warmups, meta)
        except Exception:
            pass

        for tick in step_session(session):
            try:
                strategy.on_candle(tick.get("candle"), tick.get("state"), tick.get("events"))
            except Exception:
                pass
            last_state = tick.get("state") or last_state
            state = tick.get("state") or {}
            candle = tick.get("candle") or {}
            eq = state.get("equity")
            if eq is not None:
                equities.append(float(eq))
                times.append(str(candle.get("open_time")))
            events = tick.get("events") or []
            if events:
                flat_events.extend(list(events))

    metrics: Dict[str, Any] = {}
    if last_state is not None:
        try:
            metrics = compute_metrics(session, times, equities, flat_events, last_state)
            last_state["metrics"] = metrics
        except Exception:
            metrics = {}

    try:
        _clear_engine(session)
    except Exception:
        pass
    try:
        _clear_indicators(session)
    except Exception:
        pass

    return {
        "final_state": last_state,
        "metrics": metrics,
        "equity_times": times,
        "equities": _round_list(equities, 6),
        "returns": _round_list(_returns_from_equities(equities), 8),
        "events": flat_events,
    }
