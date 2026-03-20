from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Dict, List, Optional


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_dt(s: str) -> Optional[datetime]:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


@dataclass
class _Series:
    times: List[str]
    equity: List[float]
    returns: List[float]
    dd: List[float]
    max_dd: float
    ulcer: float
    periods_per_year: float
    years: float


def _build_series(session, times: List[str], equities: List[float]) -> _Series:
    eq = [e for e in map(_to_float, equities) if e == e]
    ts = list(times)
    rets: List[float] = []
    for i in range(1, len(eq)):
        if eq[i - 1] > 0:
            rets.append(eq[i] / eq[i - 1] - 1.0)
    high = eq[0] if eq else 0.0
    dds: List[float] = []
    for e in eq:
        if high <= 0:
            dds.append(0.0)
            high = max(high, e)
            continue
        high = max(high, e)
        dds.append((e / high) - 1.0)
    max_dd = min(dds) if dds else 0.0
    ulcer = sqrt(sum((d * 100.0) ** 2 for d in dds if d < 0) / max(1, len(dds)))
    ppyr = (365.25 * 24.0 * 60.0) / float(getattr(session, "timeframe_minutes", 1) or 1)
    dt0 = getattr(session, "start_date", None)
    dt1 = getattr(session, "end_date", None)
    years = ((dt1 - dt0).total_seconds() / (365.25 * 24 * 3600.0)) if (dt0 and dt1) else (len(eq) / ppyr)
    return _Series(ts, eq, rets, dds, abs(max_dd), ulcer, ppyr, max(1e-9, years))


def _mean(xs: List[float]) -> float:
    n = len(xs)
    return sum(xs) / n if n > 0 else 0.0


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return sqrt(max(0.0, var))


def _semidev(xs: List[float]) -> float:
    neg = [min(0.0, x) for x in xs]
    if not xs:
        return 0.0
    return sqrt(sum((x) ** 2 for x in neg) / len(xs))


def _cagr(start_eq: float, end_eq: float, years: float) -> float:
    if start_eq <= 0 or end_eq <= 0 or years <= 0:
        return 0.0
    return (end_eq / start_eq) ** (1.0 / years) - 1.0


def _format_hhmm(total_minutes: float) -> str:
    mins = int(round(total_minutes))
    h = mins // 60
    m = mins % 60
    return f"{h}h {m}m"


def compute_metrics(session, equity_times: List[str], equities: List[float], flat_events: List[Dict[str, object]], final_state: Dict[str, object]) -> Dict[str, object]:
    s = _build_series(session, equity_times, equities)
    start_eq = float(getattr(session, "starting_cash", 0.0) or (s.equity[0] if s.equity else 0.0))
    finish_eq = float(_to_float((final_state or {}).get("equity")))
    if finish_eq <= 0 and s.equity:
        finish_eq = s.equity[-1]

    # Bar returns stats
    mean_r = _mean(s.returns)
    std_r = _std(s.returns)
    down_r = _semidev(s.returns)
    ann_factor = sqrt(s.periods_per_year)
    sharpe = (mean_r / std_r * ann_factor) if std_r > 1e-12 else 0.0
    sortino = (mean_r / down_r * ann_factor) if down_r > 1e-12 else 0.0
    n = max(1, len(s.returns))
    penalty = (1.0 - 3.0 / max(3.0, (4.0 * n - 1.0)))  # small-sample penalty
    smart_sharpe = sharpe * penalty
    smart_sortino = sortino * penalty

    # Calmar, Omega, Serenity
    cagr = _cagr(start_eq, finish_eq, s.years)
    calmar = (cagr / s.max_dd) if s.max_dd > 1e-12 else 0.0
    pos_sum = sum(max(0.0, r) for r in s.returns)
    neg_sum = sum(max(0.0, -r) for r in s.returns)
    omega = (pos_sum / neg_sum) if neg_sum > 1e-12 else 0.0
    serenity = (cagr / s.ulcer) if s.ulcer > 1e-12 else 0.0

    # Trades/fees from events
    total_fees = 0.0
    total_funding = 0.0
    opens_longs = 0
    opens_shorts = 0
    trade_pnls: List[float] = []
    trade_side: List[str] = []
    holding_minutes: List[float] = []
    funding_events: List[Dict[str, object]] = []

    for ev in flat_events:
        t = (ev or {}).get("type")
        if t == "fill_open":
            side = str((ev or {}).get("side") or "")
            if side == "LONG":
                opens_longs += 1
            elif side == "SHORT":
                opens_shorts += 1
            total_fees += _to_float((ev or {}).get("fee"))
        elif t in ("fill_close", "tp", "sl", "liquidation"):
            trade_pnls.append(_to_float((ev or {}).get("pnl")))
            sd = (ev or {}).get("side")
            if not sd and t == "liquidation":
                sd = (ev or {}).get("side")
            trade_side.append(str(sd or ""))
            total_fees += _to_float((ev or {}).get("fee") or (ev or {}).get("liq_fee"))
            holding_minutes.append(float(_to_float((ev or {}).get("holding_minutes"))))
        elif t == "funding":
            total_funding += _to_float((ev or {}).get("funding_cost"))
            funding_events.append(ev)

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    total_trades = len(trade_pnls)
    wins_n = len(wins)
    losses_n = len(losses)
    win_rate = (wins_n / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_win = _mean(wins) if wins else 0.0
    avg_loss = _mean(losses) if losses else 0.0
    avg_wl = (abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    expectancy = (win_rate / 100.0) * avg_win + (1.0 - win_rate / 100.0) * (avg_loss if avg_loss < 0 else 0.0)
    expected_net = expectancy * total_trades
    avg_hold_min = _mean(holding_minutes) if holding_minutes else 0.0

    # Streaks
    best_win_streak = 0
    best_lose_streak = 0
    cur_streak = 0
    last_sign = 0
    for p in trade_pnls:
        sign = 1 if p > 0 else (-1 if p < 0 else 0)
        if sign == 0:
            cur_streak = 0
            last_sign = 0
            continue
        if sign == last_sign:
            cur_streak += sign
        else:
            cur_streak = sign
            last_sign = sign
        best_win_streak = max(best_win_streak, cur_streak if cur_streak > 0 else 0)
        best_lose_streak = min(best_lose_streak, cur_streak if cur_streak < 0 else 0)

    # Output
    pnl = finish_eq - start_eq
    pnl_pct = (pnl / start_eq * 100.0) if start_eq > 0 else 0.0
    return {
        "performance": {
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "win_rate_pct": win_rate,
            "sharpe": sharpe,
            "smart_sharpe": smart_sharpe,
            "sortino": sortino,
            "smart_sortino": smart_sortino,
            "calmar": calmar,
            "omega": omega,
            "serenity": serenity,
            "avg_win_loss_ratio": avg_wl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        },
        "risk": {
            "total_losing_streak": abs(best_lose_streak),
            "largest_losing_trade": min(losses) if losses else 0.0,
            "largest_winning_trade": max(wins) if wins else 0.0,
            "total_winning_streak": best_win_streak,
            "current_streak": cur_streak,
            "expectancy": expectancy,
            "expected_net_profit": expected_net,
            "average_holding_period_minutes": avg_hold_min,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "max_drawdown_pct": s.max_dd * 100.0,
        },
        "trade": {
            "total_trades": total_trades,
            "total_winning_trades": wins_n,
            "total_losing_trades": losses_n,
            "starting_balance": start_eq,
            "finishing_balance": finish_eq,
            "longs_count": opens_longs,
            "longs_pct": (opens_longs / max(1, opens_longs + opens_shorts) * 100.0),
            "shorts_pct": (opens_shorts / max(1, opens_longs + opens_shorts) * 100.0),
            "shorts_count": opens_shorts,
            "fee_total": total_fees,
            "funding_total": total_funding,
            "funding_events_count": len(funding_events),
            "total_costs": total_fees + total_funding,
            "total_open_trades": 1 if (final_state or {}).get("position") else 0,
            "open_pl": float(_to_float((final_state or {}).get("unrealized_pnl"))),
        },
    }


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def format_metrics_report(metrics: Dict[str, object]) -> str:
    p = metrics.get("performance", {})
    r = metrics.get("risk", {})
    t = metrics.get("trade", {})
    lines: List[str] = []
    lines.append("Performance Metrics")
    lines.append(f"PNL: {_fmt_money(_to_float(p.get('pnl')))}")
    lines.append(f"Win rate: {_fmt_pct(_to_float(p.get('win_rate_pct')))}")
    lines.append(f"Sharpe ratio: {_to_float(p.get('sharpe')):.3f}")
    lines.append(f"Smart Sharpe: {_to_float(p.get('smart_sharpe')):.3f}")
    lines.append(f"Sortino ratio: {_to_float(p.get('sortino')):.3f}")
    lines.append(f"Smart Sortino: {_to_float(p.get('smart_sortino')):.3f}")
    lines.append(f"Calmar ratio: {_to_float(p.get('calmar')):.3f}")
    lines.append(f"Omega ratio: {_to_float(p.get('omega')):.3f}")
    lines.append(f"Serenity index: {_to_float(p.get('serenity')):.3f}")
    lines.append(f"Average win/loss: {_to_float(p.get('avg_win_loss_ratio')):.3f}")
    lines.append(f"Average win: {_fmt_money(_to_float(p.get('avg_win')))}")
    lines.append(f"Average loss: {_fmt_money(_to_float(p.get('avg_loss')))}\n")
    lines.append("Risk Metrics")
    lines.append(f"Total losing streak: {int(_to_float(r.get('total_losing_streak')))}")
    lines.append(f"Largest losing trade: {_fmt_money(_to_float(r.get('largest_losing_trade')))}")
    lines.append(f"Largest winning trade: {_fmt_money(_to_float(r.get('largest_winning_trade')))}")
    lines.append(f"Total winning streak: {int(_to_float(r.get('total_winning_streak')))}")
    lines.append(f"Current streak: {int(_to_float(r.get('current_streak')))}")
    lines.append(f"Expectancy: {_fmt_money(_to_float(r.get('expectancy')))}")
    lines.append(f"Expected net profit: {_fmt_money(_to_float(r.get('expected_net_profit')))}")
    hold_min = _to_float(r.get("average_holding_period_minutes"))
    lines.append(f"Average holding period: {_format_hhmm(hold_min)}")
    lines.append(f"Gross profit: {_fmt_money(_to_float(r.get('gross_profit')))}")
    lines.append(f"Gross loss: {_fmt_money(_to_float(r.get('gross_loss')))}")
    lines.append(f"Max drawdown: {_fmt_pct(_to_float(r.get('max_drawdown_pct')))}\n")
    lines.append("Trade Metrics")
    lines.append(f"Total trades: {int(_to_float(t.get('total_trades')))}")
    lines.append(f"Total winning trades: {int(_to_float(t.get('total_winning_trades')))}")
    lines.append(f"Total losing trades: {int(_to_float(t.get('total_losing_trades')))}")
    lines.append(f"Starting balance: {_fmt_money(_to_float(t.get('starting_balance')))}")
    lines.append(f"Finishing balance: {_fmt_money(_to_float(t.get('finishing_balance')))}")
    lines.append(f"Longs count: {int(_to_float(t.get('longs_count')))}")
    lines.append(f"Longs percentage: {_fmt_pct(_to_float(t.get('longs_pct')))}")
    lines.append(f"Shorts percentage: {_fmt_pct(_to_float(t.get('shorts_pct')))}")
    lines.append(f"Shorts count: {int(_to_float(t.get('shorts_count')))}")
    lines.append(f"Fee: {_fmt_money(_to_float(t.get('fee_total')))}")
    lines.append(f"Funding: {_fmt_money(_to_float(t.get('funding_total')))}")
    lines.append(f"Funding events: {int(_to_float(t.get('funding_events_count')))}")
    lines.append(f"Total costs: {_fmt_money(_to_float(t.get('total_costs')))}")
    lines.append(f"Total open trades: {int(_to_float(t.get('total_open_trades')))}")
    lines.append(f"Open PL: {_fmt_money(_to_float(t.get('open_pl')))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vectorized fast-path metrics (from run_signals_backtest result dict)
# ---------------------------------------------------------------------------


def compute_metrics_from_result(
    result: Dict[str, object],
    starting_cash: float,
    timeframe_minutes: int = 60,
) -> Dict[str, object]:
    """Compute the same metrics dict as ``compute_metrics`` but from the
    result dict returned by ``_rust_core.run_signals_backtest``.

    This avoids creating synthetic event dicts — it works directly with the
    arrays and trade lists that Rust already collected.
    """
    import numpy as _np

    equities_arr = result.get("equities")
    if equities_arr is None:
        equities_arr = []
    equities: List[float] = list(_np.asarray(equities_arr, dtype=float))

    start_eq = starting_cash
    finish_eq = float(result.get("final_equity", 0.0))
    if finish_eq <= 0.0 and equities:
        finish_eq = equities[-1]

    # --- Bar returns / drawdown (same logic as _build_series) ---
    rets: List[float] = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            rets.append(equities[i] / equities[i - 1] - 1.0)
    high = equities[0] if equities else 0.0
    dds: List[float] = []
    for e in equities:
        if high <= 0:
            dds.append(0.0)
            high = max(high, e)
            continue
        high = max(high, e)
        dds.append((e / high) - 1.0)
    max_dd = abs(min(dds)) if dds else 0.0
    ulcer = sqrt(sum((d * 100.0) ** 2 for d in dds if d < 0) / max(1, len(dds)))
    ppyr = (365.25 * 24.0 * 60.0) / float(timeframe_minutes or 1)
    years = max(1e-9, len(equities) / ppyr)

    # --- Performance stats ---
    mean_r = _mean(rets)
    std_r = _std(rets)
    down_r = _semidev(rets)
    ann_factor = sqrt(ppyr)
    sharpe = (mean_r / std_r * ann_factor) if std_r > 1e-12 else 0.0
    sortino = (mean_r / down_r * ann_factor) if down_r > 1e-12 else 0.0
    n_rets = max(1, len(rets))
    penalty = 1.0 - 3.0 / max(3.0, (4.0 * n_rets - 1.0))
    smart_sharpe = sharpe * penalty
    smart_sortino = sortino * penalty
    cagr = _cagr(start_eq, finish_eq, years)
    calmar = (cagr / max_dd) if max_dd > 1e-12 else 0.0
    pos_sum = sum(max(0.0, r) for r in rets)
    neg_sum = sum(max(0.0, -r) for r in rets)
    omega = (pos_sum / neg_sum) if neg_sum > 1e-12 else 0.0
    serenity = (cagr / ulcer) if ulcer > 1e-12 else 0.0

    # --- Trade stats (from Rust arrays) ---
    trade_pnls: List[float] = list(result.get("trade_pnls", []))
    trade_sides: List[str] = list(result.get("trade_sides", []))
    holding_minutes: List[float] = list(result.get("trade_holding_minutes", []))
    opens_longs = int(result.get("open_longs", 0))
    opens_shorts = int(result.get("open_shorts", 0))
    total_fees = float(result.get("total_fees", 0.0))
    total_funding = float(result.get("total_funding", 0.0))

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    total_trades = len(trade_pnls)
    wins_n = len(wins)
    losses_n = len(losses)
    win_rate = (wins_n / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_win = _mean(wins) if wins else 0.0
    avg_loss = _mean(losses) if losses else 0.0
    avg_wl = (abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    expectancy = (win_rate / 100.0) * avg_win + (1.0 - win_rate / 100.0) * (avg_loss if avg_loss < 0 else 0.0)
    expected_net = expectancy * total_trades
    avg_hold_min = _mean(holding_minutes) if holding_minutes else 0.0

    # Streaks
    best_win_streak = 0
    best_lose_streak = 0
    cur_streak = 0
    last_sign = 0
    for p in trade_pnls:
        sign = 1 if p > 0 else (-1 if p < 0 else 0)
        if sign == 0:
            cur_streak = 0
            last_sign = 0
            continue
        if sign == last_sign:
            cur_streak += sign
        else:
            cur_streak = sign
            last_sign = sign
        best_win_streak = max(best_win_streak, cur_streak if cur_streak > 0 else 0)
        best_lose_streak = min(best_lose_streak, cur_streak if cur_streak < 0 else 0)

    pnl = finish_eq - start_eq
    pnl_pct = (pnl / start_eq * 100.0) if start_eq > 0 else 0.0

    return {
        "performance": {
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "win_rate_pct": win_rate,
            "sharpe": sharpe,
            "smart_sharpe": smart_sharpe,
            "sortino": sortino,
            "smart_sortino": smart_sortino,
            "calmar": calmar,
            "omega": omega,
            "serenity": serenity,
            "avg_win_loss_ratio": avg_wl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        },
        "risk": {
            "total_losing_streak": abs(best_lose_streak),
            "largest_losing_trade": min(losses) if losses else 0.0,
            "largest_winning_trade": max(wins) if wins else 0.0,
            "total_winning_streak": best_win_streak,
            "current_streak": cur_streak,
            "expectancy": expectancy,
            "expected_net_profit": expected_net,
            "average_holding_period_minutes": avg_hold_min,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "max_drawdown_pct": max_dd * 100.0,
        },
        "trade": {
            "total_trades": total_trades,
            "total_winning_trades": wins_n,
            "total_losing_trades": losses_n,
            "starting_balance": start_eq,
            "finishing_balance": finish_eq,
            "longs_count": opens_longs,
            "longs_pct": (opens_longs / max(1, opens_longs + opens_shorts) * 100.0),
            "shorts_pct": (opens_shorts / max(1, opens_longs + opens_shorts) * 100.0),
            "shorts_count": opens_shorts,
            "fee_total": total_fees,
            "funding_total": total_funding,
            "funding_events_count": 0,
            "total_costs": total_fees + total_funding,
            "total_open_trades": 0,
            "open_pl": 0.0,
        },
    }