"""Core trading engine utilities for forge_engine."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional, Union, List, Callable, cast, Tuple

import csv
import os
import io
import json
import hashlib
from uuid import UUID, uuid4
from .indexer import ensure_index, find_seek_offset as _idx_find
import numpy as np



__all__ = ["Session", "create_session", "step_session", "step_session_single_pass", "get_warmup_candles", "preload_candle_data", "preload_candle_data_memmap", "preload_candle_data_aggregated", "CandleData", "CandleDataMemmap", "CandleDataAggregated", "iter_candles_from_preloaded", "iter_candles_from_aggregated", "get_intra_candles"]



def _parse_iso8601_utc(value: Union[str, datetime]) -> datetime:
    """Parse an ISO8601 string like '2025-09-01T00:00:00Z' into an aware UTC datetime."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)



def _isoformat_z(dt: datetime) -> str:
    """Return ISO8601 string with 'Z' suffix (seconds precision)."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_iso8601(value: str) -> str:
    """Normalize ISO8601 string to UTC Z format for safe comparisons."""
    return _isoformat_z(_parse_iso8601_utc(value))


def _idx_get(idx_map: Dict[str, int], *names: str) -> int:
    for name in names:
        idx = idx_map.get(name, -1)
        if idx >= 0:
            return idx
    return -1


def _index_guard(*indices: int) -> bool:
    return all(idx >= 0 for idx in indices)


def _coerce_ts(ts: str) -> str:
    try:
        return _normalize_iso8601(ts)
    except Exception:
        return ts



def _parse_timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe strings like '1m', '5m', '1h', '4h', '1d' to minutes."""
    t = tf.strip().lower()
    if t.endswith("m"):
        n = int(t[:-1])
        if n <= 0:
            raise ValueError("Timeframe minutes must be positive")
        return n
    if t.endswith("h"):
        n = int(t[:-1])
        if n <= 0:
            raise ValueError("Timeframe hours must be positive")
        return n * 60
    if t.endswith("d"):
        n = int(t[:-1])
        if n <= 0:
            raise ValueError("Timeframe days must be positive")
        return n * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")


def _floor_to_timeframe(dt: datetime, minutes: int, anchor: Optional[datetime] = None) -> datetime:
    """Floor dt to the start of its timeframe window.

    If anchor is provided, the flooring is aligned to multiples of 'minutes' from the anchor.
    Otherwise, it is aligned to Unix epoch (UTC).
    """
    step = minutes * 60
    if anchor is None:
        seconds = int(dt.timestamp())
        floored = (seconds // step) * step
        return datetime.fromtimestamp(floored, tz=timezone.utc)
    # Ensure anchor is UTC-aware
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    else:
        anchor = anchor.astimezone(timezone.utc)
    delta_seconds = int((dt - anchor).total_seconds())
    k = delta_seconds // step  # floor division works for negatives
    floored_dt = anchor + timedelta(seconds=k * step)
    return floored_dt.astimezone(timezone.utc)


def _load_funding_data(symbol: str, data_dir: str) -> Optional[Dict[str, float]]:
    """Load funding rate data from CSV if symbol is a perpetual and file exists.

    Returns dict mapping funding_time (ISO8601) to funding_rate, or None if not applicable.
    """
    # Only load funding for perpetual futures
    if "PERP" not in symbol.upper():
        return None

    funding_file = os.path.join(data_dir, f"{symbol}_funding.csv")
    if not os.path.exists(funding_file):
        return None

    funding_data: Dict[str, float] = {}
    try:
        with open(funding_file, "r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            if not header:
                return None

            # Find column indices
            names = [h.strip().lower() for h in header]
            idx_time = names.index("funding_time") if "funding_time" in names else -1
            idx_rate = names.index("funding_rate") if "funding_rate" in names else -1

            if idx_time < 0 or idx_rate < 0:
                return None

            for row in rdr:
                if idx_time >= len(row) or idx_rate >= len(row):
                    continue
                try:
                    time_str = _coerce_ts(row[idx_time].strip())
                    rate = float(row[idx_rate])
                    funding_data[time_str] = rate
                except (ValueError, IndexError):
                    continue

    except Exception:
        return None

    return funding_data if funding_data else None


@dataclass
class Session:
    id: UUID
    symbol: str
    start_date: datetime
    end_date: datetime
    starting_cash: float
    leverage: float
    margin_mode: str
    warmup_candles: int
    timeframe: str
    timeframe_minutes: int
    data_dir: str
    base_timeframe: str
    base_timeframe_minutes: int
    visual_url: Optional[str] = None
    close_at_end: bool = False
    enable_visual: bool = False
    slippage_pct: float = 0.0  # slippage applied to SL and liquidation fills
    funding_data: Optional[Dict[str, float]] = None  # {iso_time: rate} - loaded from funding CSV


def create_session(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    starting_cash: float,
    leverage: float = 1.0,  # Default 1x for spot, set higher for perps
    margin_mode: str = "cross",  # Default cross, ignored for spot (leverage=1)
    warmup_candles: int = 0,
    timeframe: str = "1m",
    base_timeframe: str = "1m",
    data_dir: Optional[str] = None,
    close_at_end: bool = False,
    enable_visual: bool = False,
    slippage_pct: float = 0.0,
) -> Session:
    """Create a session configuration."""
    timeframe_minutes = _parse_timeframe_to_minutes(timeframe)
    base_minutes = _parse_timeframe_to_minutes(base_timeframe)
    if timeframe_minutes < base_minutes or timeframe_minutes % base_minutes != 0:
        raise ValueError(
            f"timeframe '{timeframe}' must be a multiple of base_timeframe '{base_timeframe}'"
        )
    sdt = _parse_iso8601_utc(start_date)
    edt = _parse_iso8601_utc(end_date)
    if edt <= sdt:
        raise ValueError("end_date must be after start_date")
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    # Load funding rate data if available (only for perpetual futures)
    funding_data = _load_funding_data(symbol, data_dir)

    sess = Session(
        id=uuid4(),
        symbol=symbol,
        start_date=sdt,
        end_date=edt,
        starting_cash=starting_cash,
        leverage=leverage,
        margin_mode=margin_mode,
        warmup_candles=warmup_candles,
        timeframe=timeframe,
        timeframe_minutes=timeframe_minutes,
        data_dir=data_dir,
        base_timeframe=base_timeframe,
        base_timeframe_minutes=base_minutes,
        close_at_end=close_at_end,
        enable_visual=enable_visual,
        slippage_pct=slippage_pct,
        funding_data=funding_data,
    )

    if enable_visual:
        try:
            from .visual import ensure_server
            port = ensure_server(sess)
            sess.visual_url = f"http://127.0.0.1:{port}"
        except Exception:
            pass

    return sess


def get_intra_candles(session: Session, start_time: str, end_time: str) -> List[Dict[str, Any]]:
    """Fetch 1m candles between start_time and end_time for intra-candle resolution.

    Used to determine which trigger (TP/SL/LIQ) was hit first within a higher timeframe candle.
    Uses indexed seek for fast access.

    Args:
        session: The trading session
        start_time: Start time in ISO8601 format (inclusive)
        end_time: End time in ISO8601 format (exclusive)

    Returns:
        List of 1m candle dicts with open_time, open, high, low, close
    """
    base_filename = f"{session.symbol}_{session.base_timeframe}.csv"
    base_file = os.path.join(session.data_dir, base_filename)

    if not os.path.exists(base_file):
        return []

    candles: List[Dict[str, Any]] = []
    start_iso = _coerce_ts(start_time)
    end_iso = _coerce_ts(end_time)

    # Use indexed seek for fast access

    seek_offset = None
    try:
        idx_info = ensure_index(base_file, session.data_dir)
        if idx_info.get("valid"):
            seek_offset = _idx_find(str(idx_info.get("path")), start_iso)

    except Exception:
        seek_offset = None

    try:
        if seek_offset is not None:
            fb = open(base_file, "rb")
            tf = None
            try:
                # Read header to build column indices
                header_bytes = fb.readline()
                header_text = header_bytes.decode("utf-8").rstrip("\r\n")
                header_cols = next(csv.reader([header_text]))
                idx_map = {h.strip(): i for i, h in enumerate(header_cols)}

                i_ot = _idx_get(idx_map, "open_time")
                i_o = _idx_get(idx_map, "open")
                i_h = _idx_get(idx_map, "high")
                i_l = _idx_get(idx_map, "low")
                i_c = _idx_get(idx_map, "close")

                if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
                    return []

                fb.seek(seek_offset)
                tf = io.TextIOWrapper(fb, encoding="utf-8", newline="")
                rdr = csv.reader(tf)

                for row in rdr:
                    if i_ot < 0 or i_ot >= len(row):
                        continue
                    ts = _coerce_ts(row[i_ot])
                    if ts >= end_iso:
                        break
                    if ts < start_iso:
                        continue

                    try:
                        candles.append({
                            "open_time": ts,
                            "open": float(row[i_o]),
                            "high": float(row[i_h]),
                            "low": float(row[i_l]),
                            "close": float(row[i_c]),
                        })
                    except (ValueError, IndexError):
                        continue
            finally:
                try:
                    if tf is not None:
                        tf.detach()
                except Exception:
                    pass
                fb.close()


        else:
            # Fallback to full scan
            with open(base_file, "r", encoding="utf-8", newline="") as f:
                rdr = csv.reader(f)
                header = next(rdr, None)
                if not header:
                    return []

                names = [h.strip() for h in header]
                idx_map = {name: i for i, name in enumerate(names)}

                i_ot = _idx_get(idx_map, "open_time")
                i_o = _idx_get(idx_map, "open")
                i_h = _idx_get(idx_map, "high")
                i_l = _idx_get(idx_map, "low")
                i_c = _idx_get(idx_map, "close")

                if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
                    return []

                for row in rdr:
                    if i_ot < 0 or i_ot >= len(row):
                        continue
                    ts = _coerce_ts(row[i_ot])
                    if ts < start_iso:
                        continue
                    if ts >= end_iso:
                        break

                    try:
                        candles.append({
                            "open_time": ts,
                            "open": float(row[i_o]),
                            "high": float(row[i_h]),
                            "low": float(row[i_l]),
                            "close": float(row[i_c]),
                        })
                    except (ValueError, IndexError):
                        continue

    except Exception:
        return []

    return candles


def _iter_aggregated_candles(
    session: Session, read_from: datetime, end_exclusive: datetime
) -> Iterable[Dict[str, object]]:
    """Yield aggregated candles for [read_from, end_exclusive), aligned like step_session.

    Yields items shaped as:
      {
        "symbol": session.symbol,
        "timeframe": session.timeframe,
        "candle": {
            "open_time": ISO8601Z,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "quote_asset_volume": float,
            "number_of_trades": int,
            "taker_buy_base_asset_volume": float,
            "taker_buy_quote_asset_volume": float,
        }
      }
    """
    base_filename = f"{session.symbol}_{session.base_timeframe}.csv"
    base_file = os.path.join(session.data_dir, base_filename)
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Data file not found: {base_file}")

    # Align base read to base timeframe boundary
    read_from_aligned = _floor_to_timeframe(read_from, session.base_timeframe_minutes)

    current_bucket_start: Optional[datetime] = None
    agg_open: Optional[float] = None
    agg_high: Optional[float] = None
    agg_low: Optional[float] = None
    agg_close: Optional[float] = None
    agg_volume: float = 0.0
    agg_quote_volume: float = 0.0
    agg_trades: int = 0
    agg_taker_base: float = 0.0
    agg_taker_quote: float = 0.0

    # Precompute anchor for multi-day timeframes (aligned to session's calendar start)
    anchor: Optional[datetime] = None
    if session.timeframe_minutes % 1440 == 0:
        anchor = session.start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    def flush():
        nonlocal current_bucket_start, agg_open, agg_high, agg_low, agg_close
        nonlocal agg_volume, agg_quote_volume, agg_trades, agg_taker_base, agg_taker_quote
        if current_bucket_start is None:
            return None
        if current_bucket_start >= end_exclusive:
            return None
        if current_bucket_start >= read_from:
            candle = {
                "open_time": _isoformat_z(current_bucket_start),
                "open": agg_open,
                "high": agg_high,
                "low": agg_low,
                "close": agg_close,
                "volume": agg_volume,
                "quote_asset_volume": agg_quote_volume,
                "number_of_trades": agg_trades,
                "taker_buy_base_asset_volume": agg_taker_base,
                "taker_buy_quote_asset_volume": agg_taker_quote,
            }
            return {"symbol": session.symbol, "timeframe": session.timeframe, "candle": candle}
        return None

    # Fast-path when timeframe == base_timeframe: skip aggregation and datetime parsing
    no_agg = (session.timeframe_minutes == session.base_timeframe_minutes)
    end_iso = _isoformat_z(end_exclusive)

    # Indexed fast-path using dedicated sidecar index under data/_index; fallback to full scan

    target_iso = _isoformat_z(read_from_aligned)
    seek_offset = None
    try:
        idx_info = ensure_index(base_file, session.data_dir)
        if idx_info.get("valid"):
            seek_offset = _idx_find(str(idx_info.get("path")), target_iso)
    except Exception:
        seek_offset = None

    # Open CSV and iterate from seek_offset if available; otherwise full scan
    if seek_offset is not None:
        fb = open(base_file, "rb")
        tf: Optional[io.TextIOWrapper] = None
        try:
            # Read header to build column indices
            header_bytes = fb.readline()
            header_text = header_bytes.decode("utf-8").rstrip("\r\n")
            header_cols = next(csv.reader([header_text]))
            idx_map = {h.strip(): i for i, h in enumerate(header_cols)}

            i_ot = _idx_get(idx_map, "open_time")
            i_o = _idx_get(idx_map, "open")
            i_h = _idx_get(idx_map, "high")
            i_l = _idx_get(idx_map, "low")
            i_c = _idx_get(idx_map, "close")
            i_v = _idx_get(idx_map, "volume")
            i_qv = _idx_get(idx_map, "quote_asset_volume")
            i_nt = _idx_get(idx_map, "number_of_trades")
            i_tb = _idx_get(idx_map, "taker_buy_base_asset_volume")
            i_tq = _idx_get(idx_map, "taker_buy_quote_asset_volume")

            if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
                return

            fb.seek(seek_offset)
            tf = io.TextIOWrapper(fb, encoding="utf-8", newline="")
            rdr = csv.reader(tf)

            if no_agg:
                # Fast-path: stream rows directly without aggregation or datetime parsing
                for row in rdr:
                    if i_ot < 0 or i_ot >= len(row):
                        continue
                    ts = _coerce_ts(row[i_ot])
                    if ts >= end_iso:
                        break
                    if ts < target_iso:
                        continue

                    try:
                        o = float(row[i_o])
                        h = float(row[i_h])
                        l = float(row[i_l])
                        c = float(row[i_c])
                    except Exception:
                        continue

                    v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
                    qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
                    nt = int(float(row[i_nt])) if 0 <= i_nt < len(row) and row[i_nt] else 0
                    tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
                    tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0

                    yield {
                        "symbol": session.symbol,
                        "timeframe": session.timeframe,
                        "candle": {
                            "open_time": ts,
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": v,
                            "quote_asset_volume": qv,
                            "number_of_trades": nt,
                            "taker_buy_base_asset_volume": tb,
                            "taker_buy_quote_asset_volume": tq,
                        }
                    }
                return

            # Aggregation path
            parse_dt = _parse_iso8601_utc
            floor_tf = _floor_to_timeframe
            tfm = session.timeframe_minutes

            for row in rdr:
                try:
                    if i_ot < 0 or i_ot >= len(row):
                        continue
                    dt = parse_dt(row[i_ot])
                except Exception:
                    continue

                if dt >= end_exclusive:
                    out = flush()
                    if out is not None:
                        yield out
                    break

                try:
                    o = float(row[i_o]) if 0 <= i_o < len(row) and row[i_o] else None
                    h = float(row[i_h]) if 0 <= i_h < len(row) and row[i_h] else None
                    l = float(row[i_l]) if 0 <= i_l < len(row) and row[i_l] else None
                    c = float(row[i_c]) if 0 <= i_c < len(row) and row[i_c] else None
                    if o is None or h is None or l is None or c is None:
                        continue
                    v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
                    qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
                    nt = int(float(row[i_nt])) if 0 <= i_nt < len(row) and row[i_nt] else 0
                    tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
                    tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0
                except Exception:
                    continue

                if anchor is not None:
                    bucket_start = floor_tf(dt, tfm, anchor=anchor)
                else:
                    bucket_start = floor_tf(dt, tfm)

                if current_bucket_start is None:
                    current_bucket_start = bucket_start
                    agg_open = o
                    agg_high = h
                    agg_low = l
                    agg_close = c
                    agg_volume = v
                    agg_quote_volume = qv
                    agg_trades = nt
                    agg_taker_base = tb
                    agg_taker_quote = tq
                    continue

                if bucket_start != current_bucket_start:
                    out = flush()
                    current_bucket_start = bucket_start
                    agg_open = o
                    agg_high = h
                    agg_low = l
                    agg_close = c
                    agg_volume = v
                    agg_quote_volume = qv
                    agg_trades = nt
                    agg_taker_base = tb
                    agg_taker_quote = tq
                    if out is not None:
                        yield out
                else:
                    if agg_high is None or h > agg_high:
                        agg_high = h
                    if agg_low is None or l < agg_low:
                        agg_low = l
                    agg_close = c
                    agg_volume += v
                    agg_quote_volume += qv
                    agg_trades += nt
                    agg_taker_base += tb
                    agg_taker_quote += tq
            else:
                out = flush()
                if out is not None:
                    yield out
        finally:
            try:
                if tf is not None:
                    tf.detach()  # type: ignore
            except Exception:
                pass
            fb.close()

    else:
        # Fall back to full scan from start (fast-path reader)
        with open(base_file, "r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            if not header:
                return
            names = [h.strip() for h in header]
            idx_map = {name: i for i, name in enumerate(names)}

            i_ot = _idx_get(idx_map, "open_time")
            i_o = _idx_get(idx_map, "open")
            i_h = _idx_get(idx_map, "high")
            i_l = _idx_get(idx_map, "low")
            i_c = _idx_get(idx_map, "close")
            i_v = _idx_get(idx_map, "volume")
            i_qv = _idx_get(idx_map, "quote_asset_volume")
            i_nt = _idx_get(idx_map, "number_of_trades")
            i_tb = _idx_get(idx_map, "taker_buy_base_asset_volume")
            i_tq = _idx_get(idx_map, "taker_buy_quote_asset_volume")

            if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
                return

            if no_agg:
                # Fast-path: stream rows directly without aggregation or datetime parsing
                for row in rdr:
                    if i_ot < 0 or i_ot >= len(row):
                        continue
                    ts = _coerce_ts(row[i_ot])
                    if ts < target_iso:
                        continue
                    if ts >= end_iso:
                        break

                    try:
                        o = float(row[i_o])
                        h = float(row[i_h])
                        l = float(row[i_l])
                        c = float(row[i_c])
                    except Exception:
                        continue

                    v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
                    qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
                    nt = int(float(row[i_nt])) if 0 <= i_nt < len(row) and row[i_nt] else 0
                    tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
                    tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0

                    yield {
                        "symbol": session.symbol,
                        "timeframe": session.timeframe,
                        "candle": {
                            "open_time": ts,
                            "open": o,
                            "high": h,
                            "low": l,
                            "close": c,
                            "volume": v,
                            "quote_asset_volume": qv,
                            "number_of_trades": nt,
                            "taker_buy_base_asset_volume": tb,
                            "taker_buy_quote_asset_volume": tq,
                        }
                    }
                return



            # Aggregation path
            parse_dt = _parse_iso8601_utc
            floor_tf = _floor_to_timeframe
            tfm = session.timeframe_minutes

            for row in rdr:
                try:
                    if i_ot < 0 or i_ot >= len(row):
                        continue
                    dt = parse_dt(row[i_ot])
                except Exception:
                    continue

                if dt < read_from_aligned:
                    continue

                if dt >= end_exclusive:
                    out = flush()
                    if out is not None:
                        yield out
                    break

                try:
                    o = float(row[i_o]) if 0 <= i_o < len(row) and row[i_o] else None
                    h = float(row[i_h]) if 0 <= i_h < len(row) and row[i_h] else None
                    l = float(row[i_l]) if 0 <= i_l < len(row) and row[i_l] else None
                    c = float(row[i_c]) if 0 <= i_c < len(row) and row[i_c] else None
                    if o is None or h is None or l is None or c is None:
                        continue

                    v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
                    qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
                    nt = int(float(row[i_nt])) if 0 <= i_nt < len(row) and row[i_nt] else 0
                    tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
                    tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0
                except Exception:
                    continue

                if anchor is not None:
                    bucket_start = floor_tf(dt, tfm, anchor=anchor)
                else:
                    bucket_start = floor_tf(dt, tfm)

                if current_bucket_start is None:
                    current_bucket_start = bucket_start
                    agg_open = o
                    agg_high = h
                    agg_low = l
                    agg_close = c
                    agg_volume = v
                    agg_quote_volume = qv
                    agg_trades = nt
                    agg_taker_base = tb
                    agg_taker_quote = tq
                    continue

                if bucket_start != current_bucket_start:
                    out = flush()
                    current_bucket_start = bucket_start
                    agg_open = o
                    agg_high = h
                    agg_low = l
                    agg_close = c
                    agg_volume = v
                    agg_quote_volume = qv
                    agg_trades = nt
                    agg_taker_base = tb
                    agg_taker_quote = tq
                    if out is not None:
                        yield out
                else:
                    if agg_high is None or h > agg_high:
                        agg_high = h
                    if agg_low is None or l < agg_low:
                        agg_low = l
                    agg_close = c
                    agg_volume += v
                    agg_quote_volume += qv
                    agg_trades += nt
                    agg_taker_base += tb
                    agg_taker_quote += tq
            else:
                out = flush()
                if out is not None:
                    yield out


def get_warmup_candles(session: Session) -> List[Dict[str, object]]:
    """Return warmup candles aggregated to the session timeframe.

    Behavior:
      1) Attempt to collect warmup_candles prior to session.start_date over:
            [ start_date - warmup_candles*timeframe, start_date )
      2) If there are not enough pre-start candles (e.g., dataset begins at start),
         fill the remainder from the earliest available forward candles, i.e. from:
            [ start_date, start_date + remainder*timeframe )
      3) Return at most exactly 'warmup_candles' items when possible.

    Alignment and aggregation are identical to step_session.
    """
    count = session.warmup_candles
    if count <= 0:
        return []

    warmup_minutes = count * session.timeframe_minutes

    # First, try gathering candles before the session start.
    back_read_from = session.start_date - timedelta(minutes=warmup_minutes)
    back_end_exclusive = session.start_date
    back: List[Dict[str, object]] = []
    for item in _iter_aggregated_candles(session, back_read_from, back_end_exclusive):
        back.append(cast(Dict[str, object], item["candle"]))

    if len(back) >= count:
        # Ensure exactly 'count' most recent warmups (in case of partial alignment)
        return back[-count:]

    # Not enough back candles: fill the remainder from the start going forward.
    remainder = count - len(back)
    fwd_read_from = session.start_date
    fwd_end_exclusive = session.start_date + timedelta(minutes=remainder * session.timeframe_minutes)
    fwd: List[Dict[str, object]] = []
    for item in _iter_aggregated_candles(session, fwd_read_from, fwd_end_exclusive):
        fwd.append(cast(Dict[str, object], item["candle"]))

    combined = back + fwd
    # Return up to 'count' items
    trimmed = combined[-count:] if len(combined) >= count else combined
    return trimmed




def step_session(session: Session) -> Iterable[Dict[str, object]]:
    """Yield aggregated candles for the session window with trading state and events.

    If not enough pre-start warmup candles exist in the dataset, the warmup is
    filled from the beginning of the session forward. In that case, iteration
    starts after those forward-filled warmup candles to avoid duplication.
    """
    # Lazy import to avoid circular dependency at module import time
    from . import trading
    try:
        from .indicators import update_indicators_for_session as _update_inds  # type: ignore
    except Exception:
        _update_inds = None  # type: ignore

    count = session.warmup_candles
    if count <= 0:
        start_iter_from = session.start_date
    else:
        warmup_minutes = count * session.timeframe_minutes

        # Count how many warmup candles exist strictly before the session start.
        back_read_from = session.start_date - timedelta(minutes=warmup_minutes)
        back_end_exclusive = session.start_date
        back_len = 0
        for _ in _iter_aggregated_candles(session, back_read_from, back_end_exclusive):
            back_len += 1

        remainder = max(0, count - back_len)
        start_iter_from = session.start_date + timedelta(minutes=remainder * session.timeframe_minutes)

    eng = trading.ensure_engine(session)
    publish_tick = None  # type: ignore
    if getattr(session, "enable_visual", False):
        try:
            from .visual import ensure_server, publish_tick as _publish_tick
            ensure_server(session)
            publish_tick = _publish_tick  # type: ignore
        except Exception:
            publish_tick = None  # type: ignore

    candle_index = 0
    last_candle: Optional[Dict[str, object]] = None
    for item in _iter_aggregated_candles(session, start_iter_from, session.end_date):
        candle = cast(Dict[str, object], item["candle"])
        events, state = trading.on_candle(session, candle, candle_index)

        if _update_inds:
            try:
                ind_vals = _update_inds(session, candle, candle_index)  # type: ignore
            except Exception:
                ind_vals = None
            if ind_vals:
                state["indicators"] = ind_vals
        try:
            if publish_tick:
                publish_tick(session, candle, state, events)  # type: ignore
        except Exception:
            pass
        yield {"candle": candle, "state": state, "events": events}
        candle_index += 1
        last_candle = candle

    # Force-close at end if requested and a position remains
    if session.close_at_end and last_candle is not None:
        try:
            snap = eng.snapshot()
            if snap.get("position") is not None:
                close_val = last_candle.get("close")
                close_price = float(close_val) if isinstance(close_val, (int, float)) else 0.0

                # Queue the close then replay one more candle tick to fill it.
                eng.close_order(close_price)
                events_close, state = eng.on_candle(last_candle, candle_index)

                state["candle_index"] = max(state.get("candle_index", -1), candle_index - 1)
                try:
                    if publish_tick:
                        publish_tick(session, last_candle, state, events_close)  # type: ignore
                except Exception:
                    pass
                try:
                    yield {"candle": last_candle, "state": state, "events": events_close}
                except GeneratorExit:
                    return
        except Exception:
            pass



def step_session_single_pass(session: Session) -> Iterable[Dict[str, object]]:
    """Single-pass warmup + session iteration.

    Yields a first item with warmups, then candle/state/events ticks for the session.
    Falls back to warmups + step_session if the fast path cannot proceed.
    """
    try:
        warmups = get_warmup_candles(session)
        yield {"warmups": warmups}
        for tick in step_session(session):
            yield tick
    except Exception:
        warmups = []
        try:
            warmups = get_warmup_candles(session)
        except Exception:
            warmups = []
        yield {"warmups": warmups}
        for tick in step_session(session):
            yield tick


# ---------------------------------------------------------------------------
# Data Preloading for Parallel Backtesting
# ---------------------------------------------------------------------------

@dataclass
class CandleData:

    """Preloaded candle data for efficient parallel backtesting."""
    symbol: str
    base_timeframe: str
    candles: List[Dict[str, Any]]  # List of {open_time, open, high, low, close, volume, ...}
    start_time: str  # ISO8601
    end_time: str  # ISO8601

    def __len__(self) -> int:
        return len(self.candles)


@dataclass
class CandleDataMemmap:
    """Memmapped candle data shared across processes."""
    symbol: str
    base_timeframe: str
    cache_dir: str
    start_time: str
    end_time: str
    ts_path: str
    val_path: str
    count: int
    _timestamps: Optional[np.memmap] = field(default=None, repr=False, compare=False)
    _values: Optional[np.memmap] = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.count)

    def load(self) -> Tuple[np.memmap, np.memmap]:
        if self._timestamps is None or self._values is None:
            self._timestamps = np.load(self.ts_path, mmap_mode="r")
            self._values = np.load(self.val_path, mmap_mode="r")
        return self._timestamps, self._values


@dataclass
class CandleDataAggregated:
    """
    Pre-aggregated candle data optimized for fast time-range queries.

    Stores candles already aggregated to the target timeframe with Unix timestamps
    for O(log n) binary search lookups instead of O(n) string comparisons.
    """
    symbol: str
    timeframe_minutes: int
    timestamps_unix: np.ndarray  # int64 Unix timestamps (seconds)
    values: np.ndarray  # float64 [n, 9] - open, high, low, close, volume, qv, nt, tb, tq
    start_unix: int
    end_unix: int

    def __len__(self) -> int:
        return len(self.timestamps_unix)

    def find_range_indices(self, start_unix: int, end_unix: int) -> Tuple[int, int]:
        """Binary search to find start/end indices for a time range. O(log n)."""
        import bisect
        start_idx = bisect.bisect_left(self.timestamps_unix, start_unix)
        end_idx = bisect.bisect_left(self.timestamps_unix, end_unix)
        return start_idx, end_idx

    def get_candles_in_range(self, start_unix: int, end_unix: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get timestamps and values for a time range. Returns views, not copies."""
        start_idx, end_idx = self.find_range_indices(start_unix, end_unix)
        return self.timestamps_unix[start_idx:end_idx], self.values[start_idx:end_idx]



def _memmap_cache_key(symbol: str, base_timeframe: str, data_dir: str, warmup_candles: int, timeframe_minutes: int) -> str:
    payload = f"{symbol}|{base_timeframe}|{os.path.abspath(data_dir)}|{warmup_candles}|{timeframe_minutes}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _memmap_cache_paths(cache_dir: str, key: str) -> Tuple[str, str, str]:
    ts_path = os.path.join(cache_dir, f"candles_{key}.ts.npy")
    val_path = os.path.join(cache_dir, f"candles_{key}.val.npy")
    meta_path = os.path.join(cache_dir, f"candles_{key}.meta.json")
    return ts_path, val_path, meta_path


def preload_candle_data_memmap(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    base_timeframe: str = "1m",
    data_dir: Optional[str] = None,
    warmup_candles: int = 0,
    timeframe_minutes: int = 60,
) -> CandleDataMemmap:
    """Preload candle data into memmapped arrays for multiprocessing."""
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")
    cache_dir = os.path.join(data_dir, ".forge_cache")
    os.makedirs(cache_dir, exist_ok=True)

    sdt = _parse_iso8601_utc(start_date)
    edt = _parse_iso8601_utc(end_date)

    warmup_minutes = warmup_candles * timeframe_minutes
    read_from = sdt - timedelta(minutes=warmup_minutes)

    base_filename = f"{symbol}_{base_timeframe}.csv"
    base_file = os.path.join(data_dir, base_filename)
    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Data file not found: {base_file}")

    read_from_iso = _isoformat_z(read_from)
    end_iso = _isoformat_z(edt)

    key = _memmap_cache_key(symbol, base_timeframe, data_dir, warmup_candles, timeframe_minutes)
    ts_path, val_path, meta_path = _memmap_cache_paths(cache_dir, key)
    csv_mtime = os.path.getmtime(base_file)

    if os.path.exists(ts_path) and os.path.exists(val_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                meta.get("symbol") == symbol
                and meta.get("base_timeframe") == base_timeframe
                and meta.get("start_time") == read_from_iso
                and meta.get("end_time") == end_iso
                and float(meta.get("csv_mtime", -1)) == csv_mtime
            ):
                ts_arr = np.load(ts_path, mmap_mode="r")
                val_arr = np.load(val_path, mmap_mode="r")
                return CandleDataMemmap(
                    symbol=symbol,
                    base_timeframe=base_timeframe,
                    cache_dir=cache_dir,
                    start_time=read_from_iso,
                    end_time=end_iso,
                    ts_path=ts_path,
                    val_path=val_path,
                    count=int(ts_arr.shape[0]),
                )
        except Exception:
            pass

    timestamps: List[str] = []
    values: List[List[float]] = []
    with open(base_file, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            raise ValueError("CSV header missing")
        names = [h.strip() for h in header]
        idx_map = {name: i for i, name in enumerate(names)}

        i_ot = _idx_get(idx_map, "open_time")
        i_o = _idx_get(idx_map, "open")
        i_h = _idx_get(idx_map, "high")
        i_l = _idx_get(idx_map, "low")
        i_c = _idx_get(idx_map, "close")
        i_v = _idx_get(idx_map, "volume")
        i_qv = _idx_get(idx_map, "quote_asset_volume")
        i_nt = _idx_get(idx_map, "number_of_trades")
        i_tb = _idx_get(idx_map, "taker_buy_base_asset_volume")
        i_tq = _idx_get(idx_map, "taker_buy_quote_asset_volume")

        if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
            raise ValueError("CSV missing required columns")

        for row in rdr:
            if i_ot < 0 or i_ot >= len(row):
                continue
            ts = _coerce_ts(row[i_ot])
            if ts < read_from_iso:
                continue
            if ts >= end_iso:
                break
            try:
                o = float(row[i_o])
                h = float(row[i_h])
                l = float(row[i_l])
                c = float(row[i_c])
            except Exception:
                continue
            v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
            qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
            nt = float(row[i_nt]) if 0 <= i_nt < len(row) and row[i_nt] else 0.0
            tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
            tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0

            timestamps.append(ts)
            values.append([o, h, l, c, v, qv, nt, tb, tq])

    ts_arr = np.array(timestamps, dtype="U27")
    val_arr = np.array(values, dtype=np.float64)

    np.save(ts_path, ts_arr)
    np.save(val_path, val_arr)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": symbol,
                "base_timeframe": base_timeframe,
                "start_time": read_from_iso,
                "end_time": end_iso,
                "csv_mtime": csv_mtime,
            },
            f,
        )

    ts_arr = np.load(ts_path, mmap_mode="r")
    np.load(val_path, mmap_mode="r")
    return CandleDataMemmap(
        symbol=symbol,
        base_timeframe=base_timeframe,
        cache_dir=cache_dir,
        start_time=read_from_iso,
        end_time=end_iso,
        ts_path=ts_path,
        val_path=val_path,
        count=int(ts_arr.shape[0]),
    )


def preload_candle_data(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    base_timeframe: str = "1m",
    data_dir: Optional[str] = None,
    warmup_candles: int = 0,
    timeframe_minutes: int = 60,
) -> CandleData:

    """
    Preload candle data from CSV into memory for fast parallel backtesting.

    This function reads the CSV once and returns a CandleData object that can be
    passed to multiple worker processes without re-reading the file.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT_PERP")
        start_date: Start of data range (include warmup buffer before this)
        end_date: End of data range
        base_timeframe: Base timeframe of the CSV data (e.g., "1m")
        data_dir: Directory containing data files
        warmup_candles: Number of warmup candles to include before start_date
        timeframe_minutes: Target timeframe in minutes (for warmup calculation)

    Returns:
        CandleData object with all candles loaded into memory
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    sdt = _parse_iso8601_utc(start_date)
    edt = _parse_iso8601_utc(end_date)

    # Calculate how far back to read for warmup
    warmup_minutes = warmup_candles * timeframe_minutes
    read_from = sdt - timedelta(minutes=warmup_minutes)

    base_filename = f"{symbol}_{base_timeframe}.csv"
    base_file = os.path.join(data_dir, base_filename)

    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Data file not found: {base_file}")

    candles: List[Dict[str, Any]] = []
    read_from_iso = _isoformat_z(read_from)
    end_iso = _isoformat_z(edt)

    with open(base_file, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            return CandleData(symbol=symbol, base_timeframe=base_timeframe, candles=[],
                            start_time=read_from_iso, end_time=end_iso)

        names = [h.strip() for h in header]
        idx_map = {name: i for i, name in enumerate(names)}

        i_ot = _idx_get(idx_map, "open_time")
        i_o = _idx_get(idx_map, "open")
        i_h = _idx_get(idx_map, "high")
        i_l = _idx_get(idx_map, "low")
        i_c = _idx_get(idx_map, "close")
        i_v = _idx_get(idx_map, "volume")
        i_qv = _idx_get(idx_map, "quote_asset_volume")
        i_nt = _idx_get(idx_map, "number_of_trades")
        i_tb = _idx_get(idx_map, "taker_buy_base_asset_volume")
        i_tq = _idx_get(idx_map, "taker_buy_quote_asset_volume")

        if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
            return CandleData(
                symbol=symbol,
                base_timeframe=base_timeframe,
                candles=[],
                start_time=read_from_iso,
                end_time=end_iso,
            )


        for row in rdr:
            if i_ot < 0 or i_ot >= len(row):
                continue
            ts = _coerce_ts(row[i_ot])

            # Skip if before our read range
            if ts < read_from_iso:
                continue
            # Stop if after our end range
            if ts >= end_iso:
                break


            try:
                o = float(row[i_o])
                h = float(row[i_h])
                l = float(row[i_l])
                c = float(row[i_c])
            except Exception:
                continue

            v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
            qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
            nt = int(float(row[i_nt])) if 0 <= i_nt < len(row) and row[i_nt] else 0
            tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
            tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0

            candles.append({
                "open_time": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "quote_asset_volume": qv,
                "number_of_trades": nt,
                "taker_buy_base_asset_volume": tb,
                "taker_buy_quote_asset_volume": tq,
            })

    return CandleData(
        symbol=symbol,
        base_timeframe=base_timeframe,
        candles=candles,
        start_time=read_from_iso,
        end_time=end_iso,
    )


def preload_candle_data_aggregated(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    base_timeframe: str = "1m",
    target_timeframe_minutes: int = 60,
    data_dir: Optional[str] = None,
    warmup_candles: int = 0,
    session_start: Optional[datetime] = None,
) -> CandleDataAggregated:
    """
    Preload and pre-aggregate candle data for maximum optimization speed.

    This function:
    1. Reads the CSV once
    2. Aggregates candles to the target timeframe immediately
    3. Stores timestamps as Unix integers for O(log n) binary search
    4. Returns a CandleDataAggregated object that supports fast time-range queries

    This is ~10-50x faster than iter_candles_from_preloaded for repeated queries
    because aggregation happens once, not on every iteration.

    Args:
        symbol: Trading symbol
        start_date: Start date (warmup is subtracted from this)
        end_date: End date
        base_timeframe: Base CSV timeframe (e.g., "1m")
        target_timeframe_minutes: Target aggregation timeframe in minutes
        data_dir: Data directory
        warmup_candles: Number of warmup candles to include
        session_start: Optional anchor for multi-day timeframe alignment

    Returns:
        CandleDataAggregated with pre-aggregated candles and Unix timestamps
    """
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data")

    sdt = _parse_iso8601_utc(start_date)
    edt = _parse_iso8601_utc(end_date)

    # Calculate warmup buffer
    warmup_minutes = warmup_candles * target_timeframe_minutes
    read_from = sdt - timedelta(minutes=warmup_minutes)

    base_filename = f"{symbol}_{base_timeframe}.csv"
    base_file = os.path.join(data_dir, base_filename)

    if not os.path.exists(base_file):
        raise FileNotFoundError(f"Data file not found: {base_file}")

    base_minutes = _parse_timeframe_to_minutes(base_timeframe)
    no_agg = (target_timeframe_minutes == base_minutes)

    # Anchor for multi-day timeframes
    anchor: Optional[datetime] = None
    if target_timeframe_minutes % 1440 == 0 and session_start:
        anchor = session_start.replace(hour=0, minute=0, second=0, microsecond=0)
    elif target_timeframe_minutes % 1440 == 0:
        anchor = sdt.replace(hour=0, minute=0, second=0, microsecond=0)

    read_from_iso = _isoformat_z(read_from)
    end_iso = _isoformat_z(edt)

    # Pre-allocate lists for aggregated data
    agg_timestamps: List[int] = []
    agg_values: List[List[float]] = []

    # Aggregation state
    current_bucket_start: Optional[datetime] = None
    current_bucket_unix: int = 0
    agg_open: float = 0.0
    agg_high: float = 0.0
    agg_low: float = 0.0
    agg_close: float = 0.0
    agg_volume: float = 0.0
    agg_quote_volume: float = 0.0
    agg_trades: int = 0
    agg_taker_base: float = 0.0
    agg_taker_quote: float = 0.0

    def flush_bucket():
        nonlocal current_bucket_start
        if current_bucket_start is not None:
            agg_timestamps.append(current_bucket_unix)
            agg_values.append([
                agg_open, agg_high, agg_low, agg_close,
                agg_volume, agg_quote_volume, float(agg_trades),
                agg_taker_base, agg_taker_quote
            ])

    with open(base_file, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if not header:
            return CandleDataAggregated(
                symbol=symbol,
                timeframe_minutes=target_timeframe_minutes,
                timestamps_unix=np.array([], dtype=np.int64),
                values=np.zeros((0, 9), dtype=np.float64),
                start_unix=int(read_from.timestamp()),
                end_unix=int(edt.timestamp()),
            )

        names = [h.strip() for h in header]
        idx_map = {name: i for i, name in enumerate(names)}

        i_ot = _idx_get(idx_map, "open_time")
        i_o = _idx_get(idx_map, "open")
        i_h = _idx_get(idx_map, "high")
        i_l = _idx_get(idx_map, "low")
        i_c = _idx_get(idx_map, "close")
        i_v = _idx_get(idx_map, "volume")
        i_qv = _idx_get(idx_map, "quote_asset_volume")
        i_nt = _idx_get(idx_map, "number_of_trades")
        i_tb = _idx_get(idx_map, "taker_buy_base_asset_volume")
        i_tq = _idx_get(idx_map, "taker_buy_quote_asset_volume")

        if not _index_guard(i_ot, i_o, i_h, i_l, i_c):
            return CandleDataAggregated(
                symbol=symbol,
                timeframe_minutes=target_timeframe_minutes,
                timestamps_unix=np.array([], dtype=np.int64),
                values=np.zeros((0, 9), dtype=np.float64),
                start_unix=int(read_from.timestamp()),
                end_unix=int(edt.timestamp()),
            )

        for row in rdr:
            if i_ot < 0 or i_ot >= len(row):
                continue

            ts_str = _coerce_ts(row[i_ot])
            if ts_str < read_from_iso:
                continue
            if ts_str >= end_iso:
                break

            try:
                o = float(row[i_o])
                h = float(row[i_h])
                l = float(row[i_l])
                c = float(row[i_c])
            except Exception:
                continue

            v = float(row[i_v]) if 0 <= i_v < len(row) and row[i_v] else 0.0
            qv = float(row[i_qv]) if 0 <= i_qv < len(row) and row[i_qv] else 0.0
            nt = int(float(row[i_nt])) if 0 <= i_nt < len(row) and row[i_nt] else 0
            tb = float(row[i_tb]) if 0 <= i_tb < len(row) and row[i_tb] else 0.0
            tq = float(row[i_tq]) if 0 <= i_tq < len(row) and row[i_tq] else 0.0

            if no_agg:
                # No aggregation - store directly
                dt = _parse_iso8601_utc(ts_str)
                agg_timestamps.append(int(dt.timestamp()))
                agg_values.append([o, h, l, c, v, qv, float(nt), tb, tq])
            else:
                # Aggregate to target timeframe
                dt = _parse_iso8601_utc(ts_str)

                if anchor is not None:
                    bucket_start = _floor_to_timeframe(dt, target_timeframe_minutes, anchor=anchor)
                else:
                    bucket_start = _floor_to_timeframe(dt, target_timeframe_minutes)

                if current_bucket_start is None:
                    # First candle
                    current_bucket_start = bucket_start
                    current_bucket_unix = int(bucket_start.timestamp())
                    agg_open, agg_high, agg_low, agg_close = o, h, l, c
                    agg_volume, agg_quote_volume = v, qv
                    agg_trades = nt
                    agg_taker_base, agg_taker_quote = tb, tq
                elif bucket_start != current_bucket_start:
                    # New bucket - flush previous
                    flush_bucket()
                    current_bucket_start = bucket_start
                    current_bucket_unix = int(bucket_start.timestamp())
                    agg_open, agg_high, agg_low, agg_close = o, h, l, c
                    agg_volume, agg_quote_volume = v, qv
                    agg_trades = nt
                    agg_taker_base, agg_taker_quote = tb, tq
                else:
                    # Same bucket - aggregate
                    agg_high = max(agg_high, h)
                    agg_low = min(agg_low, l)
                    agg_close = c
                    agg_volume += v
                    agg_quote_volume += qv
                    agg_trades += nt
                    agg_taker_base += tb
                    agg_taker_quote += tq

        # Flush last bucket
        if not no_agg:
            flush_bucket()

    return CandleDataAggregated(
        symbol=symbol,
        timeframe_minutes=target_timeframe_minutes,
        timestamps_unix=np.array(agg_timestamps, dtype=np.int64),
        values=np.array(agg_values, dtype=np.float64) if agg_values else np.zeros((0, 9), dtype=np.float64),
        start_unix=int(read_from.timestamp()),
        end_unix=int(edt.timestamp()),
    )


def iter_candles_from_aggregated(
    candle_data: CandleDataAggregated,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> Iterable[Dict[str, object]]:
    """
    Ultra-fast iteration over pre-aggregated candle data.

    Uses binary search for O(log n) time-range lookup instead of O(n) iteration.
    No string parsing, no datetime conversions during iteration.

    Args:
        candle_data: Pre-aggregated CandleDataAggregated object
        start_date: Start of iteration range
        end_date: End of iteration range

    Yields:
        Candle dicts with pre-computed values
    """
    sdt = _parse_iso8601_utc(start_date)
    edt = _parse_iso8601_utc(end_date)

    start_unix = int(sdt.timestamp())
    end_unix = int(edt.timestamp())

    # Binary search for indices
    start_idx, end_idx = candle_data.find_range_indices(start_unix, end_unix)

    tf_label = _format_timeframe_label(candle_data.timeframe_minutes)
    symbol = candle_data.symbol

    # Direct array access - no per-candle parsing
    timestamps = candle_data.timestamps_unix
    values = candle_data.values

    for i in range(start_idx, end_idx):
        ts_unix = timestamps[i]
        row = values[i]

        # Convert Unix timestamp to ISO format only at yield time
        dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
        ts_iso = dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        yield {
            "symbol": symbol,
            "timeframe": tf_label,
            "candle": {
                "open_time": ts_iso,
                "open": row[0],
                "high": row[1],
                "low": row[2],
                "close": row[3],
                "volume": row[4],
                "quote_asset_volume": row[5],
                "number_of_trades": int(row[6]),
                "taker_buy_base_asset_volume": row[7],
                "taker_buy_quote_asset_volume": row[8],
            },
        }


def _format_timeframe_label(minutes: int) -> str:
    """Format timeframe minutes as a human-readable label (e.g., '1m', '1h', '1d')."""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes % 1440 == 0:
        # Exact days
        return f"{minutes // 1440}d"
    elif minutes % 60 == 0:
        # Exact hours
        return f"{minutes // 60}h"
    else:
        # Non-standard (e.g., 90m) - keep as minutes
        return f"{minutes}m"


def iter_candles_from_preloaded(
    candle_data: Union[CandleData, CandleDataMemmap],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    timeframe_minutes: int = 60,
    session_start: Optional[datetime] = None,
) -> Iterable[Dict[str, object]]:

    """
    Iterate over preloaded candle data, aggregating to the target timeframe.

    Args:
        candle_data: Preloaded CandleData object
        start_date: Start of iteration range
        end_date: End of iteration range
        timeframe_minutes: Target timeframe in minutes
        session_start: Optional anchor for multi-day timeframes

    Yields:
        Aggregated candle dicts
    """
    sdt = _parse_iso8601_utc(start_date)
    edt = _parse_iso8601_utc(end_date)

    start_iso = _isoformat_z(sdt)
    end_iso = _isoformat_z(edt)

    # For 1m base -> 1m target, no aggregation needed
    base_minutes = _parse_timeframe_to_minutes(candle_data.base_timeframe)
    no_agg = (timeframe_minutes == base_minutes)

    if isinstance(candle_data, CandleDataMemmap):
        ts_arr, val_arr = candle_data.load()
        if no_agg:
            for ts, row in zip(ts_arr, val_arr):
                ts_str = str(ts)
                ts_norm = _coerce_ts(ts_str)
                if ts_norm < start_iso:
                    continue
                if ts_norm >= end_iso:
                    break
                candle = {
                    "open_time": ts_norm,
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
                yield {
                    "symbol": candle_data.symbol,
                    "timeframe": _format_timeframe_label(timeframe_minutes),
                    "candle": candle,
                }
            return


    anchor: Optional[datetime] = None
    if timeframe_minutes % 1440 == 0 and session_start:
        anchor = session_start.replace(hour=0, minute=0, second=0, microsecond=0)

    if no_agg:
        # Fast path: just filter by date range
        if isinstance(candle_data, CandleDataMemmap):
            ts_arr, val_arr = candle_data.load()
            for ts, row in zip(ts_arr, val_arr):
                ts_str = str(ts)
                ts_norm = _coerce_ts(ts_str)
                if ts_norm < start_iso:
                    continue
                if ts_norm >= end_iso:
                    break
                candle = {
                    "open_time": ts_norm,
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
                yield {
                    "symbol": candle_data.symbol,
                    "timeframe": _format_timeframe_label(timeframe_minutes),
                    "candle": candle,
                }
        else:
            for c in candle_data.candles:
                ts = _coerce_ts(c["open_time"])
                if ts < start_iso:
                    continue
                if ts >= end_iso:
                    break
                yield {
                    "symbol": candle_data.symbol,
                    "timeframe": _format_timeframe_label(timeframe_minutes),
                    "candle": c,
                }
        return



    # Aggregation path
    current_bucket_start: Optional[datetime] = None
    agg_open: Optional[float] = None
    agg_high: Optional[float] = None
    agg_low: Optional[float] = None
    agg_close: Optional[float] = None
    agg_volume: float = 0.0
    agg_quote_volume: float = 0.0
    agg_trades: int = 0
    agg_taker_base: float = 0.0
    agg_taker_quote: float = 0.0

    def flush():
        nonlocal current_bucket_start
        if current_bucket_start is None:
            return None
        if current_bucket_start >= edt:
            return None
        if current_bucket_start >= sdt:
            return {
                "symbol": candle_data.symbol,
                "timeframe": _format_timeframe_label(timeframe_minutes),
                "candle": {
                    "open_time": _isoformat_z(current_bucket_start),
                    "open": agg_open,
                    "high": agg_high,
                    "low": agg_low,
                    "close": agg_close,
                    "volume": agg_volume,
                    "quote_asset_volume": agg_quote_volume,
                    "number_of_trades": agg_trades,
                    "taker_buy_base_asset_volume": agg_taker_base,
                    "taker_buy_quote_asset_volume": agg_taker_quote,
                },
            }
        return None

    if isinstance(candle_data, CandleDataMemmap):
        ts_arr, val_arr = candle_data.load()
        iterator = ((str(ts), row) for ts, row in zip(ts_arr, val_arr))
    else:
        iterator = ((c["open_time"], c) for c in cast(CandleData, candle_data).candles)

    for ts_raw, row in iterator:
        ts_norm = _coerce_ts(ts_raw)
        dt = _parse_iso8601_utc(ts_norm)

        if dt >= edt:
            out = flush()
            if out is not None:
                yield out
            break

        if isinstance(candle_data, CandleDataMemmap):
            row_vals = cast(List[float], row)
            o, h, l, cl = float(row_vals[0]), float(row_vals[1]), float(row_vals[2]), float(row_vals[3])
            v = float(row_vals[4])
            qv = float(row_vals[5])
            nt = int(row_vals[6])
            tb = float(row_vals[7])
            tq = float(row_vals[8])
        else:
            row_dict = cast(Dict[str, Any], row)
            o, h, l, cl = row_dict["open"], row_dict["high"], row_dict["low"], row_dict["close"]
            v = row_dict.get("volume", 0.0)
            qv = row_dict.get("quote_asset_volume", 0.0)
            nt = row_dict.get("number_of_trades", 0)
            tb = row_dict.get("taker_buy_base_asset_volume", 0.0)
            tq = row_dict.get("taker_buy_quote_asset_volume", 0.0)


        if anchor is not None:
            bucket_start = _floor_to_timeframe(dt, timeframe_minutes, anchor=anchor)
        else:
            bucket_start = _floor_to_timeframe(dt, timeframe_minutes)

        if current_bucket_start is None:
            current_bucket_start = bucket_start
            agg_open = o
            agg_high = h
            agg_low = l
            agg_close = cl
            agg_volume = v
            agg_quote_volume = qv
            agg_trades = nt
            agg_taker_base = tb
            agg_taker_quote = tq
            continue

        if bucket_start != current_bucket_start:
            out = flush()
            current_bucket_start = bucket_start
            agg_open = o
            agg_high = h
            agg_low = l
            agg_close = cl
            agg_volume = v
            agg_quote_volume = qv
            agg_trades = nt
            agg_taker_base = tb
            agg_taker_quote = tq
            if out is not None:
                yield out
        else:
            if agg_high is None or h > agg_high:
                agg_high = h
            if agg_low is None or l < agg_low:
                agg_low = l
            agg_close = cl
            agg_volume += v
            agg_quote_volume += qv
            agg_trades += nt
            agg_taker_base += tb
            agg_taker_quote += tq
    else:
        out = flush()
        if out is not None:
            yield out
