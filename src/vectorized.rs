//! Vectorized Strategy Fast Path
//!
//! Two standalone #[pyfunction]s:
//! - `compute_indicators_bulk`: pre-compute all indicators over a candle array
//! - `run_signals_backtest`: run a full backtest driven by a signal array
//!
//! Both avoid per-candle Python↔Rust dict construction.

use std::collections::HashMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::engine::RustTradingEngine;
use crate::indicators::*;
use crate::types::{OrderSide, PositionSide};

// ---------------------------------------------------------------------------
// Signal constants (must match Python VectorStrategy.HOLD / LONG / SHORT / CLOSE)
// ---------------------------------------------------------------------------

const SIGNAL_HOLD: i8 = 0;
const SIGNAL_LONG: i8 = 1;
const SIGNAL_SHORT: i8 = 2;
const SIGNAL_CLOSE: i8 = 3;

// ---------------------------------------------------------------------------
// Helper: build IndicatorType from a Python spec tuple
// ---------------------------------------------------------------------------

fn build_indicator(type_str: &str, label: &str, params: &Bound<'_, PyDict>) -> PyResult<IndicatorType> {
    match type_str.to_lowercase().as_str() {
        "sma" => {
            let period: usize = params.get_item("period")?.unwrap().extract()?;
            let source_str: String = params
                .get_item("source")?
                .map(|v| v.extract::<String>().unwrap_or_else(|_| "close".into()))
                .unwrap_or_else(|| "close".into());
            let source = CandleSource::from_str(&source_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(IndicatorType::Sma(SmaIndicator::new(
                label.to_string(),
                period,
                source,
            )))
        }
        "ema" => {
            let period: usize = params.get_item("period")?.unwrap().extract()?;
            let source_str: String = params
                .get_item("source")?
                .map(|v| v.extract::<String>().unwrap_or_else(|_| "close".into()))
                .unwrap_or_else(|| "close".into());
            let source = CandleSource::from_str(&source_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(IndicatorType::Ema(EmaIndicator::new(
                label.to_string(),
                period,
                source,
            )))
        }
        "rsi" => {
            let period: usize = params.get_item("period")?.unwrap().extract()?;
            let source_str: String = params
                .get_item("source")?
                .map(|v| v.extract::<String>().unwrap_or_else(|_| "close".into()))
                .unwrap_or_else(|| "close".into());
            let source = CandleSource::from_str(&source_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(IndicatorType::Rsi(RsiIndicator::new(
                label.to_string(),
                period,
                source,
            )))
        }
        "atr" => {
            let period: usize = params.get_item("period")?.unwrap().extract()?;
            Ok(IndicatorType::Atr(AtrIndicator::new(
                label.to_string(),
                period,
            )))
        }
        "bollinger_bands" | "bb" => {
            let period: usize = params.get_item("period")?.unwrap().extract()?;
            let multiplier: f64 = params
                .get_item("multiplier")?
                .map(|v| v.extract::<f64>().unwrap_or(2.0))
                .unwrap_or(2.0);
            let source_str: String = params
                .get_item("source")?
                .map(|v| v.extract::<String>().unwrap_or_else(|_| "close".into()))
                .unwrap_or_else(|| "close".into());
            let source = CandleSource::from_str(&source_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(IndicatorType::BollingerBands(
                BollingerBandsIndicator::new(label.to_string(), period, multiplier, source),
            ))
        }
        "macd" => {
            let fast: usize = params
                .get_item("fast_period")?
                .map(|v| v.extract::<usize>().unwrap_or(12))
                .unwrap_or(12);
            let slow: usize = params
                .get_item("slow_period")?
                .map(|v| v.extract::<usize>().unwrap_or(26))
                .unwrap_or(26);
            let signal: usize = params
                .get_item("signal_period")?
                .map(|v| v.extract::<usize>().unwrap_or(9))
                .unwrap_or(9);
            let source_str: String = params
                .get_item("source")?
                .map(|v| v.extract::<String>().unwrap_or_else(|_| "close".into()))
                .unwrap_or_else(|| "close".into());
            let source = CandleSource::from_str(&source_str)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(IndicatorType::Macd(MacdIndicator::new(
                label.to_string(),
                fast,
                slow,
                signal,
                source,
            )))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown indicator type: '{}'",
            type_str
        ))),
    }
}

// ---------------------------------------------------------------------------
// compute_indicators_bulk
// ---------------------------------------------------------------------------

/// Compute all indicators over a candle array in a single Rust loop.
///
/// Args:
///     candle_data: 2D float64 numpy array, shape (N, >=5).
///                  Columns: [open_time, open, high, low, close, ...].
///     indicator_specs: list of (type_str, label, params_dict) tuples.
///
/// Returns:
///     dict mapping label strings to 1D float64 numpy arrays (NaN where
///     indicator hasn't warmed up yet).
#[pyfunction]
pub fn compute_indicators_bulk<'py>(
    py: Python<'py>,
    candle_data: PyReadonlyArray2<'py, f64>,
    indicator_specs: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    let arr = candle_data.as_array();
    let n_candles = arr.nrows();

    // Parse indicator specs
    let mut indicators: Vec<IndicatorType> = Vec::new();
    let mut all_labels: Vec<Vec<String>> = Vec::new(); // labels per indicator (may be >1 for BB, MACD)

    for item in indicator_specs.iter() {
        let tuple = item.downcast::<pyo3::types::PyTuple>()?;
        let type_str: String = tuple.get_item(0)?.extract()?;
        let label: String = tuple.get_item(1)?.extract()?;
        let params = tuple.get_item(2)?.downcast::<PyDict>()?.clone();

        // Determine output labels for this indicator type
        let labels: Vec<String> = match type_str.to_lowercase().as_str() {
            "bollinger_bands" | "bb" => vec![
                format!("{}.upper", label),
                format!("{}.mid", label),
                format!("{}.lower", label),
            ],
            "macd" => vec![
                format!("{}.line", label),
                format!("{}.signal", label),
                format!("{}.hist", label),
            ],
            _ => vec![label.clone()],
        };

        let ind = build_indicator(&type_str, &label, &params)?;
        indicators.push(ind);
        all_labels.push(labels);
    }

    // Allocate output buffers (NaN-filled)
    let mut outputs: HashMap<String, Vec<f64>> = HashMap::new();
    for labels in &all_labels {
        for lbl in labels {
            outputs.insert(lbl.clone(), vec![f64::NAN; n_candles]);
        }
    }

    // Run all indicators through all candles
    for i in 0..n_candles {
        let row = arr.row(i);
        let open = row[1];
        let high = row[2];
        let low = row[3];
        let close = row[4];

        for (ind_idx, ind) in indicators.iter_mut().enumerate() {
            let kvs = ind.push_candle(open, high, low, close);
            for (key, val) in kvs {
                if let Some(buf) = outputs.get_mut(&key) {
                    buf[i] = val;
                }
            }
            // If no output (warmup), NaN is already in place
            let _ = ind_idx; // suppress unused warning
        }
    }

    // Build return dict of numpy arrays
    let result = PyDict::new(py);
    for (key, values) in &outputs {
        let np_arr = PyArray1::from_vec(py, values.clone());
        result.set_item(key, np_arr)?;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// run_signals_backtest
// ---------------------------------------------------------------------------

/// Run a full backtest driven by a signal array — entirely in Rust.
///
/// The signal array uses the same action codes as the RL discrete actions:
///   0 = HOLD, 1 = OPEN_LONG, 2 = OPEN_SHORT, 3 = CLOSE.
///
/// Returns a dict with equity curve, trade data, and final state — everything
/// needed to compute performance metrics.
#[pyfunction]
#[pyo3(signature = (
    candle_data,
    signals,
    starting_cash,
    leverage,
    margin_mode,
    slippage_pct,
    margin_pct,
    sl_pct,
    tp_pct,
    close_at_end,
    warmup_count,
    timeframe_minutes=60,
    symbol="BTCUSDT_PERP".to_string()
))]
#[allow(clippy::too_many_arguments)]
pub fn run_signals_backtest<'py>(
    py: Python<'py>,
    candle_data: PyReadonlyArray2<'py, f64>,
    signals: PyReadonlyArray1<'py, i8>,
    starting_cash: f64,
    leverage: f64,
    margin_mode: String,
    slippage_pct: f64,
    margin_pct: f64,
    sl_pct: f64,
    tp_pct: f64,
    close_at_end: bool,
    warmup_count: i64,
    timeframe_minutes: i64,
    symbol: String,
) -> PyResult<Bound<'py, PyDict>> {
    let arr = candle_data.as_array();
    let sig = signals.as_array();
    let n = arr.nrows();

    if sig.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "candle_data has {} rows but signals has {} elements",
            n,
            sig.len()
        )));
    }

    // Normalise margin mode
    let mode = margin_mode.trim().to_lowercase();
    let normalized_mode = match mode.as_str() {
        "cross" | "crossed" => "cross",
        "isolated" | "iso" => "isolated",
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported margin_mode: {}",
                margin_mode
            )))
        }
    };

    // Create engine directly (no PyO3 constructor overhead)
    let mut engine = RustTradingEngine::new_internal(
        symbol,
        normalized_mode.to_string(),
        leverage,
        starting_cash,
        slippage_pct,
        timeframe_minutes,
    );

    // Tracking vectors
    let effective_start = warmup_count.max(0) as usize;
    let n_post_warmup = if n > effective_start { n - effective_start } else { 0 };
    let mut equities: Vec<f64> = Vec::with_capacity(n_post_warmup);
    let mut timestamps: Vec<f64> = Vec::with_capacity(n_post_warmup);
    let mut trade_pnls: Vec<f64> = Vec::new();
    let mut trade_sides: Vec<String> = Vec::new();
    let mut trade_holding_mins: Vec<f64> = Vec::new();
    let mut open_longs: i64 = 0;
    let mut open_shorts: i64 = 0;

    for i in 0..n {
        let row = arr.row(i);
        let candle: [f64; 10] = [
            row[0], // open_time (unix)
            row[1], // open
            row[2], // high
            row[3], // low
            row[4], // close
            if arr.ncols() > 5 { row[5] } else { 0.0 }, // volume
            if arr.ncols() > 6 { row[6] } else { 0.0 }, // quote_asset_volume
            if arr.ncols() > 7 { row[7] } else { 0.0 }, // number_of_trades
            if arr.ncols() > 8 { row[8] } else { 0.0 }, // taker_buy_base
            if arr.ncols() > 9 { row[9] } else { 0.0 }, // taker_buy_quote
        ];
        let close_price = candle[4];

        // Snapshot position state before processing
        let prev_has_pos = engine.position.is_some();
        let prev_pos_side = engine.position.as_ref().map(|p| p.side);
        let prev_pos_opened_idx = engine.position.as_ref().map(|p| p.opened_index);
        let prev_realized_pnl = engine.realized_pnl;

        // Apply signal (only after warmup)
        if (i as i64) >= warmup_count {
            let signal = sig[i];
            match signal {
                SIGNAL_LONG => {
                    if !engine.position.is_some() {
                        let sl = if sl_pct > 0.0 {
                            Some(close_price * (1.0 - sl_pct))
                        } else {
                            None
                        };
                        let tp = if tp_pct > 0.0 {
                            Some(close_price * (1.0 + tp_pct))
                        } else {
                            None
                        };
                        engine.create_order_internal(
                            OrderSide::LONG,
                            close_price,
                            margin_pct,
                            tp,
                            sl,
                        );
                    }
                }
                SIGNAL_SHORT => {
                    if !engine.position.is_some() {
                        let sl = if sl_pct > 0.0 {
                            Some(close_price * (1.0 + sl_pct))
                        } else {
                            None
                        };
                        let tp = if tp_pct > 0.0 {
                            Some(close_price * (1.0 - tp_pct))
                        } else {
                            None
                        };
                        engine.create_order_internal(
                            OrderSide::SHORT,
                            close_price,
                            margin_pct,
                            tp,
                            sl,
                        );
                    }
                }
                SIGNAL_CLOSE => {
                    if engine.position.is_some() {
                        engine.close_order_internal(close_price);
                    }
                }
                SIGNAL_HOLD | _ => {} // do nothing
            }
        }

        // Process the candle through the state machine
        let (_had_tp, _had_sl, _had_liq) = engine.on_candle_raw(py, &candle)?;

        // Detect trade open
        if !prev_has_pos && engine.position.is_some() {
            match engine.position.as_ref().unwrap().side {
                PositionSide::LONG => open_longs += 1,
                PositionSide::SHORT => open_shorts += 1,
            }
        }

        // Detect trade close
        if prev_has_pos && engine.position.is_none() {
            let pnl = engine.realized_pnl - prev_realized_pnl;
            trade_pnls.push(pnl);
            if let Some(side) = prev_pos_side {
                trade_sides.push(side.as_str().to_string());
            }
            if let Some(opened_idx) = prev_pos_opened_idx {
                let holding_bars = engine.current_index - opened_idx;
                trade_holding_mins.push((holding_bars * timeframe_minutes) as f64);
            }
        }

        // Track equity (post-warmup only)
        if (i as i64) >= warmup_count {
            equities.push(engine.cash + engine.unrealized_pnl);
            timestamps.push(candle[0]); // unix timestamp
        }
    }

    // Close at end if requested
    if close_at_end && engine.position.is_some() {
        let last_close = arr.row(n - 1)[4];
        let prev_realized = engine.realized_pnl;
        let prev_side = engine.position.as_ref().map(|p| p.side);
        let prev_opened_idx = engine.position.as_ref().map(|p| p.opened_index);

        engine.close_order_internal(last_close);
        // Process one more candle tick to fill the close order
        // Re-use the last candle data
        let last_row = arr.row(n - 1);
        let last_candle: [f64; 10] = [
            last_row[0],
            last_row[1],
            last_row[2],
            last_row[3],
            last_row[4],
            if arr.ncols() > 5 { last_row[5] } else { 0.0 },
            if arr.ncols() > 6 { last_row[6] } else { 0.0 },
            if arr.ncols() > 7 { last_row[7] } else { 0.0 },
            if arr.ncols() > 8 { last_row[8] } else { 0.0 },
            if arr.ncols() > 9 { last_row[9] } else { 0.0 },
        ];
        let _ = engine.on_candle_raw(py, &last_candle)?;

        // Record the close trade
        if engine.position.is_none() {
            let pnl = engine.realized_pnl - prev_realized;
            trade_pnls.push(pnl);
            if let Some(side) = prev_side {
                trade_sides.push(side.as_str().to_string());
            }
            if let Some(opened_idx) = prev_opened_idx {
                let holding_bars = engine.current_index - opened_idx;
                trade_holding_mins.push((holding_bars * timeframe_minutes) as f64);
            }
        }

        // Update last equity
        if !equities.is_empty() {
            let last_idx = equities.len() - 1;
            equities[last_idx] = engine.cash + engine.unrealized_pnl;
        }
    }

    // Build return dict
    let result = PyDict::new(py);
    result.set_item("equities", PyArray1::from_vec(py, equities))?;
    result.set_item("timestamps", PyArray1::from_vec(py, timestamps))?;
    result.set_item("final_equity", engine.cash + engine.unrealized_pnl)?;
    result.set_item("final_cash", engine.cash)?;
    result.set_item("realized_pnl", engine.realized_pnl)?;

    // Trade data as Python lists
    let pnl_list = PyList::new(py, &trade_pnls)?;
    result.set_item("trade_pnls", pnl_list)?;
    let sides_list = PyList::new(py, &trade_sides)?;
    result.set_item("trade_sides", sides_list)?;
    let holding_list = PyList::new(py, &trade_holding_mins)?;
    result.set_item("trade_holding_minutes", holding_list)?;
    result.set_item("open_longs", open_longs)?;
    result.set_item("open_shorts", open_shorts)?;
    result.set_item("total_funding", 0.0_f64)?; // funding skipped in fast path
    // Total fees: cash = starting_cash - fees + realized_pnl  =>  fees = starting_cash + realized_pnl - cash
    let total_fees = starting_cash + engine.realized_pnl - engine.cash;
    result.set_item("total_fees", total_fees)?;

    Ok(result)
}
