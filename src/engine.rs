//! Trading Engine — Phase 4: Order Execution & Candle Loop
//!
//! Full state machine: create_order, close_order, cancel_order, on_candle,
//! trigger resolution with in-memory binary search for intra-candle optimization,
//! liquidation, TP/SL, slippage, funding, and snapshot.

use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use numpy::PyArray1;

use crate::indicators::*;
use crate::rl::{
    ObsFeatureType, ObsHistory, ObservationSpace, RlActionSpace, RlConfig, RewardFn,
    observations::{DrawdownState, EngineState, IndicatorObsConfig, RawCandle, SmaRatioConfig},
    actions::{
        ContinuousActionSpace, DiscreteActionSpace, DiscreteWithSizingSpace,
    },
    rewards::*,
};
use crate::types::*;

// ---------------------------------------------------------------------------
// EngineConfig — static params extracted at init, avoiding FFI per tick
// ---------------------------------------------------------------------------

pub struct EngineConfig {
    pub symbol: String,
    pub margin_mode: String, // "cross" or "isolated"
    pub leverage: f64,
    pub slippage_pct: f64,
    pub timeframe_minutes: i64,
    /// Funding data indexed by Unix timestamp for fast BTreeMap range queries.
    /// Each entry stores (original_iso_string, funding_rate).
    pub funding_data: Option<BTreeMap<i64, (String, f64)>>,
}

// ---------------------------------------------------------------------------
// LastCandleMeta — tracks metadata of the most recent candle
// ---------------------------------------------------------------------------

pub struct LastCandleMeta {
    pub open_time: Option<String>,
    pub volume: f64,
    pub quote_asset_volume: f64,
    pub number_of_trades: i64,
    pub taker_buy_base_asset_volume: f64,
    pub taker_buy_quote_asset_volume: f64,
}

impl Default for LastCandleMeta {
    fn default() -> Self {
        Self {
            open_time: None,
            volume: 0.0,
            quote_asset_volume: 0.0,
            number_of_trades: 0,
            taker_buy_base_asset_volume: 0.0,
            taker_buy_quote_asset_volume: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// CandleData — extracted from PyDict for internal use (no FFI per field)
// ---------------------------------------------------------------------------

struct CandleData {
    open_time: Option<String>,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

// ---------------------------------------------------------------------------
// Liquidation price calculations (free functions)
// ---------------------------------------------------------------------------

/// Isolated-margin liquidation price.
pub fn calc_isolated_liq_price(side: PositionSide, entry: f64, leverage: f64) -> f64 {
    match side {
        PositionSide::LONG => r_price(entry * (1.0 - 1.0 / leverage + MMR)),
        PositionSide::SHORT => r_price(entry * (1.0 + 1.0 / leverage - MMR)),
    }
}

/// Cross-margin liquidation price.
pub fn calc_cross_liq_price(side: PositionSide, entry: f64, size: f64, eq0: f64) -> f64 {
    if size <= 0.0 {
        return f64::INFINITY;
    }
    match side {
        PositionSide::LONG => {
            let p = (entry * size - eq0) / (size * (1.0 - MMR));
            r_price(p)
        }
        PositionSide::SHORT => {
            let p = (eq0 / size + entry) / (1.0 + MMR);
            r_price(p)
        }
    }
}

// ---------------------------------------------------------------------------
// ISO timestamp parsing helpers
// ---------------------------------------------------------------------------

fn parse_iso_to_unix(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.and_utc().timestamp());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc().timestamp());
    }
    None
}

fn parse_funding_dict(py_dict: &Bound<'_, PyDict>) -> PyResult<BTreeMap<i64, (String, f64)>> {
    let mut map = BTreeMap::new();
    for (key, value) in py_dict.iter() {
        let time_str: String = key.extract()?;
        let rate: f64 = value.extract()?;
        if let Some(ts) = parse_iso_to_unix(&time_str) {
            map.insert(ts, (time_str, rate));
        }
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// RustTradingEngine
// ---------------------------------------------------------------------------

#[pyclass]
pub struct RustTradingEngine {
    config: EngineConfig,

    // Scalar state
    pub(crate) cash: f64,
    pub(crate) current_index: i64,
    pub(crate) realized_pnl: f64,
    pub(crate) unrealized_pnl: f64,
    pub(crate) insurance_fund: f64,
    pub(crate) total_funding_paid: f64,
    pub(crate) cross_used_margin: f64,

    // Position & order slots
    pub(crate) position: Option<Position>,
    pub(crate) open_order: Option<Order>,
    pub(crate) close_request: Option<Order>,
    pub(crate) tp: Option<Order>,
    pub(crate) sl: Option<Order>,

    pub(crate) pending_open_order: Option<Order>,
    pub(crate) pending_close_request: Option<Order>,
    pub(crate) pending_tp: Option<Order>,
    pub(crate) pending_sl: Option<Order>,

    // Scheduled cancels: Vec preserves insertion order (matches Python dict)
    // Each entry: (order_id, eligible_index, role)
    pub(crate) scheduled_cancels: Vec<(String, i64, String)>,
    pub(crate) last_candle_meta: LastCandleMeta,

    // In-memory 1m candles for intra-candle trigger resolution (binary search)
    // Each entry: (unix_timestamp, low, high) — must be sorted by timestamp
    pub(crate) intra_candles: Option<Vec<(i64, f64, f64)>>,

    // Phase 5: Native indicators
    pub(crate) indicators: BTreeMap<String, f64>,
    pub(crate) registered_indicators: Vec<IndicatorType>,

    // Phase 6: RL fast-path config (None until setup_rl_env is called)
    rl_config: Option<RlConfig>,
}

// -- Internal (non-PyO3) helpers -------------------------------------------

impl RustTradingEngine {
    /// Refresh unrealized PnL based on current mark price.
    pub(crate) fn refresh_upnl(&mut self, mark_price: f64) {
        match self.position {
            None => {
                self.unrealized_pnl = 0.0;
            }
            Some(ref mut pos) => {
                let upnl = match pos.side {
                    PositionSide::LONG => (mark_price - pos.entry_price) * pos.size,
                    PositionSide::SHORT => (pos.entry_price - mark_price) * pos.size,
                };
                pos.unrealized_pnl = upnl;
                self.unrealized_pnl = upnl;
            }
        }
    }

    /// Apply funding payments for all funding events within this candle window.
    pub(crate) fn apply_funding<'py>(
        &mut self,
        py: Python<'py>,
        candle_open_time: &str,
        events: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        let funding_data = match self.config.funding_data.as_ref() {
            Some(fd) => fd,
            None => return Ok(()),
        };
        if self.position.is_none() {
            return Ok(());
        }
        let candle_start_ts = match parse_iso_to_unix(candle_open_time) {
            Some(ts) => ts,
            None => return Ok(()),
        };
        let candle_end_ts = candle_start_ts + self.config.timeframe_minutes * 60;

        let entries: Vec<(String, f64)> = funding_data
            .range(candle_start_ts..candle_end_ts)
            .map(|(_, (s, r))| (s.clone(), *r))
            .collect();

        for (time_str, rate) in entries {
            let (notional, is_long, side_str) = {
                let pos = self.position.as_ref().unwrap();
                (
                    pos.notional,
                    pos.side == PositionSide::LONG,
                    pos.side.as_str(),
                )
            };
            let funding_cost = if is_long {
                r_usd(notional * rate)
            } else {
                r_usd(-notional * rate)
            };
            self.cash = r_usd(self.cash - funding_cost);
            self.total_funding_paid = r_usd(self.total_funding_paid + funding_cost);

            let ev = PyDict::new(py);
            ev.set_item("type", "funding")?;
            ev.set_item("funding_time", &time_str)?;
            ev.set_item("funding_rate", rate)?;
            ev.set_item("notional", r_usd(notional))?;
            ev.set_item("funding_cost", r_usd(funding_cost))?;
            ev.set_item("side", side_str)?;
            ev.set_item("total_funding_paid", r_usd(self.total_funding_paid))?;
            events.append(ev)?;
        }
        Ok(())
    }

    // -- Validation helpers ------------------------------------------------

    fn validate_params(&self, price: f64, qty: f64) -> Option<String> {
        if price <= 0.0 {
            return Some("price must be positive".to_string());
        }
        if qty <= 0.0 {
            return Some(
                "Computed quantity must be positive (check margin_pct and equity)".to_string(),
            );
        }
        let notional = price * qty;
        if notional < MIN_NOTIONAL {
            return Some(format!("Min notional ${} not met", MIN_NOTIONAL));
        }
        None
    }

    fn opposite_side_position(&self, side: OrderSide) -> bool {
        match &self.position {
            None => false,
            Some(pos) => pos.side.as_str() != side.as_str(),
        }
    }

    /// Risk check used by both order creation and fill.
    fn risk_check(&self, side: OrderSide, price: f64, qty: f64) -> Option<String> {
        let notional = price * qty;
        let open_fee = notional * OPEN_FEE_RATE;

        if self.config.margin_mode == "isolated" {
            let required = notional / self.config.leverage + open_fee;
            if self.cash < required - 1e-12 {
                return Some("Insufficient margin".to_string());
            }
            return None;
        }

        // Cross
        let equity_after_open_fee = (self.cash - open_fee) + self.unrealized_pnl;
        let existing_notional = self
            .position
            .as_ref()
            .map_or(0.0, |p| p.entry_price * p.size);
        let existing_size = self.position.as_ref().map_or(0.0, |p| p.size);
        let existing_entry = self.position.as_ref().map_or(0.0, |p| p.entry_price);

        let combined_notional = (price * qty) + existing_notional;
        let required_margin = combined_notional / self.config.leverage;
        if equity_after_open_fee < required_margin - 1e-12 {
            return Some("Insufficient margin".to_string());
        }

        let new_total_size = existing_size + qty;
        let new_entry = if new_total_size > 0.0 {
            (existing_entry * existing_size + price * qty) / new_total_size
        } else {
            price
        };
        let eq0_after_open_fee = self.cash - open_fee;
        let p_liq = calc_cross_liq_price(
            PositionSide::from(side),
            new_entry,
            new_total_size,
            eq0_after_open_fee,
        );

        match side {
            OrderSide::LONG => {
                if p_liq >= price - 1e-12 {
                    return Some("Immediate liquidation risk".to_string());
                }
            }
            OrderSide::SHORT => {
                if p_liq <= price + 1e-12 {
                    return Some("Immediate liquidation risk".to_string());
                }
            }
        }
        None
    }

    // -- Slippage ----------------------------------------------------------

    fn apply_slippage(&self, price: f64, position_side: PositionSide) -> f64 {
        if self.config.slippage_pct <= 0.0 {
            return price;
        }
        match position_side {
            PositionSide::LONG => r_price(price * (1.0 - self.config.slippage_pct)),
            PositionSide::SHORT => r_price(price * (1.0 + self.config.slippage_pct)),
        }
    }

    // -- Schedule cancel helper --------------------------------------------

    fn schedule_cancel(&mut self, order_id: String, eligible_index: i64, role: String) {
        if let Some(entry) = self
            .scheduled_cancels
            .iter_mut()
            .find(|(id, _, _)| *id == order_id)
        {
            entry.1 = eligible_index;
            entry.2 = role;
        } else {
            self.scheduled_cancels
                .push((order_id, eligible_index, role));
        }
    }

    // -- Fill liquidation --------------------------------------------------

    fn fill_liquidation<'py>(
        &mut self,
        py: Python<'py>,
        liq_price: f64,
        candle: &CandleData,
        events: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        let pos = match &self.position {
            Some(p) => p,
            None => return Ok(()),
        };

        let price = self.apply_slippage(r_price(liq_price), pos.side);
        let size = pos.size;
        let pnl = r_usd(match pos.side {
            PositionSide::LONG => (price - pos.entry_price) * size,
            PositionSide::SHORT => (pos.entry_price - price) * size,
        });
        let liq_fee = r_usd(price * size * LIQ_FEE_RATE);

        // Capture entry info before mutation
        let entry_time_iso = pos.opened_at.clone();
        let entry_price_val = r_price(pos.entry_price);
        let holding_bars = (self.current_index - pos.opened_index).max(0);
        let holding_minutes = holding_bars * self.config.timeframe_minutes;
        let side_val = pos.side.as_str().to_string();

        if self.config.margin_mode == "isolated" {
            self.realized_pnl = r_usd(self.realized_pnl + pnl);
            self.position = None;
            self.insurance_fund = r_usd(self.insurance_fund + liq_fee);
        } else {
            self.cash = r_usd(self.cash + pnl - liq_fee);
            self.realized_pnl = r_usd(self.realized_pnl + pnl);
            self.position = None;
            self.insurance_fund = r_usd(self.insurance_fund + liq_fee);
            self.cross_used_margin = 0.0;
        }

        self.tp = None;
        self.sl = None;
        self.close_request = None;

        let ev = PyDict::new(py);
        ev.set_item("type", "liquidation")?;
        Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
        ev.set_item("price", r_price(price))?;
        ev.set_item("quantity", r_qty(size))?;
        ev.set_item("notional", r_usd(price * size))?;
        ev.set_item("liq_fee", r_usd(liq_fee))?;
        ev.set_item("fee", r_usd(liq_fee))?;
        ev.set_item("pnl", r_usd(pnl))?;
        ev.set_item("side", &*side_val)?;
        ev.set_item("entry_price", entry_price_val)?;
        ev.set_item("entry_time", &*entry_time_iso)?;
        ev.set_item("holding_bars", holding_bars)?;
        ev.set_item("holding_minutes", holding_minutes)?;
        ev.set_item("open", r_price(candle.open))?;
        ev.set_item("high", r_price(candle.high))?;
        ev.set_item("low", r_price(candle.low))?;
        ev.set_item("close", r_price(candle.close))?;
        events.append(ev)?;
        Ok(())
    }

    // -- Execute close (user close, TP, SL) --------------------------------

    fn execute_close<'py>(
        &mut self,
        py: Python<'py>,
        order: &Order,
        candle: &CandleData,
        events: &Bound<'py, PyList>,
        trigger: &str,
    ) -> PyResult<()> {
        if self.position.is_none() {
            let mut rejected = order.clone();
            rejected.status = OrderStatus::REJECTED;
            let ev = PyDict::new(py);
            ev.set_item("type", "rejected")?;
            ev.set_item("reason", "No open position")?;
            ev.set_item("order", rejected.to_dict(py)?)?;
            Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
            events.append(ev)?;
            return Ok(());
        }

        {
            let pos = self.position.as_ref().unwrap();
            let expected_side = match pos.side {
                PositionSide::LONG => PositionSide::SHORT,
                PositionSide::SHORT => PositionSide::LONG,
            };
            if PositionSide::from(order.side) != expected_side {
                let mut rejected = order.clone();
                rejected.status = OrderStatus::REJECTED;
                let ev = PyDict::new(py);
                ev.set_item("type", "rejected")?;
                ev.set_item("reason", "Close side mismatch")?;
                ev.set_item("order", rejected.to_dict(py)?)?;
                Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
                events.append(ev)?;
                return Ok(());
            }
        }

        // Snapshot entry info before mutation
        let (entry_time_iso, entry_price_val, holding_bars, holding_minutes) = {
            let pos = self.position.as_ref().unwrap();
            (
                pos.opened_at.clone(),
                r_price(pos.entry_price),
                (self.current_index - pos.opened_index).max(0),
                (self.current_index - pos.opened_index).max(0) * self.config.timeframe_minutes,
            )
        };

        let mut price = r_price(order.price);
        if trigger == "sl" {
            let pos_side = self.position.as_ref().unwrap().side;
            price = self.apply_slippage(price, pos_side);
        }

        let qty = {
            let pos = self.position.as_ref().unwrap();
            r_qty(pos.size.min(order.quantity))
        };
        let notional = r_usd(price * qty);

        let pnl = {
            let pos = self.position.as_ref().unwrap();
            r_usd(match pos.side {
                PositionSide::LONG => (price - pos.entry_price) * qty,
                PositionSide::SHORT => (pos.entry_price - price) * qty,
            })
        };

        let close_fee = r_usd(notional * CLOSE_FEE_RATE);
        self.cash = r_usd(self.cash - close_fee);

        // Margin release
        if self.config.margin_mode == "isolated" {
            let (margin_release,) = {
                let pos = self.position.as_ref().unwrap();
                (if pos.size > 0.0 {
                    r_usd(pos.margin * (qty / pos.size))
                } else {
                    0.0
                },)
            };
            {
                let pos = self.position.as_mut().unwrap();
                pos.margin = r_usd(pos.margin - margin_release);
            }
            self.cash = r_usd(self.cash + margin_release);
        } else {
            let margin_release = {
                let pos = self.position.as_ref().unwrap();
                if pos.size > 0.0 {
                    r_usd(pos.margin * (qty / pos.size))
                } else {
                    0.0
                }
            };
            {
                let pos = self.position.as_mut().unwrap();
                pos.margin = r_usd((pos.margin - margin_release).max(0.0));
            }
            self.cross_used_margin = r_usd((self.cross_used_margin - margin_release).max(0.0));
        }

        self.cash = r_usd(self.cash + pnl);
        self.realized_pnl = r_usd(self.realized_pnl + pnl);

        let new_size = r_qty(self.position.as_ref().unwrap().size - qty);
        if new_size <= 0.0 {
            self.position = None;
            self.tp = None;
            self.sl = None;
            self.close_request = None;
            if self.config.margin_mode == "cross" {
                self.cross_used_margin = 0.0;
            }
        } else {
            {
                let pos = self.position.as_mut().unwrap();
                pos.size = new_size;
                pos.notional = r_usd(pos.entry_price * pos.size);
            }
            if self.config.margin_mode == "isolated" {
                let (side, entry) = {
                    let pos = self.position.as_ref().unwrap();
                    (pos.side, pos.entry_price)
                };
                let new_liq = calc_isolated_liq_price(side, entry, self.config.leverage);
                self.position.as_mut().unwrap().liquidation_price = new_liq;
            }
        }

        // Clear close_request if this was the close_request order
        let order_id = order.id.clone();
        if self
            .close_request
            .as_ref()
            .map_or(false, |cr| cr.id == order_id)
        {
            self.close_request = None;
        }

        // Build event
        let ev = PyDict::new(py);
        let event_type = if trigger == "close" {
            "fill_close"
        } else {
            trigger
        };
        ev.set_item("type", event_type)?;
        Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
        ev.set_item("order_id", &*order_id)?;
        ev.set_item("side", order.side.as_str())?;
        ev.set_item("price", r_price(price))?;
        ev.set_item("quantity", r_qty(qty))?;
        ev.set_item("notional", r_usd(notional))?;
        ev.set_item("fee", r_usd(close_fee))?;
        ev.set_item("pnl", r_usd(pnl))?;
        ev.set_item("entry_price", entry_price_val)?;
        ev.set_item("entry_time", &*entry_time_iso)?;
        ev.set_item("holding_bars", holding_bars)?;
        ev.set_item("holding_minutes", holding_minutes)?;
        events.append(ev)?;

        Ok(())
    }

    // -- Execute open (STOP_MARKET fill) -----------------------------------

    fn execute_open<'py>(
        &mut self,
        py: Python<'py>,
        order: &Order,
        candle: &CandleData,
        events: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        let price = r_price(order.price);
        let qty = r_qty(order.quantity);
        let notional = r_usd(price * qty);

        // Risk check at fill time
        if let Some(rej) = self.risk_check(order.side, price, qty) {
            let mut rejected = order.clone();
            rejected.status = OrderStatus::REJECTED;
            let ev = PyDict::new(py);
            ev.set_item("type", "rejected")?;
            ev.set_item("reason", &*rej)?;
            ev.set_item("order", rejected.to_dict(py)?)?;
            Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
            events.append(ev)?;
            if self
                .open_order
                .as_ref()
                .map_or(false, |oo| oo.id == order.id)
            {
                self.open_order = None;
            }
            return Ok(());
        }

        let open_fee = r_usd(notional * OPEN_FEE_RATE);
        self.cash = r_usd(self.cash - open_fee);
        let margin_add = r_usd(notional / self.config.leverage);

        if self.position.is_none() {
            let entry = price;
            let size = qty;
            let notional_pos = r_usd(entry * size);
            let (margin, liq_price) = if self.config.margin_mode == "isolated" {
                self.cash = r_usd(self.cash - margin_add);
                let lp = calc_isolated_liq_price(
                    PositionSide::from(order.side),
                    entry,
                    self.config.leverage,
                );
                (margin_add, lp)
            } else {
                self.cross_used_margin = r_usd(self.cross_used_margin + margin_add);
                (margin_add, 0.0)
            };
            self.position = Some(Position::new(
                self.config.symbol.clone(),
                PositionSide::from(order.side),
                r_price(entry),
                r_qty(size),
                notional_pos,
                margin,
                self.config.leverage,
                liq_price,
                self.realized_pnl,
                self.current_index,
            ));
        } else {
            let pos = self.position.as_mut().unwrap();
            let new_size = r_qty(pos.size + qty);
            let new_entry = r_price((pos.entry_price * pos.size + price * qty) / new_size);
            pos.size = new_size;
            pos.entry_price = new_entry;
            pos.notional = r_usd(new_entry * new_size);
            if self.config.margin_mode == "isolated" {
                pos.margin = r_usd(pos.margin + margin_add);
                self.cash = r_usd(self.cash - margin_add);
                pos.liquidation_price =
                    calc_isolated_liq_price(pos.side, pos.entry_price, self.config.leverage);
            } else {
                pos.margin = r_usd(pos.margin + margin_add);
                self.cross_used_margin = r_usd(self.cross_used_margin + margin_add);
            }
        }

        // Clear open_order slot
        let order_id = order.id.clone();
        if self
            .open_order
            .as_ref()
            .map_or(false, |oo| oo.id == order_id)
        {
            self.open_order = None;
        }

        // Attached TP
        if let Some(tp_price) = order.tp {
            if let Some(ref existing_tp) = self.tp {
                let tid = existing_tp.id.clone();
                self.schedule_cancel(tid, self.current_index + 1, "tp".to_string());
            }
            let (pos_size, pos_opened_index, pos_side, pos_entry) = {
                let pos = self.position.as_ref().unwrap();
                (pos.size, pos.opened_index, pos.side, pos.entry_price)
            };
            let tp_order = Order::new(
                self.config.symbol.clone(),
                order.side.opposite(),
                OrderType::TP,
                r_price(tp_price),
                r_qty(pos_size),
                r_usd(tp_price * pos_size),
                None,
                None,
                Some(pos_opened_index + 1),
                None,
                None,
            );
            let invalid = (pos_side == PositionSide::LONG && tp_order.price <= pos_entry)
                || (pos_side == PositionSide::SHORT && tp_order.price >= pos_entry);
            if invalid {
                let ev = PyDict::new(py);
                ev.set_item("type", "tp_invalid")?;
                ev.set_item("reason", "Invalid TP relative to entry")?;
                ev.set_item("price", r_price(tp_order.price))?;
                Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
                events.append(ev)?;
            } else {
                self.pending_tp = Some(tp_order);
            }
        }

        // Attached SL
        if let Some(sl_price) = order.sl {
            if let Some(ref existing_sl) = self.sl {
                let sid = existing_sl.id.clone();
                self.schedule_cancel(sid, self.current_index + 1, "sl".to_string());
            }
            let (pos_size, pos_opened_index, pos_side, pos_entry) = {
                let pos = self.position.as_ref().unwrap();
                (pos.size, pos.opened_index, pos.side, pos.entry_price)
            };
            let sl_order = Order::new(
                self.config.symbol.clone(),
                order.side.opposite(),
                OrderType::SL,
                r_price(sl_price),
                r_qty(pos_size),
                r_usd(sl_price * pos_size),
                None,
                None,
                Some(pos_opened_index + 1),
                None,
                None,
            );
            let invalid = (pos_side == PositionSide::LONG && sl_order.price >= pos_entry)
                || (pos_side == PositionSide::SHORT && sl_order.price <= pos_entry);
            if invalid {
                let ev = PyDict::new(py);
                ev.set_item("type", "sl_invalid")?;
                ev.set_item("reason", "Invalid SL relative to entry")?;
                ev.set_item("price", r_price(sl_order.price))?;
                Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
                events.append(ev)?;
            } else {
                self.pending_sl = Some(sl_order);
            }
        }

        // Fill event
        let ev = PyDict::new(py);
        ev.set_item("type", "fill_open")?;
        Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
        ev.set_item("order_id", &*order_id)?;
        ev.set_item("side", order.side.as_str())?;
        ev.set_item("price", r_price(price))?;
        ev.set_item("quantity", r_qty(qty))?;
        ev.set_item("notional", r_usd(notional))?;
        ev.set_item("margin", r_usd(margin_add))?;
        ev.set_item("fee", r_usd(open_fee))?;
        events.append(ev)?;

        Ok(())
    }

    // -- Intra-candle trigger resolution (binary search) --------------------

    fn determine_first_trigger(
        &self,
        candle: &CandleData,
        liq_price: f64,
        tp_price: Option<f64>,
        sl_price: Option<f64>,
    ) -> Option<&'static str> {
        let tf_minutes = self.config.timeframe_minutes;
        if tf_minutes <= 1 {
            return None;
        }

        let candle_open_time = candle.open_time.as_deref()?;
        let candle_start_ts = parse_iso_to_unix(candle_open_time)?;
        let candle_end_ts = candle_start_ts + tf_minutes * 60;

        let intra = self.intra_candles.as_ref()?;
        if intra.is_empty() {
            return None;
        }

        // Binary search for window boundaries
        let start_idx = intra.partition_point(|(ts, _, _)| *ts < candle_start_ts);
        let end_idx = intra.partition_point(|(ts, _, _)| *ts < candle_end_ts);

        for &(_, m1_low, m1_high) in &intra[start_idx..end_idx] {
            let mut count = 0u8;
            let mut first = None;
            if in_range(liq_price, m1_low, m1_high) {
                count += 1;
                first = Some("liq");
            }
            if let Some(tp) = tp_price {
                if in_range(tp, m1_low, m1_high) {
                    count += 1;
                    if first.is_none() {
                        first = Some("tp");
                    }
                }
            }
            if let Some(sl) = sl_price {
                if in_range(sl, m1_low, m1_high) {
                    count += 1;
                    if first.is_none() {
                        first = Some("sl");
                    }
                }
            }

            if count == 0 {
                continue;
            }
            if count == 1 {
                return first;
            }
            // Multiple triggers in same 1m candle — fall back to priority
            return None;
        }
        None
    }

    // -- Try triggers (LIQ / SL / TP) -------------------------------------

    fn try_triggers<'py>(
        &mut self,
        py: Python<'py>,
        candle: &CandleData,
        index: i64,
        events: &Bound<'py, PyList>,
    ) -> PyResult<bool> {
        if self.position.is_none() {
            return Ok(false);
        }

        let low = candle.low;
        let high = candle.high;

        let liq_price = {
            let pos = self.position.as_ref().unwrap();
            if self.config.margin_mode == "isolated" {
                calc_isolated_liq_price(pos.side, pos.entry_price, self.config.leverage)
            } else {
                calc_cross_liq_price(pos.side, pos.entry_price, pos.size, self.cash)
            }
        };

        let tp_price = self.tp.as_ref().and_then(|tp| {
            tp.eligible_from_index
                .and_then(|efi| if efi <= index { Some(tp.price) } else { None })
        });
        let sl_price = self.sl.as_ref().and_then(|sl| {
            sl.eligible_from_index
                .and_then(|efi| if efi <= index { Some(sl.price) } else { None })
        });

        let mut triggers_in_range: Vec<(&str, f64)> = Vec::new();
        if in_range(liq_price, low, high) {
            triggers_in_range.push(("liq", liq_price));
        }
        if let Some(sp) = sl_price {
            if in_range(sp, low, high) {
                triggers_in_range.push(("sl", sp));
            }
        }
        if let Some(tp) = tp_price {
            if in_range(tp, low, high) {
                triggers_in_range.push(("tp", tp));
            }
        }

        if triggers_in_range.is_empty() {
            return Ok(false);
        }

        if triggers_in_range.len() == 1 {
            match triggers_in_range[0].0 {
                "liq" => self.fill_liquidation(py, liq_price, candle, events)?,
                "sl" => {
                    let order = self.sl.clone().unwrap();
                    self.execute_close(py, &order, candle, events, "sl")?;
                }
                "tp" => {
                    let order = self.tp.clone().unwrap();
                    self.execute_close(py, &order, candle, events, "tp")?;
                }
                _ => {}
            }
            return Ok(true);
        }

        // Multiple triggers — use 1m data to determine which hit first
        let first_trigger = if self.config.timeframe_minutes > 1 {
            self.determine_first_trigger(candle, liq_price, tp_price, sl_price)
        } else {
            None
        };

        match first_trigger {
            Some("liq") => {
                self.fill_liquidation(py, liq_price, candle, events)?;
                Ok(true)
            }
            Some("sl") => {
                let order = self.sl.clone().unwrap();
                self.execute_close(py, &order, candle, events, "sl")?;
                Ok(true)
            }
            Some("tp") => {
                let order = self.tp.clone().unwrap();
                self.execute_close(py, &order, candle, events, "tp")?;
                Ok(true)
            }
            _ => {
                // Fallback priority: LIQ > SL > TP
                if in_range(liq_price, low, high) {
                    self.fill_liquidation(py, liq_price, candle, events)?;
                    return Ok(true);
                }
                if let Some(sp) = sl_price {
                    if in_range(sp, low, high) {
                        let order = self.sl.clone().unwrap();
                        self.execute_close(py, &order, candle, events, "sl")?;
                        return Ok(true);
                    }
                }
                if let Some(tp) = tp_price {
                    if in_range(tp, low, high) {
                        let order = self.tp.clone().unwrap();
                        self.execute_close(py, &order, candle, events, "tp")?;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    // -- Apply scheduled cancels -------------------------------------------

    fn apply_cancels<'py>(
        &mut self,
        py: Python<'py>,
        index: i64,
        events: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        // Collect eligible cancels preserving insertion order
        let eligible: Vec<(String, String)> = self
            .scheduled_cancels
            .iter()
            .filter(|(_, eidx, _)| *eidx <= index)
            .map(|(oid, _, role)| (oid.clone(), role.clone()))
            .collect();

        // Remove processed entries
        self.scheduled_cancels.retain(|(_, eidx, _)| *eidx > index);

        // Process each cancel
        for (oid, role) in eligible {
            let found = match role.as_str() {
                "open_order" => {
                    if self
                        .open_order
                        .as_ref()
                        .map_or(false, |o| o.id == oid)
                    {
                        self.open_order = None;
                        true
                    } else if self
                        .pending_open_order
                        .as_ref()
                        .map_or(false, |o| o.id == oid)
                    {
                        self.pending_open_order = None;
                        true
                    } else {
                        false
                    }
                }
                "close_request" => {
                    if self
                        .close_request
                        .as_ref()
                        .map_or(false, |o| o.id == oid)
                    {
                        self.close_request = None;
                        true
                    } else if self
                        .pending_close_request
                        .as_ref()
                        .map_or(false, |o| o.id == oid)
                    {
                        self.pending_close_request = None;
                        true
                    } else {
                        false
                    }
                }
                "tp" => {
                    if self.tp.as_ref().map_or(false, |o| o.id == oid) {
                        self.tp = None;
                        true
                    } else if self
                        .pending_tp
                        .as_ref()
                        .map_or(false, |o| o.id == oid)
                    {
                        self.pending_tp = None;
                        true
                    } else {
                        false
                    }
                }
                "sl" => {
                    if self.sl.as_ref().map_or(false, |o| o.id == oid) {
                        self.sl = None;
                        true
                    } else if self
                        .pending_sl
                        .as_ref()
                        .map_or(false, |o| o.id == oid)
                    {
                        self.pending_sl = None;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if found {
                let ev = PyDict::new(py);
                ev.set_item("type", "cancel")?;
                ev.set_item("order_id", &*oid)?;
                ev.set_item("role", &*role)?;
                Self::set_opt_str(&ev, "open_time", &self.last_candle_meta.open_time.clone())?;
                events.append(ev)?;
            }
        }
        Ok(())
    }

    // -- Promote pending orders --------------------------------------------

    fn promote_pending(&mut self, index: i64) {
        if self
            .pending_open_order
            .as_ref()
            .map_or(false, |o| o.eligible_from_index.map_or(false, |e| e <= index))
        {
            self.open_order = self.pending_open_order.take();
        }
        if self
            .pending_close_request
            .as_ref()
            .map_or(false, |o| o.eligible_from_index.map_or(false, |e| e <= index))
        {
            self.close_request = self.pending_close_request.take();
        }
        if self
            .pending_tp
            .as_ref()
            .map_or(false, |o| o.eligible_from_index.map_or(false, |e| e <= index))
        {
            self.tp = self.pending_tp.take();
        }
        if self
            .pending_sl
            .as_ref()
            .map_or(false, |o| o.eligible_from_index.map_or(false, |e| e <= index))
        {
            self.sl = self.pending_sl.take();
        }
    }

    // -- RL fast-path: internal order creation (no PyDict, no events) --------

    /// Create an order internally (used by RL action translation).
    /// This mirrors create_order but without PyDict return or FFI overhead.
    pub(crate) fn create_order_internal(
        &mut self,
        side: OrderSide,
        price: f64,
        margin_pct: f64,
        tp: Option<f64>,
        sl: Option<f64>,
    ) {
        let price = r_price(price);
        if margin_pct <= 0.0 || margin_pct > 1.0 {
            return;
        }

        let equity_now = self.cash + self.unrealized_pnl;
        let allocated_margin_raw = equity_now * margin_pct;
        let notional_target = allocated_margin_raw * self.config.leverage;
        let qty = r_qty(if price > 0.0 {
            notional_target / price
        } else {
            0.0
        });
        let notional = r_usd(price * qty);

        if self.validate_params(price, qty).is_some() {
            return;
        }
        if self.opposite_side_position(side) {
            return;
        }

        // Validate TP
        let tp_val = tp.map(|tp_raw| r_price(tp_raw)).and_then(|tp_r| {
            if side == OrderSide::LONG && tp_r <= price {
                None
            } else if side == OrderSide::SHORT && tp_r >= price {
                None
            } else {
                Some(tp_r)
            }
        });

        // Validate SL
        let sl_val = sl.map(|sl_raw| r_price(sl_raw)).and_then(|sl_r| {
            if side == OrderSide::LONG && sl_r >= price {
                None
            } else if side == OrderSide::SHORT && sl_r <= price {
                None
            } else {
                Some(sl_r)
            }
        });

        if self.risk_check(side, price, qty).is_some() {
            return;
        }

        let allocated_margin_eff = r_usd(notional / self.config.leverage);
        let ord_obj = Order::new(
            self.config.symbol.clone(),
            side,
            OrderType::STOP_MARKET,
            price,
            qty,
            notional,
            tp_val,
            sl_val,
            Some(self.current_index + 1),
            Some(margin_pct),
            Some(allocated_margin_eff),
        );

        if let Some(ref existing) = self.open_order {
            let eid = existing.id.clone();
            self.schedule_cancel(eid, self.current_index + 1, "open_order".to_string());
        }

        self.pending_open_order = Some(ord_obj);
    }

    /// Close the current position internally (used by RL action translation).
    pub(crate) fn close_order_internal(&mut self, price: f64) {
        let price = r_price(price);
        let pos = match &self.position {
            Some(p) if p.size > 0.0 => p,
            _ => return,
        };

        let side = match pos.side {
            PositionSide::SHORT => OrderSide::LONG,
            PositionSide::LONG => OrderSide::SHORT,
        };
        let qty = r_qty(pos.size);

        let ord_obj = Order::new(
            self.config.symbol.clone(),
            side,
            OrderType::CLOSE,
            price,
            qty,
            r_usd(price * qty),
            None,
            None,
            Some(self.current_index + 1),
            None,
            None,
        );

        if let Some(ref existing) = self.close_request {
            let eid = existing.id.clone();
            self.schedule_cancel(eid, self.current_index + 1, "close_request".to_string());
        }

        self.pending_close_request = Some(ord_obj);
    }

    // -- Process user orders (open_order / close_request candidates) --------

    fn process_user_orders<'py>(
        &mut self,
        py: Python<'py>,
        candle: &CandleData,
        index: i64,
        open_p: f64,
        low: f64,
        high: f64,
        events: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        // Collect eligible candidates as full clones
        let mut candidates: Vec<(Order, bool)> = Vec::new(); // (order, is_open_order)

        if let Some(ref oo) = self.open_order {
            if oo
                .eligible_from_index
                .map_or(false, |e| e <= index)
            {
                candidates.push((oo.clone(), true));
            }
        }
        if let Some(ref cr) = self.close_request {
            if cr
                .eligible_from_index
                .map_or(false, |e| e <= index)
            {
                candidates.push((cr.clone(), false));
            }
        }

        // Sort by (created_at, price if LONG else -price) — matches Python exactly
        candidates.sort_by(|a, b| {
            let ca = a.0.created_at.cmp(&b.0.created_at);
            if ca != std::cmp::Ordering::Equal {
                return ca;
            }
            let pa = if a.0.side == OrderSide::LONG {
                a.0.price
            } else {
                -a.0.price
            };
            let pb = if b.0.side == OrderSide::LONG {
                b.0.price
            } else {
                -b.0.price
            };
            pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter open-tick exclusion
        let eligible: Vec<&(Order, bool)> = candidates
            .iter()
            .filter(|(o, _)| {
                let efi = o.eligible_from_index.unwrap_or(-1);
                !(efi == index && (open_p - o.price).abs() < 1e-9)
            })
            .collect();

        for (order, _is_open) in eligible {
            if in_range(order.price, low, high) {
                if order.order_type == OrderType::STOP_MARKET {
                    if self.opposite_side_position(order.side) {
                        // Reject: position open on opposite side
                        let mut rejected = order.clone();
                        rejected.status = OrderStatus::REJECTED;
                        let ev = PyDict::new(py);
                        ev.set_item("type", "rejected")?;
                        ev.set_item("reason", "Position open (reduce-only required)")?;
                        ev.set_item("order", rejected.to_dict(py)?)?;
                        Self::set_opt_str(&ev, "open_time", &candle.open_time)?;
                        events.append(ev)?;
                        if self
                            .open_order
                            .as_ref()
                            .map_or(false, |oo| oo.id == order.id)
                        {
                            self.open_order = None;
                        }
                    } else {
                        self.execute_open(py, order, candle, events)?;
                    }
                } else if order.order_type == OrderType::CLOSE {
                    self.execute_close(py, order, candle, events, "close")?;
                }
            }
        }
        Ok(())
    }

    // -- Helper: set Option<String> on PyDict (None → Python None) ----------

    fn set_opt_str(dict: &Bound<'_, PyDict>, key: &str, val: &Option<String>) -> PyResult<()> {
        match val {
            Some(s) => dict.set_item(key, s.as_str()),
            None => dict.set_item(key, dict.py().None()),
        }
    }
}

// -- PyO3 methods (Python-visible) -----------------------------------------

#[pymethods]
impl RustTradingEngine {
    #[new]
    #[pyo3(signature = (symbol, margin_mode, leverage, starting_cash, slippage_pct=0.0, timeframe_minutes=60, funding_data=None))]
    fn new(
        symbol: String,
        margin_mode: String,
        leverage: f64,
        starting_cash: f64,
        slippage_pct: f64,
        timeframe_minutes: i64,
        funding_data: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mode = margin_mode.trim().to_lowercase();
        let normalized = match mode.as_str() {
            "cross" | "crossed" => "cross",
            "isolated" | "iso" => "isolated",
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported margin_mode: {}",
                    margin_mode
                )))
            }
        };

        let parsed_funding = match funding_data {
            Some(fd) => Some(parse_funding_dict(fd)?),
            None => None,
        };

        Ok(Self {
            config: EngineConfig {
                symbol,
                margin_mode: normalized.to_string(),
                leverage,
                slippage_pct,
                timeframe_minutes,
                funding_data: parsed_funding,
            },
            cash: starting_cash,
            current_index: -1,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            insurance_fund: 0.0,
            total_funding_paid: 0.0,
            cross_used_margin: 0.0,
            position: None,
            open_order: None,
            close_request: None,
            tp: None,
            sl: None,
            pending_open_order: None,
            pending_close_request: None,
            pending_tp: None,
            pending_sl: None,
            scheduled_cancels: Vec::new(),
            last_candle_meta: LastCandleMeta::default(),
            intra_candles: None,
            indicators: BTreeMap::new(),
            registered_indicators: Vec::new(),
            rl_config: None,
        })
    }

    // -- Snapshot ----------------------------------------------------------

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = PyDict::new(py);

        let available = f64::max(
            0.0,
            self.cash
                - if self.config.margin_mode == "cross" {
                    self.cross_used_margin
                } else {
                    0.0
                },
        );
        let used_im = if self.config.margin_mode == "cross" {
            self.cross_used_margin
        } else {
            self.position.as_ref().map_or(0.0, |p| p.margin)
        };

        state.set_item("cash", self.cash)?;
        state.set_item("available_cash", available)?;
        state.set_item("used_initial_margin", used_im)?;
        state.set_item("equity", self.cash + self.unrealized_pnl)?;
        state.set_item("realized_pnl", self.realized_pnl)?;
        state.set_item("unrealized_pnl", self.unrealized_pnl)?;

        match self.position.as_ref() {
            Some(pos) => {
                let pos_dict = pos.to_dict(py, &self.config.margin_mode, None)?;
                state.set_item("position", pos_dict)?;
            }
            None => {
                state.set_item("position", py.None())?;
            }
        }

        state.set_item("margin_mode", &self.config.margin_mode)?;
        state.set_item("leverage", self.config.leverage)?;
        state.set_item("candle_index", std::cmp::max(self.current_index, -1))?;

        // Phase 5: indicators
        let ind_dict = PyDict::new(py);
        for (k, v) in &self.indicators {
            ind_dict.set_item(k.as_str(), *v)?;
        }
        state.set_item("indicators", ind_dict)?;

        Ok(state)
    }

    // -- Capacity ---------------------------------------------------------

    #[pyo3(signature = (side, price))]
    fn compute_open_capacity<'py>(
        &self,
        py: Python<'py>,
        side: &str,
        price: f64,
    ) -> PyResult<Py<PyDict>> {
        let side =
            OrderSide::from_str(side).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let price = r_price(price);
        let l = self.config.leverage;
        let f = OPEN_FEE_RATE;
        let mmr = MMR;

        let e = self.cash + self.unrealized_pnl;
        let s0 = self.position.as_ref().map_or(0.0, |p| p.size);
        let e0 = self.position.as_ref().map_or(0.0, |p| p.entry_price);
        let existing_notional = e0 * s0;

        let used_im = if l > 0.0 {
            existing_notional / l
        } else {
            f64::INFINITY
        };
        let im_headroom = e - used_im;

        let denom_im = if l > 0.0 {
            price * (1.0 / l + f)
        } else {
            f64::INFINITY
        };
        let q_im_max = if im_headroom <= 0.0 || denom_im <= 0.0 {
            0.0
        } else {
            im_headroom / denom_im
        };

        let eq00 = self.cash;
        let num_liq = match side {
            OrderSide::LONG => eq00 + price * s0 - price * mmr * s0 - e0 * s0,
            OrderSide::SHORT => eq00 + (e0 - price) * s0 - price * mmr * s0,
        };
        let denom_liq = price * (mmr + f);
        let q_liq_max = if denom_liq <= 0.0 {
            0.0
        } else {
            num_liq / denom_liq
        };

        let q_max = f64::max(0.0, f64::min(q_im_max, q_liq_max));
        let n_max = q_max * price;
        let im_needed_max = if l > 0.0 { n_max / l } else { 0.0 };

        let mut m_max_im = 0.0;
        if e > 0.0 && l > 0.0 {
            m_max_im = f64::max(0.0, f64::min(1.0, (e - used_im) / (e * (1.0 + l * f))));
        }
        let mut m_max_from_q = 0.0;
        if e > 0.0 && l > 0.0 {
            m_max_from_q = (q_max * price) / (e * l);
            m_max_from_q = f64::max(0.0, f64::min(1.0, m_max_from_q));
        }

        let result = PyDict::new(py);
        result.set_item("price", r_price(price))?;
        result.set_item("equity", r_usd(e))?;
        result.set_item(
            "used_initial_margin",
            r_usd(f64::max(
                0.0,
                if used_im.is_infinite() { 0.0 } else { used_im },
            )),
        )?;
        result.set_item("im_headroom", r_usd(f64::max(0.0, im_headroom)))?;
        result.set_item("q_im_max", r_qty(f64::max(0.0, q_im_max)))?;
        result.set_item("q_liq_max", r_qty(f64::max(0.0, q_liq_max)))?;
        result.set_item("q_max", r_qty(q_max))?;
        result.set_item("notional_max", r_usd(n_max))?;
        result.set_item("im_needed_max", r_usd(f64::max(0.0, im_needed_max)))?;
        result.set_item(
            "margin_pct_max_im",
            format!("{:.6}", m_max_im).parse::<f64>().unwrap(),
        )?;
        result.set_item(
            "margin_pct_max",
            format!("{:.6}", f64::min(m_max_im, m_max_from_q))
                .parse::<f64>()
                .unwrap(),
        )?;

        Ok(result.unbind())
    }

    // -- Candle processing -------------------------------------------------

    #[pyo3(signature = (candle, index))]
    fn on_candle<'py>(
        &mut self,
        py: Python<'py>,
        candle: &Bound<'py, PyDict>,
        index: i64,
    ) -> PyResult<(Py<PyList>, Py<PyDict>)> {
        self.current_index = index;
        let events = PyList::empty(py);

        // Extract candle data
        let open_time: Option<String> = candle
            .get_item("open_time")?
            .and_then(|v| v.extract::<String>().ok());
        let open_p: f64 = candle.get_item("open")?.unwrap().extract()?;
        let high: f64 = candle.get_item("high")?.unwrap().extract()?;
        let low: f64 = candle.get_item("low")?.unwrap().extract()?;
        let close: f64 = candle.get_item("close")?.unwrap().extract()?;

        let candle_data = CandleData {
            open_time: open_time.clone(),
            open: open_p,
            high,
            low,
            close,
        };

        // Update candle metadata
        self.last_candle_meta = LastCandleMeta {
            open_time,
            volume: candle
                .get_item("volume")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            quote_asset_volume: candle
                .get_item("quote_asset_volume")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            number_of_trades: candle
                .get_item("number_of_trades")?
                .and_then(|v| v.extract::<i64>().ok())
                .unwrap_or(0),
            taker_buy_base_asset_volume: candle
                .get_item("taker_buy_base_asset_volume")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
            taker_buy_quote_asset_volume: candle
                .get_item("taker_buy_quote_asset_volume")?
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0),
        };

        // 1: refresh UPNL at close
        self.refresh_upnl(close);

        // 1.5: apply funding
        if let Some(ref ot) = candle_data.open_time {
            self.apply_funding(py, ot, &events)?;
        }

        // 2: triggers
        let consumed_by_trigger = self.try_triggers(py, &candle_data, index, &events)?;

        // 3: apply cancels
        self.apply_cancels(py, index, &events)?;

        // 4: promote pending
        self.promote_pending(index);

        // 5: user orders
        if !consumed_by_trigger {
            self.process_user_orders(py, &candle_data, index, open_p, low, high, &events)?;
        }

        // 6: refresh UPNL at close
        self.refresh_upnl(close);

        // 7: update registered indicators
        for ind in &mut self.registered_indicators {
            let outputs = ind.push_candle(open_p, high, low, close);
            for (key, val) in outputs {
                self.indicators.insert(key, val);
            }
        }

        let state = self.snapshot(py)?;
        state.set_item("candle_index", index)?;

        Ok((events.unbind(), state.unbind()))
    }

    // -- Order API ---------------------------------------------------------

    #[pyo3(signature = (side, price, margin_pct, tp=None, sl=None))]
    fn create_order<'py>(
        &mut self,
        py: Python<'py>,
        side: &str,
        price: f64,
        margin_pct: f64,
        tp: Option<f64>,
        sl: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let side = OrderSide::from_str(side)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let price = r_price(price);

        if margin_pct <= 0.0 || margin_pct > 1.0 {
            let result = PyDict::new(py);
            result.set_item("status", "rejected")?;
            result.set_item(
                "reason",
                "margin_pct must be in (0, 1]. Example: 0.1 for 10% of equity",
            )?;
            return Ok(result.unbind());
        }

        let equity_now = self.cash + self.unrealized_pnl;
        let allocated_margin_raw = equity_now * margin_pct;
        let notional_target = allocated_margin_raw * self.config.leverage;
        let qty = r_qty(if price > 0.0 {
            notional_target / price
        } else {
            0.0
        });
        let notional = r_usd(price * qty);
        let allocated_margin_eff = r_usd(notional / self.config.leverage);

        if let Some(rej) = self.validate_params(price, qty) {
            let result = PyDict::new(py);
            result.set_item("status", "rejected")?;
            result.set_item("reason", &*rej)?;
            return Ok(result.unbind());
        }

        if self.opposite_side_position(side) {
            let result = PyDict::new(py);
            result.set_item("status", "rejected")?;
            result.set_item("reason", "Position open (reduce-only required)")?;
            return Ok(result.unbind());
        }

        // Validate TP
        let mut tp_val = tp;
        if let Some(tp_raw) = tp_val {
            let tp_rounded = r_price(tp_raw);
            if side == OrderSide::LONG && tp_rounded <= price {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "Invalid TP for LONG (must be > entry)")?;
                return Ok(result.unbind());
            }
            if side == OrderSide::SHORT && tp_rounded >= price {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "Invalid TP for SHORT (must be < entry)")?;
                return Ok(result.unbind());
            }
            tp_val = Some(tp_rounded);
        }

        // Validate SL
        let mut sl_val = sl;
        if let Some(sl_raw) = sl_val {
            let sl_rounded = r_price(sl_raw);
            if side == OrderSide::LONG && sl_rounded >= price {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "Invalid SL for LONG (must be < entry)")?;
                return Ok(result.unbind());
            }
            if side == OrderSide::SHORT && sl_rounded <= price {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "Invalid SL for SHORT (must be > entry)")?;
                return Ok(result.unbind());
            }
            sl_val = Some(sl_rounded);
        }

        // Risk check
        if let Some(rej) = self.risk_check(side, price, qty) {
            let result = PyDict::new(py);
            result.set_item("status", "rejected")?;
            result.set_item("reason", &*rej)?;
            return Ok(result.unbind());
        }

        // Create order
        let ord_obj = Order::new(
            self.config.symbol.clone(),
            side,
            OrderType::STOP_MARKET,
            price,
            qty,
            notional,
            tp_val,
            sl_val,
            Some(self.current_index + 1),
            Some(margin_pct),
            Some(allocated_margin_eff),
        );

        // Schedule cancel of existing open_order
        if let Some(ref existing) = self.open_order {
            let eid = existing.id.clone();
            self.schedule_cancel(eid, self.current_index + 1, "open_order".to_string());
        }

        let result = PyDict::new(py);
        result.set_item("status", "accepted")?;
        result.set_item("order_id", &ord_obj.id)?;
        result.set_item("eligible_from_index", ord_obj.eligible_from_index.unwrap())?;
        result.set_item("price", r_price(price))?;
        result.set_item("quantity", r_qty(qty))?;
        result.set_item("notional", r_usd(notional))?;
        result.set_item(
            "margin_pct",
            format!("{:.6}", margin_pct).parse::<f64>().unwrap(),
        )?;
        result.set_item("allocated_margin", r_usd(allocated_margin_eff))?;

        self.pending_open_order = Some(ord_obj);

        Ok(result.unbind())
    }

    #[pyo3(signature = (price,))]
    fn close_order<'py>(&mut self, py: Python<'py>, price: f64) -> PyResult<Py<PyDict>> {
        let price = r_price(price);

        let pos = match &self.position {
            Some(p) if p.size > 0.0 => p,
            _ => {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "No open position")?;
                return Ok(result.unbind());
            }
        };

        let side = match pos.side {
            PositionSide::SHORT => OrderSide::LONG,
            PositionSide::LONG => OrderSide::SHORT,
        };
        let qty = r_qty(pos.size);

        let ord_obj = Order::new(
            self.config.symbol.clone(),
            side,
            OrderType::CLOSE,
            price,
            qty,
            r_usd(price * qty),
            None,
            None,
            Some(self.current_index + 1),
            None,
            None,
        );

        // Schedule cancel of existing close_request
        if let Some(ref existing) = self.close_request {
            let eid = existing.id.clone();
            self.schedule_cancel(eid, self.current_index + 1, "close_request".to_string());
        }

        let result = PyDict::new(py);
        result.set_item("status", "accepted")?;
        result.set_item("order_id", &ord_obj.id)?;
        result.set_item("eligible_from_index", ord_obj.eligible_from_index.unwrap())?;
        result.set_item("price", r_price(price))?;
        result.set_item("quantity", r_qty(qty))?;

        self.pending_close_request = Some(ord_obj);

        Ok(result.unbind())
    }

    #[pyo3(signature = (order_id,))]
    fn cancel_order<'py>(
        &mut self,
        py: Python<'py>,
        order_id: &str,
    ) -> PyResult<Py<PyDict>> {
        let oid = order_id.to_string();

        // Validate UUID format (matches Python's try: UUID(str(order_id)))
        if uuid::Uuid::parse_str(order_id).is_err() {
            let result = PyDict::new(py);
            result.set_item("status", "rejected")?;
            result.set_item("reason", "Invalid order_id")?;
            return Ok(result.unbind());
        }

        let role = if self.open_order.as_ref().map_or(false, |o| o.id == oid) {
            Some("open_order")
        } else if self
            .close_request
            .as_ref()
            .map_or(false, |o| o.id == oid)
        {
            Some("close_request")
        } else if self.tp.as_ref().map_or(false, |o| o.id == oid) {
            Some("tp")
        } else if self.sl.as_ref().map_or(false, |o| o.id == oid) {
            Some("sl")
        } else if self
            .pending_open_order
            .as_ref()
            .map_or(false, |o| o.id == oid)
        {
            Some("open_order")
        } else if self
            .pending_close_request
            .as_ref()
            .map_or(false, |o| o.id == oid)
        {
            Some("close_request")
        } else if self
            .pending_tp
            .as_ref()
            .map_or(false, |o| o.id == oid)
        {
            Some("tp")
        } else if self
            .pending_sl
            .as_ref()
            .map_or(false, |o| o.id == oid)
        {
            Some("sl")
        } else {
            None
        };

        match role {
            None => {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "Order not found")?;
                Ok(result.unbind())
            }
            Some(role) => {
                self.schedule_cancel(oid.clone(), self.current_index + 1, role.to_string());
                let result = PyDict::new(py);
                result.set_item("status", "accepted")?;
                result.set_item("order_id", &*oid)?;
                result.set_item("effective_index", self.current_index + 1)?;
                Ok(result.unbind())
            }
        }
    }

    /// Load 1m intra-candle data for binary-search trigger resolution.
    /// Each tuple: (unix_timestamp_seconds, low, high).
    /// Data MUST be sorted by timestamp ascending.
    #[pyo3(signature = (data,))]
    fn load_intra_candles(&mut self, data: Vec<(i64, f64, f64)>) {
        self.intra_candles = Some(data);
    }

    // -- Phase 5: Indicator Registration -----------------------------------

    /// Register an SMA indicator.
    #[pyo3(signature = (label, period, source="close"))]
    fn register_sma(&mut self, label: String, period: usize, source: &str) -> PyResult<()> {
        let src = CandleSource::from_str(source)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        self.registered_indicators
            .push(IndicatorType::Sma(SmaIndicator::new(label, period, src)));
        Ok(())
    }

    /// Register an EMA indicator.
    #[pyo3(signature = (label, period, source="close"))]
    fn register_ema(&mut self, label: String, period: usize, source: &str) -> PyResult<()> {
        let src = CandleSource::from_str(source)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        self.registered_indicators
            .push(IndicatorType::Ema(EmaIndicator::new(label, period, src)));
        Ok(())
    }

    /// Register an RSI indicator.
    #[pyo3(signature = (label, period, source="close"))]
    fn register_rsi(&mut self, label: String, period: usize, source: &str) -> PyResult<()> {
        let src = CandleSource::from_str(source)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        self.registered_indicators
            .push(IndicatorType::Rsi(RsiIndicator::new(label, period, src)));
        Ok(())
    }

    /// Register an ATR indicator (always uses high/low/close).
    #[pyo3(signature = (label, period))]
    fn register_atr(&mut self, label: String, period: usize) -> PyResult<()> {
        self.registered_indicators
            .push(IndicatorType::Atr(AtrIndicator::new(label, period)));
        Ok(())
    }

    /// Register a Bollinger Bands indicator.
    #[pyo3(signature = (label, period, multiplier=2.0, source="close"))]
    fn register_bollinger_bands(
        &mut self,
        label: String,
        period: usize,
        multiplier: f64,
        source: &str,
    ) -> PyResult<()> {
        let src = CandleSource::from_str(source)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        self.registered_indicators
            .push(IndicatorType::BollingerBands(
                BollingerBandsIndicator::new(label, period, multiplier, src),
            ));
        Ok(())
    }

    /// Register a MACD indicator.
    #[pyo3(signature = (label, fast=12, slow=26, signal=9, source="close"))]
    fn register_macd(
        &mut self,
        label: String,
        fast: usize,
        slow: usize,
        signal: usize,
        source: &str,
    ) -> PyResult<()> {
        let src = CandleSource::from_str(source)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        self.registered_indicators
            .push(IndicatorType::Macd(MacdIndicator::new(
                label, fast, slow, signal, src,
            )));
        Ok(())
    }

    // -- Phase 6: RL Fast-Path API -----------------------------------------

    /// Configure the RL environment within the engine.
    ///
    /// Args:
    ///   action_type: "discrete", "discrete_with_sizing", or "continuous"
    ///   obs_features: list of feature name strings (e.g. ["ohlcv", "position_info", "rsi_14"])
    ///   reward_type: "pnl", "log_return", "differential_sharpe", "risk_adjusted", "sortino", "advanced"
    ///   action_config: dict with action-specific params
    ///   reward_config: dict with reward-specific params
    #[pyo3(signature = (action_type, obs_features, reward_type, action_config=None, reward_config=None))]
    fn setup_rl_env(
        &mut self,
        action_type: &str,
        obs_features: Vec<String>,
        reward_type: &str,
        action_config: Option<&Bound<'_, PyDict>>,
        reward_config: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<usize>> {
        // Build action space
        let action_space = match action_type {
            "discrete" => {
                let margin_pct = extract_f64(action_config, "margin_pct", 0.1);
                let sl_pct = extract_f64(action_config, "sl_pct", 0.0);
                let tp_pct = extract_f64(action_config, "tp_pct", 0.0);
                let actions_list = extract_string_list(action_config, "actions");
                let action_strs: Vec<&str> = if actions_list.is_empty() {
                    vec!["hold", "open_long", "open_short", "close"]
                } else {
                    actions_list.iter().map(|s| s.as_str()).collect()
                };
                RlActionSpace::Discrete(DiscreteActionSpace::new(
                    &action_strs,
                    margin_pct,
                    sl_pct,
                    tp_pct,
                ))
            }
            "discrete_with_sizing" => {
                let sl_pct = extract_f64(action_config, "sl_pct", 0.0);
                let tp_pct = extract_f64(action_config, "tp_pct", 0.0);
                // Default sizes
                let sizes: Vec<(&str, f64)> =
                    vec![("small", 0.05), ("medium", 0.1), ("large", 0.2)];
                RlActionSpace::DiscreteWithSizing(DiscreteWithSizingSpace::new(
                    &sizes, sl_pct, tp_pct,
                ))
            }
            "continuous" => {
                let max_margin_pct = extract_f64(action_config, "max_margin_pct", 0.2);
                let threshold = extract_f64(action_config, "threshold", 0.1);
                let sl_pct = extract_f64(action_config, "sl_pct", 0.0);
                let tp_pct = extract_f64(action_config, "tp_pct", 0.0);
                RlActionSpace::Continuous(ContinuousActionSpace::new(
                    max_margin_pct,
                    threshold,
                    sl_pct,
                    tp_pct,
                ))
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown action_type: '{}'. Use 'discrete', 'discrete_with_sizing', or 'continuous'",
                    action_type
                )));
            }
        };

        // Build observation features
        let mut features: Vec<ObsFeatureType> = Vec::new();
        for spec in &obs_features {
            let s = spec.trim().to_lowercase();
            match s.as_str() {
                "ohlcv" => features.push(ObsFeatureType::Ohlcv),
                "returns" => features.push(ObsFeatureType::Returns),
                "position_info" => features.push(ObsFeatureType::PositionInfo),
                "equity_curve" => features.push(ObsFeatureType::EquityCurve),
                "drawdown" => features.push(ObsFeatureType::Drawdown(DrawdownState::new())),
                "volume_profile" => features.push(ObsFeatureType::VolumeProfile),
                _ => {
                    // SMA ratio: "sma_ratio_20_50"
                    if s.starts_with("sma_ratio_") {
                        let parts: Vec<&str> = s.split('_').collect();
                        if parts.len() == 4 {
                            if let (Ok(fast), Ok(slow)) =
                                (parts[2].parse::<usize>(), parts[3].parse::<usize>())
                            {
                                features.push(ObsFeatureType::SmaRatio(SmaRatioConfig::new(
                                    fast, slow,
                                )));
                                continue;
                            }
                        }
                    }
                    // Indicator shorthand: rsi_14, sma_20, ema_50, atr_14
                    if s.starts_with("rsi_") {
                        features.push(ObsFeatureType::IndicatorObs(IndicatorObsConfig::new(
                            spec.clone(),
                            0.0,
                            100.0,
                        )));
                    } else if s.starts_with("sma_") || s.starts_with("ema_") {
                        features.push(ObsFeatureType::IndicatorObs(IndicatorObsConfig::new(
                            spec.clone(),
                            0.0,
                            200000.0,
                        )));
                    } else if s.starts_with("atr_") {
                        features.push(ObsFeatureType::IndicatorObs(IndicatorObsConfig::new(
                            spec.clone(),
                            0.0,
                            10000.0,
                        )));
                    } else {
                        // Direct indicator key
                        features.push(ObsFeatureType::IndicatorObs(IndicatorObsConfig::new(
                            spec.clone(),
                            0.0,
                            100.0,
                        )));
                    }
                }
            }
        }

        let obs_space = ObservationSpace::new(features);

        // Build reward function
        let reward_fn = match reward_type {
            "pnl" => RewardFn::PnL(PnLState::new()),
            "log_return" => RewardFn::LogReturn,
            "differential_sharpe" => {
                let eta = extract_f64(reward_config, "eta", 0.01);
                let scale = extract_f64(reward_config, "scale", 1.0);
                RewardFn::DifferentialSharpe(DifferentialSharpeState::new(eta, scale))
            }
            "risk_adjusted" => {
                let dd_penalty = extract_f64(reward_config, "drawdown_penalty", 2.0);
                RewardFn::RiskAdjusted(RiskAdjustedState::new(dd_penalty))
            }
            "sortino" => {
                let eta = extract_f64(reward_config, "eta", 0.01);
                let min_std = extract_f64(reward_config, "min_std", 1e-4);
                RewardFn::Sortino(SortinoState::new(eta, min_std))
            }
            "advanced" => {
                let pnl_w = extract_f64(reward_config, "pnl_weight", 1.0);
                let dd_p = extract_f64(reward_config, "drawdown_penalty", 3.0);
                let time_p = extract_f64(reward_config, "time_penalty", 0.001);
                let sl_p = extract_f64(reward_config, "sl_penalty", 2.0);
                let tp_b = extract_f64(reward_config, "tp_bonus", 1.5);
                let liq_p = extract_f64(reward_config, "liq_penalty", 10.0);
                RewardFn::Advanced(AdvancedState::new(pnl_w, dd_p, time_p, sl_p, tp_b, liq_p))
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown reward_type: '{}'. Use 'pnl', 'log_return', 'differential_sharpe', 'risk_adjusted', 'sortino', or 'advanced'",
                    reward_type
                )));
            }
        };

        // Return obs space shape info: [total_size, num_actions]
        let total_size = obs_space.total_size();
        let num_actions = action_space.num_actions();

        let initial_equity = self.cash + self.unrealized_pnl;

        self.rl_config = Some(RlConfig {
            action_space,
            observation_space: obs_space,
            reward_fn,
            history: ObsHistory::new(200),
            step_count: 0,
            prev_equity: initial_equity,
            last_was_liquidation: false,
        });

        // Initialize history
        self.rl_config.as_mut().unwrap().history.reset(initial_equity);

        Ok(vec![total_size, num_actions])
    }

    /// Reset the RL environment state (call on env.reset()).
    fn reset_rl_env(&mut self) {
        let initial_equity = self.cash + self.unrealized_pnl;
        if let Some(ref mut rl) = self.rl_config {
            rl.reset(initial_equity);
        }
    }

    /// Get the observation space bounds as (low, high) numpy arrays.
    fn get_obs_bounds<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        let rl = self.rl_config.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("setup_rl_env not called")
        })?;
        let low = PyArray1::from_vec(py, rl.observation_space.low_bounds.clone());
        let high = PyArray1::from_vec(py, rl.observation_space.high_bounds.clone());
        Ok((low, high))
    }

    /// The RL fast-path step function.
    ///
    /// Accepts a raw f64 action and a flat [f64; 10] candle array to avoid
    /// all PyDict overhead. Returns (obs, reward, terminated, truncated, info).
    ///
    /// candle_data layout:
    ///   [open_time_unix, open, high, low, close, volume, quote_asset_volume,
    ///    trades, taker_buy_base, taker_buy_quote]
    #[pyo3(signature = (action_val, candle_data))]
    fn step_rl<'py>(
        &mut self,
        py: Python<'py>,
        action_val: f64,
        candle_data: [f64; 10],
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, f64, bool, bool, Bound<'py, PyDict>)> {
        if self.rl_config.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "setup_rl_env not called",
            ));
        }

        let close_price = candle_data[4];

        // 1. Translate action -> engine orders
        //    We need to temporarily take rl_config to avoid borrow conflicts.
        {
            let rl = self.rl_config.as_ref().unwrap();
            // Read action_space params (no mutation needed for translation dispatch)
            match &rl.action_space {
                RlActionSpace::Discrete(d) => {
                    let idx = action_val as usize;
                    if idx < d.actions.len() {
                        let action = d.actions[idx];
                        let has_position = self.position.is_some();
                        match action {
                            crate::rl::actions::DiscreteAction::Hold => {}
                            crate::rl::actions::DiscreteAction::OpenLong => {
                                if !has_position {
                                    let sl = if d.sl_pct > 0.0 { Some(close_price * (1.0 - d.sl_pct)) } else { None };
                                    let tp = if d.tp_pct > 0.0 { Some(close_price * (1.0 + d.tp_pct)) } else { None };
                                    self.create_order_internal(OrderSide::LONG, close_price, d.margin_pct, tp, sl);
                                }
                            }
                            crate::rl::actions::DiscreteAction::OpenShort => {
                                if !has_position {
                                    let sl = if d.sl_pct > 0.0 { Some(close_price * (1.0 + d.sl_pct)) } else { None };
                                    let tp = if d.tp_pct > 0.0 { Some(close_price * (1.0 - d.tp_pct)) } else { None };
                                    self.create_order_internal(OrderSide::SHORT, close_price, d.margin_pct, tp, sl);
                                }
                            }
                            crate::rl::actions::DiscreteAction::Close => {
                                if has_position {
                                    self.close_order_internal(close_price);
                                }
                            }
                        }
                    }
                }
                RlActionSpace::DiscreteWithSizing(d) => {
                    let idx = action_val as usize;
                    if idx < d.actions.len() {
                        let has_position = self.position.is_some();
                        match &d.actions[idx] {
                            crate::rl::actions::SizedAction::Hold => {}
                            crate::rl::actions::SizedAction::Close => {
                                if has_position { self.close_order_internal(close_price); }
                            }
                            crate::rl::actions::SizedAction::OpenLong(mpct) => {
                                if !has_position {
                                    let sl = if d.sl_pct > 0.0 { Some(close_price * (1.0 - d.sl_pct)) } else { None };
                                    let tp = if d.tp_pct > 0.0 { Some(close_price * (1.0 + d.tp_pct)) } else { None };
                                    self.create_order_internal(OrderSide::LONG, close_price, *mpct, tp, sl);
                                }
                            }
                            crate::rl::actions::SizedAction::OpenShort(mpct) => {
                                if !has_position {
                                    let sl = if d.sl_pct > 0.0 { Some(close_price * (1.0 + d.sl_pct)) } else { None };
                                    let tp = if d.tp_pct > 0.0 { Some(close_price * (1.0 - d.tp_pct)) } else { None };
                                    self.create_order_internal(OrderSide::SHORT, close_price, *mpct, tp, sl);
                                }
                            }
                        }
                    }
                }
                RlActionSpace::Continuous(c) => {
                    let signal = action_val.clamp(-1.0, 1.0);
                    let has_position = self.position.is_some();
                    if signal.abs() <= c.threshold {
                        // Dead zone: do nothing (hold current state).
                        // Positions exit via SL/TP or opposite-direction signal.
                    } else if signal > c.threshold {
                        if has_position {
                            let is_short = self.position.as_ref().map_or(false, |p| p.side == PositionSide::SHORT);
                            if is_short { self.close_order_internal(close_price); }
                            // already long → hold
                        } else {
                            let margin = signal.abs() * c.max_margin_pct;
                            let sl = if c.sl_pct > 0.0 { Some(close_price * (1.0 - c.sl_pct)) } else { None };
                            let tp = if c.tp_pct > 0.0 { Some(close_price * (1.0 + c.tp_pct)) } else { None };
                            self.create_order_internal(OrderSide::LONG, close_price, margin, tp, sl);
                        }
                    } else {
                        // signal < -threshold → want short
                        if has_position {
                            let is_long = self.position.as_ref().map_or(false, |p| p.side == PositionSide::LONG);
                            if is_long { self.close_order_internal(close_price); }
                            // already short → hold
                        } else {
                            let margin = signal.abs() * c.max_margin_pct;
                            let sl = if c.sl_pct > 0.0 { Some(close_price * (1.0 + c.sl_pct)) } else { None };
                            let tp = if c.tp_pct > 0.0 { Some(close_price * (1.0 - c.tp_pct)) } else { None };
                            self.create_order_internal(OrderSide::SHORT, close_price, margin, tp, sl);
                        }
                    }
                }
            }
        }

        // 2. Step the engine with raw candle data (on_candle_raw)
        let (had_tp, had_sl, had_liquidation) = self.on_candle_raw(py, &candle_data)?;

        // 3. Compute reward
        let eq_curr = self.cash + self.unrealized_pnl;
        let has_position = self.position.is_some();

        let rl = self.rl_config.as_mut().unwrap();
        let eq_prev = rl.prev_equity;

        let reward = rl.reward_fn.compute(
            eq_prev,
            eq_curr,
            has_position,
            had_tp,
            had_sl,
            had_liquidation,
        );

        // 4. Check termination
        let terminated = eq_curr <= 0.0 || had_liquidation;
        let truncated = false; // truncation is managed by Python (max_steps)

        // 5. Push volumes into history (so mean_vol includes current candle,
        //    matching Python where history.push() happens before observe).
        //    But DON'T update prev_close yet — Returns needs the previous close.
        let raw_candle = RawCandle::from_raw(&candle_data);
        rl.history.push_volumes(raw_candle.volume, raw_candle.taker_buy_base);

        // 6. Build observation (prev_close still reflects the PREVIOUS candle)
        let engine_state = EngineState {
            equity: eq_curr,
            unrealized_pnl: self.unrealized_pnl,
            used_initial_margin: if self.config.margin_mode == "cross" {
                self.cross_used_margin
            } else {
                self.position.as_ref().map_or(0.0, |p| p.margin)
            },
            has_position,
            position_side: self.position.as_ref().map(|p| p.side),
        };

        let obs_vec = rl.observation_space.observe(
            &raw_candle,
            &engine_state,
            &rl.history,
            &self.indicators,
        );

        // 7. Update prev_close and step counter (for next step)
        rl.history.update_prev_close(raw_candle.close);
        rl.prev_equity = eq_curr;
        rl.step_count += 1;
        rl.last_was_liquidation = had_liquidation;
        let step_count = rl.step_count;

        let obs = PyArray1::from_vec(py, obs_vec);

        // 7. Build minimal info dict
        let info = PyDict::new(py);
        info.set_item("step", step_count)?;
        info.set_item("equity", eq_curr)?;
        info.set_item("reward", reward)?;
        info.set_item("is_liquidated", had_liquidation)?;

        Ok((obs, reward, terminated, truncated, info))
    }
}

// ---------------------------------------------------------------------------
// Internal constructors and helpers (used by vectorized.rs, rl, etc.)
// ---------------------------------------------------------------------------

impl RustTradingEngine {
    /// Create an engine instance from Rust without PyO3/Python overhead.
    /// Used by `run_signals_backtest` and other internal callers.
    pub(crate) fn new_internal(
        symbol: String,
        margin_mode: String,
        leverage: f64,
        starting_cash: f64,
        slippage_pct: f64,
        timeframe_minutes: i64,
    ) -> Self {
        Self {
            config: EngineConfig {
                symbol,
                margin_mode,
                leverage,
                slippage_pct,
                timeframe_minutes,
                funding_data: None,
            },
            cash: starting_cash,
            current_index: -1,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            insurance_fund: 0.0,
            total_funding_paid: 0.0,
            cross_used_margin: 0.0,
            position: None,
            open_order: None,
            close_request: None,
            tp: None,
            sl: None,
            pending_open_order: None,
            pending_close_request: None,
            pending_tp: None,
            pending_sl: None,
            scheduled_cancels: Vec::new(),
            last_candle_meta: LastCandleMeta::default(),
            intra_candles: None,
            indicators: BTreeMap::new(),
            registered_indicators: Vec::new(),
            rl_config: None,
        }
    }

    /// Process one candle from raw f64 array. Returns (had_tp, had_sl, had_liquidation).
    /// This is the hot-path equivalent of on_candle without PyDict or event list overhead.
    pub(crate) fn on_candle_raw<'py>(
        &mut self,
        py: Python<'py>,
        data: &[f64; 10],
    ) -> PyResult<(bool, bool, bool)> {
        let _open_time_unix = data[0];
        let open_p = data[1];
        let high = data[2];
        let low = data[3];
        let close = data[4];
        let volume = data[5];
        let quote_asset_volume = data[6];
        let trades = data[7] as i64;
        let taker_buy_base = data[8];
        let taker_buy_quote = data[9];

        self.current_index += 1;
        let index = self.current_index;

        // Build CandleData with open_time as None (we use unix timestamps in RL)
        let candle_data = CandleData {
            open_time: None,
            open: open_p,
            high,
            low,
            close,
        };

        // Update metadata
        self.last_candle_meta = LastCandleMeta {
            open_time: None,
            volume,
            quote_asset_volume,
            number_of_trades: trades,
            taker_buy_base_asset_volume: taker_buy_base,
            taker_buy_quote_asset_volume: taker_buy_quote,
        };

        // 1: refresh UPNL at close
        self.refresh_upnl(close);

        // 1.5: funding — skip in RL fast path (no ISO timestamps available for lookup)
        // Funding is a minor effect and not worth the overhead in RL training.

        // 2: triggers — we need to detect what type of trigger occurred
        let mut had_tp = false;
        let mut had_sl = false;
        let mut had_liquidation = false;

        let consumed_by_trigger = if self.position.is_some() {
            // Inline trigger detection to capture event types
            let liq_price = {
                let pos = self.position.as_ref().unwrap();
                if self.config.margin_mode == "isolated" {
                    calc_isolated_liq_price(pos.side, pos.entry_price, self.config.leverage)
                } else {
                    calc_cross_liq_price(pos.side, pos.entry_price, pos.size, self.cash)
                }
            };

            let tp_price = self.tp.as_ref().and_then(|tp| {
                tp.eligible_from_index
                    .and_then(|efi| if efi <= index { Some(tp.price) } else { None })
            });
            let sl_price = self.sl.as_ref().and_then(|sl| {
                sl.eligible_from_index
                    .and_then(|efi| if efi <= index { Some(sl.price) } else { None })
            });

            let mut triggers: Vec<(&str, f64)> = Vec::new();
            if in_range(liq_price, low, high) {
                triggers.push(("liq", liq_price));
            }
            if let Some(sp) = sl_price {
                if in_range(sp, low, high) {
                    triggers.push(("sl", sp));
                }
            }
            if let Some(tp) = tp_price {
                if in_range(tp, low, high) {
                    triggers.push(("tp", tp));
                }
            }

            if triggers.is_empty() {
                false
            } else {
                // Use dummy PyList for event recording (we discard events in RL)
                let events = PyList::empty(py);

                if triggers.len() == 1 {
                    match triggers[0].0 {
                        "liq" => {
                            self.fill_liquidation(py, liq_price, &candle_data, &events)?;
                            had_liquidation = true;
                        }
                        "sl" => {
                            let order = self.sl.clone().unwrap();
                            self.execute_close(py, &order, &candle_data, &events, "sl")?;
                            had_sl = true;
                        }
                        "tp" => {
                            let order = self.tp.clone().unwrap();
                            self.execute_close(py, &order, &candle_data, &events, "tp")?;
                            had_tp = true;
                        }
                        _ => {}
                    }
                } else {
                    // Multiple triggers — use 1m data or fallback priority
                    let first_trigger = if self.config.timeframe_minutes > 1 {
                        self.determine_first_trigger(&candle_data, liq_price, tp_price, sl_price)
                    } else {
                        None
                    };

                    match first_trigger {
                        Some("liq") => {
                            self.fill_liquidation(py, liq_price, &candle_data, &events)?;
                            had_liquidation = true;
                        }
                        Some("sl") => {
                            let order = self.sl.clone().unwrap();
                            self.execute_close(py, &order, &candle_data, &events, "sl")?;
                            had_sl = true;
                        }
                        Some("tp") => {
                            let order = self.tp.clone().unwrap();
                            self.execute_close(py, &order, &candle_data, &events, "tp")?;
                            had_tp = true;
                        }
                        _ => {
                            // Fallback priority: LIQ > SL > TP
                            if in_range(liq_price, low, high) {
                                self.fill_liquidation(py, liq_price, &candle_data, &events)?;
                                had_liquidation = true;
                            } else if let Some(sp) = sl_price {
                                if in_range(sp, low, high) {
                                    let order = self.sl.clone().unwrap();
                                    self.execute_close(py, &order, &candle_data, &events, "sl")?;
                                    had_sl = true;
                                }
                            } else if let Some(tp) = tp_price {
                                if in_range(tp, low, high) {
                                    let order = self.tp.clone().unwrap();
                                    self.execute_close(py, &order, &candle_data, &events, "tp")?;
                                    had_tp = true;
                                }
                            }
                        }
                    }
                }
                true
            }
        } else {
            false
        };

        // 3: apply cancels (no events recorded in RL)
        self.scheduled_cancels.retain(|(_, eidx, _)| *eidx > index);
        // Simplified cancel processing: just clear the slots
        {
            let eligible: Vec<(String, String)> = self
                .scheduled_cancels
                .iter()
                .filter(|(_, eidx, _)| *eidx <= index)
                .map(|(oid, _, role)| (oid.clone(), role.clone()))
                .collect();
            self.scheduled_cancels.retain(|(_, eidx, _)| *eidx > index);
            for (oid, role) in eligible {
                match role.as_str() {
                    "open_order" => {
                        if self.open_order.as_ref().map_or(false, |o| o.id == oid) {
                            self.open_order = None;
                        } else if self.pending_open_order.as_ref().map_or(false, |o| o.id == oid) {
                            self.pending_open_order = None;
                        }
                    }
                    "close_request" => {
                        if self.close_request.as_ref().map_or(false, |o| o.id == oid) {
                            self.close_request = None;
                        } else if self.pending_close_request.as_ref().map_or(false, |o| o.id == oid) {
                            self.pending_close_request = None;
                        }
                    }
                    "tp" => {
                        if self.tp.as_ref().map_or(false, |o| o.id == oid) {
                            self.tp = None;
                        } else if self.pending_tp.as_ref().map_or(false, |o| o.id == oid) {
                            self.pending_tp = None;
                        }
                    }
                    "sl" => {
                        if self.sl.as_ref().map_or(false, |o| o.id == oid) {
                            self.sl = None;
                        } else if self.pending_sl.as_ref().map_or(false, |o| o.id == oid) {
                            self.pending_sl = None;
                        }
                    }
                    _ => {}
                }
            }
        }

        // 4: promote pending
        self.promote_pending(index);

        // 5: user orders (open_order / close_request)
        if !consumed_by_trigger {
            let events = PyList::empty(py);
            self.process_user_orders(py, &candle_data, index, open_p, low, high, &events)?;
        }

        // 6: refresh UPNL at close
        self.refresh_upnl(close);

        // 7: update registered indicators
        for ind in &mut self.registered_indicators {
            let outputs = ind.push_candle(open_p, high, low, close);
            for (key, val) in outputs {
                self.indicators.insert(key, val);
            }
        }

        Ok((had_tp, had_sl, had_liquidation))
    }
}

// ---------------------------------------------------------------------------
// Helper: extract f64 from optional PyDict
// ---------------------------------------------------------------------------

fn extract_f64(dict: Option<&Bound<'_, PyDict>>, key: &str, default: f64) -> f64 {
    dict.and_then(|d| {
        d.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f64>().ok())
    })
    .unwrap_or(default)
}

fn extract_string_list(dict: Option<&Bound<'_, PyDict>>, key: &str) -> Vec<String> {
    dict.and_then(|d| {
        d.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<Vec<String>>().ok())
    })
    .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolated_liq_long() {
        let liq = calc_isolated_liq_price(PositionSide::LONG, 100.0, 10.0);
        assert_eq!(liq, 90.5);
    }

    #[test]
    fn test_isolated_liq_short() {
        let liq = calc_isolated_liq_price(PositionSide::SHORT, 100.0, 10.0);
        assert_eq!(liq, 109.5);
    }

    #[test]
    fn test_cross_liq_long() {
        let liq = calc_cross_liq_price(PositionSide::LONG, 100.0, 1.0, 100.0);
        assert_eq!(liq, 0.0);
    }

    #[test]
    fn test_cross_liq_short() {
        let liq = calc_cross_liq_price(PositionSide::SHORT, 100.0, 1.0, 100.0);
        assert_eq!(liq, r_price(200.0 / 1.005));
    }

    #[test]
    fn test_cross_liq_zero_size() {
        let liq = calc_cross_liq_price(PositionSide::LONG, 100.0, 0.0, 100.0);
        assert!(liq.is_infinite());
    }

    #[test]
    fn test_parse_iso_z_suffix() {
        assert_eq!(parse_iso_to_unix("2024-01-01T00:00:00Z"), Some(1704067200));
    }

    #[test]
    fn test_parse_iso_offset_suffix() {
        assert_eq!(
            parse_iso_to_unix("2024-01-01T00:00:00+00:00"),
            Some(1704067200)
        );
    }

    #[test]
    fn test_parse_iso_naive_t() {
        assert_eq!(
            parse_iso_to_unix("2024-01-01T00:00:00"),
            Some(1704067200)
        );
    }

    #[test]
    fn test_parse_iso_naive_space() {
        assert_eq!(
            parse_iso_to_unix("2024-01-01 00:00:00"),
            Some(1704067200)
        );
    }

    #[test]
    fn test_parse_iso_empty() {
        assert_eq!(parse_iso_to_unix(""), None);
        assert_eq!(parse_iso_to_unix("   "), None);
    }
}
