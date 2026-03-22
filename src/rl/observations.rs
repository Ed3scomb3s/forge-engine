//! RL Observation Features — Phase 6
//!
//! Ports: OHLCV, Returns, PositionInfo, EquityCurve, Drawdown,
//!        IndicatorObs, SMA_Ratio, VolumeProfile.
//!
//! All features produce f32 values concatenated into a flat Vec<f32>.
//! Bounds clipping is enforced per feature.

use std::collections::BTreeMap;
use std::collections::VecDeque;

use crate::types::PositionSide;

// ---------------------------------------------------------------------------
// ObsHistory — lightweight Rust-native history buffer
// ---------------------------------------------------------------------------

/// Rolling volume history + initial equity for observation features.
/// Tracks only what the obs features actually need (volumes, prev close, taker buy vol).
pub struct ObsHistory {
    /// Rolling window of recent candle volumes (max 200).
    pub volumes: VecDeque<f64>,
    /// Rolling window of recent taker buy base volumes (max 200).
    pub taker_buy_vols: VecDeque<f64>,
    /// Previous candle's close price (for Returns calculation).
    pub prev_close: f64,
    /// Initial equity (set on first step).
    pub initial_equity: f64,
    /// Maximum history length.
    max_len: usize,
}

impl ObsHistory {
    pub fn new(max_len: usize) -> Self {
        Self {
            volumes: VecDeque::with_capacity(max_len),
            taker_buy_vols: VecDeque::with_capacity(max_len),
            prev_close: 0.0,
            initial_equity: 1.0,
            max_len,
        }
    }

    pub fn reset(&mut self, initial_equity: f64) {
        self.volumes.clear();
        self.taker_buy_vols.clear();
        self.prev_close = 0.0;
        self.initial_equity = if initial_equity > 0.0 {
            initial_equity
        } else {
            1.0
        };
    }

    /// Push a candle's volume into history and update prev_close.
    ///
    /// Call sequence in step_rl:
    ///   1. push_volumes() — includes current candle vol in mean for OHLCV/VolumeProfile
    ///   2. observe() — uses prev_close (still the PREVIOUS candle's close)
    ///   3. update_prev_close() — sets prev_close to current for next step
    pub fn push_volumes(&mut self, volume: f64, taker_buy_vol: f64) {
        if self.volumes.len() == self.max_len {
            self.volumes.pop_front();
        }
        self.volumes.push_back(volume);

        if self.taker_buy_vols.len() == self.max_len {
            self.taker_buy_vols.pop_front();
        }
        self.taker_buy_vols.push_back(taker_buy_vol);
    }

    /// Update prev_close after observation is computed.
    pub fn update_prev_close(&mut self, close: f64) {
        self.prev_close = close;
    }

    /// Mean of positive volumes in the buffer.
    pub fn mean_volume(&self, current_vol: f64) -> f64 {
        let mut sum = 0.0;
        let mut count = 0u64;
        for &v in &self.volumes {
            if v > 0.0 {
                sum += v;
                count += 1;
            }
        }
        if count > 0 {
            sum / count as f64
        } else {
            current_vol.max(1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Candle data snapshot (extracted from [f64; 10] for obs features)
// ---------------------------------------------------------------------------

/// Raw candle data extracted from the [f64; 10] array.
pub struct RawCandle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub taker_buy_base: f64,
}

impl RawCandle {
    /// Extract from the flat [f64; 10] array.
    /// Layout: [open_time_unix, open, high, low, close, volume, quote_asset_volume, trades, taker_buy_base, taker_buy_quote]
    pub fn from_raw(data: &[f64; 10]) -> Self {
        Self {
            open: data[1],
            high: data[2],
            low: data[3],
            close: data[4],
            volume: data[5],
            taker_buy_base: data[8],
        }
    }
}

// ---------------------------------------------------------------------------
// Engine state snapshot (extracted without PyDict)
// ---------------------------------------------------------------------------

/// Minimal engine state needed by observation features.
pub struct EngineState {
    pub equity: f64,
    pub unrealized_pnl: f64,
    pub used_initial_margin: f64,
    pub has_position: bool,
    pub position_side: Option<PositionSide>,
}

// ---------------------------------------------------------------------------
// ObsFeatureType — enum of all concrete observation features
// ---------------------------------------------------------------------------

pub enum ObsFeatureType {
    Ohlcv,
    Returns,
    PositionInfo,
    EquityCurve,
    Drawdown(DrawdownState),
    IndicatorObs(IndicatorObsConfig),
    SmaRatio(SmaRatioConfig),
    MacdNormalized(MacdNormalizedConfig),
    VolumeProfile,
}

impl ObsFeatureType {
    /// Number of f32 values this feature produces.
    pub fn size(&self) -> usize {
        match self {
            Self::Ohlcv => 5,
            Self::Returns => 1,
            Self::PositionInfo => 4,
            Self::EquityCurve => 1,
            Self::Drawdown(_) => 1,
            Self::IndicatorObs(_) => 1,
            Self::SmaRatio(_) => 1,
            Self::MacdNormalized(_) => 1,
            Self::VolumeProfile => 2,
        }
    }

    /// Compute observation values and append to `out`.
    pub fn observe(
        &mut self,
        candle: &RawCandle,
        state: &EngineState,
        history: &ObsHistory,
        indicators: &BTreeMap<String, f64>,
        out: &mut Vec<f32>,
    ) {
        match self {
            Self::Ohlcv => observe_ohlcv(candle, history, out),
            Self::Returns => observe_returns(candle, history, out),
            Self::PositionInfo => observe_position_info(state, history, out),
            Self::EquityCurve => observe_equity_curve(state, history, out),
            Self::Drawdown(dd) => observe_drawdown(state, dd, out),
            Self::IndicatorObs(cfg) => observe_indicator(indicators, cfg, out),
            Self::SmaRatio(cfg) => observe_sma_ratio(indicators, cfg, out),
            Self::MacdNormalized(cfg) => observe_macd_normalized(candle, indicators, cfg, out),
            Self::VolumeProfile => observe_volume_profile(candle, history, out),
        }
    }

    /// Reset any internal state (called on env reset).
    pub fn reset(&mut self) {
        if let Self::Drawdown(dd) = self {
            dd.hwm = 0.0;
        }
    }

    /// Low bounds for this feature.
    pub fn low(&self) -> Vec<f32> {
        match self {
            Self::Ohlcv => vec![-1.0, -1.0, -1.0, -1.0, 0.0],
            Self::Returns => vec![-1.0],
            Self::PositionInfo => vec![0.0, -1.0, -1.0, 0.0],
            Self::EquityCurve => vec![-1.0],
            Self::Drawdown(_) => vec![0.0],
            Self::IndicatorObs(_) => vec![0.0],
            Self::SmaRatio(_) => vec![-0.5],
            Self::MacdNormalized(_) => vec![-10.0],
            Self::VolumeProfile => vec![0.0, -1.0],
        }
    }

    /// High bounds for this feature.
    pub fn high(&self) -> Vec<f32> {
        match self {
            Self::Ohlcv => vec![1.0, 1.0, 1.0, 1.0, 10.0],
            Self::Returns => vec![1.0],
            Self::PositionInfo => vec![1.0, 1.0, 1.0, 1.0],
            Self::EquityCurve => vec![5.0],
            Self::Drawdown(_) => vec![1.0],
            Self::IndicatorObs(_) => vec![1.0],
            Self::SmaRatio(_) => vec![0.5],
            Self::MacdNormalized(_) => vec![10.0],
            Self::VolumeProfile => vec![1.0, 5.0],
        }
    }
}

// ---------------------------------------------------------------------------
// Drawdown internal state
// ---------------------------------------------------------------------------

pub struct DrawdownState {
    pub hwm: f64,
}

impl DrawdownState {
    pub fn new() -> Self {
        Self { hwm: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// IndicatorObs config
// ---------------------------------------------------------------------------

pub struct IndicatorObsConfig {
    /// The resolved indicator key (e.g., "RSI(14)[close]").
    pub key: String,
    pub raw_low: f64,
    pub raw_high: f64,
}

impl IndicatorObsConfig {
    pub fn new(key: String, raw_low: f64, raw_high: f64) -> Self {
        // Expand shorthand: rsi_14 -> RSI(14)[close], etc.
        let resolved = expand_shorthand(&key);
        Self {
            key: resolved,
            raw_low,
            raw_high,
        }
    }
}

// ---------------------------------------------------------------------------
// SMA_Ratio config
// ---------------------------------------------------------------------------

pub struct SmaRatioConfig {
    pub fast_key: String,
    pub slow_key: String,
}

impl SmaRatioConfig {
    pub fn new(fast: usize, slow: usize) -> Self {
        Self {
            fast_key: format!("SMA({})[close]", fast),
            slow_key: format!("SMA({})[close]", slow),
        }
    }
}

pub struct MacdNormalizedConfig {
    pub key: String,
}

impl MacdNormalizedConfig {
    pub fn from_spec(spec: &str) -> Option<Self> {
        let lowered = spec.trim().to_lowercase();
        let parts: Vec<&str> = lowered.split('_').collect();
        let (component, fast, slow, signal) = match parts.as_slice() {
            ["macd", component] if matches!(*component, "line" | "signal" | "hist") => {
                (*component, 12usize, 26usize, 9usize)
            }
            ["macd", component, fast, slow, signal]
                if matches!(*component, "line" | "signal" | "hist") =>
            {
                let fast = fast.parse::<usize>().ok()?;
                let slow = slow.parse::<usize>().ok()?;
                let signal = signal.parse::<usize>().ok()?;
                (*component, fast, slow, signal)
            }
            _ => return None,
        };

        Some(Self {
            key: format!("MACD({},{},{})[close].{}", fast, slow, signal, component),
        })
    }
}

// ---------------------------------------------------------------------------
// Shorthand expansion (matches Python IndicatorObs._expand_shorthand)
// ---------------------------------------------------------------------------

fn expand_shorthand(key: &str) -> String {
    let k = key.trim();
    let low = k.to_lowercase();
    if let Some(rest) = low.strip_prefix("rsi_") {
        return format!("RSI({})[close]", rest);
    }
    if let Some(rest) = low.strip_prefix("sma_") {
        return format!("SMA({})[close]", rest);
    }
    if let Some(rest) = low.strip_prefix("ema_") {
        return format!("EMA({})[close]", rest);
    }
    if let Some(rest) = low.strip_prefix("atr_") {
        return format!("ATR({})", rest);
    }
    k.to_string()
}

// ---------------------------------------------------------------------------
// ObservationSpace — holds all features and produces flat Vec<f32>
// ---------------------------------------------------------------------------

pub struct ObservationSpace {
    pub features: Vec<ObsFeatureType>,
    /// Precomputed low/high bounds for the full observation vector.
    pub low_bounds: Vec<f32>,
    pub high_bounds: Vec<f32>,
}

impl ObservationSpace {
    pub fn new(features: Vec<ObsFeatureType>) -> Self {
        let mut low_bounds = Vec::new();
        let mut high_bounds = Vec::new();
        for f in &features {
            low_bounds.extend(f.low());
            high_bounds.extend(f.high());
        }
        Self {
            features,
            low_bounds,
            high_bounds,
        }
    }

    /// Total observation size.
    pub fn total_size(&self) -> usize {
        self.features.iter().map(|f| f.size()).sum()
    }

    /// Compute the full observation vector with bounds clipping.
    pub fn observe(
        &mut self,
        candle: &RawCandle,
        state: &EngineState,
        history: &ObsHistory,
        indicators: &BTreeMap<String, f64>,
    ) -> Vec<f32> {
        let total = self.total_size();
        let mut obs = Vec::with_capacity(total);

        for feature in &mut self.features {
            feature.observe(candle, state, history, indicators, &mut obs);
        }

        // Clip to bounds
        for (i, val) in obs.iter_mut().enumerate() {
            if i < self.low_bounds.len() {
                *val = val.clamp(self.low_bounds[i], self.high_bounds[i]);
            }
        }

        obs
    }

    /// Reset all features (for env reset).
    pub fn reset(&mut self) {
        for f in &mut self.features {
            f.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete observation functions
// ---------------------------------------------------------------------------

const EPS: f64 = 1e-10;

fn observe_ohlcv(candle: &RawCandle, history: &ObsHistory, out: &mut Vec<f32>) {
    let c = if candle.close > 0.0 {
        candle.close
    } else {
        1.0
    };
    let o = if candle.open > 0.0 { candle.open } else { c };
    let h = if candle.high > 0.0 { candle.high } else { c };
    let lo = if candle.low > 0.0 { candle.low } else { c };
    let v = if candle.volume > 0.0 {
        candle.volume
    } else {
        0.0
    };

    let log_oc = (o.max(EPS) / c.max(EPS)).ln();
    let log_hc = (h.max(EPS) / c.max(EPS)).ln();
    let log_lc = (lo.max(EPS) / c.max(EPS)).ln();

    let mean_vol = history.mean_volume(v);
    let norm_v = v / mean_vol.max(EPS);

    out.push(log_oc as f32);
    out.push(log_hc as f32);
    out.push(log_lc as f32);
    out.push(0.0_f32);
    out.push(norm_v as f32);
}

fn observe_returns(candle: &RawCandle, history: &ObsHistory, out: &mut Vec<f32>) {
    if history.prev_close <= 0.0 {
        out.push(0.0);
        return;
    }
    let curr_c = if candle.close > 0.0 {
        candle.close
    } else {
        1.0
    };
    let lr = (curr_c.max(EPS) / history.prev_close.max(EPS)).ln();
    out.push((lr as f32).clamp(-1.0, 1.0));
}

fn observe_position_info(state: &EngineState, history: &ObsHistory, out: &mut Vec<f32>) {
    let has_pos = if state.has_position { 1.0_f32 } else { 0.0 };
    let side = match state.position_side {
        Some(PositionSide::LONG) => 1.0_f32,
        Some(PositionSide::SHORT) => -1.0_f32,
        None => 0.0_f32,
    };
    let upnl_pct = (state.unrealized_pnl / history.initial_equity.max(EPS)).clamp(-1.0, 1.0);
    let margin_pct =
        (state.used_initial_margin / state.equity.max(EPS)).clamp(0.0, 1.0);

    out.push(has_pos);
    out.push(side);
    out.push(upnl_pct as f32);
    out.push(margin_pct as f32);
}

fn observe_equity_curve(state: &EngineState, history: &ObsHistory, out: &mut Vec<f32>) {
    let rel = state.equity / history.initial_equity.max(EPS) - 1.0;
    out.push((rel as f32).clamp(-1.0, 5.0));
}

fn observe_drawdown(state: &EngineState, dd: &mut DrawdownState, out: &mut Vec<f32>) {
    if state.equity > dd.hwm {
        dd.hwm = state.equity;
    }
    let drawdown = if dd.hwm <= 0.0 {
        0.0
    } else {
        (dd.hwm - state.equity) / dd.hwm
    };
    out.push((drawdown as f32).clamp(0.0, 1.0));
}

fn observe_indicator(
    indicators: &BTreeMap<String, f64>,
    cfg: &IndicatorObsConfig,
    out: &mut Vec<f32>,
) {
    match indicators.get(&cfg.key) {
        None => out.push(0.5), // neutral default
        Some(&raw) => {
            let span = cfg.raw_high - cfg.raw_low;
            let normed = if span <= 0.0 {
                0.5
            } else {
                (raw - cfg.raw_low) / span
            };
            out.push((normed as f32).clamp(0.0, 1.0));
        }
    }
}

fn observe_sma_ratio(
    indicators: &BTreeMap<String, f64>,
    cfg: &SmaRatioConfig,
    out: &mut Vec<f32>,
) {
    let f_val = indicators.get(&cfg.fast_key);
    let s_val = indicators.get(&cfg.slow_key);
    match (f_val, s_val) {
        (Some(&f), Some(&s)) if s != 0.0 => {
            let ratio = f / s - 1.0;
            out.push((ratio as f32).clamp(-0.5, 0.5));
        }
        _ => out.push(0.0),
    }
}

fn observe_macd_normalized(
    candle: &RawCandle,
    indicators: &BTreeMap<String, f64>,
    cfg: &MacdNormalizedConfig,
    out: &mut Vec<f32>,
) {
    let raw = indicators.get(&cfg.key).copied().unwrap_or(0.0);
    let close = candle.close.max(EPS);
    let normalized_pct = (raw / close) * 100.0;
    out.push((normalized_pct as f32).clamp(-10.0, 10.0));
}

fn observe_volume_profile(candle: &RawCandle, history: &ObsHistory, out: &mut Vec<f32>) {
    let v = candle.volume;
    let tb = candle.taker_buy_base;

    let buy_ratio = if v > 0.0 { tb / v.max(EPS) } else { 0.5 };

    let mean_vol = history.mean_volume(v);
    let vol_change = v / mean_vol.max(EPS) - 1.0;

    out.push((buy_ratio as f32).clamp(0.0, 1.0));
    out.push((vol_change as f32).clamp(-1.0, 5.0));
}
