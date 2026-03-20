//! Native Rust Indicators — Phase 5
//!
//! SMA, EMA, RSI, ATR, BollingerBands, MACD.
//! All outputs use `python_round(val, 8)` for exact parity with Python's `round(x, 8)`.

use std::collections::VecDeque;

use crate::types::python_round;

// ---------------------------------------------------------------------------
// CandleSource — which OHLC field to feed into a single-source indicator
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum CandleSource {
    Open,
    High,
    Low,
    Close,
}

impl CandleSource {
    #[inline]
    pub fn extract(&self, open: f64, high: f64, low: f64, close: f64) -> f64 {
        match self {
            Self::Open => open,
            Self::High => high,
            Self::Low => low,
            Self::Close => close,
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "open" => Ok(Self::Open),
            "high" => Ok(Self::High),
            "low" => Ok(Self::Low),
            "close" => Ok(Self::Close),
            _ => Err(format!(
                "Invalid source: '{}'. Must be one of: open, high, low, close",
                s
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// SMA
// ---------------------------------------------------------------------------

pub struct SmaIndicator {
    pub label: String,
    pub source: CandleSource,
    period: usize,
    buf: VecDeque<f64>,
    sum: f64,
}

impl SmaIndicator {
    pub fn new(label: String, period: usize, source: CandleSource) -> Self {
        Self {
            label,
            source,
            period,
            buf: VecDeque::with_capacity(period),
            sum: 0.0,
        }
    }

    fn push(&mut self, value: f64) -> Option<f64> {
        if self.buf.len() == self.period {
            self.sum -= self.buf.pop_front().unwrap();
        }
        self.buf.push_back(value);
        self.sum += value;
        if self.buf.len() == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    pub fn push_candle(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        let v = self.source.extract(open, high, low, close);
        match self.push(v) {
            Some(val) => vec![(self.label.clone(), python_round(val, 8))],
            None => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// EMA
// ---------------------------------------------------------------------------

pub struct EmaIndicator {
    pub label: String,
    pub source: CandleSource,
    period: usize,
    alpha: f64,
    ema: Option<f64>,
    seed_sum: f64,
    seed_count: usize,
}

impl EmaIndicator {
    pub fn new(label: String, period: usize, source: CandleSource) -> Self {
        let p = period.max(1);
        Self {
            label,
            source,
            period: p,
            alpha: 2.0 / (p as f64 + 1.0),
            ema: None,
            seed_sum: 0.0,
            seed_count: 0,
        }
    }

    fn push(&mut self, value: f64) -> Option<f64> {
        if self.ema.is_none() {
            if self.seed_count < self.period {
                self.seed_sum += value;
                self.seed_count += 1;
                if self.seed_count == self.period {
                    self.ema = Some(self.seed_sum / self.period as f64);
                }
                return None;
            }
            // Fallback for period == 1 edge case
            if self.period == 1 {
                self.ema = Some(value);
                return self.ema;
            }
        } else {
            let ema = self.ema.unwrap();
            self.ema = Some((value - ema) * self.alpha + ema);
        }
        self.ema
    }

    pub fn push_candle(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        let v = self.source.extract(open, high, low, close);
        match self.push(v) {
            Some(val) => vec![(self.label.clone(), python_round(val, 8))],
            None => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// RSI
// ---------------------------------------------------------------------------

pub struct RsiIndicator {
    pub label: String,
    pub source: CandleSource,
    period: usize,
    prev: Option<f64>,
    avg_gain: Option<f64>,
    avg_loss: Option<f64>,
    seed_g: f64,
    seed_l: f64,
    seed_count: usize,
}

impl RsiIndicator {
    pub fn new(label: String, period: usize, source: CandleSource) -> Self {
        let p = period.max(1);
        Self {
            label,
            source,
            period: p,
            prev: None,
            avg_gain: None,
            avg_loss: None,
            seed_g: 0.0,
            seed_l: 0.0,
            seed_count: 0,
        }
    }

    fn push(&mut self, value: f64) -> Option<f64> {
        if self.prev.is_none() {
            self.prev = Some(value);
            return None;
        }
        let delta = value - self.prev.unwrap();
        self.prev = Some(value);
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };

        if self.avg_gain.is_none() || self.avg_loss.is_none() {
            if self.seed_count < self.period {
                self.seed_g += gain;
                self.seed_l += loss;
                self.seed_count += 1;
                if self.seed_count == self.period {
                    self.avg_gain = Some(self.seed_g / self.period as f64);
                    self.avg_loss = Some(self.seed_l / self.period as f64);
                }
                return None;
            }
        } else {
            let ag = self.avg_gain.unwrap();
            let al = self.avg_loss.unwrap();
            self.avg_gain =
                Some((ag * (self.period - 1) as f64 + gain) / self.period as f64);
            self.avg_loss =
                Some((al * (self.period - 1) as f64 + loss) / self.period as f64);
        }

        if self.avg_gain.is_none() || self.avg_loss.is_none() {
            return None;
        }

        let ag = self.avg_gain.unwrap();
        let al = self.avg_loss.unwrap();
        let rsi = if al == 0.0 {
            100.0
        } else {
            let rs = ag / al;
            100.0 - (100.0 / (1.0 + rs))
        };
        Some(rsi)
    }

    pub fn push_candle(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        let v = self.source.extract(open, high, low, close);
        match self.push(v) {
            Some(val) => vec![(self.label.clone(), python_round(val, 8))],
            None => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// ATR
// ---------------------------------------------------------------------------

pub struct AtrIndicator {
    pub label: String,
    period: usize,
    prev_close: Option<f64>,
    atr: Option<f64>,
    seed_sum: f64,
    seed_count: usize,
}

impl AtrIndicator {
    pub fn new(label: String, period: usize) -> Self {
        let p = period.max(1);
        Self {
            label,
            period: p,
            prev_close: None,
            atr: None,
            seed_sum: 0.0,
            seed_count: 0,
        }
    }

    fn push(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if self.prev_close.is_none() {
            high - low
        } else {
            let pc = self.prev_close.unwrap();
            f64::max(
                high - low,
                f64::max((high - pc).abs(), (low - pc).abs()),
            )
        };
        self.prev_close = Some(close);

        if self.atr.is_none() {
            if self.seed_count < self.period {
                self.seed_sum += tr;
                self.seed_count += 1;
                if self.seed_count == self.period {
                    self.atr = Some(self.seed_sum / self.period as f64);
                }
                return None;
            }
        } else {
            let a = self.atr.unwrap();
            self.atr = Some((a * (self.period - 1) as f64 + tr) / self.period as f64);
        }
        self.atr
    }

    pub fn push_candle(
        &mut self,
        _open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        match self.push(high, low, close) {
            Some(val) => vec![(self.label.clone(), python_round(val, 8))],
            None => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// BollingerBands
// ---------------------------------------------------------------------------

pub struct BollingerBandsIndicator {
    pub label: String,
    pub source: CandleSource,
    period: usize,
    multiplier: f64,
    buf: VecDeque<f64>,
    sum: f64,
    sumsq: f64,
}

impl BollingerBandsIndicator {
    pub fn new(label: String, period: usize, multiplier: f64, source: CandleSource) -> Self {
        let p = period.max(1);
        Self {
            label,
            source,
            period: p,
            multiplier,
            buf: VecDeque::with_capacity(p),
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    fn push(&mut self, value: f64) -> Option<(f64, f64, f64)> {
        if self.buf.len() == self.period {
            let old = self.buf.pop_front().unwrap();
            self.sum -= old;
            self.sumsq -= old * old;
        }
        self.buf.push_back(value);
        self.sum += value;
        self.sumsq += value * value;

        if self.buf.len() < self.period {
            return None;
        }

        let n = self.period as f64;
        let mean = self.sum / n;
        // Population variance (divide by n, not n-1)
        let var = f64::max(0.0, (self.sumsq - (self.sum * self.sum) / n) / n);
        let std = var.sqrt();
        let upper = mean + self.multiplier * std;
        let lower = mean - self.multiplier * std;
        Some((upper, mean, lower))
    }

    pub fn push_candle(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        let v = self.source.extract(open, high, low, close);
        match self.push(v) {
            Some((upper, mid, lower)) => vec![
                (format!("{}.upper", self.label), python_round(upper, 8)),
                (format!("{}.mid", self.label), python_round(mid, 8)),
                (format!("{}.lower", self.label), python_round(lower, 8)),
            ],
            None => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// MACD
// ---------------------------------------------------------------------------

pub struct MacdIndicator {
    pub label: String,
    pub source: CandleSource,
    fast_period: usize,
    slow_period: usize,
    sig_period: usize,
    alpha_f: f64,
    alpha_s: f64,
    alpha_g: f64,
    // Fast EMA state
    ema_f: Option<f64>,
    seed_f_sum: f64,
    seed_f_count: usize,
    // Slow EMA state
    ema_s: Option<f64>,
    seed_s_sum: f64,
    seed_s_count: usize,
    // Signal EMA state (seeded on MACD line values)
    ema_sig: Option<f64>,
    sig_seed_sum: f64,
    sig_seed_count: usize,
}

impl MacdIndicator {
    pub fn new(
        label: String,
        fast: usize,
        slow: usize,
        signal: usize,
        source: CandleSource,
    ) -> Self {
        let f = fast.max(1);
        let s = slow.max(1);
        let g = signal.max(1);
        Self {
            label,
            source,
            fast_period: f,
            slow_period: s,
            sig_period: g,
            alpha_f: 2.0 / (f as f64 + 1.0),
            alpha_s: 2.0 / (s as f64 + 1.0),
            alpha_g: 2.0 / (g as f64 + 1.0),
            ema_f: None,
            seed_f_sum: 0.0,
            seed_f_count: 0,
            ema_s: None,
            seed_s_sum: 0.0,
            seed_s_count: 0,
            ema_sig: None,
            sig_seed_sum: 0.0,
            sig_seed_count: 0,
        }
    }

    fn push(&mut self, v: f64) -> Option<(f64, f64, f64)> {
        // Seed or update fast EMA
        if self.ema_f.is_none() {
            if self.seed_f_count < self.fast_period {
                self.seed_f_sum += v;
                self.seed_f_count += 1;
                if self.seed_f_count == self.fast_period {
                    self.ema_f = Some(self.seed_f_sum / self.fast_period as f64);
                }
            }
        } else {
            let ef = self.ema_f.unwrap();
            self.ema_f = Some((v - ef) * self.alpha_f + ef);
        }

        // Seed or update slow EMA
        if self.ema_s.is_none() {
            if self.seed_s_count < self.slow_period {
                self.seed_s_sum += v;
                self.seed_s_count += 1;
                if self.seed_s_count == self.slow_period {
                    self.ema_s = Some(self.seed_s_sum / self.slow_period as f64);
                }
            }
        } else {
            let es = self.ema_s.unwrap();
            self.ema_s = Some((v - es) * self.alpha_s + es);
        }

        // Both EMAs needed for MACD line
        if self.ema_f.is_none() || self.ema_s.is_none() {
            return None;
        }

        let macd_line = self.ema_f.unwrap() - self.ema_s.unwrap();

        // Seed or update signal EMA on MACD line
        if self.ema_sig.is_none() {
            if self.sig_seed_count < self.sig_period {
                self.sig_seed_sum += macd_line;
                self.sig_seed_count += 1;
                if self.sig_seed_count == self.sig_period {
                    self.ema_sig = Some(self.sig_seed_sum / self.sig_period as f64);
                }
            }
            return None;
        } else {
            let es = self.ema_sig.unwrap();
            self.ema_sig = Some((macd_line - es) * self.alpha_g + es);
        }

        let signal = self.ema_sig.unwrap();
        let hist = macd_line - signal;
        Some((macd_line, signal, hist))
    }

    pub fn push_candle(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        let v = self.source.extract(open, high, low, close);
        match self.push(v) {
            Some((line, signal, hist)) => vec![
                (format!("{}.line", self.label), python_round(line, 8)),
                (format!("{}.signal", self.label), python_round(signal, 8)),
                (format!("{}.hist", self.label), python_round(hist, 8)),
            ],
            None => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// IndicatorType — dispatch enum for engine integration
// ---------------------------------------------------------------------------

pub enum IndicatorType {
    Sma(SmaIndicator),
    Ema(EmaIndicator),
    Rsi(RsiIndicator),
    Atr(AtrIndicator),
    BollingerBands(BollingerBandsIndicator),
    Macd(MacdIndicator),
}

impl IndicatorType {
    /// Push a candle and return key-value pairs of indicator outputs.
    /// Returns empty vec if indicator hasn't warmed up yet.
    pub fn push_candle(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> Vec<(String, f64)> {
        match self {
            Self::Sma(ind) => ind.push_candle(open, high, low, close),
            Self::Ema(ind) => ind.push_candle(open, high, low, close),
            Self::Rsi(ind) => ind.push_candle(open, high, low, close),
            Self::Atr(ind) => ind.push_candle(open, high, low, close),
            Self::BollingerBands(ind) => ind.push_candle(open, high, low, close),
            Self::Macd(ind) => ind.push_candle(open, high, low, close),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: push a slice of close values through a single-source indicator
    fn push_closes(
        ind: &mut IndicatorType,
        values: &[f64],
    ) -> Vec<Vec<(String, f64)>> {
        values
            .iter()
            .map(|&v| ind.push_candle(0.0, 0.0, 0.0, v))
            .collect()
    }

    // ── SMA ──────────────────────────────────────────────────────────────

    #[test]
    fn test_sma_basic() {
        let mut sma = SmaIndicator::new("sma3".into(), 3, CandleSource::Close);
        assert!(sma.push(10.0).is_none());
        assert!(sma.push(20.0).is_none());
        assert_eq!(sma.push(30.0), Some(20.0));
        assert_eq!(sma.push(40.0), Some(30.0));
        assert_eq!(sma.push(50.0), Some(40.0));
    }

    #[test]
    fn test_sma_rounding() {
        // SMA(3) of [1.0, 2.0, 3.1] = 6.1/3 = 2.0333333...
        let mut ind = IndicatorType::Sma(SmaIndicator::new("s".into(), 3, CandleSource::Close));
        let results = push_closes(&mut ind, &[1.0, 2.0, 3.1]);
        assert_eq!(results[0].len(), 0);
        assert_eq!(results[1].len(), 0);
        assert_eq!(results[2].len(), 1);
        assert_eq!(results[2][0].1, python_round(6.1 / 3.0, 8));
    }

    // ── EMA ──────────────────────────────────────────────────────────────

    #[test]
    fn test_ema_seeding_and_output() {
        // period=3, alpha=0.5
        let mut ema = EmaIndicator::new("ema3".into(), 3, CandleSource::Close);
        assert!(ema.push(10.0).is_none()); // seed 1
        assert!(ema.push(20.0).is_none()); // seed 2
        assert!(ema.push(30.0).is_none()); // seed 3 → ema=20.0, return None
        // Next push: ema = (40-20)*0.5 + 20 = 30.0
        assert_eq!(ema.push(40.0), Some(30.0));
        // ema = (50-30)*0.5 + 30 = 40.0
        assert_eq!(ema.push(50.0), Some(40.0));
    }

    #[test]
    fn test_ema_period_1() {
        let mut ema = EmaIndicator::new("ema1".into(), 1, CandleSource::Close);
        // period=1: seed with 1 value (returns None), then output on 2nd push
        assert!(ema.push(5.0).is_none()); // seed → ema=5.0, return None
        // alpha=2/2=1.0, so ema = (10-5)*1.0 + 5 = 10.0
        assert_eq!(ema.push(10.0), Some(10.0));
    }

    // ── RSI ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rsi_timing() {
        // period=3: first output at push #(period+2) = 5th value
        let mut rsi = RsiIndicator::new("rsi3".into(), 3, CandleSource::Close);
        assert!(rsi.push(44.0).is_none());   // prev set
        assert!(rsi.push(44.34).is_none());  // seed 1
        assert!(rsi.push(44.09).is_none());  // seed 2
        assert!(rsi.push(43.61).is_none());  // seed 3 → avg set, return None
        let val = rsi.push(44.33);           // first RSI output
        assert!(val.is_some());
        let rsi_val = val.unwrap();
        assert!(rsi_val > 60.0 && rsi_val < 70.0, "RSI={}", rsi_val);
    }

    #[test]
    fn test_rsi_all_up() {
        let mut rsi = RsiIndicator::new("rsi2".into(), 2, CandleSource::Close);
        rsi.push(10.0); // prev
        rsi.push(20.0); // seed 1 (gain=10, loss=0)
        rsi.push(30.0); // seed 2 (gain=10, loss=0) → avg_gain=10, avg_loss=0. None.
        let val = rsi.push(40.0); // Wilder: avg_loss stays 0 → RSI=100
        assert_eq!(val, Some(100.0));
    }

    // ── ATR ──────────────────────────────────────────────────────────────

    #[test]
    fn test_atr_basic() {
        let mut atr = AtrIndicator::new("atr2".into(), 2);
        // Candle 1: h=12, l=10, c=11. No prev_close → TR=2. seed 1.
        assert!(atr.push(12.0, 10.0, 11.0).is_none());
        // Candle 2: h=13, l=10, c=12. TR=3. seed 2 → atr=2.5. Returns None (seed completion).
        assert!(atr.push(13.0, 10.0, 12.0).is_none());
        // Candle 3: h=14, l=11, c=13. TR=3. atr=(2.5*1+3)/2=2.75. First output.
        assert_eq!(atr.push(14.0, 11.0, 13.0), Some(2.75));
        // Candle 4: h=15, l=12, c=14. TR=3. atr=(2.75*1+3)/2=2.875.
        assert_eq!(atr.push(15.0, 12.0, 14.0), Some(2.875));
    }

    // ── BollingerBands ──────────────────────────────────────────────────

    #[test]
    fn test_bb_basic() {
        // period=3, mult=2, values=[10, 20, 30]
        // mean = 20, pop_var = ((100+400+900) - 3600/3)/3 = (1400-1200)/3 = 200/3
        // std = sqrt(200/3) ≈ 8.16496..
        // upper = 20 + 2*8.16496.. = 36.32993..
        // lower = 20 - 2*8.16496.. = 3.67006..
        let mut bb = BollingerBandsIndicator::new("bb".into(), 3, 2.0, CandleSource::Close);
        assert!(bb.push(10.0).is_none());
        assert!(bb.push(20.0).is_none());
        let result = bb.push(30.0);
        assert!(result.is_some());
        let (upper, mid, lower) = result.unwrap();
        assert_eq!(mid, 20.0);
        assert!((upper - 36.32993162).abs() < 1e-6, "upper={}", upper);
        assert!((lower - 3.67006838).abs() < 1e-6, "lower={}", lower);
    }

    // ── MACD ─────────────────────────────────────────────────────────────

    #[test]
    fn test_macd_timing() {
        // fast=2, slow=3, signal=2
        // Push 1: seed both → None
        // Push 2: fast seeded (ema_f = avg of 2), slow seed 2 → None
        // Push 3: fast updates via EMA, slow seeded (ema_s = avg of 3) → MACD line exists.
        //         sig seed 1 → None
        // Push 4: MACD line. sig seed 2 → sig seeded. None.
        // Push 5: MACD line. sig updates → first output.
        let mut macd = MacdIndicator::new("m".into(), 2, 3, 2, CandleSource::Close);
        assert!(macd.push(10.0).is_none()); // 1
        assert!(macd.push(20.0).is_none()); // 2
        assert!(macd.push(30.0).is_none()); // 3 → sig seed 1
        assert!(macd.push(40.0).is_none()); // 4 → sig seed 2 → sig seeded, None
        let result = macd.push(50.0);       // 5 → first output
        assert!(result.is_some());
        let (line, signal, hist) = result.unwrap();
        // line = ema_f - ema_s
        // hist = line - signal
        assert!((hist - (line - signal)).abs() < 1e-12);
    }

    // ── IndicatorType dispatch ───────────────────────────────────────────

    #[test]
    fn test_indicator_type_sma() {
        let mut ind = IndicatorType::Sma(SmaIndicator::new("test_sma".into(), 2, CandleSource::Close));
        let r1 = ind.push_candle(0.0, 0.0, 0.0, 10.0);
        assert!(r1.is_empty());
        let r2 = ind.push_candle(0.0, 0.0, 0.0, 20.0);
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].0, "test_sma");
        assert_eq!(r2[0].1, 15.0);
    }

    #[test]
    fn test_indicator_type_bb_keys() {
        let mut ind = IndicatorType::BollingerBands(
            BollingerBandsIndicator::new("bb".into(), 2, 2.0, CandleSource::Close),
        );
        ind.push_candle(0.0, 0.0, 0.0, 10.0);
        let r = ind.push_candle(0.0, 0.0, 0.0, 20.0);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].0, "bb.upper");
        assert_eq!(r[1].0, "bb.mid");
        assert_eq!(r[2].0, "bb.lower");
    }

    #[test]
    fn test_indicator_type_macd_keys() {
        // fast=1, slow=1, signal=1 → outputs quickly
        let mut ind = IndicatorType::Macd(
            MacdIndicator::new("macd".into(), 1, 1, 1, CandleSource::Close),
        );
        // push 1: fast seeded, slow seeded → macd_line. sig seed 1 → sig seeded. None.
        let r1 = ind.push_candle(0.0, 0.0, 0.0, 10.0);
        assert!(r1.is_empty());
        // push 2: fast updated, slow updated, macd_line, sig updated → output
        let r2 = ind.push_candle(0.0, 0.0, 0.0, 20.0);
        assert_eq!(r2.len(), 3);
        assert_eq!(r2[0].0, "macd.line");
        assert_eq!(r2[1].0, "macd.signal");
        assert_eq!(r2[2].0, "macd.hist");
    }
}
