//! RL Reward Functions — Phase 6
//!
//! Ports: PnLReward, LogReturnReward, DifferentialSharpeReward,
//!        RiskAdjustedReward, SortinoReward, AdvancedReward.
//!
//! All internal state lives in Rust — no Python allocations per step.

// ---------------------------------------------------------------------------
// RewardFn — enum dispatching to concrete reward functions
// ---------------------------------------------------------------------------

pub enum RewardFn {
    PnL(PnLState),
    LogReturn,
    DifferentialSharpe(DifferentialSharpeState),
    RiskAdjusted(RiskAdjustedState),
    Sortino(SortinoState),
    Advanced(AdvancedState),
}

impl RewardFn {
    /// Compute reward given previous and current equity.
    /// `has_position` and `events` are used only by AdvancedReward.
    pub fn compute(
        &mut self,
        eq_prev: f64,
        eq_curr: f64,
        has_position: bool,
        had_tp: bool,
        had_sl: bool,
        had_liquidation: bool,
    ) -> f64 {
        match self {
            Self::PnL(s) => s.compute(eq_prev, eq_curr),
            Self::LogReturn => compute_log_return(eq_prev, eq_curr),
            Self::DifferentialSharpe(s) => s.compute(eq_prev, eq_curr),
            Self::RiskAdjusted(s) => s.compute(eq_prev, eq_curr),
            Self::Sortino(s) => s.compute(eq_prev, eq_curr),
            Self::Advanced(s) => {
                s.compute(eq_prev, eq_curr, has_position, had_tp, had_sl, had_liquidation)
            }
        }
    }

    /// Reset internal state (on env reset).
    pub fn reset(&mut self) {
        match self {
            Self::PnL(s) => s.reset(),
            Self::LogReturn => {}
            Self::DifferentialSharpe(s) => s.reset(),
            Self::RiskAdjusted(s) => s.reset(),
            Self::Sortino(s) => s.reset(),
            Self::Advanced(s) => s.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// PnLReward
// ---------------------------------------------------------------------------

pub struct PnLState {
    pub initial_equity: f64,
}

impl PnLState {
    pub fn new() -> Self {
        Self {
            initial_equity: 0.0,
        }
    }

    fn reset(&mut self) {
        self.initial_equity = 0.0;
    }

    fn compute(&mut self, eq_prev: f64, eq_curr: f64) -> f64 {
        if self.initial_equity == 0.0 {
            self.initial_equity = if eq_prev > 0.0 { eq_prev } else { 1.0 };
        }
        (eq_curr - eq_prev) / self.initial_equity.max(1e-10)
    }
}

// ---------------------------------------------------------------------------
// LogReturnReward
// ---------------------------------------------------------------------------

fn compute_log_return(eq_prev: f64, eq_curr: f64) -> f64 {
    const EPS: f64 = 1e-10;
    (eq_curr.max(EPS) / eq_prev.max(EPS)).ln()
}

// ---------------------------------------------------------------------------
// DifferentialSharpeReward
// ---------------------------------------------------------------------------

pub struct DifferentialSharpeState {
    pub eta: f64,
    pub scale: f64,
    pub a: f64, // EMA of returns
    pub b: f64, // EMA of squared returns
}

impl DifferentialSharpeState {
    pub fn new(eta: f64, scale: f64) -> Self {
        Self {
            eta,
            scale,
            a: 0.0,
            b: 0.0,
        }
    }

    fn reset(&mut self) {
        self.a = 0.0;
        self.b = 0.0;
    }

    fn compute(&mut self, eq_prev: f64, eq_curr: f64) -> f64 {
        if eq_prev <= 0.0 {
            return 0.0;
        }

        let r = eq_curr / eq_prev - 1.0;

        let delta_a = r - self.a;
        let delta_b = r * r - self.b;

        let denom = self.b - self.a * self.a;
        let ds = if denom <= 1e-12 {
            r
        } else {
            (self.b * delta_a - 0.5 * self.a * delta_b) / denom.powf(1.5)
        };

        self.a += self.eta * delta_a;
        self.b += self.eta * delta_b;

        (ds * self.scale).clamp(-10.0, 10.0)
    }
}

// ---------------------------------------------------------------------------
// RiskAdjustedReward
// ---------------------------------------------------------------------------

pub struct RiskAdjustedState {
    pub drawdown_penalty: f64,
    pub hwm: f64,
    pub prev_dd: f64,
    pub initial_equity: f64,
}

impl RiskAdjustedState {
    pub fn new(drawdown_penalty: f64) -> Self {
        Self {
            drawdown_penalty,
            hwm: 0.0,
            prev_dd: 0.0,
            initial_equity: 0.0,
        }
    }

    fn reset(&mut self) {
        self.hwm = 0.0;
        self.prev_dd = 0.0;
        self.initial_equity = 0.0;
    }

    fn compute(&mut self, eq_prev: f64, eq_curr: f64) -> f64 {
        if self.initial_equity == 0.0 {
            self.initial_equity = if eq_prev > 0.0 { eq_prev } else { 1.0 };
        }

        let ret = (eq_curr - eq_prev) / self.initial_equity.max(1e-10);

        if eq_curr > self.hwm {
            self.hwm = eq_curr;
        }
        let dd = if self.hwm <= 0.0 {
            0.0
        } else {
            (self.hwm - eq_curr) / self.hwm
        };
        let dd_increase = (dd - self.prev_dd).max(0.0);
        self.prev_dd = dd;

        (ret - self.drawdown_penalty * dd_increase).clamp(-10.0, 10.0)
    }
}

// ---------------------------------------------------------------------------
// SortinoReward
// ---------------------------------------------------------------------------

pub struct SortinoState {
    pub eta: f64,
    pub min_std: f64,
    pub ema_neg_sq: f64,
}

impl SortinoState {
    pub fn new(eta: f64, min_std: f64) -> Self {
        Self {
            eta,
            min_std,
            ema_neg_sq: 0.0,
        }
    }

    fn reset(&mut self) {
        self.ema_neg_sq = 0.0;
    }

    fn compute(&mut self, eq_prev: f64, eq_curr: f64) -> f64 {
        if eq_prev <= 0.0 {
            return 0.0;
        }

        let r = eq_curr / eq_prev - 1.0;
        let neg_r = r.min(0.0);
        self.ema_neg_sq += self.eta * (neg_r * neg_r - self.ema_neg_sq);

        let downside_std = self.ema_neg_sq.max(0.0).sqrt();
        let denom = downside_std.max(self.min_std);

        (r / denom).clamp(-10.0, 10.0)
    }
}

// ---------------------------------------------------------------------------
// AdvancedReward
// ---------------------------------------------------------------------------

pub struct AdvancedState {
    pub pnl_weight: f64,
    pub drawdown_penalty: f64,
    pub time_penalty: f64,
    pub sl_penalty: f64,
    pub tp_bonus: f64,
    pub liq_penalty: f64,
    pub hwm: f64,
    pub prev_dd: f64,
    pub initial_equity: f64,
}

impl AdvancedState {
    pub fn new(
        pnl_weight: f64,
        drawdown_penalty: f64,
        time_penalty: f64,
        sl_penalty: f64,
        tp_bonus: f64,
        liq_penalty: f64,
    ) -> Self {
        Self {
            pnl_weight,
            drawdown_penalty,
            time_penalty,
            sl_penalty,
            tp_bonus,
            liq_penalty,
            hwm: 0.0,
            prev_dd: 0.0,
            initial_equity: 0.0,
        }
    }

    fn reset(&mut self) {
        self.hwm = 0.0;
        self.prev_dd = 0.0;
        self.initial_equity = 0.0;
    }

    fn compute(
        &mut self,
        eq_prev: f64,
        eq_curr: f64,
        has_position: bool,
        had_tp: bool,
        had_sl: bool,
        had_liquidation: bool,
    ) -> f64 {
        if self.initial_equity == 0.0 {
            self.initial_equity = if eq_prev > 0.0 { eq_prev } else { 1.0 };
        }

        // 1. Base PnL
        let ret = (eq_curr - eq_prev) / self.initial_equity.max(1e-10);
        let mut reward = ret * self.pnl_weight;

        // 2. Drawdown penalty
        if eq_curr > self.hwm {
            self.hwm = eq_curr;
        }
        let dd = if self.hwm <= 0.0 {
            0.0
        } else {
            (self.hwm - eq_curr) / self.hwm
        };
        let dd_increase = (dd - self.prev_dd).max(0.0);
        self.prev_dd = dd;
        reward -= dd_increase * self.drawdown_penalty;

        // 3. Time penalty
        if has_position {
            reward -= self.time_penalty;
        }

        // 4. Event-driven
        if had_tp {
            reward += self.tp_bonus;
        }
        if had_sl {
            reward -= self.sl_penalty;
        }
        if had_liquidation {
            reward -= self.liq_penalty;
        }

        reward.clamp(-15.0, 15.0)
    }
}
