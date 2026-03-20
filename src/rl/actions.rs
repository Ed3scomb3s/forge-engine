//! RL Action Spaces — Phase 6
//!
//! Translates f64 action values from Gymnasium into native engine order calls.
//! Ports: DiscreteActions, DiscreteActionsWithSizing, ContinuousActions.

use crate::engine::RustTradingEngine;
use crate::types::*;

// ---------------------------------------------------------------------------
// RlActionSpace — enum dispatching to the three concrete spaces
// ---------------------------------------------------------------------------

pub enum RlActionSpace {
    Discrete(DiscreteActionSpace),
    DiscreteWithSizing(DiscreteWithSizingSpace),
    Continuous(ContinuousActionSpace),
}

impl RlActionSpace {
    /// Translate a raw f64 action into engine order calls.
    /// `close_price` is extracted from the current candle close.
    pub fn translate(&self, engine: &mut RustTradingEngine, action_val: f64, close_price: f64) {
        match self {
            Self::Discrete(d) => d.translate(engine, action_val, close_price),
            Self::DiscreteWithSizing(d) => d.translate(engine, action_val, close_price),
            Self::Continuous(c) => c.translate(engine, action_val, close_price),
        }
    }

    /// Number of discrete actions (for Gymnasium space definition).
    pub fn num_actions(&self) -> usize {
        match self {
            Self::Discrete(d) => d.actions.len(),
            Self::DiscreteWithSizing(d) => d.actions.len(),
            Self::Continuous(_) => 1, // Box(1,)
        }
    }
}

// ---------------------------------------------------------------------------
// DiscreteActionSpace
// ---------------------------------------------------------------------------

/// Action names for discrete space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiscreteAction {
    Hold,
    OpenLong,
    OpenShort,
    Close,
}

impl DiscreteAction {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "hold" => Some(Self::Hold),
            "open_long" => Some(Self::OpenLong),
            "open_short" => Some(Self::OpenShort),
            "close" => Some(Self::Close),
            _ => None,
        }
    }
}

pub struct DiscreteActionSpace {
    pub actions: Vec<DiscreteAction>,
    pub margin_pct: f64,
    pub sl_pct: f64,
    pub tp_pct: f64,
}

impl DiscreteActionSpace {
    pub fn new(action_names: &[&str], margin_pct: f64, sl_pct: f64, tp_pct: f64) -> Self {
        let actions: Vec<DiscreteAction> = action_names
            .iter()
            .filter_map(|s| DiscreteAction::from_str(s))
            .collect();
        Self {
            actions,
            margin_pct,
            sl_pct,
            tp_pct,
        }
    }

    pub fn default() -> Self {
        Self::new(
            &["hold", "open_long", "open_short", "close"],
            0.1,
            0.0,
            0.0,
        )
    }

    fn translate(&self, engine: &mut RustTradingEngine, action_val: f64, close_price: f64) {
        let idx = action_val as usize;
        if idx >= self.actions.len() {
            return; // invalid -> hold
        }
        let action = self.actions[idx];
        let has_position = engine.position.is_some();

        match action {
            DiscreteAction::Hold => {}
            DiscreteAction::OpenLong => {
                if has_position {
                    return;
                }
                let sl = if self.sl_pct > 0.0 {
                    Some(close_price * (1.0 - self.sl_pct))
                } else {
                    None
                };
                let tp = if self.tp_pct > 0.0 {
                    Some(close_price * (1.0 + self.tp_pct))
                } else {
                    None
                };
                engine.create_order_internal(
                    OrderSide::LONG,
                    close_price,
                    self.margin_pct,
                    tp,
                    sl,
                );
            }
            DiscreteAction::OpenShort => {
                if has_position {
                    return;
                }
                let sl = if self.sl_pct > 0.0 {
                    Some(close_price * (1.0 + self.sl_pct))
                } else {
                    None
                };
                let tp = if self.tp_pct > 0.0 {
                    Some(close_price * (1.0 - self.tp_pct))
                } else {
                    None
                };
                engine.create_order_internal(
                    OrderSide::SHORT,
                    close_price,
                    self.margin_pct,
                    tp,
                    sl,
                );
            }
            DiscreteAction::Close => {
                if !has_position {
                    return;
                }
                engine.close_order_internal(close_price);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DiscreteWithSizingSpace
// ---------------------------------------------------------------------------

/// An action entry for discrete-with-sizing: side + margin_pct, or Hold/Close.
#[derive(Clone, Debug)]
pub enum SizedAction {
    Hold,
    OpenLong(f64),  // margin_pct
    OpenShort(f64), // margin_pct
    Close,
}

pub struct DiscreteWithSizingSpace {
    pub actions: Vec<SizedAction>,
    pub sl_pct: f64,
    pub tp_pct: f64,
}

impl DiscreteWithSizingSpace {
    /// Build from a list of (label, margin_pct) pairs, e.g. [("small", 0.05), ("medium", 0.1)].
    pub fn new(sizes: &[(&str, f64)], sl_pct: f64, tp_pct: f64) -> Self {
        let mut actions = vec![SizedAction::Hold];
        for (_label, mpct) in sizes {
            actions.push(SizedAction::OpenLong(*mpct));
            actions.push(SizedAction::OpenShort(*mpct));
        }
        actions.push(SizedAction::Close);
        Self {
            actions,
            sl_pct,
            tp_pct,
        }
    }

    pub fn default() -> Self {
        Self::new(
            &[("small", 0.05), ("medium", 0.1), ("large", 0.2)],
            0.0,
            0.0,
        )
    }

    fn translate(&self, engine: &mut RustTradingEngine, action_val: f64, close_price: f64) {
        let idx = action_val as usize;
        if idx >= self.actions.len() {
            return;
        }
        let has_position = engine.position.is_some();

        match &self.actions[idx] {
            SizedAction::Hold => {}
            SizedAction::Close => {
                if has_position {
                    engine.close_order_internal(close_price);
                }
            }
            SizedAction::OpenLong(margin_pct) => {
                if has_position {
                    return;
                }
                let sl = if self.sl_pct > 0.0 {
                    Some(close_price * (1.0 - self.sl_pct))
                } else {
                    None
                };
                let tp = if self.tp_pct > 0.0 {
                    Some(close_price * (1.0 + self.tp_pct))
                } else {
                    None
                };
                engine.create_order_internal(OrderSide::LONG, close_price, *margin_pct, tp, sl);
            }
            SizedAction::OpenShort(margin_pct) => {
                if has_position {
                    return;
                }
                let sl = if self.sl_pct > 0.0 {
                    Some(close_price * (1.0 + self.sl_pct))
                } else {
                    None
                };
                let tp = if self.tp_pct > 0.0 {
                    Some(close_price * (1.0 - self.tp_pct))
                } else {
                    None
                };
                engine.create_order_internal(OrderSide::SHORT, close_price, *margin_pct, tp, sl);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ContinuousActionSpace
// ---------------------------------------------------------------------------

pub struct ContinuousActionSpace {
    pub max_margin_pct: f64,
    pub threshold: f64,
    pub sl_pct: f64,
    pub tp_pct: f64,
}

impl ContinuousActionSpace {
    pub fn new(max_margin_pct: f64, threshold: f64, sl_pct: f64, tp_pct: f64) -> Self {
        Self {
            max_margin_pct,
            threshold,
            sl_pct,
            tp_pct,
        }
    }

    pub fn default() -> Self {
        Self::new(0.2, 0.1, 0.0, 0.0)
    }

    fn translate(&self, engine: &mut RustTradingEngine, action_val: f64, close_price: f64) {
        let signal = action_val.clamp(-1.0, 1.0);
        let has_position = engine.position.is_some();

        // Dead zone: do nothing (hold current state).
        // Positions exit via SL/TP or opposite-direction signal.
        if signal.abs() <= self.threshold {
            return;
        }

        if signal > self.threshold {
            // Want long
            if has_position {
                let is_short = engine
                    .position
                    .as_ref()
                    .map_or(false, |p| p.side == PositionSide::SHORT);
                if is_short {
                    engine.close_order_internal(close_price);
                }
                return; // already long or just closed short
            }
            let margin = signal.abs() * self.max_margin_pct;
            let sl = if self.sl_pct > 0.0 {
                Some(close_price * (1.0 - self.sl_pct))
            } else {
                None
            };
            let tp = if self.tp_pct > 0.0 {
                Some(close_price * (1.0 + self.tp_pct))
            } else {
                None
            };
            engine.create_order_internal(OrderSide::LONG, close_price, margin, tp, sl);
        } else {
            // signal < -threshold -> want short
            if has_position {
                let is_long = engine
                    .position
                    .as_ref()
                    .map_or(false, |p| p.side == PositionSide::LONG);
                if is_long {
                    engine.close_order_internal(close_price);
                }
                return; // already short or just closed long
            }
            let margin = signal.abs() * self.max_margin_pct;
            let sl = if self.sl_pct > 0.0 {
                Some(close_price * (1.0 + self.sl_pct))
            } else {
                None
            };
            let tp = if self.tp_pct > 0.0 {
                Some(close_price * (1.0 - self.tp_pct))
            } else {
                None
            };
            engine.create_order_internal(OrderSide::SHORT, close_price, margin, tp, sl);
        }
    }
}
