//! Forge Engine Rust Core
//!
//! This module exposes `RustTradingEngine` to Python via PyO3.
//! Phase 1: stub shell.  Phase 2: core types & exact rounding (types.rs).
//! Phase 3: engine math & state (engine.rs).  Phase 4+: full state machine.

pub mod types;
pub mod engine;
pub mod indicators;
pub mod data;
pub mod rl;
pub mod vectorized;

use pyo3::prelude::*;

use types::{r_price, r_qty, r_usd};

// ─────────────────────────────────────────────────────────────────────────────
// Exported rounding functions (for cross-validation and direct use)
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
fn py_r_price(x: f64) -> f64 {
    r_price(x)
}

#[pyfunction]
fn py_r_qty(x: f64) -> f64 {
    r_qty(x)
}

#[pyfunction]
fn py_r_usd(x: f64) -> f64 {
    r_usd(x)
}

/// The compiled Rust extension, importable as `forge_engine._rust_core`.
#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::RustTradingEngine>()?;
    m.add_class::<data::RustCandleDataAggregated>()?;
    m.add_function(wrap_pyfunction!(py_r_price, m)?)?;
    m.add_function(wrap_pyfunction!(py_r_qty, m)?)?;
    m.add_function(wrap_pyfunction!(py_r_usd, m)?)?;
    m.add_function(wrap_pyfunction!(vectorized::compute_indicators_bulk, m)?)?;
    m.add_function(wrap_pyfunction!(vectorized::run_signals_backtest, m)?)?;
    Ok(())
}
